// Required CUDA headers
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>

// 单个 Warp 处理的 tile 维度
// Volta, Turing, Ampere 架构均支持
#define M 16
#define N 16
#define K 16

// 整个矩阵乘法的维度
#define WMMA_M 32
#define WMMA_N 32
#define WMMA_K 32

// 用于检查 CUDA 错误的辅助函数
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}


__global__ void wmma_32x32_gemm_kernel(half *a, half *b, float *c) {

    // 识别当前 Warp 负责计算 C 矩阵的哪一个 16x16 的 tile
    // 我们启动了 2x2 的 Warp 网格，因此有 warp 0, 1, 2, 3
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize / (WMMA_N / N);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize % (WMMA_N / N);

    // --- 这是安全的、Warp 级别的逻辑 ---
    // 整个 Warp 会计算出相同的基地址指针。
    // 尽管在 load/store 时每个线程的最终地址不同，
    // 但决定计算哪个 tile 的逻辑对于 Warp 内所有线程是统一的。

    // 指向当前 Warp 要计算的 C tile 左上角的指针
    float *c_tile_ptr = c + warpM * M * WMMA_N + warpN * N;

    // 指向第一次乘法所需的 A 和 B tiles 的指针
    half *a1_tile_ptr = a + warpM * M * WMMA_K;
    half *b1_tile_ptr = b + warpN * N;

    // 指向第二次（累加）乘法所需的 A 和 B tiles 的指针
    half *a2_tile_ptr = a + warpM * M * WMMA_K + K;
    half *b2_tile_ptr = b + K * WMMA_N + warpN * N;


    // --- WMMA 临界区开始 ---
    // Warp 内的所有线程必须严格执行相同的指令序列。
    
    // 1. 为矩阵 A, B 和累加器 C 声明分片 (fragment)
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, float> c_frag;

    // 2. 将累加器分片所有元素初始化为 0
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    // 3. 第一次乘法: C_tile = A1 * B1
    nvcuda::wmma::load_matrix_sync(a_frag, a1_tile_ptr, WMMA_K);
    nvcuda::wmma::load_matrix_sync(b_frag, b1_tile_ptr, WMMA_N);
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // 4. 第二次乘法并累加: C_tile = (A1 * B1) + A2 * B2
    // mma_sync 本身就是 D = A*B+C 的操作，结果会累加回 c_frag
    nvcuda::wmma::load_matrix_sync(a_frag, a2_tile_ptr, WMMA_K);
    nvcuda::wmma::load_matrix_sync(b_frag, b2_tile_ptr, WMMA_N);
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // 5. 将分片中的最终结果写回到全局内存
    nvcuda::wmma::store_matrix_sync(c_tile_ptr, c_frag, WMMA_N, nvcuda::wmma::mem_row_major);
    
    // --- WMMA 临界区结束 ---
}

// 用于运行 Kernel 的主机端 main 函数
int main() {
    // 分配主机内存
    half *h_a = new half[WMMA_M * WMMA_K];
    half *h_b = new half[WMMA_K * WMMA_N];
    float *h_c = new float[WMMA_M * WMMA_N];
    float *h_c_ref = new float[WMMA_M * WMMA_N](); // 初始化为 0

    // 初始化主机矩阵
    for (int i = 0; i < WMMA_M * WMMA_K; i++) h_a[i] = __float2half(1.0f);
    for (int i = 0; i < WMMA_K * WMMA_N; i++) h_b[i] = __float2half(1.0f);

    // 基于 CPU 的参考计算
    for (int i = 0; i < WMMA_M; i++) {
        for (int j = 0; j < WMMA_N; j++) {
            for (int l = 0; l < WMMA_K; l++) {
                h_c_ref[i * WMMA_N + j] += __half2float(h_a[i * WMMA_K + l]) * __half2float(h_b[l * WMMA_N + j]);
            }
        }
    }

    // 分配设备内存
    half *d_a, *d_b;
    float *d_c;
    checkCudaError(cudaMalloc(&d_a, sizeof(half) * WMMA_M * WMMA_K), "cudaMalloc d_a");
    checkCudaError(cudaMalloc(&d_b, sizeof(half) * WMMA_K * WMMA_N), "cudaMalloc d_b");
    checkCudaError(cudaMalloc(&d_c, sizeof(float) * WMMA_M * WMMA_N), "cudaMalloc d_c");

    // 从主机拷贝数据到设备
    checkCudaError(cudaMemcpy(d_a, h_a, sizeof(half) * WMMA_M * WMMA_K, cudaMemcpyHostToDevice), "cudaMemcpy d_a");
    checkCudaError(cudaMemcpy(d_b, h_b, sizeof(half) * WMMA_K * WMMA_N, cudaMemcpyHostToDevice), "cudaMemcpy d_b");

    // 我们需要 4 个 Warps (2x2) 来覆盖 32x32 的矩阵，即 4 * 32 = 128 个线程
    int num_threads = 128;
    int num_blocks = 1;

    // 启动 Kernel
    wmma_32x32_gemm_kernel<<<num_blocks, num_threads>>>(d_a, d_b, d_c);
    checkCudaError(cudaGetLastError(), "Kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // 从设备将结果拷贝回主机
    checkCudaError(cudaMemcpy(h_c, d_c, sizeof(float) * WMMA_M * WMMA_N, cudaMemcpyDeviceToHost), "cudaMemcpy d_c");

    // 验证结果
    bool success = true;
    for (int i = 0; i < WMMA_M * WMMA_N; i++) {
        if (abs(h_c[i] - h_c_ref[i]) > 1e-5) {
            std::cout << "Verification FAILED at index " << i << "! GPU result: " << h_c[i] << ", CPU ref: " << h_c_ref[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Verification PASSED!" << std::endl;
        std::cout << "Result matrix [0][0] = " << h_c[0] << " (Expected " << WMMA_K << ")" << std::endl;
    }

    // 释放内存
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_c_ref;
    checkCudaError(cudaFree(d_a), "cudaFree d_a");
    checkCudaError(cudaFree(d_b), "cudaFree d_b");
    checkCudaError(cudaFree(d_c), "cudaFree d_c");

    return 0;
}