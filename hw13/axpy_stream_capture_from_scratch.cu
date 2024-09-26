#include <stdio.h>
#include <cuda_runtime_api.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#define N 500000

// Simple short kernels
__global__
void kernel_a(float * x, float * y){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = 2.0*x[idx] + y[idx];

}

__global__
void kernel_b(float * x, float * y){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = 2.0*x[idx] + y[idx];

}

__global__
void kernel_c(float * x, float * y){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = 2.0*x[idx] + y[idx];

}

__global__
void kernel_d(float * x, float * y){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = 2.0*x[idx] + y[idx];

}

int main(){

// Set up and create events
cudaEvent_t event1;
cudaEvent_t event2;

cudaEventCreateWithFlags(&event1, cudaEventDisableTiming);
cudaEventCreateWithFlags(&event2, cudaEventDisableTiming);

// Set up and create streams
const int num_streams = 2;

cudaStream_t streams[num_streams];

for (int i = 0; i < num_streams; ++i){
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
}

// Set up and initialize host data
float* h_x;
float* h_y;

h_x = (float*) malloc(N * sizeof(float));
h_y = (float*) malloc(N * sizeof(float));

for (int i = 0; i < N; ++i){
    h_x[i] = (float)i;
    h_y[i] = (float)i;
//    printf("%2.0f ", h_x[i]);
}
printf("\n");

// Set up device data
float* d_x;
float* d_y;

cudaMalloc((void**) &d_x, N * sizeof(float));
cudaMalloc((void**) &d_y, N * sizeof(float));
cudaCheckErrors("cudaMalloc failed");

cudaMemcpy(d_x, h_x, N, cudaMemcpyHostToDevice);
cudaMemcpy(d_y, h_y, N, cudaMemcpyHostToDevice);
cudaCheckErrors("cudaMalloc failed");

int threads = 512;
int blocks = (N + (threads - 1) / threads);

// Launching work
for (int i = 0; i < 100; ++i){
    kernel_a<<<blocks, threads, 0, streams[0]>>>(d_x, d_y);
    cudaCheckErrors("Kernel a failed");

    cudaEventRecord(event1, streams[0]);
    cudaCheckErrors("Event record failed");

    kernel_b<<<blocks, threads, 0, streams[0]>>>(d_x, d_y);
    cudaCheckErrors("Kernel b failed");

    cudaStreamWaitEvent(streams[1], event1);
    cudaCheckErrors("Event wait failed");

    kernel_c<<<blocks, threads, 0, streams[1]>>>(d_x, d_y);
    cudaCheckErrors("Kernel c failed");

    cudaEventRecord(event2, streams[1]);
    cudaCheckErrors("Event record failed");

    cudaStreamWaitEvent(streams[0], event2);
    cudaCheckErrors("Event wait failed");

    kernel_d<<<blocks, threads, 0, streams[0]>>>(d_x, d_y);
    cudaCheckErrors("Kernel d failed");

    cudaStreamSynchronize(streams[0]);
}

// Copy data back to host
cudaMemcpy(h_y, d_y, N, cudaMemcpyDeviceToHost);
cudaCheckErrors("Finishing memcpy failed");

cudaDeviceSynchronize();

// Print out the first 25 values of h_y
for (int i = 0; i < 25; ++i){
    printf("%2.0f ", h_y[i]);
}
printf("\n");

return 0;
}
