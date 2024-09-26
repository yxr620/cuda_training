# 1. reduction profiling

The first task is compiling and profiling the reduction.cu program. Using the following command to compile and profile.

```bash
nvcc -o reduction reduction.cu
sudo /usr/local/cuda/bin/ncu ./reduction
```
On my machine, I get the following result.

```
atomic_red(const float *, float *)
Duration    msecond 19.51

reduce_a(float *, float *)
Duration    usecond 78.85

reduce_ws(float *, float *)
Duration    usecond 78.50
```
The result shows that both `reduce_a` and `reduce_ws` kernel's execution time is less than `atomic_red`. For the `reduce_a` and `reduce_ws` kernels, they use a tree structure of to get the sum result. `atomic_red` kernel use the atomic add operation which involes more waiting for each thread. The reason that `atomic_red` and `reduce_a` kernels cost almost same time may be that the grid stripe loop is the main overhead of the kernel which is the same for two kernels.

The N is 8M in this program. Therefore, the bandwidth is $ 8MB / 78.85 us = 101 GB/s$.

If we change the N from 8M to 163840 (256*640), we get the following results.

```
atomic_red(const float *, float *)
Duration    usecond 383.33

reduce_a(float *, float *)
Duration    usecond 7.17

reduce_ws(float *, float *)
Duration    usecond 5.38
```
This time the bandwidth calculated by `reduce_ws` kernel is 30.45 GB/s which means that the program has not reached the GPU memory bandwidth. 

If we change 8M to 32M, we get the follwoing results. The error occur because we can not using +1 from 0 to 32M on float format. The precision of the float format does not support +1 on big number (over 16777216.000000).

```
atomic sum reduction incorrect!

(the reduce_a is correct)
(the reduce_ws is correct)
```

# 2. change sum to max

We change the reduce kernel from sum to max and delete the atomicadd operation. The program will launch the kernel two times to get the max result. The kernel is as follow:

```c
__global__ void reduce(float *gdata, float *out, size_t n){
  __shared__ float sdata[BLOCK_SIZE];
  int tid = threadIdx.x;
  sdata[tid] = 0.0f;
  size_t idx = threadIdx.x+blockDim.x*blockIdx.x;

  while (idx < n) {  // grid stride loop to load data
    // max
    sdata[tid] = max(sdata[tid], gdata[idx]);
    // sdata[tid] += gdata[idx];
    idx += gridDim.x*blockDim.x;  
    }

  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    __syncthreads();
    if (tid < s)  // parallel sweep reduction
      sdata[tid] = max(sdata[tid], sdata[tid + s]);
      // sdata[tid] += sdata[tid + s];
    }
  if (tid == 0) out[blockIdx.x] = sdata[0];
  }
```
The kernel does not need to use atomic operations to reduce results from defferent blocks but use a second kernel launch to reduce the results.

# 3. row_sum

In previous homework4, the `row_sums` kernels cost more time then `col_rows` kernels. The ncu profiling result is as follows:

```
row_sums(const float *, float *, unsigned long)
Duration    msecond  6.91

column_sums(const float *, float *, unsigned long)
Duration    msecond  2.28
```

In order to speedup the `row_sums` kernels, we modify the kernel to `row_sums_mod` as follows:

```c
__global__ void row_sums_mod(const float *A, float *sums, size_t ds){

  // int idx = threadIdx.x+blockDim.x*blockIdx.x; // create typical 1D thread index from built-in variables
  int bidx = blockIdx.x;
  int tid = threadIdx.x;
  __shared__ float sdata[block_size];

  // move element in one ine to sdata
  size_t start = 0;
  sdata[tid] = 0.0f;
  while(tid + start < ds) {
    sdata[tid] += A[bidx * ds + tid + start];
    start += blockDim.x;
  }

  for (unsigned int s = blockDim.x / 2; s > 0; s >>=1) {
    __syncthreads();
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
  }
  if (tid == 0) atomicAdd(&sums[bidx], sdata[0]);
}
```
In the `row_sums_mod` kernel, every block handles the sum of one row. The Therefore the kernel launch code needs to be modified as follow:

```c
row_sums_mod<<<DSIZE, block_size>>>(d_A, d_sums, DSIZE);
```

The result of row_sums_mod kernel is as follow:

```
row_sums(const float *, float *, unsigned long)
Duration    msecond  1.92

column_sums(const float *, float *, unsigned long)
Duration    msecond  2.28
```

The `row_sums_mod` kernel even became faster than the `column_sums` kernel.