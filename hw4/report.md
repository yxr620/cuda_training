# Code completing

For `row_sums` function, we need to add the elements in one row and assign the result in d_sums. The complete kernel function looks like follow:

```c
__global__ void row_sums(const float *A, float *sums, size_t ds){
  int idx = threadIdx.x + blockIdx.x * blockDim.x; // create typical 1D thread index from built-in variables
  if (idx < ds){
    float sum = 0.0f;
    for (size_t i = 0; i < ds; i++)
      sum += A[idx * ds + i];
    sums[idx] = sum;
}}
```
For `column_sums` function, we need to add the elements in one column and assign the result in d_sums. The complete kernel function looks like follow:


```c
__global__ void column_sums(const float *A, float *sums, size_t ds){
  int idx = threadIdx.x + blockIdx.x * blockDim.x; // create typical 1D thread index from built-in variables
  if (idx < ds){
    float sum = 0.0f;
    for (size_t i = 0; i < ds; i++)
      sum += A[i * ds + idx];
    sums[idx] = sum;
}}
```
# Profiling

Firstly, using command `ncu ./mat_sum` to profile the overall metrics of two kernels. From the results shown below it is obvious that `column_sums` is much faster than `row_sums`

```
row_sums(const float *, float *, unsigned long)
Duration    msecond  6.91

column_sums(const float *, float *, unsigned long)
Duration    msecond  2.28
```

When we use ncu to profile two kernels from a cache loading perspective, we get the following result:

```

row_sums(const float *, float *, unsigned long), 2024-Aug-27 08:24:37, Context 1, Stream 7
  Section: Command line profiler metrics
  ----------------------------------------------- ------- -----------
  l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum request   8,388,608
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum   sector 268,414,598
  ----------------------------------------------- ------- -----------
column_sums(const float *, float *, unsigned long), 2024-Aug-27 08:24:38, Context 1, Stream 7
  Section: Command line profiler metrics
  ----------------------------------------------- ------- ----------
  l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum request  8,388,608
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum   sector 33,554,432
  ----------------------------------------------- ------- ----------
```

From the result it is obvious that `column_sums` load much fewer sectors than `row_sums` under same request number, i.e. 8,388,608. This means that `column_sums` requires less data loading time from global memory to finish the calculation. From the coding perspective, in `column_sums` the 32 threads from one warp visit 32 * 4 Byte continues memory at a time. In contrast, 32 threads iin one warp in `row_sums` visit 32 * 4 Byte in a range of 32 * ds * 4 Byte (32 * 16384 * 4 Byte in this case). Therefore, differentiating from the single thread scenario, `column_sums` work faster in CUDA programming interface.
