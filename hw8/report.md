# task1

task1.cu just uses the navie matrix transpose in CUDA kernel. The ncu profiling result is as follow:

```
naive_cuda_transpose(int, const double *, double *), 2024-Sep-26 08:49:37, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio    sector/request                             32
  l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio    sector/request                              8
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                                                            0
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                                                            0
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                                                                0
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                                                                0
  l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                                request                        524,288
  l1tex__t_requests_pipe_lsu_mem_global_op_st.sum                                request                        524,288
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                                  sector                     16,777,216
  l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                                  sector                      4,194,304
  smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct                        %                             25
  smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct                        %                            100
  smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct                           %                              0
  ---------------------------------------------------------------------- --------------- ------------------------------
```

From the readme.md, the minimum load per request for this program is 256 Byte / 32 Byte = 8. This is because 32 threads in a warp need to access 32 float variables each of which is 8 Byte. One transaction is 32 Byte. Therefore the utilization percentage is 8 / 32 = 25%, meaning that 75% transaction is wasted.


# task2

task2.cu use shared memory to solve the load inefficiency problem. The ncu profiling result is as follow:

```
smem_cuda_transpose(int, const double *, double *), 2024-Sep-26 09:19:08, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio    sector/request                              8
  l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio    sector/request                              8
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                                                            0
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                                                   15,728,640
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                                                        1,048,576
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                                                       16,777,216
  l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                                request                        524,288
  l1tex__t_requests_pipe_lsu_mem_global_op_st.sum                                request                        524,288
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                                  sector                      4,194,304
  l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                                  sector                      4,194,304
  smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct                        %                            100
  smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct                        %                            100
  smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct                           %                          11.76
  ---------------------------------------------------------------------- --------------- ------------------------------
```

From the result, we can see that the load/store efficiency is 100 percent. However the st bank conflict number raises.

# task3

After solving the bank conflict problem, the ncu profiling result is as follow:

```
smem_cuda_transpose(int, const double *, double *), 2024-Sep-26 09:26:19, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio    sector/request                              8
  l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio    sector/request                              8
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                                                            0
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                                                            0
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                                                        1,048,576
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                                                        1,048,576
  l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                                request                        524,288
  l1tex__t_requests_pipe_lsu_mem_global_op_st.sum                                request                        524,288
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                                  sector                      4,194,304
  l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                                  sector                      4,194,304
  smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct                        %                            100
  smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct                        %                            100
  smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct                           %                            100
  ---------------------------------------------------------------------- --------------- ------------------------------

```

I need to explore the bank conflict problem more. This problem is interesting.




