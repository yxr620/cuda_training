# 2. array increment

a.

The original kernel under nsys profiling looks like follows:

```
Total Time (ns)            Name           
---------------  -------------------------
        555,393  inc(int *, unsigned long)


 Time (%)  Total Time (ns)  Count      Operation     
 --------  ---------------  -----  ------------------
     50.1       18,846,150      1  [CUDA memcpy HtoD]
     49.9       18,773,318      1  [CUDA memcpy DtoH]
```

b. 

After changing the cudaMalloc to cudaMallocManaged, we get the following result for kernel execution time. The profiling result also shows that the DtoH and HtoD operation counts increase dramatically.

```
Total Time (ns)            Name           
---------------  -------------------------
     44,397,382  inc(int *, unsigned long)

 Time (%)  Total Time (ns)  Count             Operation            
 --------  ---------------  ----- ---------------------------------
     57.1       14,633,687  2,003 [CUDA Unified Memory memcpy HtoD]
     42.9       11,014,560    768 [CUDA Unified Memory memcpy DtoH]

```

c.

After adding the prefecting operation to the program, we get the following result. The result shows that the kernel execution time return to normal as (a). Besides the DtoH and HtoD memory operation also decrease.

```
 Time (%)  Total Time (ns)            Name           
 --------  ---------------  -------------------------
    100.0          546,560  inc(int *, unsigned long)

 Time (%)  Total Time (ns)  Count              Operation            
 --------  ---------------  -----  ---------------------------------
     57.9       17,368,884     64  [CUDA Unified Memory memcpy DtoH]
     42.1       12,623,146     64  [CUDA Unified Memory memcpy HtoD]
```


