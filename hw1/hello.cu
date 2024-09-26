#include <stdio.h>

__global__ void hello(){
  uint blockid = blockIdx.x;
  uint threadid = threadIdx.x;
  printf("Hello from block: %u, thread: %u\n", blockid, threadid);
}

int main(){

  hello<<<2, 2>>>();
  cudaDeviceSynchronize();

  return 0;
}

