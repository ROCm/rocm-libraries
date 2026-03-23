// Compile for a target architecture with:
//   hipcc --cuda-device-only -S --offload-arch=<arch> -O2 lds_reverse.cpp -o
//   ../<arch>/lds_reverse_2.s
#include <hip/hip_runtime.h>

__global__ void lds_reverse(int *data) {
  __shared__ int temp[256];
  int tid = threadIdx.x;
  temp[tid] = data[threadIdx.x];
  __syncthreads();
  data[tid] = temp[256 - tid - 1];
}
