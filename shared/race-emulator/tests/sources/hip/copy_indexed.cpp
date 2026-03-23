// Compile for a target architecture with:
//   hipcc --cuda-device-only -S --offload-arch=<arch> -O2 copy_indexed.cpp -o
//   ../<arch>/copy_indexed.s
#include <hip/hip_runtime.h>

__global__ void copy_kernel(int *out, const int *in) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = in[i];
}
