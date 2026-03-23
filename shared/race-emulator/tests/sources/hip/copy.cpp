// Compile for a target architecture with:
//   hipcc --cuda-device-only -S --offload-arch=<arch> -O2 copy.cpp -o
//   ../<arch>/copy.s
#include <hip/hip_runtime.h>

__global__ void copy_kernel(int *out, const int *in) {
  int i = threadIdx.x;
  out[i] = in[i];
}
