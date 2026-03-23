// Compile for a target architecture with:
//   hipcc --cuda-device-only -S --offload-arch=<arch> -O2 simple_adder_f16.cpp
//   -o ../<arch>/simple_adder_f16.s
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

__global__ void simple_adder_f16(__half *c, float toAdd, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    c[idx] = __float2half(__half2float(c[idx]) + toAdd);
  }
}
