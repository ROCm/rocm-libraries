// Compile for a target architecture with:
//   hipcc --cuda-device-only -S --offload-arch=<arch> -O2 double_evens.cpp -o
//   ../<arch>/double_evens.s
#include <hip/hip_runtime.h>

__global__ void double_evens(float *c, float scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int elements_per_thread = 32;
  int start = idx * elements_per_thread;
  for (int i = 0; i < elements_per_thread; ++i) {
    int current_index = start + i;
    if (current_index % 2 == 0) {
      c[current_index] *= scale;
    }
  }
}
