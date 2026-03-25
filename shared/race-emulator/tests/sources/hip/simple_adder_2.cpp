// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Compile for a target architecture with:
//   hipcc --cuda-device-only -S --offload-arch=<arch> -O2 simple_adder_2.cpp -o
//   ../<arch>/simple_adder_2.s
#include <hip/hip_runtime.h>

__global__ void simple_adder(float *c, float toAdd, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int elements_per_thread = 1;
  int start = idx * elements_per_thread;

  for (int i = 0; i < elements_per_thread; ++i) {
    int current_index = start + i;
    if (current_index < N) {
      c[current_index] += toAdd;
    }
  }
}
