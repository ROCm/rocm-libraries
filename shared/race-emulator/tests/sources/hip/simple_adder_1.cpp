// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Compile for a target architecture with:
//   hipcc --cuda-device-only -S --offload-arch=<arch> -O2 simple_adder_1.cpp -o
//   ../<arch>/simple_adder_1.s
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

__global__ void simple_adder(hip_bfloat16 *c, float toAdd, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int elements_per_thread = 1;
  int start = idx * elements_per_thread;

  for (int i = 0; i < elements_per_thread; ++i) {
    int current_index = start + i;
    if (current_index < N) {
      c[current_index] += hip_bfloat16(toAdd);
    }
  }
}
