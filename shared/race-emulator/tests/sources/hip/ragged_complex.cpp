// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Compile for a target architecture with:
//   hipcc --cuda-device-only -S --offload-arch=<arch> -O2 ragged_complex.cpp -o
//   ../<arch>/ragged_complex.s
//
// Like ragged.cpp but with a more complex loop body that prevents
// the compiler from replacing the loop with a closed-form expression.
// This exercises divergent control flow: each thread iterates a
// different number of times based on limits[tid].
#include <hip/hip_runtime.h>

__global__ void ragged_complex_kernel(int *data, const int *limits) {
  int tid = threadIdx.x;
  int limit = limits[tid];
  int sum = 0;
  for (int j = 0; j < limit; ++j) {
    sum += j * j + 2 * j + j % 3;
  }
  data[tid] = sum;
}
