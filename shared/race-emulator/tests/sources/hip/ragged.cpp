// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Compile for a target architecture with:
//   hipcc --cuda-device-only -S --offload-arch=<arch> -O2 ragged.cpp -o
//   ../<arch>/ragged.s
#include <hip/hip_runtime.h>

__global__ void ragged_kernel(int *data, const int *limits) {
  int tid = threadIdx.x;
  int limit = limits[tid];
  int sum = 0;
  for (int j = 0; j < limit; ++j) {
    sum += j;
  }
  data[tid] = sum;
}
