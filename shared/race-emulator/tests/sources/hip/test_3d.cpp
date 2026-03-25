// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Compile for a target architecture with:
//   hipcc --cuda-device-only -S --offload-arch=<arch> -O2 test_3d.cpp -o
//   ../<arch>/test_3d.s
#include <hip/hip_runtime.h>

__global__ void test_3d_kernel(float *output) {
  int flat_idx = threadIdx.x + (threadIdx.y * blockDim.x) +
                 (threadIdx.z * blockDim.x * blockDim.y);
  output[flat_idx] = flat_idx + 1.0f;
}
