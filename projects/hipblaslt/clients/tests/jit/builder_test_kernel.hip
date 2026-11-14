// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>

extern "C" __global__ void hipblaslt_jit_builder_test_scale(float* x, float a, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        x[i] *= a;
}
