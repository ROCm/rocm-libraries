// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>

extern "C" __global__ void HsacoFixtureAdd(const float* a, const float* b, float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        c[i] = a[i] + b[i];
}

extern "C" __global__ void HsacoFixtureScale(float* x, float s, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        x[i] *= s;
}
