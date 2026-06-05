// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// This kernel is used for testing in TestHipProgramAndKernel.cpp

extern "C" __global__ void vector_add(const float* a, const float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}
