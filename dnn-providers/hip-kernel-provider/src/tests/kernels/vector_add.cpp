// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// This kernel is used for testing in TestProgramAndKernel.cpp

extern "C" __global__ void vector_add(const FLOAT* a, const FLOAT* b, FLOAT* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}
