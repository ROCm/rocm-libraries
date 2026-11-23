// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Two entry points in one translation unit. Used in TestLaunchKernelDefinition.cpp to launch
// both out of one compiled program through entryPointKernelDefinition.

extern "C" __global__ void two_entry_points_fill(int* out, int value)
{
    out[threadIdx.x] = value;
}

extern "C" __global__ void two_entry_points_add(int* out, int value)
{
    out[threadIdx.x] += value;
}
