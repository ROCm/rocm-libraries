// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GPU kernel for element-wise ReLU forward activation.
// Compiled at runtime via HIPRTC and launched by ReluPlan.

#include "IndexType.hpp"

extern "C" __global__ void relu_forward_kernel(const float* input,
                                               float* output,
                                               unsigned int numElements,
                                               float negativeSlope)
{
    IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < numElements)
    {
        float val = input[idx];
        output[idx] = (val >= 0.0f) ? val : negativeSlope * val;
    }
}
