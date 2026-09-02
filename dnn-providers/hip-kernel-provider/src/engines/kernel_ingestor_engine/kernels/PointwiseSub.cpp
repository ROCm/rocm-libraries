// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Elementwise subtract over a single-element tensor. Block size metadata is unused
// here but must reach the compiler for ranking and knob reporting.

extern "C" __global__ void PointwiseSub(const HIP_PLUGIN_POINTWISE_TYPE* a,
                                        const HIP_PLUGIN_POINTWISE_TYPE* b,
                                        HIP_PLUGIN_POINTWISE_TYPE* c)
{
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        c[0] = a[0] - b[0];
    }
}
