// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Skeleton kernel behind the pointwise-add descriptor pack: elementwise add over a
// single-element tensor.
//
// HIP_PLUGIN_POINTWISE_ADD_TYPE is the element type from the kernel descriptor's dtype
// metadata. HIP_PLUGIN_POINTWISE_ADD_BLOCK_SIZE is the descriptor's block size; unused
// by the kernel itself (one element needs one thread), but must reach the compiler for
// ranking and knob reporting.

extern "C" __global__ void PointwiseAdd(const HIP_PLUGIN_POINTWISE_ADD_TYPE* a,
                                        const HIP_PLUGIN_POINTWISE_ADD_TYPE* b,
                                        HIP_PLUGIN_POINTWISE_ADD_TYPE* c)
{
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        c[0] = a[0] + b[0];
    }
}
