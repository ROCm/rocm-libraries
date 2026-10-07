// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Elementwise add over a single-element tensor, compiled ahead of time by the packager.
// Behaves like the add the shipped pointwise engine compiles, so its result is comparable
// against the same CPU reference.
//
// HIP_PLUGIN_POINTWISE_BLOCK_SIZE is unused here but must still reach the compiler: it is
// what makes the block-size variants distinct (source, build) inputs, and therefore distinct
// blobs under distinct toc_keys in the archive.
//
// The include is explicit: the packager drives clang directly with `-x hip` and no
// force-included runtime header, so blockIdx and threadIdx would otherwise be undeclared.

#include <hip/hip_runtime.h>

extern "C" __global__ void PointwiseAdd(const HIP_PLUGIN_POINTWISE_TYPE* a,
                                        const HIP_PLUGIN_POINTWISE_TYPE* b,
                                        HIP_PLUGIN_POINTWISE_TYPE* c)
{
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        c[0] = a[0] + b[0];
    }
}
