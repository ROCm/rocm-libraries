// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// One dtype-agnostic source, compiled by hipRTC at prepare() time from a
// bundle dropped beside the descriptors. The two things that vary between
// variants -- element type and block size -- arrive as -D flags bound from
// each kernel's own metadata (`kernel_source.defines` in hiprtc_dropin.yaml),
// so two descriptors over THIS ONE FILE produce two distinct binaries with no
// native code and no rebuild.
//
// Nothing here is derived or conditional: the substituter replaces a token
// with a value and evaluates nothing. A kernel needing `block_size * 2` or a
// dtype-dependent choice gets it from the pack's dispatch handler, which is
// native code and therefore a rebuild.

#include "ScaleAddCommon.h"

extern "C" __global__ void ScaleAddHiprtc(const ScaleAddElement* x,
                                          const ScaleAddElement* b,
                                          ScaleAddElement* y,
                                          ScaleAddElement a,
                                          unsigned int count)
{
    const unsigned int index = blockIdx.x * SCALE_ADD_BLOCK_SIZE + threadIdx.x;
    if(index < count)
    {
        y[index] = a * x[index] + b[index];
    }
}
