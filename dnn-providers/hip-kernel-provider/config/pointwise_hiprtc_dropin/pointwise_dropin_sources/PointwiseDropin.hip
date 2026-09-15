// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// One source file, two kernel descriptors. The element type is not written here: it
// arrives as -DHIPDNN_DROPIN_DTYPE=<FLOAT|HALF>, bound from each descriptor's own
// metadata.dtype by the `$kernel.dtype` token in kernel_source.defines, and the bundle's
// header turns that tag into a type.
//
// The argument list is the pointwise dispatch handler's, verbatim: PointwiseDispatchHandler
// ::launch passes exactly three pointers in (inputA, inputB, output) order, and grid/block
// geometry is the handler's too. Wrong arity is diagnosed nowhere, so this list is derived
// from that launch body and from nothing else.

#include "PointwiseDropinTypes.h"

extern "C" __global__ void PointwiseDropin(const DropinElement* a,
                                           const DropinElement* b,
                                           DropinElement* c)
{
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        c[0] = a[0] + b[0];
    }
}
