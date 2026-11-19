// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "GpuRefTypes.h"

using namespace gpu_ref;

extern "C" __global__ void BlockScaleDequantizeRef(BlockScaleDequantizeArgs args)
{
    const long long index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index >= args.elements)
    {
        return;
    }

    auto* x = static_cast<const X_TYPE*>(args.x);
    auto* scale = static_cast<const SCALE_TYPE*>(args.scale);
    auto* y = static_cast<Y_TYPE*>(args.y);

    long long remaining = index;
    long long xOffset = 0;
    long long yOffset = 0;
    long long scaleOffset = 0;
    for(int dim = args.nDim - 1; dim >= 0; --dim)
    {
        const long long coordinate = remaining % args.xDims[dim];
        remaining /= args.xDims[dim];
        xOffset += coordinate * args.xStrides[dim];
        yOffset += coordinate * args.yStrides[dim];
        scaleOffset += coordinate / args.effectiveBlockSize[dim] * args.scaleStrides[dim];
    }

    const auto xValue = toAccum(x[xOffset]);
    const auto scaleValue = toAccum(scale[scaleOffset]);
    const auto output
        = args.isNegativeScale != 0 ? xValue * exp2(-scaleValue) : xValue * scaleValue;
    y[yOffset] = fromAccum<Y_TYPE>(output);
}
