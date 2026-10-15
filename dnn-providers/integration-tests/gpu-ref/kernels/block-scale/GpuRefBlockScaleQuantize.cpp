// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "GpuRefTypes.h"

using namespace gpu_ref;

extern "C" __global__ void BlockScaleQuantizeRef(BlockScaleQuantizeArgs args)
{
    const long long scaleIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if(scaleIndex >= args.scaleElements)
    {
        return;
    }

    auto* x = static_cast<const X_TYPE*>(args.x);
    auto* y = static_cast<Y_TYPE*>(args.y);
    auto* scale = static_cast<SCALE_TYPE*>(args.scale);

    long long coordinates[5] = {};
    long long remaining = scaleIndex;
    long long scaleOffset = 0;
    for(int dim = args.nDim - 1; dim >= 0; --dim)
    {
        coordinates[dim] = remaining % args.scaleDims[dim];
        remaining /= args.scaleDims[dim];
        scaleOffset += coordinates[dim] * args.scaleStrides[dim];
    }

    const long long blockStart = coordinates[args.axis] * args.blockSize;
    const long long blockEnd = min(blockStart + args.blockSize, args.xDims[args.axis]);
    COMPUTE_TYPE maxAbs = COMPUTE_TYPE{0};
    long long xOffset = 0;
    long long yOffset = 0;
    for(int dim = 0; dim < args.nDim; ++dim)
    {
        if(dim != args.axis)
        {
            xOffset += coordinates[dim] * args.xStrides[dim];
            yOffset += coordinates[dim] * args.yStrides[dim];
        }
    }

    for(long long element = blockStart; element < blockEnd; ++element)
    {
        const auto value = toAccum(x[xOffset + element * args.xStrides[args.axis]]);
        maxAbs = max(maxAbs, static_cast<COMPUTE_TYPE>(fabs(value)));
    }

    auto scaleValue = static_cast<COMPUTE_TYPE>(maxAbs / args.outputMax);
    SCALE_TYPE storedScale = roundScaleUp<SCALE_TYPE>(static_cast<float>(scaleValue));
    scale[scaleOffset] = storedScale;

    const auto computeScale = static_cast<COMPUTE_TYPE>(storedScale);
    for(long long element = blockStart; element < blockEnd; ++element)
    {
        const auto input = toAccum(x[xOffset + element * args.xStrides[args.axis]]);
        auto output = computeScale != COMPUTE_TYPE{0} ? input / computeScale : COMPUTE_TYPE{0};
        output = max(static_cast<COMPUTE_TYPE>(args.outputLowest),
                     min(output, static_cast<COMPUTE_TYPE>(args.outputMax)));
        if constexpr(__is_integral(Y_TYPE))
        {
            output = nearbyint(output);
        }
        y[yOffset + element * args.yStrides[args.axis]] = static_cast<Y_TYPE>(output);
    }
}
