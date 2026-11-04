// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GPU reference resample backward kernel.
// Compiled via HipRTC with -DDY_TYPE=<type> -DDX_TYPE=<type> -DCOMPUTE_TYPE=<type>
// -DINDEX_TYPE=<type> -DSPATIAL_DIMS=<2|3> -DRESAMPLE_MODE=<mode> -DPADDING_MODE=<mode>.
// Each thread computes one dx element by iterating over the resample window in the dy tensor
// and applying the backward resample operation to produce the dx element.

#include "GpuRefTypes.h"
#include "ResampleCommon.hpp"

using namespace gpu_ref;

__device__ __forceinline__ long long validCount1D(
    long long out, long long stride, long long prePad, long long window, long long dxDim)
{
    long long count = 0;
    for(long long k = 0; k < window; ++k)
    {
        const long long in = out * stride + k - prePad;
        if(in >= 0 && in < dxDim)
        {
            ++count;
        }
    }
    return count;
}

extern "C" __global__ void ResampleBwdRef(ResampleBwdArgs args)
{
    const long long tid = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                          + static_cast<long long>(threadIdx.x);
    constexpr long long spatialDims = static_cast<long long>(SPATIAL_DIMS);

    long long dxSpatialSize = 1;
    for(long long i = 0; i < spatialDims; ++i)
    {
        dxSpatialSize *= args.dxSpatialDims[i];
    }

    if(tid >= args.n * args.c * dxSpatialSize)
    {
        return;
    }

    auto* dy = static_cast<const DY_TYPE*>(args.dy);
    [[maybe_unused]] auto* index = static_cast<const INDEX_TYPE*>(args.index);
    auto* dx = static_cast<DX_TYPE*>(args.dx);

    constexpr ResampleMode resampleMode = static_cast<ResampleMode>(RESAMPLE_MODE);

    // Unpack spatial arrays
    long long dySpatial[3] = {1, 1, 1};
    long long dxSpatial[3] = {1, 1, 1};
    long long window[3] = {1, 1, 1};
    long long stride[3] = {1, 1, 1};
    long long prePad[3] = {0, 0, 0};

    const long long offset = (spatialDims == 2) ? 1 : 0;
    for(long long i = 0; i < spatialDims; ++i)
    {
        dySpatial[offset + i] = args.dySpatialDims[i];
        dxSpatial[offset + i] = args.dxSpatialDims[i];
        window[offset + i] = args.window[i];
        stride[offset + i] = args.stride[i];
        prePad[offset + i] = args.prePadding[i];
    }

    // Compute the dx indices for this thread
    long long remaining = tid;
    const long long dxW = remaining % dxSpatial[2];
    remaining /= dxSpatial[2];
    const long long dxH = remaining % dxSpatial[1];
    remaining /= dxSpatial[1];

    long long dxD = 0;
    if constexpr(spatialDims == 3)
    {
        dxD = remaining % dxSpatial[0];
        remaining /= dxSpatial[0];
    }

    const long long c = remaining % args.c;
    remaining /= args.c;
    const long long n = remaining;

    [[maybe_unused]] const INDEX_TYPE currentFlattenedIndex
        = flattenSpatialIndex<INDEX_TYPE, spatialDims>(dxD, dxH, dxW, dxSpatial[1], dxSpatial[2]);
    COMPUTE_TYPE result = static_cast<COMPUTE_TYPE>(0);

    // Iterate over the window and accumulate the contribution of every
    // dy element included in the window of the current dx element
    for(long long kd = 0; kd < window[0]; ++kd)
    {
        const long long numD = dxD + prePad[0] - kd;
        const long long outD = numD / stride[0];
        const bool validD
            = (spatialDims == 2) || (numD >= 0 && outD < dySpatial[0] && (numD % stride[0]) == 0);
        for(long long kh = 0; kh < window[1]; ++kh)
        {
            const long long numH = dxH + prePad[1] - kh;
            const long long outH = numH / stride[1];
            const bool validH = (numH >= 0 && outH < dySpatial[1] && (numH % stride[1]) == 0);
            for(long long kw = 0; kw < window[2]; ++kw)
            {
                const long long numW = dxW + prePad[2] - kw;
                const long long outW = numW / stride[2];
                const bool validW = (numW >= 0 && outW < dySpatial[2] && (numW % stride[2]) == 0);

                // Skip if this dx coordinate does not map to a valid dy coordinate
                if(!(validD && validH && validW))
                {
                    continue;
                }

                const long long dyOffset
                    = getOffset<spatialDims>(n, c, outD, outH, outW, args.dyStrides);

                if constexpr(resampleMode == ResampleMode::MAXPOOL)
                {
                    // Only the element selected as max in the forward pass receives gradient
                    if(index[dyOffset] != currentFlattenedIndex)
                    {
                        continue;
                    }
                    result += toAccum(dy[dyOffset]);
                }
                else
                {
                    long long divisor;
                    if constexpr(resampleMode == ResampleMode::AVGPOOL_EXCLUDE_PADDING)
                    {
                        // Number of valid elements in the pooling window are computed
                        // separately for each dimension and multiplied together to get
                        // the total valid count. This approach works for rectangular
                        // windows since the valid count for each dimension is independent
                        // of the other dimensions. But the same does not hold for non-rectangular
                        // windows, so a different approach should be used in that case.
                        const long long cD
                            = (spatialDims == 2)
                                  ? 1
                                  : validCount1D(
                                        outD, stride[0], prePad[0], window[0], dxSpatial[0]);
                        const long long cH
                            = validCount1D(outH, stride[1], prePad[1], window[1], dxSpatial[1]);
                        const long long cW
                            = validCount1D(outW, stride[2], prePad[2], window[2], dxSpatial[2]);
                        const long long validCount = cD * cH * cW;
                        divisor = (validCount == 0) ? 1 : validCount;
                    }
                    else
                    {
                        divisor = window[0] * window[1] * window[2];
                    }

                    result += toAccum(dy[dyOffset]) / static_cast<COMPUTE_TYPE>(divisor);
                }
            }
        }
    }

    // Store the computed gradient
    const long long dxOffset = getOffset<spatialDims>(n, c, dxD, dxH, dxW, args.dxStrides);
    DX_TYPE* tag = nullptr;
    dx[dxOffset] = fromAccum(result, tag);
}
