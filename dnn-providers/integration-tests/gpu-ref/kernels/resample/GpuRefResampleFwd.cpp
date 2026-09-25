// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GPU reference resample forward kernel.
// Compiled via HipRTC with -DX_TYPE=<type> -DY_TYPE=<type> -DCOMPUTE_TYPE=<type>
// -DINDEX_TYPE=<type> -DHAS_INDEX=<0|1>.
// Each thread computes one output element by iterating over the resample window in the input tensor
// and applying the resample operation to produce the output element.

#include "GpuRefTypes.h"
#include "GpuRefUtils.hpp"

using namespace gpu_ref;

constexpr long long spatialDims = static_cast<long long>(SPATIAL_DIMS);

__device__ __forceinline__ long long getInputOffset(
    long long n, long long c, long long d, long long h, long long w, const long long* xStrides)
{
    if constexpr(spatialDims == 2)
    {
        return n * xStrides[0] + c * xStrides[1] + h * xStrides[2] + w * xStrides[3];
    }
    else
    {
        return n * xStrides[0] + c * xStrides[1] + d * xStrides[2] + h * xStrides[3]
               + w * xStrides[4];
    }
}

__device__ __forceinline__ long long getOutputOffset(
    long long n, long long c, long long d, long long h, long long w, const long long* yStrides)
{
    if constexpr(spatialDims == 2)
    {
        return n * yStrides[0] + c * yStrides[1] + h * yStrides[2] + w * yStrides[3];
    }
    else
    {
        return n * yStrides[0] + c * yStrides[1] + d * yStrides[2] + h * yStrides[3]
               + w * yStrides[4];
    }
}

__device__ __forceinline__ INDEX_TYPE
    flattenSpatialIndex(long long d, long long h, long long w, long long xHeight, long long xWidth)
{
    if constexpr(spatialDims == 2)
    {
        return static_cast<INDEX_TYPE>(h * xWidth + w);
    }
    else
    {
        return static_cast<INDEX_TYPE>(d * xHeight * xWidth + h * xWidth + w);
    }
}

extern "C" __global__ void ResampleFwdRef(ResampleFwdArgs args)
{
    const long long tid = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                          + static_cast<long long>(threadIdx.x);

    long long ySpatialSize = 1;
    for(long long i = 0; i < spatialDims; ++i)
    {
        ySpatialSize *= args.ySpatialDims[i];
    }

    if(tid >= args.n * args.c * ySpatialSize)
    {
        return;
    }

    auto* x = static_cast<const X_TYPE*>(args.x);
    auto* y = static_cast<Y_TYPE*>(args.y);
    [[maybe_unused]] auto* index = static_cast<INDEX_TYPE*>(args.index);

    constexpr ResampleMode resampleMode = static_cast<ResampleMode>(RESAMPLE_MODE);
    constexpr PaddingMode paddingMode = static_cast<PaddingMode>(PADDING_MODE);
    constexpr bool hasIndex = static_cast<bool>(HAS_INDEX);

    // Unpack spatial arrays
    long long xSpatial[3] = {1, 1, 1};
    long long ySpatial[3] = {1, 1, 1};
    long long window[3] = {1, 1, 1};
    long long stride[3] = {1, 1, 1};
    long long prePad[3] = {0, 0, 0};

    const long long offset = (spatialDims == 2) ? 1 : 0;
    for(long long i = 0; i < spatialDims; ++i)
    {
        xSpatial[offset + i] = args.xSpatialDims[i];
        ySpatial[offset + i] = args.ySpatialDims[i];
        window[offset + i] = args.window[i];
        stride[offset + i] = args.stride[i];
        prePad[offset + i] = args.prePadding[i];
    }

    // Compute the output indices for this thread
    long long remaining = tid;
    const long long outW = remaining % ySpatial[2];
    remaining /= ySpatial[2];
    const long long outH = remaining % ySpatial[1];
    remaining /= ySpatial[1];

    long long outD = 0;
    if constexpr(spatialDims == 3)
    {
        outD = remaining % ySpatial[0];
        remaining /= ySpatial[0];
    }

    const long long c = remaining % args.c;
    remaining /= args.c;
    const long long n = remaining;

    COMPUTE_TYPE result = resampleMode == ResampleMode::MAXPOOL
                              ? NumericLimits<COMPUTE_TYPE>::minVal
                              : static_cast<COMPUTE_TYPE>(0);
    long long validCount = 0;
    INDEX_TYPE selectedIndex = static_cast<INDEX_TYPE>(-1);

    // Iterate over the resample window and apply the resample operation
    for(long long kd = 0; kd < window[0]; ++kd)
    {
        const long long inD = outD * stride[0] + kd - prePad[0];
        const bool validD = (spatialDims == 2) || (inD >= 0 && inD < xSpatial[0]);

        for(long long kh = 0; kh < window[1]; ++kh)
        {
            const long long inH = outH * stride[1] + kh - prePad[1];
            const bool validH = (inH >= 0 && inH < xSpatial[1]);

            for(long long kw = 0; kw < window[2]; ++kw)
            {
                const long long inW = outW * stride[2] + kw - prePad[2];
                const bool validW = (inW >= 0) && (inW < xSpatial[2]);
                const bool valid = validD && validH && validW;

                COMPUTE_TYPE candidate = static_cast<COMPUTE_TYPE>(0);
                INDEX_TYPE candidateIndex = static_cast<INDEX_TYPE>(-1);

                // If the input index is valid, read the input value and compute the flattened spatial index
                if(valid)
                {
                    const long long xOffset = getInputOffset(n, c, inD, inH, inW, args.xStrides);
                    candidate = toAccum(x[xOffset]);
                    candidateIndex = flattenSpatialIndex(inD, inH, inW, xSpatial[1], xSpatial[2]);
                    ++validCount;
                }
                else if constexpr(paddingMode == PaddingMode::PAD_NEG_INF
                                  && resampleMode == ResampleMode::MAXPOOL)
                {
                    continue;
                }

                // Apply the resample operation based on the mode
                // Get the maximum value for MAXPOOL, or accumulate for AVGPOOL modes
                if constexpr(resampleMode == ResampleMode::MAXPOOL)
                {
                    if(candidate > result)
                    {
                        result = candidate;
                        selectedIndex = candidateIndex;
                    }
                }
                else
                {
                    result += candidate;
                }
            }
        }
    }

    // Post-accumulation division for average pooling
    if constexpr(resampleMode == ResampleMode::AVGPOOL_EXCLUDE_PADDING)
    {
        const long long divisor = (validCount == 0) ? 1 : validCount;
        result /= static_cast<COMPUTE_TYPE>(divisor);
    }
    else if constexpr(resampleMode == ResampleMode::AVGPOOL_INCLUDE_PADDING)
    {
        result /= static_cast<COMPUTE_TYPE>(window[0] * window[1] * window[2]);
    }

    // Store output value and max-pool index output
    const long long yOffset = getOutputOffset(n, c, outD, outH, outW, args.yStrides);
    Y_TYPE* tag = nullptr;
    y[yOffset] = fromAccum(result, tag);

    if constexpr(hasIndex && resampleMode == ResampleMode::MAXPOOL)
    {
        index[yOffset] = selectedIndex;
    }
}
