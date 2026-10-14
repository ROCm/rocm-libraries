// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <type_traits>

#include "VectorTypes.hpp"

using DyType = HIP_PLUGIN_RESAMPLE_DY_TYPE;
using DxType = HIP_PLUGIN_RESAMPLE_DX_TYPE;
using ComputeType = HIP_PLUGIN_RESAMPLE_COMPUTE_TYPE;
using IndexType = HIP_PLUGIN_RESAMPLE_INDEX_TYPE;

constexpr int SPATIAL_DIMS = HIP_PLUGIN_RESAMPLE_SPATIAL_DIMS;
constexpr int RESAMPLE_MODE = HIP_PLUGIN_RESAMPLE_MODE;
constexpr uint64_t DX_ELEMENT_COUNT = HIP_PLUGIN_RESAMPLE_DX_ELEMENT_COUNT;

constexpr int MODE_MAXPOOL = 1;
constexpr int MODE_AVGPOOL_EXCLUDE_PADDING = 2;
constexpr int MODE_AVGPOOL_INCLUDE_PADDING = 3;

__device__ __forceinline__ int64_t dxOffset(int64_t n, int64_t c, int64_t d, int64_t h, int64_t w)
{
    if constexpr(SPATIAL_DIMS == 2)
    {
        return n * HIP_PLUGIN_RESAMPLE_DX_STRIDE_N + c * HIP_PLUGIN_RESAMPLE_DX_STRIDE_C
               + h * HIP_PLUGIN_RESAMPLE_DX_STRIDE_H + w * HIP_PLUGIN_RESAMPLE_DX_STRIDE_W;
    }
    else
    {
        return n * HIP_PLUGIN_RESAMPLE_DX_STRIDE_N + c * HIP_PLUGIN_RESAMPLE_DX_STRIDE_C
               + d * HIP_PLUGIN_RESAMPLE_DX_STRIDE_D + h * HIP_PLUGIN_RESAMPLE_DX_STRIDE_H
               + w * HIP_PLUGIN_RESAMPLE_DX_STRIDE_W;
    }
}

__device__ __forceinline__ int64_t dyOffset(int64_t n, int64_t c, int64_t d, int64_t h, int64_t w)
{
    if constexpr(SPATIAL_DIMS == 2)
    {
        return n * HIP_PLUGIN_RESAMPLE_DY_STRIDE_N + c * HIP_PLUGIN_RESAMPLE_DY_STRIDE_C
               + h * HIP_PLUGIN_RESAMPLE_DY_STRIDE_H + w * HIP_PLUGIN_RESAMPLE_DY_STRIDE_W;
    }
    else
    {
        return n * HIP_PLUGIN_RESAMPLE_DY_STRIDE_N + c * HIP_PLUGIN_RESAMPLE_DY_STRIDE_C
               + d * HIP_PLUGIN_RESAMPLE_DY_STRIDE_D + h * HIP_PLUGIN_RESAMPLE_DY_STRIDE_H
               + w * HIP_PLUGIN_RESAMPLE_DY_STRIDE_W;
    }
}

__device__ __forceinline__ int64_t
    indexOffset(int64_t n, int64_t c, int64_t d, int64_t h, int64_t w)
{
    if constexpr(SPATIAL_DIMS == 2)
    {
        return n * HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_N + c * HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_C
               + h * HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_H + w * HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_W;
    }
    else
    {
        return n * HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_N + c * HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_C
               + d * HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_D + h * HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_H
               + w * HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_W;
    }
}

__device__ __forceinline__ IndexType flattenDxSpatialIndex(int64_t d, int64_t h, int64_t w)
{
    if constexpr(SPATIAL_DIMS == 2)
    {
        return static_cast<IndexType>(h * HIP_PLUGIN_RESAMPLE_DX_W + w);
    }
    else
    {
        return static_cast<IndexType>(d * HIP_PLUGIN_RESAMPLE_DX_H * HIP_PLUGIN_RESAMPLE_DX_W
                                      + h * HIP_PLUGIN_RESAMPLE_DX_W + w);
    }
}

__device__ __forceinline__ int64_t
    validCount1D(int64_t out, int64_t stride, int64_t prePad, int64_t window, int64_t dxDim)
{
    int64_t count = 0;
    for(int64_t k = 0; k < window; ++k)
    {
        const int64_t in = out * stride + k - prePad;
        if(in >= 0 && in < dxDim)
        {
            ++count;
        }
    }
    return count;
}

extern "C" __global__ void ResampleBwd(const DyType* __restrict__ dy,
                                       const IndexType* __restrict__ index,
                                       DxType* __restrict__ dx)
{
    static_assert(std::is_same<ComputeType, float>::value,
                  "ResampleBwd currently supports float compute type only");

    const uint64_t gid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(gid >= DX_ELEMENT_COUNT)
    {
        return;
    }

    // Compute the spatial coordinates for the current thread's output element
    uint64_t remaining = gid;
    const int64_t dxW = static_cast<int64_t>(remaining % HIP_PLUGIN_RESAMPLE_DX_W);
    remaining /= HIP_PLUGIN_RESAMPLE_DX_W;
    const int64_t dxH = static_cast<int64_t>(remaining % HIP_PLUGIN_RESAMPLE_DX_H);
    remaining /= HIP_PLUGIN_RESAMPLE_DX_H;

    int64_t dxD = 0;
    if constexpr(SPATIAL_DIMS == 3)
    {
        dxD = static_cast<int64_t>(remaining % HIP_PLUGIN_RESAMPLE_DX_D);
        remaining /= HIP_PLUGIN_RESAMPLE_DX_D;
    }

    const int64_t c = static_cast<int64_t>(remaining % HIP_PLUGIN_RESAMPLE_C);
    remaining /= HIP_PLUGIN_RESAMPLE_C;
    const int64_t n = static_cast<int64_t>(remaining);

    const IndexType currentFlattenedIndex = flattenDxSpatialIndex(dxD, dxH, dxW);
    float result = 0.0F;

    // Compute the gradient for the current input element by evaluating its
    // contribution to the output elements in the resampling window
    for(int64_t kd = 0; kd < HIP_PLUGIN_RESAMPLE_WINDOW_D; ++kd)
    {
        // Check if the current input depth coordinate maps to a valid output depth coordinate
        const int64_t outD
            = (dxD + HIP_PLUGIN_RESAMPLE_PRE_PAD_D - kd) / HIP_PLUGIN_RESAMPLE_STRIDE_D;
        const bool divD
            = (dxD + HIP_PLUGIN_RESAMPLE_PRE_PAD_D - kd) % HIP_PLUGIN_RESAMPLE_STRIDE_D == 0;
        const bool validD
            = SPATIAL_DIMS == 2 || (outD >= 0 && outD < HIP_PLUGIN_RESAMPLE_DY_D && divD);

        for(int64_t kh = 0; kh < HIP_PLUGIN_RESAMPLE_WINDOW_H; ++kh)
        {
            // Check if the current input height coordinate maps to a valid output height coordinate
            const int64_t outH
                = (dxH + HIP_PLUGIN_RESAMPLE_PRE_PAD_H - kh) / HIP_PLUGIN_RESAMPLE_STRIDE_H;
            const bool divH
                = (dxH + HIP_PLUGIN_RESAMPLE_PRE_PAD_H - kh) % HIP_PLUGIN_RESAMPLE_STRIDE_H == 0;
            const bool validH = outH >= 0 && outH < HIP_PLUGIN_RESAMPLE_DY_H && divH;

            for(int64_t kw = 0; kw < HIP_PLUGIN_RESAMPLE_WINDOW_W; ++kw)
            {
                // Check if the current input width coordinate maps to a valid output width coordinate
                const int64_t outW
                    = (dxW + HIP_PLUGIN_RESAMPLE_PRE_PAD_W - kw) / HIP_PLUGIN_RESAMPLE_STRIDE_W;
                const bool divW
                    = (dxW + HIP_PLUGIN_RESAMPLE_PRE_PAD_W - kw) % HIP_PLUGIN_RESAMPLE_STRIDE_W
                      == 0;
                const bool validW = outW >= 0 && outW < HIP_PLUGIN_RESAMPLE_DY_W && divW;

                // Skip if the current input coordinate does not map to a valid output coordinate
                if(!(validD && validH && validW))
                {
                    continue;
                }

                if constexpr(RESAMPLE_MODE == MODE_MAXPOOL)
                {
                    // Add the gradient from dy to dx if the current input coordinate
                    // contributed to the max pooling operation
                    const int64_t idxOff = indexOffset(n, c, outD, outH, outW);
                    if(index[idxOff] != currentFlattenedIndex)
                    {
                        continue;
                    }
                    const int64_t dyOff = dyOffset(n, c, outD, outH, outW);
                    result += hip_kernel_provider::cast<float>(dy[dyOff]);
                }
                else // MODE_AVGPOOL_EXCLUDE_PADDING or MODE_AVGPOOL_INCLUDE_PADDING
                {
                    int64_t divisor;
                    if constexpr(RESAMPLE_MODE == MODE_AVGPOOL_EXCLUDE_PADDING)
                    {
                        // Instead of using a nested loop to count the number of valid
                        // elements, we compute the valid count for each dimension separately
                        // and multiply them together to get the total valid count. This
                        // reduces the computational complexity from O(WINDOW_SIZE^3) to
                        // O(WINDOW_SIZE). This works because validity is axis-aligned: whether
                        // a given window offset is in-bounds in one spatial dimension does not
                        // depend on the offsets in the other dimensions. If validity were ever
                        // coupled across dimensions (e.g. non-rectangular window), this separable
                        // approach would break and a nested-loop (or other) approach would be
                        // needed instead.
                        const int64_t cD = SPATIAL_DIMS == 2
                                               ? 1
                                               : validCount1D(outD,
                                                              HIP_PLUGIN_RESAMPLE_STRIDE_D,
                                                              HIP_PLUGIN_RESAMPLE_PRE_PAD_D,
                                                              HIP_PLUGIN_RESAMPLE_WINDOW_D,
                                                              HIP_PLUGIN_RESAMPLE_DX_D);
                        const int64_t cH = validCount1D(outH,
                                                        HIP_PLUGIN_RESAMPLE_STRIDE_H,
                                                        HIP_PLUGIN_RESAMPLE_PRE_PAD_H,
                                                        HIP_PLUGIN_RESAMPLE_WINDOW_H,
                                                        HIP_PLUGIN_RESAMPLE_DX_H);
                        const int64_t cW = validCount1D(outW,
                                                        HIP_PLUGIN_RESAMPLE_STRIDE_W,
                                                        HIP_PLUGIN_RESAMPLE_PRE_PAD_W,
                                                        HIP_PLUGIN_RESAMPLE_WINDOW_W,
                                                        HIP_PLUGIN_RESAMPLE_DX_W);
                        const int64_t validCount = cD * cH * cW;
                        divisor = validCount == 0 ? 1 : validCount;
                    }
                    else
                    {
                        divisor = HIP_PLUGIN_RESAMPLE_WINDOW_D * HIP_PLUGIN_RESAMPLE_WINDOW_H
                                  * HIP_PLUGIN_RESAMPLE_WINDOW_W;
                    }

                    // Add the averaged dy contribution to dx for the current input coordinate
                    const int64_t dyOff = dyOffset(n, c, outD, outH, outW);
                    result += hip_kernel_provider::cast<float>(dy[dyOff])
                              / static_cast<float>(divisor);
                }
            }
        }
    }

    // Write the computed gradient to the output dx tensor
    const int64_t dxOff = dxOffset(n, c, dxD, dxH, dxW);
    dx[dxOff] = hip_kernel_provider::cast<DxType>(result);
}
