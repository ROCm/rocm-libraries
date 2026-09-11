// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <type_traits>

#include "VectorTypes.hpp"

using InputType = HIP_PLUGIN_RESAMPLE_INPUT_TYPE;
using OutputType = HIP_PLUGIN_RESAMPLE_OUTPUT_TYPE;
using ComputeType = HIP_PLUGIN_RESAMPLE_COMPUTE_TYPE;
using IndexType = HIP_PLUGIN_RESAMPLE_INDEX_TYPE;

constexpr int SPATIAL_DIMS = HIP_PLUGIN_RESAMPLE_SPATIAL_DIMS;
constexpr int RESAMPLE_MODE = HIP_PLUGIN_RESAMPLE_MODE;
constexpr int PADDING_MODE = HIP_PLUGIN_RESAMPLE_PADDING_MODE;
constexpr bool HAS_INDEX = HIP_PLUGIN_RESAMPLE_HAS_INDEX;
constexpr bool GENERATE_INDEX = HIP_PLUGIN_RESAMPLE_GENERATE_INDEX;
constexpr uint64_t OUTPUT_ELEMENT_COUNT = HIP_PLUGIN_RESAMPLE_OUTPUT_ELEMENT_COUNT;

constexpr int MODE_MAXPOOL = 1;
constexpr int MODE_AVGPOOL_EXCLUDE_PADDING = 2;
constexpr int MODE_AVGPOOL_INCLUDE_PADDING = 3;
constexpr int PAD_NEG_INF = 1;
constexpr int PAD_ZERO = 2;

__device__ __forceinline__ int64_t
    inputOffset(int64_t n, int64_t c, int64_t d, int64_t h, int64_t w)
{
    if constexpr(SPATIAL_DIMS == 2)
    {
        return n * HIP_PLUGIN_RESAMPLE_X_STRIDE_N + c * HIP_PLUGIN_RESAMPLE_X_STRIDE_C
               + h * HIP_PLUGIN_RESAMPLE_X_STRIDE_H + w * HIP_PLUGIN_RESAMPLE_X_STRIDE_W;
    }
    else
    {
        return n * HIP_PLUGIN_RESAMPLE_X_STRIDE_N + c * HIP_PLUGIN_RESAMPLE_X_STRIDE_C
               + d * HIP_PLUGIN_RESAMPLE_X_STRIDE_D + h * HIP_PLUGIN_RESAMPLE_X_STRIDE_H
               + w * HIP_PLUGIN_RESAMPLE_X_STRIDE_W;
    }
}

__device__ __forceinline__ int64_t
    outputOffset(int64_t n, int64_t c, int64_t d, int64_t h, int64_t w)
{
    if constexpr(SPATIAL_DIMS == 2)
    {
        return n * HIP_PLUGIN_RESAMPLE_Y_STRIDE_N + c * HIP_PLUGIN_RESAMPLE_Y_STRIDE_C
               + h * HIP_PLUGIN_RESAMPLE_Y_STRIDE_H + w * HIP_PLUGIN_RESAMPLE_Y_STRIDE_W;
    }
    else
    {
        return n * HIP_PLUGIN_RESAMPLE_Y_STRIDE_N + c * HIP_PLUGIN_RESAMPLE_Y_STRIDE_C
               + d * HIP_PLUGIN_RESAMPLE_Y_STRIDE_D + h * HIP_PLUGIN_RESAMPLE_Y_STRIDE_H
               + w * HIP_PLUGIN_RESAMPLE_Y_STRIDE_W;
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

__device__ __forceinline__ IndexType flattenSpatialIndex(int64_t d, int64_t h, int64_t w)
{
    if constexpr(SPATIAL_DIMS == 2)
    {
        return static_cast<IndexType>(h * HIP_PLUGIN_RESAMPLE_X_W + w);
    }
    else
    {
        return static_cast<IndexType>(d * HIP_PLUGIN_RESAMPLE_X_H * HIP_PLUGIN_RESAMPLE_X_W
                                      + h * HIP_PLUGIN_RESAMPLE_X_W + w);
    }
}

extern "C" __global__ void ResampleFwd(const InputType* __restrict__ x,
                                       OutputType* __restrict__ y,
                                       IndexType* __restrict__ index)
{
    static_assert(std::is_same<ComputeType, float>::value,
                  "ResampleFwd currently supports float compute type only");

    const uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= OUTPUT_ELEMENT_COUNT)
    {
        return;
    }

    uint64_t remaining = gid;
    const int64_t outW = static_cast<int64_t>(remaining % HIP_PLUGIN_RESAMPLE_Y_W);
    remaining /= HIP_PLUGIN_RESAMPLE_Y_W;
    const int64_t outH = static_cast<int64_t>(remaining % HIP_PLUGIN_RESAMPLE_Y_H);
    remaining /= HIP_PLUGIN_RESAMPLE_Y_H;

    int64_t outD = 0;
    if constexpr(SPATIAL_DIMS == 3)
    {
        outD = static_cast<int64_t>(remaining % HIP_PLUGIN_RESAMPLE_Y_D);
        remaining /= HIP_PLUGIN_RESAMPLE_Y_D;
    }

    const int64_t c = static_cast<int64_t>(remaining % HIP_PLUGIN_RESAMPLE_C);
    remaining /= HIP_PLUGIN_RESAMPLE_C;
    const int64_t n = static_cast<int64_t>(remaining);

    float result = RESAMPLE_MODE == MODE_MAXPOOL ? -3.402823466e+38F : 0.0F;
    int64_t validCount = 0;
    IndexType selectedIndex = static_cast<IndexType>(-1);

    for(int64_t kd = 0; kd < HIP_PLUGIN_RESAMPLE_WINDOW_D; ++kd)
    {
        const int64_t inD
            = outD * HIP_PLUGIN_RESAMPLE_STRIDE_D + kd - HIP_PLUGIN_RESAMPLE_PRE_PAD_D;
        const bool validD = SPATIAL_DIMS == 2 || (inD >= 0 && inD < HIP_PLUGIN_RESAMPLE_X_D);

        for(int64_t kh = 0; kh < HIP_PLUGIN_RESAMPLE_WINDOW_H; ++kh)
        {
            const int64_t inH
                = outH * HIP_PLUGIN_RESAMPLE_STRIDE_H + kh - HIP_PLUGIN_RESAMPLE_PRE_PAD_H;
            const bool validH = inH >= 0 && inH < HIP_PLUGIN_RESAMPLE_X_H;

            for(int64_t kw = 0; kw < HIP_PLUGIN_RESAMPLE_WINDOW_W; ++kw)
            {
                const int64_t inW
                    = outW * HIP_PLUGIN_RESAMPLE_STRIDE_W + kw - HIP_PLUGIN_RESAMPLE_PRE_PAD_W;
                const bool validW = inW >= 0 && inW < HIP_PLUGIN_RESAMPLE_X_W;
                const bool valid = validD && validH && validW;

                float candidate = 0.0F;
                IndexType candidateIndex = static_cast<IndexType>(-1);
                if(valid)
                {
                    const int64_t xOffset = inputOffset(n, c, inD, inH, inW);
                    candidate = hip_kernel_provider::cast<float>(x[xOffset]);
                    candidateIndex = flattenSpatialIndex(inD, inH, inW);
                    ++validCount;
                }
                else if constexpr(PADDING_MODE == PAD_NEG_INF && RESAMPLE_MODE == MODE_MAXPOOL)
                {
                    continue;
                }

                if constexpr(RESAMPLE_MODE == MODE_MAXPOOL)
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

    if constexpr(RESAMPLE_MODE == MODE_AVGPOOL_EXCLUDE_PADDING)
    {
        const int64_t divisor = validCount == 0 ? 1 : validCount;
        result /= static_cast<float>(divisor);
    }
    else if constexpr(RESAMPLE_MODE == MODE_AVGPOOL_INCLUDE_PADDING)
    {
        result /= static_cast<float>(HIP_PLUGIN_RESAMPLE_WINDOW_D * HIP_PLUGIN_RESAMPLE_WINDOW_H
                                     * HIP_PLUGIN_RESAMPLE_WINDOW_W);
    }

    const int64_t yOffset = outputOffset(n, c, outD, outH, outW);
    y[yOffset] = hip_kernel_provider::cast<OutputType>(result);

    if constexpr(HAS_INDEX && GENERATE_INDEX && RESAMPLE_MODE == MODE_MAXPOOL)
    {
        const int64_t idxOffset = indexOffset(n, c, outD, outH, outW);
        index[idxOffset] = selectedIndex;
    }
}
