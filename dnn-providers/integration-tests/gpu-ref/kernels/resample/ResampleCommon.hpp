// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

template <long long SpatialDims>
__device__ __forceinline__ long long getOffset(
    long long n, long long c, long long d, long long h, long long w, const long long* strides)
{
    if constexpr(SpatialDims == 2)
    {
        return n * strides[0] + c * strides[1] + h * strides[2] + w * strides[3];
    }
    else
    {
        return n * strides[0] + c * strides[1] + d * strides[2] + h * strides[3] + w * strides[4];
    }
}

template <typename IndexType, long long SpatialDims>
__device__ __forceinline__ IndexType
    flattenSpatialIndex(long long d, long long h, long long w, long long height, long long width)
{
    if constexpr(SpatialDims == 2)
    {
        return static_cast<IndexType>(h * width + w);
    }
    else
    {
        return static_cast<IndexType>(d * height * width + h * width + w);
    }
}
