// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "Bfloat16Dev.hpp"

namespace hip_kernel_provider
{
namespace detail
{

template <typename T>
struct FloatCast;

template <>
struct FloatCast<float>
{
    static __device__ __forceinline__ float to(float value)
    {
        return value;
    }

    static __device__ __forceinline__ float from(float value)
    {
        return value;
    }
};

template <>
struct FloatCast<half>
{
    static __device__ __forceinline__ float to(half value)
    {
        return __half2float(value);
    }

    static __device__ __forceinline__ half from(float value)
    {
        return __float2half(value);
    }
};

template <>
struct FloatCast<ushort>
{
    static __device__ __forceinline__ float to(ushort value)
    {
        return bfloat16_to_float(value);
    }

    static __device__ __forceinline__ ushort from(float value)
    {
        return float_to_bfloat16(value);
    }
};

} // namespace detail

template <typename T>
__device__ __forceinline__ float to_float32(T value)
{
    return detail::FloatCast<T>::to(value);
}

template <typename T>
__device__ __forceinline__ T from_float32(float value)
{
    return detail::FloatCast<T>::from(value);
}

} // namespace hip_kernel_provider
