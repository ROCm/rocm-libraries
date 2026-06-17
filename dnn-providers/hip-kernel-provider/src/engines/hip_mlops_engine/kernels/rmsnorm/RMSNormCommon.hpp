// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "Bfloat16Dev.hpp"

namespace hip_kernel_provider::rmsnorm
{

template <typename T>
struct Cast;

template <>
struct Cast<float>
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
struct Cast<half>
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
struct Cast<ushort>
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

template <typename T>
__device__ __forceinline__ float to_float32(T value)
{
    return Cast<T>::to(value);
}

template <typename T>
__device__ __forceinline__ T from_float32(float value)
{
    return Cast<T>::from(value);
}

} // namespace hip_kernel_provider::rmsnorm
