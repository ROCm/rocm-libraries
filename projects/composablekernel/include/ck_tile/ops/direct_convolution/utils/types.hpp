// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "conv_params.hpp"
#include <hip/hip_fp8.h>

namespace ck_tile::direct_conv
{

using fp16_t   = _Float16;
using fp16x4_t = fp16_t __attribute__((ext_vector_type(4)));
using fp16x8_t = fp16_t __attribute__((ext_vector_type(8)));

using bf16_t   = __bf16;
using bf16x4_t = bf16_t __attribute__((ext_vector_type(4)));
using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));

using fp32_t = float;
using fp32x4_t = float __attribute__((ext_vector_type(4)));

using fp8_t  = __hip_fp8_e4m3;
using bf8_t  = __hip_fp8_e5m2;

template <DataType type>
struct ToTypeImpl;
template <>
struct ToTypeImpl<DataType::fp16>
{
    using type = fp16_t;
};
template <>
struct ToTypeImpl<DataType::bf16>
{
    using type = bf16_t;
};
template <>
struct ToTypeImpl<DataType::fp32>
{
    using type = fp32_t;
};
template <>
struct ToTypeImpl<DataType::fp8>
{
    using type = fp8_t;
};
template <>
struct ToTypeImpl<DataType::bf8>
{
    using type = bf8_t;
};

template <DataType type>
using ToType = typename ToTypeImpl<type>::type;

inline auto mantissa_bits(DataType type)
{
    switch(type)
    {
    case DataType::fp16:
        return 10;
    case DataType::bf16:
        return 7;
    case DataType::fp32:
        return 23;
    case DataType::fp8:
        return 3;
    case DataType::bf8:
        return 2;
    }
}

// Convert a pair of fp32 values to a pair of fp16/bf16 values (packed as a single 32-bit word).
// Used in the output writer to convert MFMA fp32 accumulator results to the output element type.
template <typename ElementType>
struct ConvertFp32ToVec4;

template <>
struct ConvertFp32ToVec4<_Float16>
{
    __device__ __forceinline__ static uint32_t convert(float a, float b)
    {
        union { _Float16 h[2]; uint32_t u; } u;
        u.h[0] = static_cast<_Float16>(a);
        u.h[1] = static_cast<_Float16>(b);
        return u.u;
    }
};

template <>
struct ConvertFp32ToVec4<__bf16>
{
    __device__ __forceinline__ static uint32_t convert(float a, float b)
    {
        union { __bf16 h[2]; uint32_t u; } u;
        u.h[0] = static_cast<__bf16>(a);
        u.h[1] = static_cast<__bf16>(b);
        return u.u;
    }
};

} // namespace ck_tile::direct_conv
