// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "conv_params.hpp"
#include <hip/hip_fp8.h>

namespace ck_tile::direct_conv
{

using fp16_t   = _Float16;
using fp16x4_t = __attribute__((vector_size(4 * sizeof(fp16_t)))) fp16_t;
using fp16x8_t = __attribute__((vector_size(8 * sizeof(fp16_t)))) fp16_t;
using fp32x4_t = __attribute__((vector_size(4 * sizeof(float)))) float;

using bf16_t = __bf16;
using fp32_t = float;
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

} // namespace ck_tile::direct_conv
