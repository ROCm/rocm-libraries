#pragma once

#include "hipconv/conv2d_params.hpp"
#include "unreachable.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>

using fp16_t   = _Float16;
using fp16x2_t = __attribute__((ext_vector_type(2))) _Float16;
using fp16x4_t = __attribute__((ext_vector_type(4))) _Float16;
using fp16x8_t = __attribute__((ext_vector_type(8))) _Float16;

using bf16_t   = __bf16;
using bf16x2_t = __attribute__((ext_vector_type(2))) __bf16;
using bf16x4_t = __attribute__((ext_vector_type(4))) __bf16;
using bf16x8_t = __attribute__((ext_vector_type(8))) __bf16;

using fp32_t   = float;
using fp32x2_t = __attribute__((ext_vector_type(2))) float;
using fp32x4_t = __attribute__((ext_vector_type(4))) float;
using fp32x8_t = __attribute__((ext_vector_type(8))) float;

using int16x2_t = __attribute__((ext_vector_type(2))) short;
using int16x4_t = __attribute__((ext_vector_type(4))) short;
using int16x8_t = __attribute__((ext_vector_type(8))) short;

using fp8_t = __hip_fp8_e4m3;
using bf8_t = __hip_fp8_e5m2;

template <hipconv::DataType type>
struct ToTypeImpl;
template <>
struct ToTypeImpl<hipconv::DataType::fp16>
{
    using type = fp16_t;
};
template <>
struct ToTypeImpl<hipconv::DataType::bf16>
{
    using type = bf16_t;
};
template <>
struct ToTypeImpl<hipconv::DataType::fp32>
{
    using type = fp32_t;
};
template <>
struct ToTypeImpl<hipconv::DataType::fp8>
{
    using type = fp8_t;
};
template <>
struct ToTypeImpl<hipconv::DataType::bf8>
{
    using type = bf8_t;
};
template <>
struct ToTypeImpl<hipconv::DataType::tf32>
{
    // TF32 is stored as fp32; the BF16 decomposition happens inside MFMA dispatch.
    using type = fp32_t;
};

template <hipconv::DataType type>
using ToType = typename ToTypeImpl<type>::type;

inline auto mantissa_bits(hipconv::DataType type)
{
    switch(type)
    {
    case hipconv::DataType::fp16:
        return 10;
    case hipconv::DataType::bf16:
        return 7;
    case hipconv::DataType::fp32:
        return 23;
    case hipconv::DataType::fp8:
        return 3;
    case hipconv::DataType::bf8:
        return 2;
    case hipconv::DataType::tf32:
        // Two BF16 components combined: ~14 effective mantissa bits.
        return 14;
    }
    HIPCONV_UNREACHABLE();
}
