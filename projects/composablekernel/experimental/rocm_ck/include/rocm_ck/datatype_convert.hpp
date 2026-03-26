// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: host — float/typed conversions, verification tolerances.
//
// Host-side type conversion utilities for rocm_ck examples.
// Compiled with GCC (not hipcc), so hip's half/bfloat16 operator
// overloads are not available. Uses _Float16 (GCC built-in) for fp16
// and raw bit manipulation for bf16.

#pragma once

#include <rocm_ck/datatype_utils.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace rocm_ck {

/// Convert float to bf16 using round-to-nearest-even.
inline std::uint16_t float_to_bf16_bits(float f)
{
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    // Round to nearest even: add rounding bias based on LSB + trailing bits
    u += 0x7FFF + ((u >> 16) & 1);
    return static_cast<std::uint16_t>(u >> 16);
}

/// Convert bf16 bits back to float.
inline float bf16_bits_to_float(std::uint16_t bits)
{
    std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

/// Convert a float to the device type and store into a byte buffer.
inline void float_to_typed(DataType dt, float value, void* dst)
{
    switch(dt)
    {
    case DataType::FP64: *static_cast<double*>(dst) = static_cast<double>(value); break;
    case DataType::FP32: *static_cast<float*>(dst) = value; break;
    case DataType::FP16: *static_cast<_Float16*>(dst) = static_cast<_Float16>(value); break;
    case DataType::BF16: *static_cast<std::uint16_t*>(dst) = float_to_bf16_bits(value); break;
    case DataType::FP8_FNUZ:
    case DataType::BF8_FNUZ:
    case DataType::FP8_OCP:
    case DataType::BF8_OCP:
        std::fprintf(stderr, "%s host conversion not implemented\n", data_type_name(dt));
        std::abort();
    case DataType::I4:
    case DataType::I8:
    case DataType::I16:
    case DataType::I32:
    case DataType::I64:
    case DataType::U8:
    case DataType::U16:
    case DataType::U32:
    case DataType::U64:
        std::fprintf(stderr, "%s: use integer-specific conversion\n", data_type_name(dt));
        std::abort();
    }
}

/// Read a typed value from a byte buffer and convert to float.
inline float typed_to_float(DataType dt, const void* src)
{
    switch(dt)
    {
    case DataType::FP64: return static_cast<float>(*static_cast<const double*>(src));
    case DataType::FP32: return *static_cast<const float*>(src);
    case DataType::FP16: return static_cast<float>(*static_cast<const _Float16*>(src));
    case DataType::BF16: return bf16_bits_to_float(*static_cast<const std::uint16_t*>(src));
    case DataType::FP8_FNUZ:
    case DataType::BF8_FNUZ:
    case DataType::FP8_OCP:
    case DataType::BF8_OCP:
        std::fprintf(stderr, "%s host conversion not implemented\n", data_type_name(dt));
        std::abort();
    case DataType::I4:
    case DataType::I8:
    case DataType::I16:
    case DataType::I32:
    case DataType::I64:
    case DataType::U8:
    case DataType::U16:
    case DataType::U32:
    case DataType::U64:
        std::fprintf(stderr, "%s: use integer-specific conversion\n", data_type_name(dt));
        std::abort();
    }
    return 0.0f;
}

/// Tolerance for verification based on data type.
inline float tolerance_for(DataType dt)
{
    switch(dt)
    {
    case DataType::FP64: return 1e-12f;
    case DataType::FP32: return 1e-5f;
    case DataType::FP16: return 1e-2f;
    case DataType::BF16: return 1e-1f;
    case DataType::FP8_FNUZ:
    case DataType::BF8_FNUZ:
    case DataType::FP8_OCP:
    case DataType::BF8_OCP: return 1.0f;
    case DataType::I4:
    case DataType::I8:
    case DataType::I16:
    case DataType::I32:
    case DataType::I64:
    case DataType::U8:
    case DataType::U16:
    case DataType::U32:
    case DataType::U64: return 0.0f; // exact for integer types
    }
    return 1e-5f;
}

} // namespace rocm_ck
