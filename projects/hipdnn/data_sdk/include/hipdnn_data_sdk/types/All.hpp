// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file All.hpp
 * @brief Convenience header that includes all custom hipDNN data types.
 *
 * This header provides access to all portable floating-point types that
 * do not require the __HIPCC__ macro:
 *
 * - bfloat16: 16-bit brain floating point (1 sign, 8 exponent, 7 mantissa)
 * - half: 16-bit IEEE 754 half precision (1 sign, 5 exponent, 10 mantissa)
 * - fp8_e4m3: 8-bit floating point (1 sign, 4 exponent, 3 mantissa)
 * - fp8_e5m2: 8-bit floating point (1 sign, 5 exponent, 2 mantissa)
 *
 * Also includes forwarding functions for built-in types (float, double,
 * int8_t, uint8_t, int32_t) that enable uniform unqualified calls for
 * math functions across all types.
 *
 * All types use EXPLICIT constructors and conversion operators to prevent
 * silent precision loss and eliminate overload ambiguity issues.
 */

#include "Bfloat16.hpp"
#include "Double.hpp"
#include "Float.hpp"
#include "Fp8E4M3.hpp"
#include "Fp8E5M2.hpp"
#include "Half.hpp"
#include "Int32.hpp"
#include "Int8.hpp"
#include "Uint8.hpp"

// Cross-type constructor definitions
// These must be defined after all types are fully declared

namespace hipdnn_data_sdk::types
{

// bfloat16 cross-type constructors
inline bfloat16::bfloat16(half h) noexcept
    : data(detail::float_to_bfloat16_bits(static_cast<float>(h)))
{
}

inline bfloat16::bfloat16(fp8_e4m3 f) noexcept
    : data(detail::float_to_bfloat16_bits(static_cast<float>(f)))
{
}

inline bfloat16::bfloat16(fp8_e5m2 f) noexcept
    : data(detail::float_to_bfloat16_bits(static_cast<float>(f)))
{
}

// half cross-type constructors
inline half::half(bfloat16 b) noexcept
    : data(detail::float_to_half_bits(static_cast<float>(b)))
{
}

inline half::half(fp8_e4m3 f) noexcept
    : data(detail::float_to_half_bits(static_cast<float>(f)))
{
}

inline half::half(fp8_e5m2 f) noexcept
    : data(detail::float_to_half_bits(static_cast<float>(f)))
{
}

// fp8_e4m3 cross-type constructors
inline fp8_e4m3::fp8_e4m3(bfloat16 b) noexcept
    : data(detail::float_to_fp8_e4m3_bits(static_cast<float>(b)))
{
}

inline fp8_e4m3::fp8_e4m3(half h) noexcept
    : data(detail::float_to_fp8_e4m3_bits(static_cast<float>(h)))
{
}

inline fp8_e4m3::fp8_e4m3(fp8_e5m2 f) noexcept
    : data(detail::float_to_fp8_e4m3_bits(static_cast<float>(f)))
{
}

// fp8_e5m2 cross-type constructors
inline fp8_e5m2::fp8_e5m2(bfloat16 b) noexcept
    : data(detail::float_to_fp8_e5m2_bits(static_cast<float>(b)))
{
}

inline fp8_e5m2::fp8_e5m2(half h) noexcept
    : data(detail::float_to_fp8_e5m2_bits(static_cast<float>(h)))
{
}

inline fp8_e5m2::fp8_e5m2(fp8_e4m3 f) noexcept
    : data(detail::float_to_fp8_e5m2_bits(static_cast<float>(f)))
{
}

} // namespace hipdnn_data_sdk::types
