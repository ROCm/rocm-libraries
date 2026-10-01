// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "Common.hpp"
#include <cstdint>
#include <limits>
#include <type_traits>

namespace hipdnn_gpu_ref::types
{

namespace detail
{

// ============================================================================
// FP8 E4M3 (OCP Format) Bit Layout Constants
// ============================================================================
// OCP E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits
// - No infinity representation (uses max value for saturation)
// - NaN represented as all 1s in mantissa (0x7F positive, 0xFF negative)
//
// Bit layout: [S|EEEE|MMM]
//              7 6  3 2 0
// ============================================================================

// Sign bit mask (bit 7)
constexpr uint8_t FP8_E4M3_SIGN_MASK = 0x80;

// Absolute value mask (all bits except sign)
constexpr uint8_t FP8_E4M3_ABS_MASK = 0x7F;

// Exponent field mask (bits 3-6)
constexpr uint8_t FP8_E4M3_EXP_MASK = 0x78;

// Mantissa field mask (bits 0-2)
constexpr uint8_t FP8_E4M3_MANT_MASK = 0x07;

// Exponent bias for OCP E4M3
constexpr int FP8_E4M3_EXP_BIAS = 7;

// Number of mantissa bits
constexpr int FP8_E4M3_MANT_BITS = 3;

// Number of exponent bits
constexpr int FP8_E4M3_EXP_BITS = 4;

// Maximum biased exponent representable in EXP_BITS bits
constexpr int FP8_E4M3_MAX_BIASED_EXP = (1 << FP8_E4M3_EXP_BITS) - 1;

// NaN: all mantissa bits set (0x7F for positive sign, 0xFF for negative)
constexpr uint8_t FP8_E4M3_NAN = 0x7F;

// Maximum finite positive value: 0x7E = 448.0
constexpr uint8_t FP8_E4M3_MAX = 0x7E;

/// Minimum positive normal value: 2^-6 = 0.015625
constexpr uint8_t FP8_E4M3_MIN_NORMAL = 0x08;

/// Minimum positive denormal value
constexpr uint8_t FP8_E4M3_DENORM_MIN = 0x01;

/// Maximum finite negative value (lowest): -448.0
constexpr uint8_t FP8_E4M3_LOWEST = 0xFE;

/// Epsilon: 2^-3 = 0.125
constexpr uint8_t FP8_E4M3_EPSILON = 0x20;

/// Round error (0.5)
constexpr uint8_t FP8_E4M3_ROUND_ERROR = 0x30;

/// Rounding threshold for round-to-nearest-even (midpoint of 20-bit remainder)
constexpr uint32_t FP8_E4M3_ROUND_THRESHOLD = 0x80000;

// Convert float to FP8 E4M3 bits (OCP format: 1 sign, 4 exponent, 3 mantissa)
// Range: +/- 448, no infinity, NaN = 0x7F or 0xFF
// NOTE: The `saturate=false` paths return NaN on overflow per the OCP spec and
// are reserved for future non-saturating-mode support.
// NOLINTNEXTLINE(readability-identifier-naming)
HOST_DEVICE inline uint8_t float_to_fp8_e4m3_bits(float f, bool saturate = true) noexcept
{
    const uint32_t bits = floatToBits(f);
    const uint32_t absBits = bits & 0x7FFFFFFFu;
    const uint32_t sign = (bits >> 24) & FP8_E4M3_SIGN_MASK; // sign moved to bit 7

    // NaN (sign preserved)
    if(absBits > 0x7F800000u)
    {
        return static_cast<uint8_t>(sign | FP8_E4M3_NAN);
    }

    // Infinity: E4M3 has no infinity representation; saturate to limit values or return NaN
    if(absBits == 0x7F800000u)
    {
        return static_cast<uint8_t>(sign | (saturate ? FP8_E4M3_MAX : FP8_E4M3_NAN));
    }

    const uint32_t fp32Exp = absBits >> 23;
    int32_t exp = static_cast<int32_t>(fp32Exp) - 127 + FP8_E4M3_EXP_BIAS;
    uint32_t mant = bits & 0x007FFFFFu;

    // +/-0 and fp32 subnormals: far below the smallest E4M3 value -> signed zero
    if(fp32Exp == 0)
    {
        return static_cast<uint8_t>(sign);
    }

    // Overflow (biased exp 15 is a valid normal exponent for E4M3, so only > 15 overflows)
    if(exp > FP8_E4M3_MAX_BIASED_EXP)
    {
        return static_cast<uint8_t>(sign | (saturate ? FP8_E4M3_MAX : FP8_E4M3_NAN));
    }

    // Subnormal / underflow range (may round up into the smallest normal)
    if(exp <= 0)
    {
        mant |= 0x00800000u; // implicit 1
        const auto shift = static_cast<uint32_t>(1 - exp + 20); // 23 - 3 = 20 bits to shift
        if(shift > 24)
        {
            return static_cast<uint8_t>(sign); // too small -> zero
        }

        const uint32_t halfPoint = 1u << (shift - 1);
        const uint32_t remainder = mant & ((1u << shift) - 1u);
        mant >>= shift;

        // Round to nearest even
        if(remainder > halfPoint || (remainder == halfPoint && ((mant & 1u) != 0u)))
        {
            mant++;
            if(mant > FP8_E4M3_MANT_MASK)
            {
                return static_cast<uint8_t>(sign | FP8_E4M3_MIN_NORMAL);
            }
        }
        if(mant == 0u)
        {
            return static_cast<uint8_t>(sign);
        }
        return static_cast<uint8_t>(sign | (mant & FP8_E4M3_MANT_MASK));
    }

    // Normal range: 23 -> 3 mantissa bits, round-to-nearest-even
    uint32_t fp8Mant = (mant >> 20) & FP8_E4M3_MANT_MASK;
    const uint32_t remainder = mant & 0x000FFFFFu;

    // Round to nearest even
    if(remainder > FP8_E4M3_ROUND_THRESHOLD
       || (remainder == FP8_E4M3_ROUND_THRESHOLD && ((fp8Mant & 1u) != 0u)))
    {
        fp8Mant++;
        if(fp8Mant > FP8_E4M3_MANT_MASK)
        {
            fp8Mant = 0;
            exp++;
            if(exp > FP8_E4M3_MAX_BIASED_EXP)
            {
                return static_cast<uint8_t>(sign | (saturate ? FP8_E4M3_MAX : FP8_E4M3_NAN));
            }
        }
    }

    // exp=15 && mant=7 would be 0x7F/0xFF, which is NaN in OCP E4M3.
    // Any finite value landing there saturates to MAX (or NaN when not saturating).
    if(exp == FP8_E4M3_MAX_BIASED_EXP && fp8Mant == FP8_E4M3_MANT_MASK)
    {
        return static_cast<uint8_t>(sign | (saturate ? FP8_E4M3_MAX : FP8_E4M3_NAN));
    }

    return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 3) | fp8Mant);
}

// Convert FP8 E4M3 bits to float
// NOLINTNEXTLINE(readability-identifier-naming)
HOST_DEVICE inline float fp8_e4m3_bits_to_float(uint8_t b) noexcept
{
    const uint32_t sign = (static_cast<uint32_t>(b) & 0x80u) << 24;
    const uint32_t exp = (static_cast<uint32_t>(b) >> 3) & 0xFu;
    const uint32_t mant = static_cast<uint32_t>(b) & 0x7u;

    uint32_t out;
    if((b & FP8_E4M3_ABS_MASK) == FP8_E4M3_NAN)
    {
        out = sign | 0x7FC00000u;
    }
    else if(exp == 0u)
    {
        // Zero / subnormal: mant * 2^-9, with mant in 1..7 = 1.f * 2^(p-9)
        if(mant == 0u)
        {
            out = sign;
        }
        else
        {
            const uint32_t p = mant >= 4u ? 2u : (mant >= 2u ? 1u : 0u); // leading-bit position
            const uint32_t frac = (mant - (1u << p)) << (23u - p);
            out = sign | ((118u + p) << 23) | frac;
        }
    }
    else
    {
        // Normal: rebias 7 -> 127 (+120), mantissa into top 3 bits of the fp32 mantissa
        out = sign | ((exp + 120u) << 23) | (mant << 20);
    }

    return bitsToFloat(out);
}

} // namespace detail

/**
 * @brief Custom storage-only FP8 E4M3 type for device
 *
 * This type provides a portable FP8 E4M3 (1 sign, 4 exponent, 3 mantissa)
 * implementation. Uses OCP E4M3 format.
 *
 * This is a STORAGE-ONLY type intended for data representation and conversion,
 * not direct computation. Arithmetic operations and comparisons are
 * intentionally not provided. For computation, explicitly convert to float.
 *
 * Binary layout: 1 sign bit, 4 exponent bits, 3 mantissa bits
 * Range: +/- 448 (max normal value)
 * No infinity representation (uses NaN for overflow without saturation)
 */
// NOLINTNEXTLINE(readability-identifier-naming) - lowercase for consistency
struct fp8_e4m3
{
    uint8_t data;

    constexpr fp8_e4m3() noexcept
        : data(0)
    {
    }

    fp8_e4m3(const fp8_e4m3&) = default;
    fp8_e4m3(fp8_e4m3&&) noexcept = default;
    fp8_e4m3& operator=(const fp8_e4m3&) = default;
    fp8_e4m3& operator=(fp8_e4m3&&) noexcept = default;

    HOST_DEVICE explicit fp8_e4m3(float f) noexcept
        : data(detail::float_to_fp8_e4m3_bits(f))
    {
    }

    HOST_DEVICE explicit fp8_e4m3(double d) noexcept
        : fp8_e4m3(static_cast<float>(d))
    {
    }

    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    HOST_DEVICE explicit fp8_e4m3(T value) noexcept
        : fp8_e4m3(static_cast<float>(value))
    {
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    HOST_DEVICE static constexpr fp8_e4m3 from_bits(uint8_t bits) noexcept
    {
        fp8_e4m3 val;
        val.data = bits;
        return val;
    }

    HOST_DEVICE explicit operator float() const noexcept
    {
        return detail::fp8_e4m3_bits_to_float(data);
    }

    HOST_DEVICE explicit operator double() const noexcept
    {
        return static_cast<double>(detail::fp8_e4m3_bits_to_float(data));
    }

    HOST_DEVICE fp8_e4m3 operator-() const noexcept
    {
        return from_bits(static_cast<uint8_t>(data ^ detail::FP8_E4M3_SIGN_MASK));
    }

    HOST_DEVICE fp8_e4m3 operator+() const noexcept
    {
        return *this;
    }
};

// Static assertions for binary compatibility
static_assert(sizeof(fp8_e4m3) == sizeof(uint8_t), "fp8_e4m3 must be 1 byte");
static_assert(std::is_trivially_copyable_v<fp8_e4m3>, "fp8_e4m3 must be trivially copyable");
static_assert(std::is_standard_layout_v<fp8_e4m3>, "fp8_e4m3 must be standard layout");
static_assert(std::is_default_constructible_v<fp8_e4m3>, "fp8_e4m3 must be default constructible");
static_assert(std::is_copy_constructible_v<fp8_e4m3>, "fp8_e4m3 must be copy constructible");
static_assert(std::is_move_constructible_v<fp8_e4m3>, "fp8_e4m3 must be move constructible");

// ============================================================================
// Math functions for fp8_e4m3
// ============================================================================
// These are defined in our namespace to enable ADL (Argument Dependent Lookup).
// Use unqualified calls like: isnan(x), isinf(x), etc.
// ============================================================================

HOST_DEVICE inline bool isnan(fp8_e4m3 x)
{
    return (x.data & detail::FP8_E4M3_ABS_MASK) == detail::FP8_E4M3_NAN;
}

HOST_DEVICE inline bool isinf(fp8_e4m3 /*x*/)
{
    return false; // E4M3 has no infinity
}

HOST_DEVICE inline bool signbit(fp8_e4m3 x)
{
    return (x.data & detail::FP8_E4M3_SIGN_MASK) != 0;
}

HOST_DEVICE inline bool isfinite(fp8_e4m3 x)
{
    return !isnan(x);
}

} // namespace hipdnn_gpu_ref::types

// std::numeric_limits specialization
// NOLINTBEGIN(readability-identifier-naming) - standard library names must match exactly
template <>
class std::numeric_limits<hipdnn_gpu_ref::types::fp8_e4m3>
{
    using T = hipdnn_gpu_ref::types::fp8_e4m3;

public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = false; // OCP E4M3 has no infinity
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = false;
    static constexpr std::float_denorm_style has_denorm = std::denorm_present;
    static constexpr bool has_denorm_loss = false;
    static constexpr std::float_round_style round_style = std::round_to_nearest;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = 4; // 3 mantissa + 1 implicit
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 3;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -5;
    static constexpr int min_exponent10 = -2;
    static constexpr int max_exponent = 9;
    static constexpr int max_exponent10 = 2;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;

    HOST_DEVICE static constexpr T min() noexcept
    {
        return T::from_bits(hipdnn_gpu_ref::types::detail::FP8_E4M3_MIN_NORMAL);
    }
    HOST_DEVICE static constexpr T lowest() noexcept
    {
        return T::from_bits(hipdnn_gpu_ref::types::detail::FP8_E4M3_LOWEST);
    }
    HOST_DEVICE static constexpr T max() noexcept
    {
        return T::from_bits(hipdnn_gpu_ref::types::detail::FP8_E4M3_MAX);
    }
    HOST_DEVICE static constexpr T epsilon() noexcept
    {
        return T::from_bits(hipdnn_gpu_ref::types::detail::FP8_E4M3_EPSILON);
    }
    HOST_DEVICE static constexpr T round_error() noexcept
    {
        return T::from_bits(hipdnn_gpu_ref::types::detail::FP8_E4M3_ROUND_ERROR);
    }
    HOST_DEVICE static constexpr T infinity() noexcept
    {
        // no infinity, return max()
        return max();
    }

    HOST_DEVICE static constexpr T quiet_NaN() noexcept
    {
        return T::from_bits(hipdnn_gpu_ref::types::detail::FP8_E4M3_NAN);
    }
    HOST_DEVICE static constexpr T signaling_NaN() noexcept
    {
        return T::from_bits(hipdnn_gpu_ref::types::detail::FP8_E4M3_NAN);
    }
    HOST_DEVICE static constexpr T denorm_min() noexcept
    {
        return T::from_bits(hipdnn_gpu_ref::types::detail::FP8_E4M3_DENORM_MIN);
    }
};
// NOLINTEND(readability-identifier-naming)
