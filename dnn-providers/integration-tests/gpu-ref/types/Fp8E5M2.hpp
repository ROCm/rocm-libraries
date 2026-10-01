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
// FP8 E5M2 (OCP Format) Bit Layout Constants
// ============================================================================
// OCP E5M2 format: 1 sign bit, 5 exponent bits, 2 mantissa bits
// - Has infinity representation (exponent all 1s, mantissa 0)
// - Has NaN representation (exponent all 1s, mantissa non-zero)
//
// Bit layout: [S|EEEEE|MM]
//              7 6   2 1 0
// ============================================================================

// Sign bit mask (bit 7)
constexpr uint8_t FP8_E5M2_SIGN_MASK = 0x80;

// Absolute value mask (all bits except sign)
constexpr uint8_t FP8_E5M2_ABS_MASK = 0x7F;

// Exponent field mask (bits 2-6)
constexpr uint8_t FP8_E5M2_EXP_MASK = 0x7C;

// Mantissa field mask (bits 0-1)
constexpr uint8_t FP8_E5M2_MANT_MASK = 0x03;

// Exponent bias for OCP E5M2
constexpr int FP8_E5M2_EXP_BIAS = 15;

// Number of mantissa bits
constexpr int FP8_E5M2_MANT_BITS = 2;

// Number of exponent bits
constexpr int FP8_E5M2_EXP_BITS = 5;

// Maximum biased exponent representable in EXP_BITS bits
constexpr int FP8_E5M2_MAX_BIASED_EXP = (1 << FP8_E5M2_EXP_BITS) - 1;

// Positive infinity: exponent all 1s, mantissa 0 (0x7C)
constexpr uint8_t FP8_E5M2_POS_INF = 0x7C;

// Negative infinity
constexpr uint8_t FP8_E5M2_NEG_INF = 0xFC;

// NaN: exponent all 1s, mantissa non-zero (0x7F is)
// Note: OCP E5M2 does not distinguish between signaling and quiet NaN.
// All NaN encodings S.11111.{01, 10, 11} are treated as (quiet) NaN.
constexpr uint8_t FP8_E5M2_NAN = 0x7F;

// Smallest absolute bit pattern that decodes to NaN (0x7D).
// NaN range is [0x7D, 0x7F]; equivalently exp=31 and mant != 0.
constexpr uint8_t FP8_E5M2_NAN_MIN = 0x7D;

// Maximum finite positive value: 0x7B = 57344.0
constexpr uint8_t FP8_E5M2_MAX = 0x7B;

// Minimum positive normal value: 2^-14 = 6.1e-5
constexpr uint8_t FP8_E5M2_MIN_NORMAL = 0x04;

// Minimum positive denormal value: 2^-16 = 1.5e-5
constexpr uint8_t FP8_E5M2_DENORM_MIN = 0x01;

// Maximum finite negative value (lowest): -57344.0
constexpr uint8_t FP8_E5M2_LOWEST = 0xFB;

// Epsilon: 2^-2 = 0.25
constexpr uint8_t FP8_E5M2_EPSILON = 0x34;

// Round error (0.5)
constexpr uint8_t FP8_E5M2_ROUND_ERROR = 0x38;

// Rounding threshold for round-to-nearest-even (midpoint of 21-bit remainder)
constexpr uint32_t FP8_E5M2_ROUND_THRESHOLD = 0x100000;

// Convert float to FP8 E5M2 bits
// Range: +/- 57344, has infinity and NaN
// NOTE: The `saturate=false` paths return Inf on overflow per the OCP spec and
// are reserved for future non-saturating-mode support.
// NOLINTNEXTLINE(readability-identifier-naming)
HOST_DEVICE inline uint8_t float_to_fp8_e5m2_bits(float f, bool saturate = true) noexcept
{
    const uint32_t bits = floatToBits(f);
    const uint32_t absBits = bits & 0x7FFFFFFFu;
    const uint32_t sign = (bits >> 24) & FP8_E5M2_SIGN_MASK; // sign moved to bit 7

    // NaN (sign preserved)
    if(absBits > 0x7F800000u)
    {
        return static_cast<uint8_t>(sign | FP8_E5M2_NAN);
    }

    // Infinity
    if(absBits == 0x7F800000u)
    {
        return static_cast<uint8_t>(sign | (saturate ? FP8_E5M2_MAX : FP8_E5M2_POS_INF));
    }

    const uint32_t fp32Exp = absBits >> 23;
    int32_t exp = static_cast<int32_t>(fp32Exp) - 127 + FP8_E5M2_EXP_BIAS;
    uint32_t mant = bits & 0x007FFFFFu;

    // +/-0 and fp32 subnormals: far below the smallest E5M2 value -> signed zero
    if(fp32Exp == 0)
    {
        return static_cast<uint8_t>(sign);
    }

    // Overflow (biased exp 31 is reserved for Inf/NaN, hence >=)
    if(exp >= FP8_E5M2_MAX_BIASED_EXP)
    {
        return static_cast<uint8_t>(sign | (saturate ? FP8_E5M2_MAX : FP8_E5M2_POS_INF));
    }

    // Subnormal / underflow range (may round up into the smallest normal)
    if(exp <= 0)
    {
        mant |= 0x00800000u; // implicit 1
        const auto shift = static_cast<uint32_t>(1 - exp + 21);
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
            if(mant > FP8_E5M2_MANT_MASK)
            {
                return static_cast<uint8_t>(sign | FP8_E5M2_MIN_NORMAL);
            }
        }
        if(mant == 0u)
        {
            return static_cast<uint8_t>(sign);
        }
        return static_cast<uint8_t>(sign | (mant & FP8_E5M2_MANT_MASK));
    }

    // Normal range: 23 -> 2 mantissa bits, round-to-nearest-even
    uint32_t fp8Mant = (mant >> 21) & FP8_E5M2_MANT_MASK;
    const uint32_t remainder = mant & 0x001FFFFFu;

    // Round to nearest even
    if(remainder > FP8_E5M2_ROUND_THRESHOLD
       || (remainder == FP8_E5M2_ROUND_THRESHOLD && ((fp8Mant & 1u) != 0u)))
    {
        fp8Mant++;
        if(fp8Mant > FP8_E5M2_MANT_MASK)
        {
            fp8Mant = 0;
            exp++;
            if(exp >= FP8_E5M2_MAX_BIASED_EXP)
            {
                return static_cast<uint8_t>(sign | (saturate ? FP8_E5M2_MAX : FP8_E5M2_POS_INF));
            }
        }
    }

    return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 2) | fp8Mant);
}

// Convert FP8 E5M2 bits to float
// NOLINTNEXTLINE(readability-identifier-naming)
HOST_DEVICE inline float fp8_e5m2_bits_to_float(uint8_t b) noexcept
{
    const uint32_t sign = (static_cast<uint32_t>(b) & 0x80u) << 24;
    const uint32_t exp = (static_cast<uint32_t>(b) >> 2) & 0x1Fu;
    const uint32_t mant = static_cast<uint32_t>(b) & 0x3u;

    uint32_t out;
    if(exp == 0x1Fu)
    {
        // Inf (mant == 0) or NaN (mant != 0) with sign preserved
        out = sign | 0x7F800000u | (mant != 0u ? 0x00400000u : 0u);
    }
    else if(exp == 0u)
    {
        // Zero / subnormal: mant * 2^-16
        if(mant == 0u)
        {
            out = sign;
        }
        else if(mant == 1u)
        {
            out = sign | (111u << 23); // 2^-16
        }
        else if(mant == 2u)
        {
            out = sign | (112u << 23); // 2^-15
        }
        else
        {
            out = sign | (112u << 23) | 0x00400000u; // 1.5 * 2^-15
        }
    }
    else
    {
        // Normal: rebias 15 -> 127 (+112), mantissa into top 2 bits of the fp32 mantissa
        out = sign | ((exp + 112u) << 23) | (mant << 21);
    }

    return bitsToFloat(out);
}

} // namespace detail

/**
 * @brief Custom storage-only FP8 E5M2 type on device
 *
 * This type provides a portable FP8 E5M2 (1 sign, 5 exponent, 2 mantissa)
 * implementation. Uses OCP E5M2 format.
 *
 * This is a STORAGE-ONLY type intended for data representation and conversion,
 * not direct computation. Arithmetic operations and comparisons are
 * intentionally not provided. For computation, explicitly convert to float.
 *
 * Binary layout: 1 sign bit, 5 exponent bits, 2 mantissa bits
 * Range: +/- 57344 (max normal value)
 * Has infinity and NaN representations
 */
// NOLINTNEXTLINE(readability-identifier-naming)
struct fp8_e5m2
{
    // Raw bits representation of the fp8_e5m2 value on device
    uint8_t data;

    constexpr fp8_e5m2() noexcept
        : data(0)
    {
    }

    fp8_e5m2(const fp8_e5m2&) = default;
    fp8_e5m2(fp8_e5m2&&) noexcept = default;
    fp8_e5m2& operator=(const fp8_e5m2&) = default;
    fp8_e5m2& operator=(fp8_e5m2&&) noexcept = default;

    HOST_DEVICE explicit fp8_e5m2(float f) noexcept
        : data(detail::float_to_fp8_e5m2_bits(f))
    {
    }

    HOST_DEVICE explicit fp8_e5m2(double d) noexcept
        : fp8_e5m2(static_cast<float>(d))
    {
    }

    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    HOST_DEVICE explicit fp8_e5m2(T value) noexcept
        : fp8_e5m2(static_cast<float>(value))
    {
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    HOST_DEVICE static constexpr fp8_e5m2 from_bits(uint8_t bits) noexcept
    {
        fp8_e5m2 val;
        val.data = bits;
        return val;
    }

    HOST_DEVICE explicit operator float() const noexcept
    {
        return detail::fp8_e5m2_bits_to_float(data);
    }

    HOST_DEVICE explicit operator double() const noexcept
    {
        return static_cast<double>(detail::fp8_e5m2_bits_to_float(data));
    }

    HOST_DEVICE fp8_e5m2 operator-() const noexcept
    {
        return from_bits(static_cast<uint8_t>(data ^ detail::FP8_E5M2_SIGN_MASK));
    }

    HOST_DEVICE fp8_e5m2 operator+() const noexcept
    {
        return *this;
    }
};

// Static assertions for binary compatibility
static_assert(sizeof(fp8_e5m2) == sizeof(uint8_t), "fp8_e5m2 must be 1 byte");
static_assert(std::is_trivially_copyable_v<fp8_e5m2>, "fp8_e5m2 must be trivially copyable");
static_assert(std::is_standard_layout_v<fp8_e5m2>, "fp8_e5m2 must be standard layout");
static_assert(std::is_default_constructible_v<fp8_e5m2>, "fp8_e5m2 must be default constructible");
static_assert(std::is_copy_constructible_v<fp8_e5m2>, "fp8_e5m2 must be copy constructible");
static_assert(std::is_move_constructible_v<fp8_e5m2>, "fp8_e5m2 must be move constructible");

// ============================================================================
// Math functions for fp8_e5m2
// ============================================================================
// These are defined in our namespace to enable ADL (Argument Dependent Lookup).
// Use unqualified calls like: isnan(x), isinf(x), etc.
// ============================================================================

HOST_DEVICE inline bool isnan(fp8_e5m2 x)
{
    return (x.data & detail::FP8_E5M2_EXP_MASK) == detail::FP8_E5M2_EXP_MASK
           && (x.data & detail::FP8_E5M2_MANT_MASK) != 0;
}

HOST_DEVICE inline bool isinf(fp8_e5m2 x)
{
    return (x.data & detail::FP8_E5M2_ABS_MASK) == detail::FP8_E5M2_POS_INF;
}

HOST_DEVICE inline bool signbit(fp8_e5m2 x)
{
    return (x.data & detail::FP8_E5M2_SIGN_MASK) != 0;
}

HOST_DEVICE inline bool isfinite(fp8_e5m2 x)
{
    return !isnan(x) && !isinf(x);
}

} // namespace hipdnn_gpu_ref::types

// std::numeric_limits specialization
// NOLINTBEGIN(readability-identifier-naming) - standard library names must match exactly
template <>
class std::numeric_limits<hipdnn_gpu_ref::types::fp8_e5m2>
{
    using T = hipdnn_gpu_ref::types::fp8_e5m2;

public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = false; // OCP E5M2 does not distinguish signaling NaN
    static constexpr std::float_denorm_style has_denorm = std::denorm_present;
    static constexpr bool has_denorm_loss = false;
    static constexpr std::float_round_style round_style = std::round_to_nearest;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = 3; // 2 mantissa + 1 implicit
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 2;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -13;
    static constexpr int min_exponent10 = -4;
    static constexpr int max_exponent = 16;
    static constexpr int max_exponent10 = 4;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;

    HOST_DEVICE static constexpr T min() noexcept
    {
        return T::from_bits(hipdnn_gpu_ref::types::detail::FP8_E5M2_MIN_NORMAL);
    }
    HOST_DEVICE static constexpr T lowest() noexcept
    {
        return T::from_bits(hipdnn_gpu_ref::types::detail::FP8_E5M2_LOWEST);
    }
    HOST_DEVICE static constexpr T max() noexcept
    {
        return T::from_bits(hipdnn_gpu_ref::types::detail::FP8_E5M2_MAX);
    }
    HOST_DEVICE static constexpr T epsilon() noexcept
    {
        return T::from_bits(hipdnn_gpu_ref::types::detail::FP8_E5M2_EPSILON);
    }
    HOST_DEVICE static constexpr T round_error() noexcept
    {
        return T::from_bits(hipdnn_gpu_ref::types::detail::FP8_E5M2_ROUND_ERROR);
    }
    HOST_DEVICE static constexpr T infinity() noexcept
    {
        return T::from_bits(hipdnn_gpu_ref::types::detail::FP8_E5M2_POS_INF);
    }
    HOST_DEVICE static constexpr T quiet_NaN() noexcept
    {
        return T::from_bits(hipdnn_gpu_ref::types::detail::FP8_E5M2_NAN);
    }
    HOST_DEVICE static constexpr T signaling_NaN() noexcept
    {
        return T::from_bits(hipdnn_gpu_ref::types::detail::FP8_E5M2_NAN);
    }
    HOST_DEVICE static constexpr T denorm_min() noexcept
    {
        return T::from_bits(hipdnn_gpu_ref::types::detail::FP8_E5M2_DENORM_MIN);
    }
};
// NOLINTEND(readability-identifier-naming)
