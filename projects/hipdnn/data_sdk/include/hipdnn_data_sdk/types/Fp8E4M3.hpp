// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <ostream>
#include <type_traits>

namespace hipdnn_data_sdk::types
{

// Forward declarations for cross-type conversions
// NOLINTBEGIN(readability-identifier-naming) - lowercase to match type definitions
enum class Bfloat16RoundingMode;
template <Bfloat16RoundingMode>
struct bfloat16_t;
struct half;
struct fp8_e5m2;
// NOLINTEND(readability-identifier-naming)

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

/// Sign bit mask (bit 7)
constexpr uint8_t FP8_E4M3_SIGN_MASK = 0x80;

/// Absolute value mask (all bits except sign)
constexpr uint8_t FP8_E4M3_ABS_MASK = 0x7F;

/// Exponent field mask (bits 3-6)
constexpr uint8_t FP8_E4M3_EXP_MASK = 0x78;

/// Mantissa field mask (bits 0-2)
constexpr uint8_t FP8_E4M3_MANT_MASK = 0x07;

/// Exponent bias for OCP E4M3
constexpr int FP8_E4M3_EXP_BIAS = 7;

/// Number of mantissa bits
constexpr int FP8_E4M3_MANT_BITS = 3;

/// Number of exponent bits
constexpr int FP8_E4M3_EXP_BITS = 4;

/// Maximum exponent value (before bias)
constexpr int FP8_E4M3_MAX_EXP = 15;

// ============================================================================
// FP8 E4M3 Special Values (bit patterns)
// ============================================================================

/// NaN: all mantissa bits set (0x7F for positive sign, 0xFF for negative)
constexpr uint8_t FP8_E4M3_NAN = 0x7F;

/// Maximum finite positive value: 0x7E = 448.0
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
// NOLINTNEXTLINE(readability-identifier-naming)
inline uint8_t float_to_fp8_e4m3_bits(float f, bool saturate = true) noexcept
{
    // Handle special values first using std library functions
    if(std::isnan(f))
    {
        return std::signbit(f) ? 0xFF : 0x7F; // NaN with preserved sign
    }

    if(std::isinf(f))
    {
        // E4M3 has no infinity - saturate to max or return NaN
        if(saturate)
        {
            return std::signbit(f) ? 0xFE : 0x7E; // Max finite value
        }
        return std::signbit(f) ? 0xFF : 0x7F; // NaN
    }

    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));

    const uint32_t sign = (bits >> 24) & 0x80; // Extract sign to bit 7
    const uint32_t fp32Exp = (bits >> 23) & 0xFF;
    int32_t exp = static_cast<int32_t>(fp32Exp) - 127 + 7; // Rebias from float (127) to E4M3 (7)
    uint32_t mant = bits & 0x007FFFFF;

    // Handle overflow
    if(exp >= 15)
    {
        if(saturate)
        {
            // Saturate to max finite value (0x7E for positive, 0xFE for negative)
            return static_cast<uint8_t>(sign | 0x7E);
        }
        return static_cast<uint8_t>(sign | 0x7F); // NaN (no infinity in E4M3)
    }

    // Handle zero
    if(fp32Exp == 0 && mant == 0)
    {
        return static_cast<uint8_t>(sign); // Signed zero
    }

    // Handle denormalized or underflow
    if(exp <= 0)
    {
        // Denormalized: shift formula is (1 - exp + 20) where exp can range from
        // -17 (deeply subnormal float) to 0, giving shift values from 21 to 38.
        // The guard condition (shift >= 24) ensures we don't shift beyond the
        // 24-bit mantissa, returning zero for values too small to represent.
        mant |= 0x00800000; // Add implicit 1
        auto shift = static_cast<uint32_t>(1 - exp + 20); // 23 - 3 = 20 bits to shift
        if(shift >= 24)
        {
            return static_cast<uint8_t>(sign); // Too small, return zero
        }
        mant >>= shift;
        return static_cast<uint8_t>(sign | (mant & 0x07));
    }

    // Normal case: shift mantissa from 23 bits to 3 bits with rounding
    uint32_t fp8Mant = (mant >> 20) & 0x07;
    const uint32_t remainder = mant & 0x000FFFFF;

    // Round to nearest even
    if(remainder > FP8_E4M3_ROUND_THRESHOLD
       || (remainder == FP8_E4M3_ROUND_THRESHOLD && ((fp8Mant & 1) != 0)))
    {
        fp8Mant++;
        if(fp8Mant > 7)
        {
            fp8Mant = 0;
            exp++;
            if(exp >= 15)
            {
                if(saturate)
                {
                    return static_cast<uint8_t>(sign | 0x7E);
                }
                return static_cast<uint8_t>(sign | 0x7F); // NaN
            }
        }
    }

    return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 3) | fp8Mant);
}

// Convert FP8 E4M3 bits to float
// NOLINTNEXTLINE(readability-identifier-naming)
inline float fp8_e4m3_bits_to_float(uint8_t bits) noexcept
{
    uint32_t sign = (static_cast<uint32_t>(bits) & 0x80) << 24;
    uint32_t exp = (bits >> 3) & 0x0F;
    uint32_t mant = bits & 0x07;

    // Handle NaN (0x7F or 0xFF)
    if((bits & 0x7F) == 0x7F)
    {
        uint32_t nanBits = sign | 0x7FC00000; // Quiet NaN
        float f;
        std::memcpy(&f, &nanBits, sizeof(float));
        return f;
    }

    // Handle zero
    if(exp == 0 && mant == 0)
    {
        float f;
        std::memcpy(&f, &sign, sizeof(float));
        return f;
    }

    // Handle denormalized
    if(exp == 0)
    {
        // Denormalized: value = (-1)^sign * 2^(-6) * (0.mantissa)
        float value = static_cast<float>(mant) / 8.0f * (1.0f / 64.0f);
        if(sign != 0)
        {
            value = -value;
        }
        return value;
    }

    // Normal case: rebias from E4M3 (7) to float (127)
    exp = exp - 7 + 127;
    mant <<= 20; // Shift 3-bit mantissa to 23-bit position

    uint32_t floatBits = sign | (exp << 23) | mant;
    float f;
    std::memcpy(&f, &floatBits, sizeof(float));
    return f;
}

} // namespace detail

/**
 * @brief Custom storage-only FP8 E4M3 type for hipDNN
 *
 * This type provides a portable FP8 E4M3 (1 sign, 4 exponent, 3 mantissa) implementation
 * that does not require the __HIPCC__ macro. Uses OCP E4M3 format.
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
    /// Raw bit representation of the FP8 E4M3 value.
    /// Public to ensure binary compatibility with HIP native types.
    uint8_t data;

    // Default constructor - value-initialized to zero for constexpr support
    constexpr fp8_e4m3() noexcept
        : data(0)
    {
    }

    // Copy/move constructors - implicit
    fp8_e4m3(const fp8_e4m3&) = default;
    fp8_e4m3(fp8_e4m3&&) noexcept = default;
    fp8_e4m3& operator=(const fp8_e4m3&) = default;
    fp8_e4m3& operator=(fp8_e4m3&&) noexcept = default;

    // EXPLICIT constructor from float
    explicit fp8_e4m3(float f) noexcept
        : data(detail::float_to_fp8_e4m3_bits(f))
    {
    }

    // EXPLICIT constructor from double (via float)
    explicit fp8_e4m3(double d) noexcept
        : data(detail::float_to_fp8_e4m3_bits(static_cast<float>(d)))
    {
    }

    // EXPLICIT constructor from integral types
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    explicit fp8_e4m3(T value) noexcept
        : data(detail::float_to_fp8_e4m3_bits(static_cast<float>(value)))
    {
    }

    // EXPLICIT constructors from other custom types (via float)
    // These are defined inline but require forward declarations above
    template <Bfloat16RoundingMode M>
    inline explicit fp8_e4m3(bfloat16_t<M> b) noexcept;
    inline explicit fp8_e4m3(half h) noexcept;
    inline explicit fp8_e4m3(fp8_e5m2 f) noexcept;

    // Factory for raw bits
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr fp8_e4m3 from_bits(uint8_t bits) noexcept
    {
        fp8_e4m3 val;
        val.data = bits;
        return val;
    }

    // EXPLICIT conversion to float
    explicit operator float() const noexcept
    {
        return detail::fp8_e4m3_bits_to_float(data);
    }

    // EXPLICIT conversion to double
    explicit operator double() const noexcept
    {
        return static_cast<double>(detail::fp8_e4m3_bits_to_float(data));
    }

    // Unary negation - XOR sign bit
    fp8_e4m3 operator-() const noexcept
    {
        return from_bits(data ^ detail::FP8_E4M3_SIGN_MASK);
    }

    // Unary plus
    fp8_e4m3 operator+() const noexcept
    {
        return *this;
    }

    // Stream output
    friend std::ostream& operator<<(std::ostream& os, fp8_e4m3 val)
    {
        return os << static_cast<float>(val);
    }
};

// Static assertions for binary compatibility
static_assert(sizeof(fp8_e4m3) == sizeof(uint8_t), "fp8_e4m3 must be 1 byte");
static_assert(std::is_trivially_copyable_v<fp8_e4m3>, "fp8_e4m3 must be trivially copyable");
static_assert(std::is_standard_layout_v<fp8_e4m3>, "fp8_e4m3 must be standard layout");

// User-defined literal
// NOLINTNEXTLINE(readability-identifier-naming)
inline fp8_e4m3 operator""_e4m3(long double val)
{
    return fp8_e4m3(static_cast<float>(val));
}

// ============================================================================
// Math functions for fp8_e4m3 (in hipdnn_data_sdk::types namespace)
// ============================================================================
// These are defined in our namespace to enable ADL (Argument Dependent Lookup).
// Use unqualified calls like: isnan(x), isinf(x), etc.
// ============================================================================

inline bool isnan(fp8_e4m3 x)
{
    // E4M3 NaN: all exponent and mantissa bits set (0x7F or 0xFF)
    return (x.data & detail::FP8_E4M3_ABS_MASK) == detail::FP8_E4M3_NAN;
}

inline bool isinf(fp8_e4m3 /*x*/)
{
    // E4M3 has no infinity representation
    return false;
}

inline bool signbit(fp8_e4m3 x)
{
    return (x.data & detail::FP8_E4M3_SIGN_MASK) != 0;
}

inline bool isfinite(fp8_e4m3 x)
{
    return !isnan(x);
}

} // namespace hipdnn_data_sdk::types

// std::numeric_limits specialization
// NOLINTBEGIN(readability-identifier-naming) - standard library names must match exactly
template <>
class std::numeric_limits<hipdnn_data_sdk::types::fp8_e4m3>
{
public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = false;
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
    static constexpr int min_exponent10 = -1;
    static constexpr int max_exponent = 8;
    static constexpr int max_exponent10 = 2;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;

    static constexpr hipdnn_data_sdk::types::fp8_e4m3 min() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e4m3::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E4M3_MIN_NORMAL);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e4m3 lowest() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e4m3::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E4M3_LOWEST);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e4m3 max() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e4m3::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E4M3_MAX);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e4m3 epsilon() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e4m3::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E4M3_EPSILON);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e4m3 round_error() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e4m3::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E4M3_ROUND_ERROR);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e4m3 infinity() noexcept
    {
        // E4M3 has no infinity, return max
        return max();
    }

    static constexpr hipdnn_data_sdk::types::fp8_e4m3 quiet_NaN() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e4m3::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E4M3_NAN);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e4m3 signaling_NaN() noexcept
    {
        // E4M3 has only one NaN representation
        return hipdnn_data_sdk::types::fp8_e4m3::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E4M3_NAN);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e4m3 denorm_min() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e4m3::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E4M3_DENORM_MIN);
    }
};
// NOLINTEND(readability-identifier-naming)
