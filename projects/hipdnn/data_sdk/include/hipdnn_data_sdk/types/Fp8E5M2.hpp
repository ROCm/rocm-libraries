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
struct fp8_e4m3;
// NOLINTEND(readability-identifier-naming)

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

/// Sign bit mask (bit 7)
constexpr uint8_t FP8_E5M2_SIGN_MASK = 0x80;

/// Absolute value mask (all bits except sign)
constexpr uint8_t FP8_E5M2_ABS_MASK = 0x7F;

/// Exponent field mask (bits 2-6)
constexpr uint8_t FP8_E5M2_EXP_MASK = 0x7C;

/// Mantissa field mask (bits 0-1)
constexpr uint8_t FP8_E5M2_MANT_MASK = 0x03;

/// Exponent bias for OCP E5M2
constexpr int FP8_E5M2_EXP_BIAS = 15;

/// Number of mantissa bits
constexpr int FP8_E5M2_MANT_BITS = 2;

/// Number of exponent bits
constexpr int FP8_E5M2_EXP_BITS = 5;

/// Maximum exponent value (before bias)
constexpr int FP8_E5M2_MAX_EXP = 31;

// ============================================================================
// FP8 E5M2 Special Values (bit patterns)
// ============================================================================

/// Positive infinity: exponent all 1s, mantissa 0 (0x7C)
constexpr uint8_t FP8_E5M2_POS_INF = 0x7C;

/// Negative infinity
constexpr uint8_t FP8_E5M2_NEG_INF = 0xFC;

/// Quiet NaN: exponent all 1s, mantissa all 1s (0x7F)
constexpr uint8_t FP8_E5M2_QNAN = 0x7F;

/// Signaling NaN: exponent all 1s, mantissa non-zero but not all 1s (0x7D)
constexpr uint8_t FP8_E5M2_SNAN = 0x7D;

/// Maximum finite positive value: 0x7B = 57344.0
constexpr uint8_t FP8_E5M2_MAX = 0x7B;

/// Minimum positive normal value: 2^-14 = 6.1e-5
constexpr uint8_t FP8_E5M2_MIN_NORMAL = 0x04;

/// Minimum positive denormal value
constexpr uint8_t FP8_E5M2_DENORM_MIN = 0x01;

/// Maximum finite negative value (lowest): -57344.0
constexpr uint8_t FP8_E5M2_LOWEST = 0xFB;

/// Epsilon: 2^-2 = 0.25
constexpr uint8_t FP8_E5M2_EPSILON = 0x34;

/// Round error (0.5)
constexpr uint8_t FP8_E5M2_ROUND_ERROR = 0x38;

/// Rounding threshold for round-to-nearest-even (midpoint of 21-bit remainder)
constexpr uint32_t FP8_E5M2_ROUND_THRESHOLD = 0x100000;

// Convert float to FP8 E5M2 bits (OCP format: 1 sign, 5 exponent, 2 mantissa)
// Range: +/- 57344, has infinity and NaN
// NOLINTNEXTLINE(readability-identifier-naming)
inline uint8_t float_to_fp8_e5m2_bits(float f, bool saturate = true) noexcept
{
    // Handle special values first using std library functions
    if(std::isnan(f))
    {
        return std::signbit(f) ? 0xFF : 0x7F; // NaN with preserved sign
    }

    if(std::isinf(f))
    {
        if(saturate)
        {
            return std::signbit(f) ? 0xFB : 0x7B; // Saturate to max finite
        }
        return std::signbit(f) ? 0xFC : 0x7C; // Infinity
    }

    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));

    const uint32_t sign = (bits >> 24) & 0x80; // Extract sign to bit 7
    const uint32_t fp32Exp = (bits >> 23) & 0xFF;
    int32_t exp = static_cast<int32_t>(fp32Exp) - 127 + 15; // Rebias from float (127) to E5M2 (15)
    uint32_t mant = bits & 0x007FFFFF;

    // Handle overflow
    if(exp >= 31)
    {
        if(saturate)
        {
            return static_cast<uint8_t>(sign | 0x7B); // max finite value
        }
        return static_cast<uint8_t>(sign | 0x7C); // Infinity
    }

    // Handle zero
    if(fp32Exp == 0 && mant == 0)
    {
        return static_cast<uint8_t>(sign); // Signed zero
    }

    // Handle denormalized or underflow
    if(exp <= 0)
    {
        // Denormalized
        mant |= 0x00800000; // Add implicit 1
        auto shift = static_cast<uint32_t>(1 - exp + 21); // 23 - 2 = 21 bits to shift
        if(shift >= 24)
        {
            return static_cast<uint8_t>(sign); // Too small, return zero
        }
        mant >>= shift;
        return static_cast<uint8_t>(sign | (mant & 0x03));
    }

    // Normal case: shift mantissa from 23 bits to 2 bits with rounding
    uint32_t fp8Mant = (mant >> 21) & 0x03;
    const uint32_t remainder = mant & 0x001FFFFF;

    // Round to nearest even
    if(remainder > FP8_E5M2_ROUND_THRESHOLD
       || (remainder == FP8_E5M2_ROUND_THRESHOLD && ((fp8Mant & 1) != 0)))
    {
        fp8Mant++;
        if(fp8Mant > 3)
        {
            fp8Mant = 0;
            exp++;
            if(exp >= 31)
            {
                if(saturate)
                {
                    return static_cast<uint8_t>(sign | 0x7B);
                }
                return static_cast<uint8_t>(sign | 0x7C); // Infinity
            }
        }
    }

    return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 2) | fp8Mant);
}

// Convert FP8 E5M2 bits to float
// NOLINTNEXTLINE(readability-identifier-naming)
inline float fp8_e5m2_bits_to_float(uint8_t bits) noexcept
{
    uint32_t sign = (static_cast<uint32_t>(bits) & 0x80) << 24;
    uint32_t exp = (bits >> 2) & 0x1F;
    uint32_t mant = bits & 0x03;

    // Handle infinity (exp=31, mant=0)
    if(exp == 31 && mant == 0)
    {
        uint32_t infBits = sign | 0x7F800000;
        float f;
        std::memcpy(&f, &infBits, sizeof(float));
        return f;
    }

    // Handle NaN (exp=31, mant!=0)
    if(exp == 31)
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
        // Denormalized: value = (-1)^sign * 2^(-14) * (0.mantissa)
        float value = static_cast<float>(mant) / 4.0f * (1.0f / 16384.0f);
        if(sign != 0)
        {
            value = -value;
        }
        return value;
    }

    // Normal case: rebias from E5M2 (15) to float (127)
    exp = exp - 15 + 127;
    mant <<= 21; // Shift 2-bit mantissa to 23-bit position

    uint32_t floatBits = sign | (exp << 23) | mant;
    float f;
    std::memcpy(&f, &floatBits, sizeof(float));
    return f;
}

} // namespace detail

/**
 * @brief Custom storage-only FP8 E5M2 type for hipDNN
 *
 * This type provides a portable FP8 E5M2 (1 sign, 5 exponent, 2 mantissa) implementation
 * that does not require the __HIPCC__ macro. Uses OCP E5M2 format.
 *
 * This is a STORAGE-ONLY type intended for data representation and conversion,
 * not direct computation. Arithmetic operations and comparisons are
 * intentionally not provided. For computation, explicitly convert to float.
 *
 * Binary layout: 1 sign bit, 5 exponent bits, 2 mantissa bits
 * Range: +/- 57344 (max normal value)
 * Has infinity and NaN representations
 */
// NOLINTNEXTLINE(readability-identifier-naming) - lowercase for consistency
struct fp8_e5m2
{
    /// Raw bit representation of the FP8 E5M2 value.
    /// Public to ensure binary compatibility with HIP native types.
    uint8_t data;

    // Default constructor - value-initialized to zero for constexpr support
    constexpr fp8_e5m2() noexcept
        : data(0)
    {
    }

    // Copy/move constructors - implicit
    fp8_e5m2(const fp8_e5m2&) = default;
    fp8_e5m2(fp8_e5m2&&) noexcept = default;
    fp8_e5m2& operator=(const fp8_e5m2&) = default;
    fp8_e5m2& operator=(fp8_e5m2&&) noexcept = default;

    // EXPLICIT constructor from float
    explicit fp8_e5m2(float f) noexcept
        : data(detail::float_to_fp8_e5m2_bits(f))
    {
    }

    // EXPLICIT constructor from double (via float)
    explicit fp8_e5m2(double d) noexcept
        : data(detail::float_to_fp8_e5m2_bits(static_cast<float>(d)))
    {
    }

    // EXPLICIT constructor from integral types
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    explicit fp8_e5m2(T value) noexcept
        : data(detail::float_to_fp8_e5m2_bits(static_cast<float>(value)))
    {
    }

    // EXPLICIT constructors from other custom types (via float)
    // These are defined inline but require forward declarations above
    template <Bfloat16RoundingMode M>
    inline explicit fp8_e5m2(bfloat16_t<M> b) noexcept;
    inline explicit fp8_e5m2(half h) noexcept;
    inline explicit fp8_e5m2(fp8_e4m3 f) noexcept;

    // Factory for raw bits
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr fp8_e5m2 from_bits(uint8_t bits) noexcept
    {
        fp8_e5m2 val;
        val.data = bits;
        return val;
    }

    // EXPLICIT conversion to float
    explicit operator float() const noexcept
    {
        return detail::fp8_e5m2_bits_to_float(data);
    }

    // EXPLICIT conversion to double
    explicit operator double() const noexcept
    {
        return static_cast<double>(detail::fp8_e5m2_bits_to_float(data));
    }

    // Unary negation - XOR sign bit
    fp8_e5m2 operator-() const noexcept
    {
        return from_bits(data ^ detail::FP8_E5M2_SIGN_MASK);
    }

    // Unary plus
    fp8_e5m2 operator+() const noexcept
    {
        return *this;
    }

    // Stream output
    friend std::ostream& operator<<(std::ostream& os, fp8_e5m2 val)
    {
        return os << static_cast<float>(val);
    }
};

// Static assertions for binary compatibility
static_assert(sizeof(fp8_e5m2) == sizeof(uint8_t), "fp8_e5m2 must be 1 byte");
static_assert(std::is_trivially_copyable_v<fp8_e5m2>, "fp8_e5m2 must be trivially copyable");
static_assert(std::is_standard_layout_v<fp8_e5m2>, "fp8_e5m2 must be standard layout");

// User-defined literal
// NOLINTNEXTLINE(readability-identifier-naming)
inline fp8_e5m2 operator""_e5m2(long double val)
{
    return fp8_e5m2(static_cast<float>(val));
}

// ============================================================================
// Math functions for fp8_e5m2 (in hipdnn_data_sdk::types namespace)
// ============================================================================
// These are defined in our namespace to enable ADL (Argument Dependent Lookup).
// Use unqualified calls like: isnan(x), isinf(x), etc.
// ============================================================================

inline bool isnan(fp8_e5m2 x)
{
    // E5M2 NaN: exponent all 1s and mantissa non-zero
    return (x.data & detail::FP8_E5M2_EXP_MASK) == detail::FP8_E5M2_EXP_MASK
           && (x.data & detail::FP8_E5M2_MANT_MASK) != 0;
}

inline bool isinf(fp8_e5m2 x)
{
    // E5M2 Infinity: exponent all 1s and mantissa 0
    return (x.data & detail::FP8_E5M2_ABS_MASK) == detail::FP8_E5M2_POS_INF;
}

inline bool signbit(fp8_e5m2 x)
{
    return (x.data & detail::FP8_E5M2_SIGN_MASK) != 0;
}

inline bool isfinite(fp8_e5m2 x)
{
    return !isnan(x) && !isinf(x);
}

} // namespace hipdnn_data_sdk::types

// std::numeric_limits specialization
// NOLINTBEGIN(readability-identifier-naming) - standard library names must match exactly
template <>
class std::numeric_limits<hipdnn_data_sdk::types::fp8_e5m2>
{
public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
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

    static constexpr hipdnn_data_sdk::types::fp8_e5m2 min() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E5M2_MIN_NORMAL);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2 lowest() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E5M2_LOWEST);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2 max() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E5M2_MAX);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2 epsilon() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E5M2_EPSILON);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2 round_error() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E5M2_ROUND_ERROR);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2 infinity() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E5M2_POS_INF);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2 quiet_NaN() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E5M2_QNAN);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2 signaling_NaN() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E5M2_SNAN);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2 denorm_min() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E5M2_DENORM_MIN);
    }
};
// NOLINTEND(readability-identifier-naming)
