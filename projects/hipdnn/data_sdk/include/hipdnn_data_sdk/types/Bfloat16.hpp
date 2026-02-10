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
struct half;
struct fp8_e4m3;
struct fp8_e5m2;
// NOLINTEND(readability-identifier-naming)

namespace detail
{

// ============================================================================
// Bfloat16 Bit Layout Constants
// ============================================================================
// bfloat16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
// Same exponent range as float32, truncated mantissa
//
// Bit layout: [S|EEEEEEEE|MMMMMMM]
//              15 14    7 6     0
// ============================================================================

/// Sign bit mask (bit 15)
constexpr uint16_t BFLOAT16_SIGN_MASK = 0x8000;

/// Absolute value mask (all bits except sign)
constexpr uint16_t BFLOAT16_ABS_MASK = 0x7FFF;

/// Exponent field mask (bits 7-14)
constexpr uint16_t BFLOAT16_EXP_MASK = 0x7F80;

/// Mantissa field mask (bits 0-6)
constexpr uint16_t BFLOAT16_MANT_MASK = 0x007F;

/// Exponent bias (same as float32)
constexpr int BFLOAT16_EXP_BIAS = 127;

// ============================================================================
// Bfloat16 Special Values (bit patterns)
// ============================================================================

/// Positive infinity: exponent all 1s, mantissa all 0s
constexpr uint16_t BFLOAT16_POS_INF = 0x7F80;

/// Negative infinity
constexpr uint16_t BFLOAT16_NEG_INF = 0xFF80;

/// Quiet NaN (canonical): exponent all 1s, MSB of mantissa set
constexpr uint16_t BFLOAT16_QNAN = 0x7FC0;

/// Signaling NaN: exponent all 1s, mantissa non-zero but MSB clear
constexpr uint16_t BFLOAT16_SNAN = 0x7F81;

/// Canonical NaN for min/max operations
constexpr uint16_t BFLOAT16_CANONICAL_NAN = 0x7FFF;

/// Maximum finite positive value: 0x7F7F = 3.3895e+38
constexpr uint16_t BFLOAT16_MAX = 0x7F7F;

/// Minimum positive normal value: 2^-126 = 1.175e-38
constexpr uint16_t BFLOAT16_MIN_NORMAL = 0x0080;

/// Minimum positive denormal value
constexpr uint16_t BFLOAT16_DENORM_MIN = 0x0001;

/// Maximum finite negative value (lowest): -3.3895e+38
constexpr uint16_t BFLOAT16_LOWEST = 0xFF7F;

/// Epsilon: smallest value such that 1.0 + epsilon != 1.0 (2^-7)
constexpr uint16_t BFLOAT16_EPSILON = 0x3C00;

/// Round error (0.5)
constexpr uint16_t BFLOAT16_ROUND_ERROR = 0x3F00;

// NOLINTBEGIN(readability-identifier-naming) - using snake_case for internal detail functions

// Convert float to bfloat16 bits using truncation (matches HIP behavior)
inline uint16_t float_to_bfloat16_bits(float f) noexcept
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    // Truncate lower 16 bits (simple truncation, no rounding)
    return static_cast<uint16_t>(bits >> 16);
}

// Convert bfloat16 bits to float
inline float bfloat16_bits_to_float(uint16_t bits) noexcept
{
    uint32_t floatBits = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &floatBits, sizeof(float));
    return f;
}

// NOLINTEND(readability-identifier-naming)

} // namespace detail

/**
 * @brief Custom bfloat16 type for hipDNN
 *
 * This type provides a portable bfloat16 implementation that does not require
 * the __HIPCC__ macro. Both constructors from float/double and conversions TO
 * float/double are explicit to prevent silent precision loss and overload ambiguity.
 *
 * Binary layout is compatible with hip_bfloat16 (16-bit, same bit representation).
 */
// NOLINTNEXTLINE(readability-identifier-naming) - lowercase to match hip_bfloat16 convention
struct bfloat16
{
    uint16_t data;

    // Default constructor - value-initialized to zero for constexpr support
    constexpr bfloat16() noexcept
        : data(0)
    {
    }

    // Copy/move constructors - implicit
    bfloat16(const bfloat16&) = default;
    bfloat16(bfloat16&&) noexcept = default;
    bfloat16& operator=(const bfloat16&) = default;
    bfloat16& operator=(bfloat16&&) noexcept = default;

    // EXPLICIT constructor from float
    explicit bfloat16(float f) noexcept
        : data(detail::float_to_bfloat16_bits(f))
    {
    }

    // EXPLICIT constructor from double (via float)
    explicit bfloat16(double d) noexcept
        : data(detail::float_to_bfloat16_bits(static_cast<float>(d)))
    {
    }

    // EXPLICIT constructor from integral types
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    explicit bfloat16(T value) noexcept
        : data(detail::float_to_bfloat16_bits(static_cast<float>(value)))
    {
    }

    // EXPLICIT constructors from other custom types (via float)
    // These are defined inline but require forward declarations above
    inline explicit bfloat16(half h) noexcept;
    inline explicit bfloat16(fp8_e4m3 f) noexcept;
    inline explicit bfloat16(fp8_e5m2 f) noexcept;

    // Factory for raw bits
    // NOLINTNEXTLINE(readability-identifier-naming) - using snake_case for factory function
    static constexpr bfloat16 from_bits(uint16_t bits) noexcept
    {
        bfloat16 val;
        val.data = bits;
        return val;
    }

    // EXPLICIT conversion to float
    explicit operator float() const noexcept
    {
        return detail::bfloat16_bits_to_float(data);
    }

    // EXPLICIT conversion to double
    explicit operator double() const noexcept
    {
        return static_cast<double>(detail::bfloat16_bits_to_float(data));
    }

    // Unary negation - XOR sign bit
    bfloat16 operator-() const noexcept
    {
        return from_bits(data ^ detail::BFLOAT16_SIGN_MASK);
    }

    // Unary plus
    bfloat16 operator+() const noexcept
    {
        return *this;
    }

    // Arithmetic operators (compute in float, return bfloat16)
    friend bfloat16 operator+(bfloat16 a, bfloat16 b) noexcept
    {
        return bfloat16(static_cast<float>(a) + static_cast<float>(b));
    }

    friend bfloat16 operator-(bfloat16 a, bfloat16 b) noexcept
    {
        return bfloat16(static_cast<float>(a) - static_cast<float>(b));
    }

    friend bfloat16 operator*(bfloat16 a, bfloat16 b) noexcept
    {
        return bfloat16(static_cast<float>(a) * static_cast<float>(b));
    }

    friend bfloat16 operator/(bfloat16 a, bfloat16 b) noexcept
    {
        return bfloat16(static_cast<float>(a) / static_cast<float>(b));
    }

    // Compound assignment operators
    bfloat16& operator+=(bfloat16 other) noexcept
    {
        *this = *this + other;
        return *this;
    }

    bfloat16& operator-=(bfloat16 other) noexcept
    {
        *this = *this - other;
        return *this;
    }

    bfloat16& operator*=(bfloat16 other) noexcept
    {
        *this = *this * other;
        return *this;
    }

    bfloat16& operator/=(bfloat16 other) noexcept
    {
        *this = *this / other;
        return *this;
    }

    // Comparison operators (compare via float conversion)
    friend bool operator==(bfloat16 a, bfloat16 b) noexcept
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }

    friend bool operator!=(bfloat16 a, bfloat16 b) noexcept
    {
        return static_cast<float>(a) != static_cast<float>(b);
    }

    friend bool operator<(bfloat16 a, bfloat16 b) noexcept
    {
        return static_cast<float>(a) < static_cast<float>(b);
    }

    friend bool operator>(bfloat16 a, bfloat16 b) noexcept
    {
        return static_cast<float>(a) > static_cast<float>(b);
    }

    friend bool operator<=(bfloat16 a, bfloat16 b) noexcept
    {
        return static_cast<float>(a) <= static_cast<float>(b);
    }

    friend bool operator>=(bfloat16 a, bfloat16 b) noexcept
    {
        return static_cast<float>(a) >= static_cast<float>(b);
    }

    // Stream output
    friend std::ostream& operator<<(std::ostream& os, bfloat16 val)
    {
        return os << static_cast<float>(val);
    }
};

// Static assertions for binary compatibility
static_assert(sizeof(bfloat16) == sizeof(uint16_t), "bfloat16 must be 2 bytes");
static_assert(std::is_trivially_copyable_v<bfloat16>, "bfloat16 must be trivially copyable");
static_assert(std::is_standard_layout_v<bfloat16>, "bfloat16 must be standard layout");

// User-defined literal
inline bfloat16 operator""_bf(long double val)
{
    return bfloat16(static_cast<float>(val));
}

// ============================================================================
// Math functions for bfloat16 (in hipdnn_data_sdk::types namespace)
// ============================================================================
// These are defined in our namespace to enable ADL (Argument Dependent Lookup).
// Use unqualified calls like: fabs(x), isnan(x), etc.
// ============================================================================

// Basic math functions
inline bfloat16 abs(bfloat16 x)
{
    return bfloat16::from_bits(x.data & detail::BFLOAT16_ABS_MASK);
}

inline bfloat16 fabs(bfloat16 x)
{
    return bfloat16::from_bits(x.data & detail::BFLOAT16_ABS_MASK);
}

inline bool isnan(bfloat16 x)
{
    // NaN: exponent all 1s and non-zero mantissa
    return (x.data & detail::BFLOAT16_EXP_MASK) == detail::BFLOAT16_EXP_MASK
           && (x.data & detail::BFLOAT16_MANT_MASK) != 0;
}

inline bool isinf(bfloat16 x)
{
    // Inf: exponent all 1s and zero mantissa
    return (x.data & detail::BFLOAT16_ABS_MASK) == detail::BFLOAT16_POS_INF;
}

inline bool signbit(bfloat16 x)
{
    return (x.data & detail::BFLOAT16_SIGN_MASK) != 0;
}

inline bool isfinite(bfloat16 x)
{
    return !isnan(x) && !isinf(x);
}

inline bfloat16 copysign(bfloat16 x, bfloat16 y)
{
    uint16_t xBits = x.data & detail::BFLOAT16_ABS_MASK;
    uint16_t ySign = y.data & detail::BFLOAT16_SIGN_MASK;
    return bfloat16::from_bits(xBits | ySign);
}

// Min/max with NaN handling
inline bfloat16 max(bfloat16 a, bfloat16 b)
{
    if(isnan(a) && isnan(b))
    {
        return bfloat16::from_bits(detail::BFLOAT16_CANONICAL_NAN);
    }
    if(isnan(a))
    {
        return b;
    }
    if(isnan(b))
    {
        return a;
    }
    return a > b ? a : b;
}

inline bfloat16 min(bfloat16 a, bfloat16 b)
{
    if(isnan(a) && isnan(b))
    {
        return bfloat16::from_bits(detail::BFLOAT16_CANONICAL_NAN);
    }
    if(isnan(a))
    {
        return b;
    }
    if(isnan(b))
    {
        return a;
    }
    return a < b ? a : b;
}

// Rounding functions
inline bfloat16 floor(bfloat16 x)
{
    return bfloat16(std::floor(static_cast<float>(x)));
}

inline bfloat16 ceil(bfloat16 x)
{
    return bfloat16(std::ceil(static_cast<float>(x)));
}

inline bfloat16 round(bfloat16 x)
{
    return bfloat16(std::round(static_cast<float>(x)));
}

inline bfloat16 trunc(bfloat16 x)
{
    return bfloat16(std::trunc(static_cast<float>(x)));
}

// Exponential and logarithmic functions
inline bfloat16 exp(bfloat16 x)
{
    return bfloat16(std::exp(static_cast<float>(x)));
}

inline bfloat16 exp2(bfloat16 x)
{
    return bfloat16(std::exp2(static_cast<float>(x)));
}

inline bfloat16 log(bfloat16 x)
{
    return bfloat16(std::log(static_cast<float>(x)));
}

inline bfloat16 log2(bfloat16 x)
{
    return bfloat16(std::log2(static_cast<float>(x)));
}

inline bfloat16 log10(bfloat16 x)
{
    return bfloat16(std::log10(static_cast<float>(x)));
}

// Power functions
inline bfloat16 sqrt(bfloat16 x)
{
    return bfloat16(std::sqrt(static_cast<float>(x)));
}

inline bfloat16 rsqrt(bfloat16 x)
{
    return bfloat16(1.0f / std::sqrt(static_cast<float>(x)));
}

inline bfloat16 pow(bfloat16 x, bfloat16 y)
{
    return bfloat16(std::pow(static_cast<float>(x), static_cast<float>(y)));
}

// Trigonometric functions
inline bfloat16 sin(bfloat16 x)
{
    return bfloat16(std::sin(static_cast<float>(x)));
}

inline bfloat16 cos(bfloat16 x)
{
    return bfloat16(std::cos(static_cast<float>(x)));
}

inline bfloat16 tan(bfloat16 x)
{
    return bfloat16(std::tan(static_cast<float>(x)));
}

inline bfloat16 asin(bfloat16 x)
{
    return bfloat16(std::asin(static_cast<float>(x)));
}

inline bfloat16 acos(bfloat16 x)
{
    return bfloat16(std::acos(static_cast<float>(x)));
}

inline bfloat16 atan(bfloat16 x)
{
    return bfloat16(std::atan(static_cast<float>(x)));
}

// Hyperbolic functions
inline bfloat16 sinh(bfloat16 x)
{
    return bfloat16(std::sinh(static_cast<float>(x)));
}

inline bfloat16 cosh(bfloat16 x)
{
    return bfloat16(std::cosh(static_cast<float>(x)));
}

inline bfloat16 tanh(bfloat16 x)
{
    return bfloat16(std::tanh(static_cast<float>(x)));
}

// Error function
inline bfloat16 erf(bfloat16 x)
{
    return bfloat16(std::erf(static_cast<float>(x)));
}

// Floating-point manipulation
inline bfloat16 fmod(bfloat16 x, bfloat16 y)
{
    return bfloat16(std::fmod(static_cast<float>(x), static_cast<float>(y)));
}

// Fused multiply-add
inline bfloat16 fma(bfloat16 x, bfloat16 y, bfloat16 z)
{
    return bfloat16(std::fma(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)));
}

} // namespace hipdnn_data_sdk::types

// std::numeric_limits specialization
// NOLINTBEGIN(readability-identifier-naming) - standard library names must match exactly
template <>
class std::numeric_limits<hipdnn_data_sdk::types::bfloat16>
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
    static constexpr int digits = 8; // 7 mantissa bits + 1 implicit
    static constexpr int digits10 = 2;
    static constexpr int max_digits10 = 4;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -125;
    static constexpr int min_exponent10 = -37;
    static constexpr int max_exponent = 128;
    static constexpr int max_exponent10 = 38;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;

    static constexpr hipdnn_data_sdk::types::bfloat16 min() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16::from_bits(
            hipdnn_data_sdk::types::detail::BFLOAT16_MIN_NORMAL);
    }

    static constexpr hipdnn_data_sdk::types::bfloat16 lowest() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16::from_bits(
            hipdnn_data_sdk::types::detail::BFLOAT16_LOWEST);
    }

    static constexpr hipdnn_data_sdk::types::bfloat16 max() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16::from_bits(
            hipdnn_data_sdk::types::detail::BFLOAT16_MAX);
    }

    static constexpr hipdnn_data_sdk::types::bfloat16 epsilon() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16::from_bits(
            hipdnn_data_sdk::types::detail::BFLOAT16_EPSILON);
    }

    static constexpr hipdnn_data_sdk::types::bfloat16 round_error() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16::from_bits(
            hipdnn_data_sdk::types::detail::BFLOAT16_ROUND_ERROR);
    }

    static constexpr hipdnn_data_sdk::types::bfloat16 infinity() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16::from_bits(
            hipdnn_data_sdk::types::detail::BFLOAT16_POS_INF);
    }

    static constexpr hipdnn_data_sdk::types::bfloat16 quiet_NaN() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16::from_bits(
            hipdnn_data_sdk::types::detail::BFLOAT16_QNAN);
    }

    static constexpr hipdnn_data_sdk::types::bfloat16 signaling_NaN() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16::from_bits(
            hipdnn_data_sdk::types::detail::BFLOAT16_SNAN);
    }

    static constexpr hipdnn_data_sdk::types::bfloat16 denorm_min() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16::from_bits(
            hipdnn_data_sdk::types::detail::BFLOAT16_DENORM_MIN);
    }
};
// NOLINTEND(readability-identifier-naming)
