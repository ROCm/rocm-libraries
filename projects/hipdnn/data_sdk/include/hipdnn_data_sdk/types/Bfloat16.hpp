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
struct fp4_e2m1;
struct fp6_e2m3;
struct fp6_e3m2;
struct fp8_e4m3;
struct fp8_e5m2;
struct fp8_e8m0;
// NOLINTEND(readability-identifier-naming)

// ============================================================================
// Bfloat16 Rounding Mode
// ============================================================================

/// Rounding mode for bfloat16 float-to-bfloat16 conversions
enum class Bfloat16RoundingMode
{
    /// Round-to-nearest-even (default, matches half and fp8 types)
    // NOLINTNEXTLINE(readability-identifier-naming) - RNE is an industry-standard acronym
    RNE,
    /// Simple truncation (faster, matches some HIP implementations)
    // NOLINTNEXTLINE(readability-identifier-naming) - CamelCase for enum values
    Truncate
};

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

// Convert float to bfloat16 bits using simple truncation
// This is faster than RNE and matches some HIP implementations
// NOLINTNEXTLINE(readability-identifier-naming)
inline uint16_t float_to_bfloat16_bits_truncate(float f) noexcept
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    return static_cast<uint16_t>(bits >> 16);
}

// Convert float to bfloat16 bits using round-to-nearest-even (RNE)
// This matches the rounding behavior of half and fp8 types
// NOLINTNEXTLINE(readability-identifier-naming)
inline uint16_t float_to_bfloat16_bits_rne(float f) noexcept
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));

    // Extract upper 16 bits (this will be the result before rounding)
    auto upper = static_cast<uint16_t>(bits >> 16);

    // Check for NaN: if exponent is all 1s and mantissa is non-zero, preserve NaN
    // NaN bit patterns have exponent 0xFF and non-zero mantissa in the full float
    const uint32_t exp = (bits >> 23) & 0xFF;
    const uint32_t mant = bits & 0x007FFFFF;
    if(exp == 0xFF && mant != 0)
    {
        // Preserve NaN - ensure mantissa is non-zero in bfloat16
        // Set the quiet NaN bit if needed
        return upper | 0x0040; // Ensure at least quiet NaN bit is set
    }

    // Extract lower 16 bits (the bits being truncated)
    auto lower = static_cast<uint16_t>(bits & 0xFFFF);

    // Round-to-nearest-even:
    // - Round up if remainder > 0.5 (lower > 0x8000)
    // - Round up if remainder == 0.5 and result LSB is 1 (tie-break to even)
    if(lower > 0x8000 || (lower == 0x8000 && (upper & 1) != 0))
    {
        upper++;
        // Note: overflow from 0x7F7F to 0x7F80 produces infinity, which is correct
        // Overflow from 0xFF7F to 0xFF80 produces -infinity, also correct
    }

    return upper;
}

// Templated conversion function that selects rounding mode at compile time
// NOLINTBEGIN(readability-identifier-naming)
template <Bfloat16RoundingMode Mode>
inline uint16_t float_to_bfloat16_bits(float f) noexcept
// NOLINTEND(readability-identifier-naming)
{
    if constexpr(Mode == Bfloat16RoundingMode::Truncate)
    {
        return float_to_bfloat16_bits_truncate(f);
    }
    else
    {
        return float_to_bfloat16_bits_rne(f);
    }
}

// Default conversion uses RNE for consistency with half and fp8 types
// NOLINTNEXTLINE(readability-identifier-naming)
inline uint16_t float_to_bfloat16_bits(float f) noexcept
{
    return float_to_bfloat16_bits_rne(f);
}

// Convert bfloat16 bits to float
// NOLINTNEXTLINE(readability-identifier-naming)
inline float bfloat16_bits_to_float(uint16_t bits) noexcept
{
    uint32_t floatBits = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &floatBits, sizeof(float));
    return f;
}

} // namespace detail

/**
 * @brief Custom bfloat16 type for hipDNN with configurable rounding mode
 *
 * This type provides a portable bfloat16 implementation that does not require
 * the __HIPCC__ macro. Both constructors from float/double and conversions TO
 * float/double are explicit to prevent silent precision loss and overload ambiguity.
 *
 * Binary layout is compatible with hip_bfloat16 (16-bit, same bit representation).
 *
 * @tparam RoundMode The rounding mode used when converting from float/double.
 *                   Default is RNE (round-to-nearest-even) for consistency with
 *                   half and fp8 types. Use Bfloat16RoundingMode::Truncate for
 *                   faster conversion that matches some HIP implementations.
 *
 * Example usage:
 * @code
 *   using namespace hipdnn_data_sdk::types;
 *
 *   // Default RNE rounding (recommended)
 *   bfloat16 a(3.14f);
 *
 *   // Explicit truncation rounding (faster, HIP-compatible)
 *   bfloat16_truncate b(3.14f);
 *
 *   // Both types have the same binary representation and can be mixed
 *   float sum = static_cast<float>(a) + static_cast<float>(b);
 * @endcode
 */
template <Bfloat16RoundingMode RoundMode = Bfloat16RoundingMode::RNE>
// NOLINTNEXTLINE(readability-identifier-naming) - lowercase to match hip_bfloat16 convention
struct bfloat16_t
{
    /// Raw bit representation of the bfloat16 value.
    /// Public to ensure binary compatibility with HIP hip_bfloat16 type.
    uint16_t data;

    /// The rounding mode used by this type
    // NOLINTNEXTLINE(readability-identifier-naming) - snake_case for public static member
    static constexpr Bfloat16RoundingMode rounding_mode = RoundMode;

    // Default constructor - value-initialized to zero for constexpr support
    constexpr bfloat16_t() noexcept
        : data(0)
    {
    }

    // Copy/move constructors - implicit
    bfloat16_t(const bfloat16_t&) = default;
    bfloat16_t(bfloat16_t&&) noexcept = default;
    bfloat16_t& operator=(const bfloat16_t&) = default;
    bfloat16_t& operator=(bfloat16_t&&) noexcept = default;

    // Constructor from other rounding mode - implicit conversion allowed
    // since the binary representation is identical
    template <Bfloat16RoundingMode OtherMode, typename = std::enable_if_t<OtherMode != RoundMode>>
    // NOLINTNEXTLINE(google-explicit-constructor) - intentionally implicit for same-representation types
    constexpr bfloat16_t(bfloat16_t<OtherMode> other) noexcept
        : data(other.data)
    {
    }

    // EXPLICIT constructor from float
    explicit bfloat16_t(float f) noexcept
        : data(detail::float_to_bfloat16_bits<RoundMode>(f))
    {
    }

    // EXPLICIT constructor from double (via float)
    explicit bfloat16_t(double d) noexcept
        : data(detail::float_to_bfloat16_bits<RoundMode>(static_cast<float>(d)))
    {
    }

    // EXPLICIT constructor from integral types
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    explicit bfloat16_t(T value) noexcept
        : data(detail::float_to_bfloat16_bits<RoundMode>(static_cast<float>(value)))
    {
    }

    // EXPLICIT constructors from other custom types (via float)
    // These are defined inline but require forward declarations above
    inline explicit bfloat16_t(half h) noexcept;
    inline explicit bfloat16_t(fp4_e2m1 f) noexcept;
    inline explicit bfloat16_t(fp6_e2m3 f) noexcept;
    inline explicit bfloat16_t(fp6_e3m2 f) noexcept;
    inline explicit bfloat16_t(fp8_e4m3 f) noexcept;
    inline explicit bfloat16_t(fp8_e5m2 f) noexcept;
    inline explicit bfloat16_t(fp8_e8m0 f) noexcept;

    // Factory for raw bits
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr bfloat16_t from_bits(uint16_t bits) noexcept
    {
        bfloat16_t val;
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
    bfloat16_t operator-() const noexcept
    {
        return from_bits(data ^ detail::BFLOAT16_SIGN_MASK);
    }

    // Unary plus
    bfloat16_t operator+() const noexcept
    {
        return *this;
    }

    // Arithmetic operators (compute in float, return bfloat16_t with same rounding mode)
    friend bfloat16_t operator+(bfloat16_t a, bfloat16_t b) noexcept
    {
        return bfloat16_t(static_cast<float>(a) + static_cast<float>(b));
    }

    friend bfloat16_t operator-(bfloat16_t a, bfloat16_t b) noexcept
    {
        return bfloat16_t(static_cast<float>(a) - static_cast<float>(b));
    }

    friend bfloat16_t operator*(bfloat16_t a, bfloat16_t b) noexcept
    {
        return bfloat16_t(static_cast<float>(a) * static_cast<float>(b));
    }

    friend bfloat16_t operator/(bfloat16_t a, bfloat16_t b) noexcept
    {
        return bfloat16_t(static_cast<float>(a) / static_cast<float>(b));
    }

    // Compound assignment operators
    bfloat16_t& operator+=(bfloat16_t other) noexcept
    {
        *this = *this + other;
        return *this;
    }

    bfloat16_t& operator-=(bfloat16_t other) noexcept
    {
        *this = *this - other;
        return *this;
    }

    bfloat16_t& operator*=(bfloat16_t other) noexcept
    {
        *this = *this * other;
        return *this;
    }

    bfloat16_t& operator/=(bfloat16_t other) noexcept
    {
        *this = *this / other;
        return *this;
    }

    // Comparison operators (compare via float conversion)
    friend bool operator==(bfloat16_t a, bfloat16_t b) noexcept
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }

    friend bool operator!=(bfloat16_t a, bfloat16_t b) noexcept
    {
        return static_cast<float>(a) != static_cast<float>(b);
    }

    friend bool operator<(bfloat16_t a, bfloat16_t b) noexcept
    {
        return static_cast<float>(a) < static_cast<float>(b);
    }

    friend bool operator>(bfloat16_t a, bfloat16_t b) noexcept
    {
        return static_cast<float>(a) > static_cast<float>(b);
    }

    friend bool operator<=(bfloat16_t a, bfloat16_t b) noexcept
    {
        return static_cast<float>(a) <= static_cast<float>(b);
    }

    friend bool operator>=(bfloat16_t a, bfloat16_t b) noexcept
    {
        return static_cast<float>(a) >= static_cast<float>(b);
    }

    // Stream output
    friend std::ostream& operator<<(std::ostream& os, bfloat16_t val)
    {
        return os << static_cast<float>(val);
    }
};

// ============================================================================
// Type Aliases
// ============================================================================

/// Default bfloat16 type using round-to-nearest-even (recommended)
// NOLINTNEXTLINE(readability-identifier-naming) - lowercase to match hip_bfloat16 convention
using bfloat16 = bfloat16_t<Bfloat16RoundingMode::RNE>;

/// Bfloat16 type using truncation rounding (faster, HIP-compatible)
// NOLINTNEXTLINE(readability-identifier-naming) - lowercase to match type naming convention
using bfloat16_truncate = bfloat16_t<Bfloat16RoundingMode::Truncate>;

// Static assertions for binary compatibility
static_assert(sizeof(bfloat16) == sizeof(uint16_t), "bfloat16 must be 2 bytes");
static_assert(std::is_trivially_copyable_v<bfloat16>, "bfloat16 must be trivially copyable");
static_assert(std::is_standard_layout_v<bfloat16>, "bfloat16 must be standard layout");
static_assert(sizeof(bfloat16_truncate) == sizeof(uint16_t), "bfloat16_truncate must be 2 bytes");
static_assert(std::is_trivially_copyable_v<bfloat16_truncate>,
              "bfloat16_truncate must be trivially copyable");
static_assert(std::is_standard_layout_v<bfloat16_truncate>,
              "bfloat16_truncate must be standard layout");

// User-defined literal (uses default RNE rounding)
inline bfloat16 operator""_bf(long double val)
{
    return bfloat16(static_cast<float>(val));
}

// ============================================================================
// Math functions for bfloat16_t (in hipdnn_data_sdk::types namespace)
// ============================================================================
// These are defined in our namespace to enable ADL (Argument Dependent Lookup).
// Use unqualified calls like: fabs(x), isnan(x), etc.
// All math functions work with any rounding mode since they operate on bit patterns.
// ============================================================================

// Basic math functions
template <Bfloat16RoundingMode M>
inline bfloat16_t<M> abs(bfloat16_t<M> x)
{
    return bfloat16_t<M>::from_bits(x.data & detail::BFLOAT16_ABS_MASK);
}

template <Bfloat16RoundingMode M>
inline bfloat16_t<M> fabs(bfloat16_t<M> x)
{
    return bfloat16_t<M>::from_bits(x.data & detail::BFLOAT16_ABS_MASK);
}

template <Bfloat16RoundingMode M>
inline bool isnan(bfloat16_t<M> x)
{
    // NaN: exponent all 1s and non-zero mantissa
    return (x.data & detail::BFLOAT16_EXP_MASK) == detail::BFLOAT16_EXP_MASK
           && (x.data & detail::BFLOAT16_MANT_MASK) != 0;
}

template <Bfloat16RoundingMode M>
inline bool isinf(bfloat16_t<M> x)
{
    // Inf: exponent all 1s and zero mantissa
    return (x.data & detail::BFLOAT16_ABS_MASK) == detail::BFLOAT16_POS_INF;
}

template <Bfloat16RoundingMode M>
inline bool signbit(bfloat16_t<M> x)
{
    return (x.data & detail::BFLOAT16_SIGN_MASK) != 0;
}

template <Bfloat16RoundingMode M>
inline bool isfinite(bfloat16_t<M> x)
{
    return !isnan(x) && !isinf(x);
}

template <Bfloat16RoundingMode M>
inline bfloat16_t<M> copysign(bfloat16_t<M> x, bfloat16_t<M> y)
{
    const uint16_t xBits = x.data & detail::BFLOAT16_ABS_MASK;
    const uint16_t ySign = y.data & detail::BFLOAT16_SIGN_MASK;
    return bfloat16_t<M>::from_bits(xBits | ySign);
}

// Equivalent to std::fmax/std::fmin
// If one input is NaN and the other is not, returns the non-NaN value.
template <Bfloat16RoundingMode M>
inline bfloat16_t<M> fmax(bfloat16_t<M> a, bfloat16_t<M> b)
{
    if(isnan(a))
    {
        return isnan(b) ? bfloat16_t<M>::from_bits(detail::BFLOAT16_CANONICAL_NAN) : b;
    }
    if(isnan(b))
    {
        return a;
    }
    return a > b ? a : b;
}

template <Bfloat16RoundingMode M>
inline bfloat16_t<M> fmin(bfloat16_t<M> a, bfloat16_t<M> b)
{
    if(isnan(a))
    {
        return isnan(b) ? bfloat16_t<M>::from_bits(detail::BFLOAT16_CANONICAL_NAN) : b;
    }
    if(isnan(b))
    {
        return a;
    }
    return a < b ? a : b;
}

// Equivalent to std::max/std::min
// No NaN handling
template <Bfloat16RoundingMode M>
inline bfloat16_t<M> max(bfloat16_t<M> a, bfloat16_t<M> b)
{
    return a < b ? b : a;
}

template <Bfloat16RoundingMode M>
inline bfloat16_t<M> min(bfloat16_t<M> a, bfloat16_t<M> b)
{
    return b < a ? b : a;
}

// Rounding functions
template <Bfloat16RoundingMode M>
inline bfloat16_t<M> floor(bfloat16_t<M> x)
{
    return bfloat16_t<M>(std::floor(static_cast<float>(x)));
}

template <Bfloat16RoundingMode M>
inline bfloat16_t<M> ceil(bfloat16_t<M> x)
{
    return bfloat16_t<M>(std::ceil(static_cast<float>(x)));
}

template <Bfloat16RoundingMode M>
inline bfloat16_t<M> round(bfloat16_t<M> x)
{
    return bfloat16_t<M>(std::round(static_cast<float>(x)));
}

template <Bfloat16RoundingMode M>
inline bfloat16_t<M> trunc(bfloat16_t<M> x)
{
    return bfloat16_t<M>(std::trunc(static_cast<float>(x)));
}

// Exponential and logarithmic functions
template <Bfloat16RoundingMode M>
inline bfloat16_t<M> exp(bfloat16_t<M> x)
{
    return bfloat16_t<M>(std::exp(static_cast<float>(x)));
}

template <Bfloat16RoundingMode M>
inline bfloat16_t<M> exp2(bfloat16_t<M> x)
{
    return bfloat16_t<M>(std::exp2(static_cast<float>(x)));
}

template <Bfloat16RoundingMode M>
inline bfloat16_t<M> log(bfloat16_t<M> x)
{
    return bfloat16_t<M>(std::log(static_cast<float>(x)));
}

template <Bfloat16RoundingMode M>
inline bfloat16_t<M> log2(bfloat16_t<M> x)
{
    return bfloat16_t<M>(std::log2(static_cast<float>(x)));
}

template <Bfloat16RoundingMode M>
inline bfloat16_t<M> log10(bfloat16_t<M> x)
{
    return bfloat16_t<M>(std::log10(static_cast<float>(x)));
}

// Power functions
template <Bfloat16RoundingMode M>
inline bfloat16_t<M> sqrt(bfloat16_t<M> x)
{
    return bfloat16_t<M>(std::sqrt(static_cast<float>(x)));
}

template <Bfloat16RoundingMode M>
inline bfloat16_t<M> rsqrt(bfloat16_t<M> x)
{
    return bfloat16_t<M>(1.0f / std::sqrt(static_cast<float>(x)));
}

template <Bfloat16RoundingMode M>
inline bfloat16_t<M> pow(bfloat16_t<M> x, bfloat16_t<M> y)
{
    return bfloat16_t<M>(std::pow(static_cast<float>(x), static_cast<float>(y)));
}

// Trigonometric functions
template <Bfloat16RoundingMode M>
inline bfloat16_t<M> sin(bfloat16_t<M> x)
{
    return bfloat16_t<M>(std::sin(static_cast<float>(x)));
}

template <Bfloat16RoundingMode M>
inline bfloat16_t<M> cos(bfloat16_t<M> x)
{
    return bfloat16_t<M>(std::cos(static_cast<float>(x)));
}

template <Bfloat16RoundingMode M>
inline bfloat16_t<M> tan(bfloat16_t<M> x)
{
    return bfloat16_t<M>(std::tan(static_cast<float>(x)));
}

template <Bfloat16RoundingMode M>
inline bfloat16_t<M> asin(bfloat16_t<M> x)
{
    return bfloat16_t<M>(std::asin(static_cast<float>(x)));
}

template <Bfloat16RoundingMode M>
inline bfloat16_t<M> acos(bfloat16_t<M> x)
{
    return bfloat16_t<M>(std::acos(static_cast<float>(x)));
}

template <Bfloat16RoundingMode M>
inline bfloat16_t<M> atan(bfloat16_t<M> x)
{
    return bfloat16_t<M>(std::atan(static_cast<float>(x)));
}

// Hyperbolic functions
template <Bfloat16RoundingMode M>
inline bfloat16_t<M> sinh(bfloat16_t<M> x)
{
    return bfloat16_t<M>(std::sinh(static_cast<float>(x)));
}

template <Bfloat16RoundingMode M>
inline bfloat16_t<M> cosh(bfloat16_t<M> x)
{
    return bfloat16_t<M>(std::cosh(static_cast<float>(x)));
}

template <Bfloat16RoundingMode M>
inline bfloat16_t<M> tanh(bfloat16_t<M> x)
{
    return bfloat16_t<M>(std::tanh(static_cast<float>(x)));
}

// Error function
template <Bfloat16RoundingMode M>
inline bfloat16_t<M> erf(bfloat16_t<M> x)
{
    return bfloat16_t<M>(std::erf(static_cast<float>(x)));
}

// Floating-point manipulation
template <Bfloat16RoundingMode M>
inline bfloat16_t<M> fmod(bfloat16_t<M> x, bfloat16_t<M> y)
{
    return bfloat16_t<M>(std::fmod(static_cast<float>(x), static_cast<float>(y)));
}

// Fused multiply-add
template <Bfloat16RoundingMode M>
inline bfloat16_t<M> fma(bfloat16_t<M> x, bfloat16_t<M> y, bfloat16_t<M> z)
{
    return bfloat16_t<M>(
        std::fma(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)));
}

} // namespace hipdnn_data_sdk::types

// std::numeric_limits specialization for bfloat16_t template
// NOLINTBEGIN(readability-identifier-naming) - standard library names must match exactly
template <hipdnn_data_sdk::types::Bfloat16RoundingMode M>
class std::numeric_limits<hipdnn_data_sdk::types::bfloat16_t<M>>
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

    static constexpr hipdnn_data_sdk::types::bfloat16_t<M> min() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16_t<M>::from_bits(
            hipdnn_data_sdk::types::detail::BFLOAT16_MIN_NORMAL);
    }

    static constexpr hipdnn_data_sdk::types::bfloat16_t<M> lowest() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16_t<M>::from_bits(
            hipdnn_data_sdk::types::detail::BFLOAT16_LOWEST);
    }

    static constexpr hipdnn_data_sdk::types::bfloat16_t<M> max() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16_t<M>::from_bits(
            hipdnn_data_sdk::types::detail::BFLOAT16_MAX);
    }

    static constexpr hipdnn_data_sdk::types::bfloat16_t<M> epsilon() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16_t<M>::from_bits(
            hipdnn_data_sdk::types::detail::BFLOAT16_EPSILON);
    }

    static constexpr hipdnn_data_sdk::types::bfloat16_t<M> round_error() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16_t<M>::from_bits(
            hipdnn_data_sdk::types::detail::BFLOAT16_ROUND_ERROR);
    }

    static constexpr hipdnn_data_sdk::types::bfloat16_t<M> infinity() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16_t<M>::from_bits(
            hipdnn_data_sdk::types::detail::BFLOAT16_POS_INF);
    }

    static constexpr hipdnn_data_sdk::types::bfloat16_t<M> quiet_NaN() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16_t<M>::from_bits(
            hipdnn_data_sdk::types::detail::BFLOAT16_QNAN);
    }

    static constexpr hipdnn_data_sdk::types::bfloat16_t<M> signaling_NaN() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16_t<M>::from_bits(
            hipdnn_data_sdk::types::detail::BFLOAT16_SNAN);
    }

    static constexpr hipdnn_data_sdk::types::bfloat16_t<M> denorm_min() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16_t<M>::from_bits(
            hipdnn_data_sdk::types::detail::BFLOAT16_DENORM_MIN);
    }
};
// NOLINTEND(readability-identifier-naming)
