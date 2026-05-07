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
struct fp4_e2m1;
struct fp6_e2m3;
struct fp6_e3m2;
struct fp8_e4m3;
struct fp8_e5m2;
struct fp8_e8m0;
// NOLINTEND(readability-identifier-naming)

namespace detail
{

// ============================================================================
// Half (FP16) Bit Layout Constants
// ============================================================================
// IEEE 754 half-precision format: 1 sign bit, 5 exponent bits, 10 mantissa bits
//
// Bit layout: [S|EEEEE|MMMMMMMMMM]
//              15 14-10 9        0
// ============================================================================

/// Sign bit mask (bit 15)
constexpr uint16_t HALF_SIGN_MASK = 0x8000;

/// Absolute value mask (all bits except sign)
constexpr uint16_t HALF_ABS_MASK = 0x7FFF;

/// Exponent field mask (bits 10-14)
constexpr uint16_t HALF_EXP_MASK = 0x7C00;

/// Mantissa field mask (bits 0-9)
constexpr uint16_t HALF_MANT_MASK = 0x03FF;

/// Exponent bias for IEEE 754 half-precision
constexpr int HALF_EXP_BIAS = 15;

/// Number of mantissa bits
constexpr int HALF_MANT_BITS = 10;

/// Number of exponent bits
constexpr int HALF_EXP_BITS = 5;

// ============================================================================
// Half Special Values (bit patterns)
// ============================================================================

/// Positive infinity: exponent all 1s, mantissa all 0s
constexpr uint16_t HALF_POS_INF = 0x7C00;

/// Negative infinity
constexpr uint16_t HALF_NEG_INF = 0xFC00;

/// Quiet NaN (canonical): exponent all 1s, MSB of mantissa set
constexpr uint16_t HALF_QNAN = 0x7E00;

/// Signaling NaN: exponent all 1s, mantissa non-zero but MSB clear
constexpr uint16_t HALF_SNAN = 0x7C01;

/// Canonical NaN for min/max operations
constexpr uint16_t HALF_CANONICAL_NAN = 0x7FFF;

/// Maximum finite positive value: 0x7BFF = 65504.0
constexpr uint16_t HALF_MAX = 0x7BFF;

/// Minimum positive normal value: 2^-14 = 6.10e-5
constexpr uint16_t HALF_MIN_NORMAL = 0x0400;

/// Minimum positive denormal value
constexpr uint16_t HALF_DENORM_MIN = 0x0001;

/// Maximum finite negative value (lowest): -65504.0
constexpr uint16_t HALF_LOWEST = 0xFBFF;

/// Epsilon: smallest value such that 1.0 + epsilon != 1.0 (2^-10)
constexpr uint16_t HALF_EPSILON = 0x1400;

/// Round error (0.5)
constexpr uint16_t HALF_ROUND_ERROR = 0x3800;

// ============================================================================
// Half Conversion Constants
// ============================================================================

/// Rounding threshold for round-to-nearest (half of LSB position)
constexpr uint32_t HALF_ROUND_THRESHOLD = 0x1000;

/// Mask for remainder bits during rounding (13 LSB of float mantissa)
constexpr uint32_t HALF_REMAINDER_MASK = 0x1FFF;

// Convert float to fp16 bits using round-to-nearest-even
// NOLINTNEXTLINE(readability-identifier-naming)
inline uint16_t float_to_half_bits(float f) noexcept
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));

    const uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>(((bits >> 23) & 0xFF)) - 127 + 15;
    uint32_t mant = bits & 0x007FFFFF;

    // Handle special cases
    if(exp <= 0)
    {
        if(exp < -10)
        {
            // Too small, return signed zero
            return static_cast<uint16_t>(sign);
        }
        // Denormalized number
        mant |= 0x00800000;
        auto shift = static_cast<uint32_t>(14 - exp);
        mant >>= shift;
        return static_cast<uint16_t>(sign | (mant >> 13));
    }
    if(exp == 0xFF - 127 + 15)
    {
        // Infinity or NaN
        if(mant == 0)
        {
            return static_cast<uint16_t>(sign | 0x7C00); // Infinity
        }
        return static_cast<uint16_t>(sign | 0x7C00 | (mant >> 13)); // NaN
    }
    if(exp > 30)
    {
        // Overflow to infinity
        return static_cast<uint16_t>(sign | 0x7C00);
    }

    // Round to nearest even
    uint32_t halfMant = mant >> 13;
    const uint32_t remainder = mant & HALF_REMAINDER_MASK;
    if(remainder > HALF_ROUND_THRESHOLD
       || (remainder == HALF_ROUND_THRESHOLD && ((halfMant & 1) != 0U)))
    {
        halfMant++;
        if(halfMant > 0x3FF)
        {
            halfMant = 0;
            exp++;
            if(exp > 30)
            {
                return static_cast<uint16_t>(sign | 0x7C00); // Overflow
            }
        }
    }

    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | halfMant);
}

// Convert fp16 bits to float
// NOLINTNEXTLINE(readability-identifier-naming)
inline float half_bits_to_float(uint16_t bits) noexcept
{
    uint32_t sign = (static_cast<uint32_t>(bits) & 0x8000) << 16;
    uint32_t exp = (bits >> 10) & 0x1F;
    uint32_t mant = bits & 0x03FF;

    if(exp == 0)
    {
        if(mant == 0)
        {
            // Signed zero
            float f;
            std::memcpy(&f, &sign, sizeof(float));
            return f;
        }
        // Denormalized
        while((mant & 0x0400) == 0)
        {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x0400U;
        exp = exp + 127 - 15;
        mant <<= 13;
    }
    else if(exp == 31)
    {
        // Infinity or NaN
        exp = 255;
        mant <<= 13;
    }
    else
    {
        exp = exp + 127 - 15;
        mant <<= 13;
    }

    uint32_t floatBits = sign | (exp << 23) | mant;
    float f;
    std::memcpy(&f, &floatBits, sizeof(float));
    return f;
}

} // namespace detail

/**
 * @brief Custom half (FP16) type for hipDNN
 *
 * This type provides a portable half-precision floating point implementation that does not require
 * the __HIPCC__ macro. Both constructors from float/double and conversions TO
 * float/double are explicit to prevent silent precision loss and overload ambiguity.
 *
 * Binary layout is compatible with __half (16-bit IEEE 754 half-precision).
 */
// NOLINTNEXTLINE(readability-identifier-naming) - lowercase to match half convention
struct half
{
    /// Raw bit representation of the half-precision value.
    /// Public to ensure binary compatibility with HIP __half type.
    uint16_t data;

    // Default constructor - value-initialized to zero for constexpr support
    constexpr half() noexcept
        : data(0)
    {
    }

    // Copy/move constructors - implicit
    half(const half&) = default;
    half(half&&) noexcept = default;
    half& operator=(const half&) = default;
    half& operator=(half&&) noexcept = default;

    // EXPLICIT constructor from float
    explicit half(float f) noexcept
        : data(detail::float_to_half_bits(f))
    {
    }

    // EXPLICIT constructor from double (via float)
    explicit half(double d) noexcept
        : data(detail::float_to_half_bits(static_cast<float>(d)))
    {
    }

    // EXPLICIT constructor from integral types
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    explicit half(T value) noexcept
        : data(detail::float_to_half_bits(static_cast<float>(value)))
    {
    }

    // EXPLICIT constructors from other custom types (via float)
    // These are defined inline but require forward declarations above
    template <Bfloat16RoundingMode M>
    inline explicit half(bfloat16_t<M> b) noexcept;
    inline explicit half(fp4_e2m1 f) noexcept;
    inline explicit half(fp6_e2m3 f) noexcept;
    inline explicit half(fp6_e3m2 f) noexcept;
    inline explicit half(fp8_e4m3 f) noexcept;
    inline explicit half(fp8_e5m2 f) noexcept;
    inline explicit half(fp8_e8m0 f) noexcept;

    // Factory for raw bits
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr half from_bits(uint16_t bits) noexcept
    {
        half val;
        val.data = bits;
        return val;
    }

    // EXPLICIT conversion to float
    explicit operator float() const noexcept
    {
        return detail::half_bits_to_float(data);
    }

    // EXPLICIT conversion to double
    explicit operator double() const noexcept
    {
        return static_cast<double>(detail::half_bits_to_float(data));
    }

    // Unary negation - XOR sign bit
    half operator-() const noexcept
    {
        return from_bits(data ^ detail::HALF_SIGN_MASK);
    }

    // Unary plus
    half operator+() const noexcept
    {
        return *this;
    }

    // Arithmetic operators (compute in float, return half)
    friend half operator+(half a, half b) noexcept
    {
        return half(static_cast<float>(a) + static_cast<float>(b));
    }

    friend half operator-(half a, half b) noexcept
    {
        return half(static_cast<float>(a) - static_cast<float>(b));
    }

    friend half operator*(half a, half b) noexcept
    {
        return half(static_cast<float>(a) * static_cast<float>(b));
    }

    friend half operator/(half a, half b) noexcept
    {
        return half(static_cast<float>(a) / static_cast<float>(b));
    }

    // Compound assignment operators
    half& operator+=(half other) noexcept
    {
        *this = *this + other;
        return *this;
    }

    half& operator-=(half other) noexcept
    {
        *this = *this - other;
        return *this;
    }

    half& operator*=(half other) noexcept
    {
        *this = *this * other;
        return *this;
    }

    half& operator/=(half other) noexcept
    {
        *this = *this / other;
        return *this;
    }

    // Comparison operators (compare via float conversion)
    friend bool operator==(half a, half b) noexcept
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }

    friend bool operator!=(half a, half b) noexcept
    {
        return static_cast<float>(a) != static_cast<float>(b);
    }

    friend bool operator<(half a, half b) noexcept
    {
        return static_cast<float>(a) < static_cast<float>(b);
    }

    friend bool operator>(half a, half b) noexcept
    {
        return static_cast<float>(a) > static_cast<float>(b);
    }

    friend bool operator<=(half a, half b) noexcept
    {
        return static_cast<float>(a) <= static_cast<float>(b);
    }

    friend bool operator>=(half a, half b) noexcept
    {
        return static_cast<float>(a) >= static_cast<float>(b);
    }

    // Stream output
    friend std::ostream& operator<<(std::ostream& os, half val)
    {
        return os << static_cast<float>(val);
    }
};

// Static assertions for binary compatibility
static_assert(sizeof(half) == sizeof(uint16_t), "half must be 2 bytes");
static_assert(std::is_trivially_copyable_v<half>, "half must be trivially copyable");
static_assert(std::is_standard_layout_v<half>, "half must be standard layout");

// User-defined literal
// NOLINTNEXTLINE(readability-identifier-naming) - using _h suffix to match existing convention
inline half operator""_h(long double val)
{
    return half(static_cast<float>(val));
}

// ============================================================================
// Math functions for half (in hipdnn_data_sdk::types namespace)
// ============================================================================
// These are defined in our namespace to enable ADL (Argument Dependent Lookup).
// Use unqualified calls like: fabs(x), isnan(x), etc.
// ============================================================================

// Basic math functions
inline half abs(half x)
{
    return half::from_bits(x.data & detail::HALF_ABS_MASK);
}

inline half fabs(half x)
{
    return half::from_bits(x.data & detail::HALF_ABS_MASK);
}

inline bool isnan(half x)
{
    // NaN: exponent all 1s and non-zero mantissa
    return (x.data & detail::HALF_EXP_MASK) == detail::HALF_EXP_MASK
           && (x.data & detail::HALF_MANT_MASK) != 0;
}

inline bool isinf(half x)
{
    // Inf: exponent all 1s and zero mantissa
    return (x.data & detail::HALF_ABS_MASK) == detail::HALF_POS_INF;
}

inline bool signbit(half x)
{
    return (x.data & detail::HALF_SIGN_MASK) != 0;
}

inline bool isfinite(half x)
{
    return !isnan(x) && !isinf(x);
}

inline half copysign(half x, half y)
{
    const uint16_t xBits = x.data & detail::HALF_ABS_MASK;
    const uint16_t ySign = y.data & detail::HALF_SIGN_MASK;
    return half::from_bits(xBits | ySign);
}

// Equivalent to std::fmax/std::fmin
// If one input is NaN and the other is not, returns the non-NaN value.
inline half fmax(half a, half b)
{
    if(isnan(a))
    {
        return isnan(b) ? half::from_bits(detail::HALF_CANONICAL_NAN) : b;
    }
    if(isnan(b))
    {
        return a;
    }
    return a > b ? a : b;
}

inline half fmin(half a, half b)
{
    if(isnan(a))
    {
        return isnan(b) ? half::from_bits(detail::HALF_CANONICAL_NAN) : b;
    }
    if(isnan(b))
    {
        return a;
    }
    return a < b ? a : b;
}

// Equivalent to std::max/std::min
// No NaN handling
inline half max(half a, half b)
{
    return a < b ? b : a;
}

inline half min(half a, half b)
{
    return b < a ? b : a;
}

// Rounding functions
inline half floor(half x)
{
    return half(std::floor(static_cast<float>(x)));
}

inline half ceil(half x)
{
    return half(std::ceil(static_cast<float>(x)));
}

inline half round(half x)
{
    return half(std::round(static_cast<float>(x)));
}

inline half trunc(half x)
{
    return half(std::trunc(static_cast<float>(x)));
}

// Exponential and logarithmic functions
inline half exp(half x)
{
    return half(std::exp(static_cast<float>(x)));
}

inline half exp2(half x)
{
    return half(std::exp2(static_cast<float>(x)));
}

inline half log(half x)
{
    return half(std::log(static_cast<float>(x)));
}

inline half log2(half x)
{
    return half(std::log2(static_cast<float>(x)));
}

inline half log10(half x)
{
    return half(std::log10(static_cast<float>(x)));
}

// Power functions
inline half sqrt(half x)
{
    return half(std::sqrt(static_cast<float>(x)));
}

inline half rsqrt(half x)
{
    return half(1.0f / std::sqrt(static_cast<float>(x)));
}

inline half pow(half x, half y)
{
    return half(std::pow(static_cast<float>(x), static_cast<float>(y)));
}

// Trigonometric functions
inline half sin(half x)
{
    return half(std::sin(static_cast<float>(x)));
}

inline half cos(half x)
{
    return half(std::cos(static_cast<float>(x)));
}

inline half tan(half x)
{
    return half(std::tan(static_cast<float>(x)));
}

inline half asin(half x)
{
    return half(std::asin(static_cast<float>(x)));
}

inline half acos(half x)
{
    return half(std::acos(static_cast<float>(x)));
}

inline half atan(half x)
{
    return half(std::atan(static_cast<float>(x)));
}

// Hyperbolic functions
inline half sinh(half x)
{
    return half(std::sinh(static_cast<float>(x)));
}

inline half cosh(half x)
{
    return half(std::cosh(static_cast<float>(x)));
}

inline half tanh(half x)
{
    return half(std::tanh(static_cast<float>(x)));
}

// Error function
inline half erf(half x)
{
    return half(std::erf(static_cast<float>(x)));
}

// Floating-point manipulation
inline half fmod(half x, half y)
{
    return half(std::fmod(static_cast<float>(x), static_cast<float>(y)));
}

// Fused multiply-add
inline half fma(half x, half y, half z)
{
    return half(std::fma(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)));
}

} // namespace hipdnn_data_sdk::types

// std::numeric_limits specialization
// NOLINTBEGIN(readability-identifier-naming) - standard library names must match exactly
template <>
class std::numeric_limits<hipdnn_data_sdk::types::half>
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
    static constexpr bool is_iec559 = true;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = 11; // 10 mantissa bits + 1 implicit
    static constexpr int digits10 = 3;
    static constexpr int max_digits10 = 5;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -13;
    static constexpr int min_exponent10 = -4;
    static constexpr int max_exponent = 16;
    static constexpr int max_exponent10 = 4;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;

    static constexpr hipdnn_data_sdk::types::half min() noexcept
    {
        return hipdnn_data_sdk::types::half::from_bits(
            hipdnn_data_sdk::types::detail::HALF_MIN_NORMAL);
    }

    static constexpr hipdnn_data_sdk::types::half lowest() noexcept
    {
        return hipdnn_data_sdk::types::half::from_bits(hipdnn_data_sdk::types::detail::HALF_LOWEST);
    }

    static constexpr hipdnn_data_sdk::types::half max() noexcept
    {
        return hipdnn_data_sdk::types::half::from_bits(hipdnn_data_sdk::types::detail::HALF_MAX);
    }

    static constexpr hipdnn_data_sdk::types::half epsilon() noexcept
    {
        return hipdnn_data_sdk::types::half::from_bits(
            hipdnn_data_sdk::types::detail::HALF_EPSILON);
    }

    static constexpr hipdnn_data_sdk::types::half round_error() noexcept
    {
        return hipdnn_data_sdk::types::half::from_bits(
            hipdnn_data_sdk::types::detail::HALF_ROUND_ERROR);
    }

    static constexpr hipdnn_data_sdk::types::half infinity() noexcept
    {
        return hipdnn_data_sdk::types::half::from_bits(
            hipdnn_data_sdk::types::detail::HALF_POS_INF);
    }

    static constexpr hipdnn_data_sdk::types::half quiet_NaN() noexcept
    {
        return hipdnn_data_sdk::types::half::from_bits(hipdnn_data_sdk::types::detail::HALF_QNAN);
    }

    static constexpr hipdnn_data_sdk::types::half signaling_NaN() noexcept
    {
        return hipdnn_data_sdk::types::half::from_bits(hipdnn_data_sdk::types::detail::HALF_SNAN);
    }

    static constexpr hipdnn_data_sdk::types::half denorm_min() noexcept
    {
        return hipdnn_data_sdk::types::half::from_bits(
            hipdnn_data_sdk::types::detail::HALF_DENORM_MIN);
    }
};
// NOLINTEND(readability-identifier-naming)
