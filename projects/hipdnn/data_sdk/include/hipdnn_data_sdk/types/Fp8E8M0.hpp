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
// NOLINTEND(readability-identifier-naming)

namespace detail
{

// ============================================================================
// FP8 E8M0 (MX Scale Format) Bit Layout Constants
// ============================================================================
// See fp8_e8m0 struct documentation for full format specification.
//
// Bit layout: [EEEEEEEE]
//              7      0
// ============================================================================

/// Exponent bias for E8M0
constexpr int FP8_E8M0_EXP_BIAS = 127;

/// Number of exponent bits
constexpr int FP8_E8M0_EXP_BITS = 8;

// ============================================================================
// FP8 E8M0 Special Values (bit patterns)
// ============================================================================

/// NaN: all bits set (0xFF)
constexpr uint8_t FP8_E8M0_NAN = 0xFF;

/// Maximum finite value: 2^127 (scale = 254)
constexpr uint8_t FP8_E8M0_MAX = 0xFE;

/// Minimum positive value: 2^-127 (scale = 0)
constexpr uint8_t FP8_E8M0_MIN = 0x00;

/// Value representing 1.0 (2^0, scale = 127)
constexpr uint8_t FP8_E8M0_ONE = 0x7F;

// Convert float to FP8 E8M0 bits (MX scale format: 8 exponent bits, 0 mantissa)
// E8M0 is unsigned - negative values and zero are clamped to min (2^-127)
// Range: 2^-127 to 2^127, plus NaN
// NOLINTNEXTLINE(readability-identifier-naming)
inline uint8_t float_to_fp8_e8m0_bits(float f) noexcept
{
    // Handle special values first using std library functions
    if(std::isnan(f))
    {
        return FP8_E8M0_NAN;
    }

    // E8M0 is unsigned and has no zero - negative values and zero clamp to min (2^-127)
    if(f <= 0.0f)
    {
        return FP8_E8M0_MIN;
    }

    if(std::isinf(f))
    {
        // E8M0 has no infinity - saturate to max
        return FP8_E8M0_MAX;
    }

    // Extract float bits
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));

    // Extract float exponent (biased by 127, same as E8M0).
    // After above checks: exp in [0,254] which maps directly to E8M0.
    // Float denormals (exp=0) clamp to E8M0 min (2^-127).
    const auto exp = static_cast<uint8_t>((bits >> 23) & 0xFF);

    return exp;
}

// Convert FP8 E8M0 bits to float
// Value = 2^(scale - 127) for all values 0-254
// NOLINTNEXTLINE(readability-identifier-naming)
inline float fp8_e8m0_bits_to_float(uint8_t bits) noexcept
{
    // Handle NaN (bits=255 would produce float infinity, not NaN)
    if(bits == FP8_E8M0_NAN)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }

    // Construct float with exponent = bits, mantissa = 0.
    // All values 0-254 produce 2^(bits-127).
    // Note: bits=0 produces a float denormal (exp=0), which equals 2^-126 * 0 = 0
    // in IEEE 754, but we want 2^-127. Handle this specially.
    if(bits == 0)
    {
        // 2^-127 cannot be represented exactly as a normal float.
        // The closest is 2^-126 * 0.5 = 2^-127, which requires mantissa = 0.5.
        // Float bit pattern: sign=0, exp=0, mantissa=0x400000 (2^22 = 0.5 in denorm)
        return 0x1p-127f;
    }

    uint32_t floatBits = static_cast<uint32_t>(bits) << 23;
    float f;
    std::memcpy(&f, &floatBits, sizeof(float));
    return f;
}

} // namespace detail

/**
 * @brief Custom storage-only FP8 E8M0 type for hipDNN
 *
 * This type provides a portable FP8 E8M0 (8 exponent bits, 0 mantissa bits) implementation
 * that does not require the __HIPCC__ macro. Uses MX (Microscaling) scale format per
 * OCP Microscaling Formats (MX) Specification v1.0.
 *
 * This is a STORAGE-ONLY type intended for data representation and conversion,
 * not direct computation. Arithmetic operations and comparisons are
 * intentionally not provided. For computation, explicitly convert to float.
 *
 * E8M0 is a pure power-of-2 scale type (unsigned):
 * - Binary layout: 8 exponent bits (no sign, no mantissa)
 * - Exponent bias: 127
 * - Value = 2^(scale - 127) for scale in [0, 254]
 * - NaN at scale = 255 (0xFF)
 * - No zero representation (use scale=0 for 2^-127)
 * - No infinity representation
 * - Range: 2^-127 to 2^127
 */
// NOLINTNEXTLINE(readability-identifier-naming) - lowercase for consistency
struct fp8_e8m0
{
    /// Raw bit representation of the FP8 E8M0 value.
    uint8_t data;

    // Default constructor - value-initialized to min (2^-127) for constexpr support
    constexpr fp8_e8m0() noexcept
        : data(0)
    {
    }

    // Copy/move constructors - implicit
    fp8_e8m0(const fp8_e8m0&) = default;
    fp8_e8m0(fp8_e8m0&&) noexcept = default;
    fp8_e8m0& operator=(const fp8_e8m0&) = default;
    fp8_e8m0& operator=(fp8_e8m0&&) noexcept = default;

    // EXPLICIT constructor from float
    explicit fp8_e8m0(float f) noexcept
        : data(detail::float_to_fp8_e8m0_bits(f))
    {
    }

    // EXPLICIT constructor from double (via float)
    explicit fp8_e8m0(double d) noexcept
        : fp8_e8m0(static_cast<float>(d))
    {
    }

    // EXPLICIT constructor from integral types
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    explicit fp8_e8m0(T value) noexcept
        : fp8_e8m0(static_cast<float>(value))
    {
    }

    // EXPLICIT constructors from other custom types (via float)
    // These are defined inline but require forward declarations above
    template <Bfloat16RoundingMode M>
    inline explicit fp8_e8m0(bfloat16_t<M> b) noexcept;
    inline explicit fp8_e8m0(half h) noexcept;

    // Factory for raw bits
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr fp8_e8m0 from_bits(uint8_t bits) noexcept
    {
        fp8_e8m0 val;
        val.data = bits;
        return val;
    }

    // EXPLICIT conversion to float
    explicit operator float() const noexcept
    {
        return detail::fp8_e8m0_bits_to_float(data);
    }

    // EXPLICIT conversion to double
    explicit operator double() const noexcept
    {
        return static_cast<double>(detail::fp8_e8m0_bits_to_float(data));
    }

    // Unary plus
    fp8_e8m0 operator+() const noexcept
    {
        return *this;
    }

    // Note: Unary negation is not meaningful for E8M0 (unsigned scale type)
    // E8M0 represents only positive powers of 2

    // Stream output
    friend std::ostream& operator<<(std::ostream& os, fp8_e8m0 val)
    {
        return os << static_cast<float>(val);
    }
};

// Static assertions for type properties
static_assert(sizeof(fp8_e8m0) == sizeof(uint8_t), "fp8_e8m0 must be 1 byte");
static_assert(std::is_trivially_copyable_v<fp8_e8m0>, "fp8_e8m0 must be trivially copyable");
static_assert(std::is_standard_layout_v<fp8_e8m0>, "fp8_e8m0 must be standard layout");
static_assert(std::is_default_constructible_v<fp8_e8m0>, "fp8_e8m0 must be default constructible");
static_assert(std::is_copy_constructible_v<fp8_e8m0>, "fp8_e8m0 must be copy constructible");
static_assert(std::is_move_constructible_v<fp8_e8m0>, "fp8_e8m0 must be move constructible");

// User-defined literal
// NOLINTNEXTLINE(readability-identifier-naming)
inline fp8_e8m0 operator""_e8m0(long double val)
{
    return fp8_e8m0(static_cast<float>(val));
}

// ============================================================================
// Math functions for fp8_e8m0 (in hipdnn_data_sdk::types namespace)
// ============================================================================
// These are defined in our namespace to enable ADL (Argument Dependent Lookup).
// Use unqualified calls like: isnan(x), isinf(x), etc.
// ============================================================================

inline bool isnan(fp8_e8m0 x)
{
    // E8M0 NaN: all bits set (0xFF)
    return x.data == detail::FP8_E8M0_NAN;
}

inline bool isinf(fp8_e8m0 /*x*/)
{
    // E8M0 has no infinity representation
    return false;
}

inline bool signbit(fp8_e8m0 /*x*/)
{
    // E8M0 is unsigned (no sign bit)
    return false;
}

inline bool isfinite(fp8_e8m0 x)
{
    return !isnan(x);
}

} // namespace hipdnn_data_sdk::types

// std::numeric_limits specialization
// NOLINTBEGIN(readability-identifier-naming) - standard library names must match exactly
template <>
class std::numeric_limits<hipdnn_data_sdk::types::fp8_e8m0>
{
public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = false;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = false;
    static constexpr std::float_denorm_style has_denorm = std::denorm_absent;
    static constexpr bool has_denorm_loss = false;
    static constexpr std::float_round_style round_style = std::round_toward_zero;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = 1; // Only 1 bit of precision (power of 2)
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 2;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -126; // 2^(min_exponent-1) = 2^-127 = min()
    static constexpr int min_exponent10 = -38;
    static constexpr int max_exponent = 128; // 2^(max_exponent-1) = 2^127 = max()
    static constexpr int max_exponent10 = 38;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;

    static constexpr hipdnn_data_sdk::types::fp8_e8m0 min() noexcept
    {
        // Minimum positive value: 2^-127 (scale = 0)
        return hipdnn_data_sdk::types::fp8_e8m0::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E8M0_MIN);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e8m0 lowest() noexcept
    {
        // E8M0 is unsigned and has no zero, so lowest equals min (2^-127)
        return min();
    }

    static constexpr hipdnn_data_sdk::types::fp8_e8m0 max() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e8m0::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E8M0_MAX);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e8m0 epsilon() noexcept
    {
        // Smallest difference at 1.0: next representable is 2.0, so epsilon = 1.0
        return hipdnn_data_sdk::types::fp8_e8m0::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E8M0_ONE);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e8m0 round_error() noexcept
    {
        // Round error is 0.5 in ULP, but E8M0 only has powers of 2
        return hipdnn_data_sdk::types::fp8_e8m0::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E8M0_ONE);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e8m0 infinity() noexcept
    {
        // E8M0 has no infinity, return max
        return max();
    }

    static constexpr hipdnn_data_sdk::types::fp8_e8m0 quiet_NaN() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e8m0::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E8M0_NAN);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e8m0 signaling_NaN() noexcept
    {
        // E8M0 has only one NaN representation
        return hipdnn_data_sdk::types::fp8_e8m0::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E8M0_NAN);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e8m0 denorm_min() noexcept
    {
        // E8M0 has no denormals, return min()
        return min();
    }
};
// NOLINTEND(readability-identifier-naming)
