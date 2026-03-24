// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
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
// FP4 E2M1 (OCP MX Format) Bit Layout Constants
// ============================================================================
// See fp4_e2m1 struct documentation for full format specification.
//
// Bit layout: [S|EE|M]
//              3 2 1 0
// ============================================================================

/// Sign bit mask (bit 3)
constexpr uint8_t FP4_E2M1_SIGN_MASK = 0x08;

/// Exponent field mask (bits 1-2)
constexpr uint8_t FP4_E2M1_EXP_MASK = 0x06;

/// Mantissa field mask (bit 0)
constexpr uint8_t FP4_E2M1_MANT_MASK = 0x01;

/// Absolute value mask (bits 0-2)
constexpr uint8_t FP4_E2M1_ABS_MASK = 0x07;

/// Exponent bias for E2M1
constexpr int FP4_E2M1_EXP_BIAS = 1;

/// Number of exponent bits
constexpr int FP4_E2M1_EXP_BITS = 2;

/// Number of mantissa bits
constexpr int FP4_E2M1_MANT_BITS = 1;

// ============================================================================
// FP4 E2M1 Special Values (bit patterns)
// ============================================================================

/// Positive zero
constexpr uint8_t FP4_E2M1_POS_ZERO = 0x00;

/// Negative zero
constexpr uint8_t FP4_E2M1_NEG_ZERO = 0x08;

/// Minimum positive subnormal: 0.5
constexpr uint8_t FP4_E2M1_DENORM_MIN = 0x01;

/// Minimum positive normal: 1.0
constexpr uint8_t FP4_E2M1_MIN_NORMAL = 0x02;

/// Maximum positive value: 6.0
constexpr uint8_t FP4_E2M1_MAX = 0x07;

/// Maximum negative value (lowest): -6.0
constexpr uint8_t FP4_E2M1_LOWEST = 0x0F;

/// Epsilon: smallest difference at 1.0 = 0.5
constexpr uint8_t FP4_E2M1_EPSILON = 0x01;

/// Round error (0.5) - same as epsilon for E2M1 format
constexpr uint8_t FP4_E2M1_ROUND_ERROR = 0x01;

/// Rounding threshold for round-to-nearest-even (midpoint of 22-bit remainder)
constexpr uint32_t FP4_E2M1_ROUND_THRESHOLD = 0x200000;

// Convert float to FP4 E2M1 bits (OCP MX format: 1 sign, 2 exponent, 1 mantissa)
// Range: +/- 6, no infinity (saturates to max), no NaN (returns zero)
// NOLINTNEXTLINE(readability-identifier-naming)
inline uint8_t float_to_fp4_e2m1_bits(float f) noexcept
{
    // Handle special values first using std library functions
    if(std::isnan(f))
    {
        // Per OCP MX Specification v1.0, conversion from NaN is implementation-defined.
        // This implementation returns zero.
        return FP4_E2M1_POS_ZERO;
    }

    if(std::isinf(f))
    {
        // E2M1 has no infinity - saturate to max
        return std::signbit(f) ? FP4_E2M1_LOWEST : FP4_E2M1_MAX;
    }

    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));

    const uint32_t sign = (bits >> 28) & FP4_E2M1_SIGN_MASK; // Extract sign to bit 3
    const uint32_t fp32Exp = (bits >> 23) & 0xFF;
    // Rebias from float (127) to E2M1 (1)
    int32_t exp = static_cast<int32_t>(fp32Exp) - 127 + FP4_E2M1_EXP_BIAS;
    uint32_t mant = bits & 0x007FFFFF;

    // Handle overflow
    if(exp > 3)
    {
        return static_cast<uint8_t>(sign | FP4_E2M1_ABS_MASK); // Saturate to max finite value
    }

    // Handle zero
    if(fp32Exp == 0 && mant == 0)
    {
        return static_cast<uint8_t>(sign); // Signed zero
    }

    // Handle inputs in the subnormal/underflow range (may round up to normal)
    if(exp <= 0)
    {
        mant |= 0x00800000; // Add implicit 1
        auto shift = static_cast<uint32_t>(1 - exp + 22); // 23 - 1 = 22 bits to shift
        if(shift > 24)
        {
            return static_cast<uint8_t>(sign); // Too small, return zero
        }

        // Apply round-to-nearest-even for subnormal results
        const uint32_t halfPoint = 1u << (shift - 1);
        const uint32_t remainder = mant & ((1u << shift) - 1);
        mant >>= shift;

        // Round to nearest even: round up if above midpoint, or at midpoint with odd result
        if(remainder > halfPoint || (remainder == halfPoint && ((mant & 1) != 0)))
        {
            mant++;
            // Check if rounding caused carry into normal range
            if(mant > FP4_E2M1_MANT_MASK)
            {
                // Promoted to smallest normal: exp=1, mant=0 -> value 1.0
                return static_cast<uint8_t>(sign | FP4_E2M1_MIN_NORMAL);
            }
        }

        return static_cast<uint8_t>(sign | (mant & FP4_E2M1_MANT_MASK));
    }

    // Normal case: shift mantissa from 23 bits to 1 bit with rounding
    uint32_t fp4Mant = (mant >> 22) & FP4_E2M1_MANT_MASK;
    const uint32_t remainder = mant & 0x003FFFFF;

    // Round to nearest even
    if(remainder > FP4_E2M1_ROUND_THRESHOLD
       || (remainder == FP4_E2M1_ROUND_THRESHOLD && ((fp4Mant & 1) != 0)))
    {
        fp4Mant++;
        if(fp4Mant > 1)
        {
            fp4Mant = 0;
            exp++;
            if(exp > 3)
            {
                return static_cast<uint8_t>(sign | FP4_E2M1_ABS_MASK);
            }
        }
    }

    return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 1) | fp4Mant);
}

// Convert FP4 E2M1 bits to float using lookup table
// NOLINTNEXTLINE(readability-identifier-naming)
inline float fp4_e2m1_bits_to_float(uint8_t bits) noexcept
{
    // clang-format off
    static constexpr std::array<float, 16> TABLE = {
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    // clang-format on
    return TABLE[bits & 0x0F];
}

} // namespace detail

/**
 * @brief Custom storage-only FP4 E2M1 type for hipDNN
 *
 * This type provides a portable FP4 E2M1 (1 sign, 2 exponent, 1 mantissa) implementation
 * that does not require the __HIPCC__ macro. Uses MX (Microscaling) scale format per
 * OCP Microscaling Formats (MX) Specification v1.0.
 *
 * This is a STORAGE-ONLY type intended for data representation and conversion,
 * not direct computation. Arithmetic operations and comparisons are
 * intentionally not provided. For computation, explicitly convert to float.
 *
 * E2M1 Format Properties:
 * - Binary layout: 1 sign bit (bit 3), 2 exponent bits (bits 1-2), 1 mantissa bit (bit 0)
 * - Exponent bias: 1
 * - No infinity representation
 * - No NaN representation
 * - Subnormals supported
 *
 * Representable Values:
 * | Bits | Sign | Exp | Mant | Value  |
 * |------|------|-----|------|--------|
 * | 0000 |  +   |  0  |  0   |  +0    |
 * | 0001 |  +   |  0  |  1   |  +0.5  |
 * | 0010 |  +   |  1  |  0   |  +1.0  |
 * | 0011 |  +   |  1  |  1   |  +1.5  |
 * | 0100 |  +   |  2  |  0   |  +2.0  |
 * | 0101 |  +   |  2  |  1   |  +3.0  |
 * | 0110 |  +   |  3  |  0   |  +4.0  |
 * | 0111 |  +   |  3  |  1   |  +6.0  |
 * | 1000 |  -   |  0  |  0   |  -0    |
 * | 1001 |  -   |  0  |  1   |  -0.5  |
 * | 1010 |  -   |  1  |  0   |  -1.0  |
 * | 1011 |  -   |  1  |  1   |  -1.5  |
 * | 1100 |  -   |  2  |  0   |  -2.0  |
 * | 1101 |  -   |  2  |  1   |  -3.0  |
 * | 1110 |  -   |  3  |  0   |  -4.0  |
 * | 1111 |  -   |  3  |  1   |  -6.0  |
 *
 * Range: -6.0 to +6.0
 * Rounding: roundTiesToEven (IEEE 754 default)
 * Overflow: Saturates to ±6 (maximum magnitude)
 * Underflow: Values with magnitude below 0.25 round to zero
 */
// NOLINTNEXTLINE(readability-identifier-naming) - lowercase for consistency
struct fp4_e2m1
{
    /// Raw bit representation of the FP4 E2M1 value.
    uint8_t data;

    // Default constructor - value-initialized to zero for constexpr support
    constexpr fp4_e2m1() noexcept
        : data(0)
    {
    }

    // Copy/move constructors - implicit
    fp4_e2m1(const fp4_e2m1&) = default;
    fp4_e2m1(fp4_e2m1&&) noexcept = default;
    fp4_e2m1& operator=(const fp4_e2m1&) = default;
    fp4_e2m1& operator=(fp4_e2m1&&) noexcept = default;

    // EXPLICIT constructor from float
    explicit fp4_e2m1(float f) noexcept
        : data(detail::float_to_fp4_e2m1_bits(f))
    {
    }

    // EXPLICIT constructor from double (via float)
    explicit fp4_e2m1(double d) noexcept
        : fp4_e2m1(static_cast<float>(d))
    {
    }

    // EXPLICIT constructor from integral types
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    explicit fp4_e2m1(T value) noexcept
        : fp4_e2m1(static_cast<float>(value))
    {
    }

    // EXPLICIT constructors from other custom types (via float)
    // These are defined out-of-line in cross_types.hpp
    template <Bfloat16RoundingMode M>
    inline explicit fp4_e2m1(bfloat16_t<M> b) noexcept;
    inline explicit fp4_e2m1(half h) noexcept;

    // Factory for raw bits
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr fp4_e2m1 from_bits(uint8_t bits) noexcept
    {
        fp4_e2m1 val;
        val.data = bits & 0x0F;
        return val;
    }

    // EXPLICIT conversion to float
    explicit operator float() const noexcept
    {
        return detail::fp4_e2m1_bits_to_float(data);
    }

    // EXPLICIT conversion to double
    explicit operator double() const noexcept
    {
        return static_cast<double>(detail::fp4_e2m1_bits_to_float(data));
    }

    // Unary negation - flip sign bit
    fp4_e2m1 operator-() const noexcept
    {
        return from_bits(data ^ detail::FP4_E2M1_SIGN_MASK);
    }

    // Unary plus
    fp4_e2m1 operator+() const noexcept
    {
        return *this;
    }

    // Stream output
    friend std::ostream& operator<<(std::ostream& os, fp4_e2m1 val)
    {
        return os << static_cast<float>(val);
    }
};

// Static assertions for type properties
static_assert(sizeof(fp4_e2m1) == sizeof(uint8_t), "fp4_e2m1 must be 1 byte");
static_assert(std::is_trivially_copyable_v<fp4_e2m1>, "fp4_e2m1 must be trivially copyable");
static_assert(std::is_standard_layout_v<fp4_e2m1>, "fp4_e2m1 must be standard layout");
static_assert(std::is_default_constructible_v<fp4_e2m1>, "fp4_e2m1 must be default constructible");
static_assert(std::is_copy_constructible_v<fp4_e2m1>, "fp4_e2m1 must be copy constructible");
static_assert(std::is_move_constructible_v<fp4_e2m1>, "fp4_e2m1 must be move constructible");

// User-defined literal
// NOLINTNEXTLINE(readability-identifier-naming)
inline fp4_e2m1 operator""_e2m1(long double val)
{
    return fp4_e2m1(static_cast<float>(val));
}

// ============================================================================
// Math functions for fp4_e2m1 (in hipdnn_data_sdk::types namespace)
// ============================================================================
// These are defined in our namespace to enable ADL (Argument Dependent Lookup).
// Use unqualified calls like: isnan(x), isinf(x), etc.
// ============================================================================

inline bool isnan(fp4_e2m1 /*x*/)
{
    // FP4 E2M1 has no NaN representation
    return false;
}

inline bool isinf(fp4_e2m1 /*x*/)
{
    // FP4 E2M1 has no infinity representation
    return false;
}

inline bool signbit(fp4_e2m1 x)
{
    return (x.data & detail::FP4_E2M1_SIGN_MASK) != 0;
}

inline bool isfinite(fp4_e2m1 /*x*/)
{
    // All FP4 E2M1 values are finite (no NaN or Inf)
    return true;
}

// ============================================================================
// Packed FP4 E2M1 Storage Type (2 values per byte)
// ============================================================================

/**
 * @brief Packed storage type for 2 FP4 E2M1 values in a single byte
 *
 * This type provides efficient packed storage for FP4 E2M1 values,
 * storing 2 values per byte.
 *
 * Storage Model:
 * - Low nibble (bits 0-3): lo element
 * - High nibble (bits 4-7): hi element
 *
 * This is a STORAGE-ONLY type. For value operations, access individual
 * elements via lo() and hi() methods.
 */
// NOLINTNEXTLINE(readability-identifier-naming) - lowercase for consistency
struct fp4x2_e2m1
{
    /// Raw bit representation storing 2 packed FP4 values.
    uint8_t data;

    // Default constructor - both elements zero
    constexpr fp4x2_e2m1() noexcept
        : data(0)
    {
    }

    // Copy/move constructors - implicit
    fp4x2_e2m1(const fp4x2_e2m1&) = default;
    fp4x2_e2m1(fp4x2_e2m1&&) noexcept = default;
    fp4x2_e2m1& operator=(const fp4x2_e2m1&) = default;
    fp4x2_e2m1& operator=(fp4x2_e2m1&&) noexcept = default;

    // Single value constructor - lo = val, hi = zero
    constexpr explicit fp4x2_e2m1(fp4_e2m1 lo) noexcept
        : data(lo.data & 0x0F)
    {
    }

    // Two values constructor
    constexpr fp4x2_e2m1(fp4_e2m1 lo, fp4_e2m1 hi) noexcept
        : data(static_cast<uint8_t>((lo.data & 0x0F) | ((hi.data & 0x0F) << 4)))
    {
    }

    // Get lo element (low nibble)
    constexpr fp4_e2m1 lo() const noexcept
    {
        return fp4_e2m1::from_bits(data & 0x0F);
    }

    // Get hi element (high nibble)
    constexpr fp4_e2m1 hi() const noexcept
    {
        return fp4_e2m1::from_bits((data >> 4) & 0x0F);
    }
};

// Static assertions for packed type properties
static_assert(sizeof(fp4x2_e2m1) == sizeof(uint8_t), "fp4x2_e2m1 must be 1 byte");
static_assert(std::is_trivially_copyable_v<fp4x2_e2m1>, "fp4x2_e2m1 must be trivially copyable");
static_assert(std::is_standard_layout_v<fp4x2_e2m1>, "fp4x2_e2m1 must be standard layout");

} // namespace hipdnn_data_sdk::types

// ============================================================================
// std::numeric_limits specialization for fp4_e2m1 (single value type)
// ============================================================================
// NOLINTBEGIN(readability-identifier-naming) - standard library names must match exactly
template <>
class std::numeric_limits<hipdnn_data_sdk::types::fp4_e2m1>
{
public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = false;
    static constexpr bool has_signaling_NaN = false;
    static constexpr std::float_denorm_style has_denorm = std::denorm_present;
    static constexpr bool has_denorm_loss = false;
    static constexpr std::float_round_style round_style = std::round_to_nearest;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = 2; // 1 mantissa + 1 implicit
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 2;
    static constexpr int radix = 2;
    static constexpr int min_exponent = 1; // 2^(min_exponent-1) = 2^0 = 1.0 = min()
    static constexpr int min_exponent10 = 0;
    static constexpr int max_exponent = 3; // 2^(max_exponent-1) * 1.5 = 6 = max()
    static constexpr int max_exponent10 = 0;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;

    static constexpr hipdnn_data_sdk::types::fp4_e2m1 min() noexcept
    {
        // Minimum positive normal value: 1.0
        return hipdnn_data_sdk::types::fp4_e2m1::from_bits(
            hipdnn_data_sdk::types::detail::FP4_E2M1_MIN_NORMAL);
    }

    static constexpr hipdnn_data_sdk::types::fp4_e2m1 lowest() noexcept
    {
        // Most negative finite value: -6.0
        return hipdnn_data_sdk::types::fp4_e2m1::from_bits(
            hipdnn_data_sdk::types::detail::FP4_E2M1_LOWEST);
    }

    static constexpr hipdnn_data_sdk::types::fp4_e2m1 max() noexcept
    {
        // Maximum positive finite value: 6.0
        return hipdnn_data_sdk::types::fp4_e2m1::from_bits(
            hipdnn_data_sdk::types::detail::FP4_E2M1_MAX);
    }

    static constexpr hipdnn_data_sdk::types::fp4_e2m1 epsilon() noexcept
    {
        // Epsilon: 0.5 (smallest difference at 1.0)
        return hipdnn_data_sdk::types::fp4_e2m1::from_bits(
            hipdnn_data_sdk::types::detail::FP4_E2M1_EPSILON);
    }

    static constexpr hipdnn_data_sdk::types::fp4_e2m1 round_error() noexcept
    {
        // Round error: 0.5
        return hipdnn_data_sdk::types::fp4_e2m1::from_bits(
            hipdnn_data_sdk::types::detail::FP4_E2M1_ROUND_ERROR);
    }

    static constexpr hipdnn_data_sdk::types::fp4_e2m1 infinity() noexcept
    {
        // Infinity: returns max (no infinity in E2M1)
        return max();
    }

    static constexpr hipdnn_data_sdk::types::fp4_e2m1 quiet_NaN() noexcept
    {
        // E2M1 has no NaN representation. Returns zero for consistency
        // with float-to-E2M1 conversion behavior.
        return hipdnn_data_sdk::types::fp4_e2m1::from_bits(
            hipdnn_data_sdk::types::detail::FP4_E2M1_POS_ZERO);
    }

    static constexpr hipdnn_data_sdk::types::fp4_e2m1 signaling_NaN() noexcept
    {
        // E2M1 has no NaN representation. Returns zero for consistency
        // with float-to-E2M1 conversion behavior.
        return hipdnn_data_sdk::types::fp4_e2m1::from_bits(
            hipdnn_data_sdk::types::detail::FP4_E2M1_POS_ZERO);
    }

    static constexpr hipdnn_data_sdk::types::fp4_e2m1 denorm_min() noexcept
    {
        // Minimum positive subnormal value: 0.5
        return hipdnn_data_sdk::types::fp4_e2m1::from_bits(
            hipdnn_data_sdk::types::detail::FP4_E2M1_DENORM_MIN);
    }
};
// NOLINTEND(readability-identifier-naming)
