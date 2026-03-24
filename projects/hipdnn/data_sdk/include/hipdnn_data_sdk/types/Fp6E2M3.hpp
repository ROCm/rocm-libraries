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
// FP6 E2M3 (OCP MX Format) Bit Layout Constants
// ============================================================================
// See fp6_e2m3 struct documentation for full format specification.
//
// Bit layout: [S|EE|MMM]
//              5 4 3 2 1 0
// ============================================================================

/// Sign bit mask (bit 5)
constexpr uint8_t FP6_E2M3_SIGN_MASK = 0x20;

/// Exponent field mask (bits 3-4)
constexpr uint8_t FP6_E2M3_EXP_MASK = 0x18;

/// Mantissa field mask (bits 0-2)
constexpr uint8_t FP6_E2M3_MANT_MASK = 0x07;

/// Absolute value mask (bits 0-4)
constexpr uint8_t FP6_E2M3_ABS_MASK = 0x1F;

/// Exponent bias for E2M3
constexpr int FP6_E2M3_EXP_BIAS = 1;

/// Number of exponent bits
constexpr int FP6_E2M3_EXP_BITS = 2;

/// Number of mantissa bits
constexpr int FP6_E2M3_MANT_BITS = 3;

// ============================================================================
// FP6 E2M3 Special Values (bit patterns)
// ============================================================================

/// Positive zero
constexpr uint8_t FP6_E2M3_POS_ZERO = 0x00;

/// Negative zero
constexpr uint8_t FP6_E2M3_NEG_ZERO = 0x20;

/// Minimum positive subnormal: 0.125
constexpr uint8_t FP6_E2M3_DENORM_MIN = 0x01;

/// Minimum positive normal: 1.0
constexpr uint8_t FP6_E2M3_MIN_NORMAL = 0x08;

/// Maximum positive value: 7.5
constexpr uint8_t FP6_E2M3_MAX = 0x1F;

/// Maximum negative value (lowest): -7.5
constexpr uint8_t FP6_E2M3_LOWEST = 0x3F;

/// Epsilon: smallest difference at 1.0 = 0.125
constexpr uint8_t FP6_E2M3_EPSILON = 0x01;

/// Round error (0.5) - maximum rounding error
constexpr uint8_t FP6_E2M3_ROUND_ERROR = 0x04;

/// Rounding threshold for round-to-nearest-even (midpoint of 20-bit remainder)
constexpr uint32_t FP6_E2M3_ROUND_THRESHOLD = 0x80000;

// Convert float to FP6 E2M3 bits (OCP MX format: 1 sign, 2 exponent, 3 mantissa)
// Range: +/- 7.5, no infinity (saturates to max), no NaN (returns zero)
// NOLINTNEXTLINE(readability-identifier-naming)
inline uint8_t float_to_fp6_e2m3_bits(float f) noexcept
{
    // Handle special values first using std library functions
    if(std::isnan(f))
    {
        // Per OCP MX Specification v1.0, conversion from NaN is implementation-defined.
        // This implementation returns zero.
        return FP6_E2M3_POS_ZERO;
    }

    if(std::isinf(f))
    {
        // E2M3 has no infinity - saturate to max
        return std::signbit(f) ? FP6_E2M3_LOWEST : FP6_E2M3_MAX;
    }

    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));

    const uint32_t sign = (bits >> 26) & FP6_E2M3_SIGN_MASK; // Extract sign to bit 5
    const uint32_t fp32Exp = (bits >> 23) & 0xFF;
    // Rebias from float (127) to E2M3 (1)
    int32_t exp = static_cast<int32_t>(fp32Exp) - 127 + FP6_E2M3_EXP_BIAS;
    uint32_t mant = bits & 0x007FFFFF;

    // Handle overflow
    if(exp > 3)
    {
        return static_cast<uint8_t>(sign | FP6_E2M3_ABS_MASK); // Saturate to max finite value
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
        auto shift = static_cast<uint32_t>(1 - exp + 20); // 23 - 3 = 20 bits to shift
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
            if(mant > FP6_E2M3_MANT_MASK)
            {
                // Promoted to smallest normal: exp=1, mant=0 -> value 1.0
                return static_cast<uint8_t>(sign | FP6_E2M3_MIN_NORMAL);
            }
        }

        return static_cast<uint8_t>(sign | (mant & FP6_E2M3_MANT_MASK));
    }

    // Normal case: shift mantissa from 23 bits to 3 bits with rounding
    uint32_t fp6Mant = (mant >> 20) & FP6_E2M3_MANT_MASK;
    const uint32_t remainder = mant & 0x000FFFFF;

    // Round to nearest even
    if(remainder > FP6_E2M3_ROUND_THRESHOLD
       || (remainder == FP6_E2M3_ROUND_THRESHOLD && ((fp6Mant & 1) != 0)))
    {
        fp6Mant++;
        if(fp6Mant > 7)
        {
            fp6Mant = 0;
            exp++;
            if(exp > 3)
            {
                return static_cast<uint8_t>(sign | FP6_E2M3_ABS_MASK);
            }
        }
    }

    return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 3) | fp6Mant);
}

// Convert FP6 E2M3 bits to float using lookup table
// NOLINTNEXTLINE(readability-identifier-naming)
inline float fp6_e2m3_bits_to_float(uint8_t bits) noexcept
{
    // clang-format off
    static constexpr std::array<float, 64> TABLE = {
        // Positive values (bits 0-31)
        0.0f,    0.125f,  0.25f,   0.375f,  0.5f,    0.625f,  0.75f,   0.875f,   // exp=0 (subnormal)
        1.0f,    1.125f,  1.25f,   1.375f,  1.5f,    1.625f,  1.75f,   1.875f,   // exp=1
        2.0f,    2.25f,   2.5f,    2.75f,   3.0f,    3.25f,   3.5f,    3.75f,    // exp=2
        4.0f,    4.5f,    5.0f,    5.5f,    6.0f,    6.5f,    7.0f,    7.5f,     // exp=3
        // Negative values (bits 32-63)
        -0.0f,   -0.125f, -0.25f,  -0.375f, -0.5f,   -0.625f, -0.75f,  -0.875f,  // exp=0 (subnormal)
        -1.0f,   -1.125f, -1.25f,  -1.375f, -1.5f,   -1.625f, -1.75f,  -1.875f,  // exp=1
        -2.0f,   -2.25f,  -2.5f,   -2.75f,  -3.0f,   -3.25f,  -3.5f,   -3.75f,   // exp=2
        -4.0f,   -4.5f,   -5.0f,   -5.5f,   -6.0f,   -6.5f,   -7.0f,   -7.5f     // exp=3
    };
    // clang-format on
    return TABLE[bits & 0x3F];
}

} // namespace detail

/**
 * @brief Custom storage-only FP6 E2M3 type for hipDNN
 *
 * This type provides a portable FP6 E2M3 (1 sign, 2 exponent, 3 mantissa) implementation
 * that does not require the __HIPCC__ macro. Uses MX (Microscaling) scale format per
 * OCP Microscaling Formats (MX) Specification v1.0.
 *
 * This is a STORAGE-ONLY type intended for data representation and conversion,
 * not direct computation. Arithmetic operations and comparisons are
 * intentionally not provided. For computation, explicitly convert to float.
 *
 * E2M3 Format Properties:
 * - Binary layout: 1 sign bit (bit 5), 2 exponent bits (bits 3-4), 3 mantissa bits (bits 0-2)
 * - Exponent bias: 1
 * - No infinity representation
 * - No NaN representation
 * - Subnormals supported
 *
 * Range: -7.5 to +7.5
 * Rounding: roundTiesToEven (IEEE 754 default)
 * Overflow: Saturates to ±7.5 (maximum magnitude)
 * Underflow: Values with magnitude below 0.0625 round to zero
 */
// NOLINTNEXTLINE(readability-identifier-naming) - lowercase for consistency
struct fp6_e2m3
{
    /// Raw bit representation of the FP6 E2M3 value.
    uint8_t data;

    // Default constructor - value-initialized to zero for constexpr support
    constexpr fp6_e2m3() noexcept
        : data(0)
    {
    }

    // Copy/move constructors - implicit
    fp6_e2m3(const fp6_e2m3&) = default;
    fp6_e2m3(fp6_e2m3&&) noexcept = default;
    fp6_e2m3& operator=(const fp6_e2m3&) = default;
    fp6_e2m3& operator=(fp6_e2m3&&) noexcept = default;

    // EXPLICIT constructor from float
    explicit fp6_e2m3(float f) noexcept
        : data(detail::float_to_fp6_e2m3_bits(f))
    {
    }

    // EXPLICIT constructor from double (via float)
    explicit fp6_e2m3(double d) noexcept
        : fp6_e2m3(static_cast<float>(d))
    {
    }

    // EXPLICIT constructor from integral types
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    explicit fp6_e2m3(T value) noexcept
        : fp6_e2m3(static_cast<float>(value))
    {
    }

    // EXPLICIT constructors from other custom types (via float)
    // These are defined out-of-line in cross_types.hpp
    template <Bfloat16RoundingMode M>
    inline explicit fp6_e2m3(bfloat16_t<M> b) noexcept;
    inline explicit fp6_e2m3(half h) noexcept;

    // Factory for raw bits
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr fp6_e2m3 from_bits(uint8_t bits) noexcept
    {
        fp6_e2m3 val;
        val.data = bits & 0x3F;
        return val;
    }

    // EXPLICIT conversion to float
    explicit operator float() const noexcept
    {
        return detail::fp6_e2m3_bits_to_float(data);
    }

    // EXPLICIT conversion to double
    explicit operator double() const noexcept
    {
        return static_cast<double>(detail::fp6_e2m3_bits_to_float(data));
    }

    // Unary negation - flip sign bit
    fp6_e2m3 operator-() const noexcept
    {
        return from_bits(data ^ detail::FP6_E2M3_SIGN_MASK);
    }

    // Unary plus
    fp6_e2m3 operator+() const noexcept
    {
        return *this;
    }

    // Stream output
    friend std::ostream& operator<<(std::ostream& os, fp6_e2m3 val)
    {
        return os << static_cast<float>(val);
    }
};

// Static assertions for type properties
static_assert(sizeof(fp6_e2m3) == sizeof(uint8_t), "fp6_e2m3 must be 1 byte");
static_assert(std::is_trivially_copyable_v<fp6_e2m3>, "fp6_e2m3 must be trivially copyable");
static_assert(std::is_standard_layout_v<fp6_e2m3>, "fp6_e2m3 must be standard layout");
static_assert(std::is_default_constructible_v<fp6_e2m3>, "fp6_e2m3 must be default constructible");
static_assert(std::is_copy_constructible_v<fp6_e2m3>, "fp6_e2m3 must be copy constructible");
static_assert(std::is_move_constructible_v<fp6_e2m3>, "fp6_e2m3 must be move constructible");

// User-defined literal
// NOLINTNEXTLINE(readability-identifier-naming)
inline fp6_e2m3 operator""_e2m3(long double val)
{
    return fp6_e2m3(static_cast<float>(val));
}

// ============================================================================
// Math functions for fp6_e2m3 (in hipdnn_data_sdk::types namespace)
// ============================================================================
// These are defined in our namespace to enable ADL (Argument Dependent Lookup).
// Use unqualified calls like: isnan(x), isinf(x), etc.
// ============================================================================

inline bool isnan(fp6_e2m3 /*x*/)
{
    // FP6 E2M3 has no NaN representation
    return false;
}

inline bool isinf(fp6_e2m3 /*x*/)
{
    // FP6 E2M3 has no infinity representation
    return false;
}

inline bool signbit(fp6_e2m3 x)
{
    return (x.data & detail::FP6_E2M3_SIGN_MASK) != 0;
}

inline bool isfinite(fp6_e2m3 /*x*/)
{
    // All FP6 E2M3 values are finite (no NaN or Inf)
    return true;
}

// ============================================================================
// Packed FP6 E2M3 Storage Type (2 values per 16 bits)
// ============================================================================

/**
 * @brief Packed storage type for 2 FP6 E2M3 values in 16 bits
 *
 * This type provides packed storage for FP6 E2M3 values,
 * storing 2 values per 16-bit word.
 *
 * Storage Model:
 * - Bits 0-5: lo element
 * - Bits 6-11: hi element
 * - Bits 12-15: unused (padding)
 *
 * This is a STORAGE-ONLY type. For value operations, access individual
 * elements via lo() and hi() methods.
 */
// NOLINTNEXTLINE(readability-identifier-naming) - lowercase for consistency
struct fp6x2_e2m3
{
    /// Raw bit representation storing 2 packed FP6 values.
    /// Bits 0-5 = lo element, Bits 6-11 = hi element, Bits 12-15 = unused.
    uint16_t data;

    // Default constructor - both elements zero
    constexpr fp6x2_e2m3() noexcept
        : data(0)
    {
    }

    // Copy/move constructors - implicit
    fp6x2_e2m3(const fp6x2_e2m3&) = default;
    fp6x2_e2m3(fp6x2_e2m3&&) noexcept = default;
    fp6x2_e2m3& operator=(const fp6x2_e2m3&) = default;
    fp6x2_e2m3& operator=(fp6x2_e2m3&&) noexcept = default;

    // Single value constructor - lo = val, hi = zero
    constexpr explicit fp6x2_e2m3(fp6_e2m3 lo) noexcept
        : data(lo.data & 0x3F)
    {
    }

    // Two values constructor
    constexpr fp6x2_e2m3(fp6_e2m3 lo, fp6_e2m3 hi) noexcept
        : data(static_cast<uint16_t>((lo.data & 0x3F) | ((hi.data & 0x3F) << 6)))
    {
    }

    // Get lo element (bits 0-5)
    constexpr fp6_e2m3 lo() const noexcept
    {
        return fp6_e2m3::from_bits(static_cast<uint8_t>(data & 0x3F));
    }

    // Get hi element (bits 6-11)
    constexpr fp6_e2m3 hi() const noexcept
    {
        return fp6_e2m3::from_bits(static_cast<uint8_t>((data >> 6) & 0x3F));
    }
};

// Static assertions for packed type properties
static_assert(sizeof(fp6x2_e2m3) == sizeof(uint16_t), "fp6x2_e2m3 must be 2 bytes");
static_assert(std::is_trivially_copyable_v<fp6x2_e2m3>, "fp6x2_e2m3 must be trivially copyable");
static_assert(std::is_standard_layout_v<fp6x2_e2m3>, "fp6x2_e2m3 must be standard layout");

// ============================================================================
// Packed FP6 E2M3 Storage Type (4 values in 3 bytes)
// ============================================================================

/**
 * @brief Packed storage type for 4 FP6 E2M3 values in 3 bytes (24 bits)
 *
 * This type provides efficient packed storage for FP6 E2M3 values,
 * storing 4 values in exactly 3 bytes with no padding.
 *
 * Storage Model (24 bits total):
 * - data[0] bits 0-5: element 0 (lo of lo_pair)
 * - data[0] bits 6-7 + data[1] bits 0-3: element 1 (hi of lo_pair)
 * - data[1] bits 4-7 + data[2] bits 0-1: element 2 (lo of hi_pair)
 * - data[2] bits 2-7: element 3 (hi of hi_pair)
 *
 * This is a STORAGE-ONLY type. For value operations, access pairs
 * via lo_pair() and hi_pair() methods.
 */
// NOLINTNEXTLINE(readability-identifier-naming) - lowercase for consistency
struct fp6x4_e2m3
{
    /// Raw bit representation storing 4 packed FP6 values in 3 bytes.
    std::array<uint8_t, 3> data;

    // Default constructor - all elements zero
    constexpr fp6x4_e2m3() noexcept
        : data{{0, 0, 0}}
    {
    }

    // Copy/move constructors - implicit
    fp6x4_e2m3(const fp6x4_e2m3&) = default;
    fp6x4_e2m3(fp6x4_e2m3&&) noexcept = default;
    fp6x4_e2m3& operator=(const fp6x4_e2m3&) = default;
    fp6x4_e2m3& operator=(fp6x4_e2m3&&) noexcept = default;

    // Two pairs constructor
    constexpr fp6x4_e2m3(fp6x2_e2m3 loPair, fp6x2_e2m3 hiPair) noexcept
        : data{{static_cast<uint8_t>(loPair.data),
                static_cast<uint8_t>((loPair.data >> 8) | (hiPair.data << 4)),
                static_cast<uint8_t>(hiPair.data >> 4)}}
    {
    }

    // Get lo_pair (elements 0 and 1)
    // NOLINTNEXTLINE(readability-identifier-naming)
    constexpr fp6x2_e2m3 lo_pair() const noexcept
    {
        fp6x2_e2m3 result;
        result.data = static_cast<uint16_t>(data[0] | ((data[1] & 0x0F) << 8));
        return result;
    }

    // Get hi_pair (elements 2 and 3)
    // NOLINTNEXTLINE(readability-identifier-naming)
    constexpr fp6x2_e2m3 hi_pair() const noexcept
    {
        fp6x2_e2m3 result;
        result.data = static_cast<uint16_t>((data[1] >> 4) | (data[2] << 4));
        return result;
    }
};

// Static assertions for packed type properties
static_assert(sizeof(fp6x4_e2m3) == 3, "fp6x4_e2m3 must be 3 bytes");
static_assert(std::is_trivially_copyable_v<fp6x4_e2m3>, "fp6x4_e2m3 must be trivially copyable");
static_assert(std::is_standard_layout_v<fp6x4_e2m3>, "fp6x4_e2m3 must be standard layout");

} // namespace hipdnn_data_sdk::types

// ============================================================================
// std::numeric_limits specialization for fp6_e2m3 (single value type)
// ============================================================================
// NOLINTBEGIN(readability-identifier-naming) - standard library names must match exactly
template <>
class std::numeric_limits<hipdnn_data_sdk::types::fp6_e2m3>
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
    static constexpr int digits = 4; // 3 mantissa + 1 implicit
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 3;
    static constexpr int radix = 2;
    static constexpr int min_exponent = 1; // 2^(min_exponent-1) = 2^0 = 1.0 = min()
    static constexpr int min_exponent10 = 0;
    static constexpr int max_exponent = 3; // 2^(max_exponent-1) * 1.875 = 7.5 = max()
    static constexpr int max_exponent10 = 0;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;

    static constexpr hipdnn_data_sdk::types::fp6_e2m3 min() noexcept
    {
        // Minimum positive normal value: 1.0
        return hipdnn_data_sdk::types::fp6_e2m3::from_bits(
            hipdnn_data_sdk::types::detail::FP6_E2M3_MIN_NORMAL);
    }

    static constexpr hipdnn_data_sdk::types::fp6_e2m3 lowest() noexcept
    {
        // Most negative finite value: -7.5
        return hipdnn_data_sdk::types::fp6_e2m3::from_bits(
            hipdnn_data_sdk::types::detail::FP6_E2M3_LOWEST);
    }

    static constexpr hipdnn_data_sdk::types::fp6_e2m3 max() noexcept
    {
        // Maximum positive finite value: 7.5
        return hipdnn_data_sdk::types::fp6_e2m3::from_bits(
            hipdnn_data_sdk::types::detail::FP6_E2M3_MAX);
    }

    static constexpr hipdnn_data_sdk::types::fp6_e2m3 epsilon() noexcept
    {
        // Epsilon: 0.125 (smallest difference at 1.0)
        return hipdnn_data_sdk::types::fp6_e2m3::from_bits(
            hipdnn_data_sdk::types::detail::FP6_E2M3_EPSILON);
    }

    static constexpr hipdnn_data_sdk::types::fp6_e2m3 round_error() noexcept
    {
        // Round error: 0.5
        return hipdnn_data_sdk::types::fp6_e2m3::from_bits(
            hipdnn_data_sdk::types::detail::FP6_E2M3_ROUND_ERROR);
    }

    static constexpr hipdnn_data_sdk::types::fp6_e2m3 infinity() noexcept
    {
        // Infinity: returns max (no infinity in E2M3)
        return max();
    }

    static constexpr hipdnn_data_sdk::types::fp6_e2m3 quiet_NaN() noexcept
    {
        // E2M3 has no NaN representation. Returns zero for consistency
        // with float-to-E2M3 conversion behavior.
        return hipdnn_data_sdk::types::fp6_e2m3::from_bits(
            hipdnn_data_sdk::types::detail::FP6_E2M3_POS_ZERO);
    }

    static constexpr hipdnn_data_sdk::types::fp6_e2m3 signaling_NaN() noexcept
    {
        // E2M3 has no NaN representation. Returns zero for consistency
        // with float-to-E2M3 conversion behavior.
        return hipdnn_data_sdk::types::fp6_e2m3::from_bits(
            hipdnn_data_sdk::types::detail::FP6_E2M3_POS_ZERO);
    }

    static constexpr hipdnn_data_sdk::types::fp6_e2m3 denorm_min() noexcept
    {
        // Minimum positive subnormal value: 0.125
        return hipdnn_data_sdk::types::fp6_e2m3::from_bits(
            hipdnn_data_sdk::types::detail::FP6_E2M3_DENORM_MIN);
    }
};
// NOLINTEND(readability-identifier-naming)
