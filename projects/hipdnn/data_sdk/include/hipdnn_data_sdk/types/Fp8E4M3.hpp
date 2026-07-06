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

/// Maximum biased exponent representable in EXP_BITS bits
constexpr int FP8_E4M3_MAX_BIASED_EXP = (1 << FP8_E4M3_EXP_BITS) - 1;

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
// NOTE: The `saturate=false` paths return NaN on overflow per the OCP spec and
// are reserved for future non-saturating-mode support.
// NOLINTNEXTLINE(readability-identifier-naming)
inline uint8_t float_to_fp8_e4m3_bits(float f, bool saturate = true) noexcept
{
    // Handle NaN: preserve sign bit in the OCP NaN encoding (0x7F / 0xFF)
    if(std::isnan(f))
    {
        return std::signbit(f) ? static_cast<uint8_t>(FP8_E4M3_NAN | FP8_E4M3_SIGN_MASK)
                               : FP8_E4M3_NAN;
    }

    // Handle infinity: E4M3 OCP has no infinity; saturate to MAX or return NaN
    if(std::isinf(f))
    {
        if(saturate)
        {
            return std::signbit(f) ? FP8_E4M3_LOWEST : FP8_E4M3_MAX;
        }
        return std::signbit(f) ? static_cast<uint8_t>(FP8_E4M3_NAN | FP8_E4M3_SIGN_MASK)
                               : FP8_E4M3_NAN;
    }

    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));

    const uint32_t sign = (bits >> 24) & FP8_E4M3_SIGN_MASK; // Extract sign to bit 7
    const uint32_t fp32Exp = (bits >> 23) & 0xFF;
    // Rebias from float (127) to OCP E4M3 (7)
    int32_t exp = static_cast<int32_t>(fp32Exp) - 127 + FP8_E4M3_EXP_BIAS;
    uint32_t mant = bits & 0x007FFFFF;

    // Handle +0.0 / -0.0 and fp32 subnormal inputs:
    // fp32 subnormals (fp32Exp==0, mant!=0) are far smaller than any representable
    // OCP E4M3 value; flush to signed zero explicitly.
    if(fp32Exp == 0)
    {
        return static_cast<uint8_t>(sign);
    }

    // Handle overflow: saturate to MAX or return NaN (E4M3 OCP has no infinity).
    // Note the asymmetry vs E5M2: in OCP E4M3, biased_exp == MAX_BIASED_EXP (15) is a
    // valid normal exponent (used by MAX = 448 at exp=15, mant=6), so only exp > 15
    // overflows. In OCP E5M2, biased_exp == MAX_BIASED_EXP (31) is reserved for Inf/NaN,
    // so the equivalent check there uses >=.
    if(exp > FP8_E4M3_MAX_BIASED_EXP)
    {
        if(saturate)
        {
            return static_cast<uint8_t>(sign | FP8_E4M3_MAX);
        }
        return static_cast<uint8_t>(sign | FP8_E4M3_NAN);
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

        const uint32_t halfPoint = 1u << (shift - 1);
        const uint32_t remainder = mant & ((1u << shift) - 1u);
        mant >>= shift;

        // Round to nearest even
        if(remainder > halfPoint || (remainder == halfPoint && ((mant & 1) != 0)))
        {
            mant++;
            if(mant > FP8_E4M3_MANT_MASK)
            {
                // Rounded up into the smallest normal (exp=1, mant=0)
                return static_cast<uint8_t>(sign | FP8_E4M3_MIN_NORMAL);
            }
        }
        if(mant == 0u)
        {
            return static_cast<uint8_t>(sign); // Rounded down to zero
        }
        return static_cast<uint8_t>(sign | (mant & FP8_E4M3_MANT_MASK));
    }

    // Normal case: shift mantissa from 23 bits to 3 bits with round-to-nearest-even
    uint32_t fp8Mant = (mant >> 20) & FP8_E4M3_MANT_MASK;
    const uint32_t remainder = mant & 0x000FFFFF;

    // Round to nearest even
    if(remainder > FP8_E4M3_ROUND_THRESHOLD
       || (remainder == FP8_E4M3_ROUND_THRESHOLD && ((fp8Mant & 1) != 0)))
    {
        fp8Mant++;
        if(fp8Mant > FP8_E4M3_MANT_MASK)
        {
            fp8Mant = 0;
            exp++;
            if(exp > FP8_E4M3_MAX_BIASED_EXP)
            {
                if(saturate)
                {
                    return static_cast<uint8_t>(sign | FP8_E4M3_MAX);
                }
                return static_cast<uint8_t>(sign | FP8_E4M3_NAN);
            }
        }
    }

    // OCP E4M3 special case: exp=15 && mant=7 produces bit pattern 0x7F (or 0xFF),
    // which is the NaN encoding. Any finite float value that maps exactly to this
    // bit pattern must instead saturate to MAX (0x7E/0xFE), since 0x7F/0xFF
    // is reserved for NaN in OCP E4M3.
    if(exp == FP8_E4M3_MAX_BIASED_EXP && fp8Mant == FP8_E4M3_MANT_MASK)
    {
        if(saturate)
        {
            return static_cast<uint8_t>(sign | FP8_E4M3_MAX);
        }
        return static_cast<uint8_t>(sign | FP8_E4M3_NAN);
    }

    return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 3) | fp8Mant);
}

// Convert FP8 E4M3 bits to float using a half-size lookup table.
// Only positive values (0x00..0x7E) are stored; 0x7F is NaN.
// For bits in 0x80..0xFE the function negates the positive entry at (bits & 0x7F).
// For 0xFF it returns a negative NaN. This halves the table size from 1024 to 512 bytes.
// NOLINTNEXTLINE(readability-identifier-naming)
inline float fp8_e4m3_bits_to_float(uint8_t bits) noexcept
{
    // clang-format off
    static constexpr std::array<float, 128> TABLE = {
        0.0f,           0.001953125f,   0.00390625f,    0.005859375f,   0.0078125f,     0.009765625f,   0.01171875f,    0.013671875f,   // exp=0 (subnormal)
        0.015625f,      0.017578125f,   0.01953125f,    0.021484375f,   0.0234375f,     0.025390625f,   0.02734375f,    0.029296875f,   // exp=1
        0.03125f,       0.03515625f,    0.0390625f,     0.04296875f,    0.046875f,      0.05078125f,    0.0546875f,     0.05859375f,    // exp=2
        0.0625f,        0.0703125f,     0.078125f,      0.0859375f,     0.09375f,       0.1015625f,     0.109375f,      0.1171875f,     // exp=3
        0.125f,         0.140625f,      0.15625f,       0.171875f,      0.1875f,        0.203125f,      0.21875f,       0.234375f,      // exp=4
        0.25f,          0.28125f,       0.3125f,        0.34375f,       0.375f,         0.40625f,       0.4375f,        0.46875f,       // exp=5
        0.5f,           0.5625f,        0.625f,         0.6875f,        0.75f,          0.8125f,        0.875f,         0.9375f,        // exp=6
        1.0f,           1.125f,         1.25f,          1.375f,         1.5f,           1.625f,         1.75f,          1.875f,         // exp=7
        2.0f,           2.25f,          2.5f,           2.75f,          3.0f,           3.25f,          3.5f,           3.75f,          // exp=8
        4.0f,           4.5f,           5.0f,           5.5f,           6.0f,           6.5f,           7.0f,           7.5f,           // exp=9
        8.0f,           9.0f,           10.0f,          11.0f,          12.0f,          13.0f,          14.0f,          15.0f,          // exp=10
        16.0f,          18.0f,          20.0f,          22.0f,          24.0f,          26.0f,          28.0f,          30.0f,          // exp=11
        32.0f,          36.0f,          40.0f,          44.0f,          48.0f,          52.0f,          56.0f,          60.0f,          // exp=12
        64.0f,          72.0f,          80.0f,          88.0f,          96.0f,          104.0f,         112.0f,         120.0f,         // exp=13
        128.0f,         144.0f,         160.0f,         176.0f,         192.0f,         208.0f,         224.0f,         240.0f,         // exp=14
        256.0f,         288.0f,         320.0f,         352.0f,         384.0f,         416.0f,         448.0f,         std::numeric_limits<float>::quiet_NaN(), // exp=15: mant=0..6 normal, mant=7 (0x7F) = NaN; the early NaN check below guards this slot, so the value here is for self-documentation only
    };
    // clang-format on
    const uint8_t absBits = bits & FP8_E4M3_ABS_MASK;
    // Handle both NaN patterns: 0x7F (positive) and 0xFF (negative)
    if(absBits == FP8_E4M3_NAN)
    {
        constexpr float QUIET_NAN = std::numeric_limits<float>::quiet_NaN();
        return ((bits & FP8_E4M3_SIGN_MASK) != 0) ? std::copysign(QUIET_NAN, -1.0f) : QUIET_NAN;
    }
    if(bits < FP8_E4M3_NAN)
    {
        return TABLE[bits];
    }
    return -TABLE[absBits];
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
        : fp8_e4m3(static_cast<float>(d))
    {
    }

    // EXPLICIT constructor from integral types
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    explicit fp8_e4m3(T value) noexcept
        : fp8_e4m3(static_cast<float>(value))
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
static_assert(std::is_default_constructible_v<fp8_e4m3>, "fp8_e4m3 must be default constructible");
static_assert(std::is_copy_constructible_v<fp8_e4m3>, "fp8_e4m3 must be copy constructible");
static_assert(std::is_move_constructible_v<fp8_e4m3>, "fp8_e4m3 must be move constructible");

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
