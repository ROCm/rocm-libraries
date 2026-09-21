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
struct fp8_e4m3_fnuz;
// NOLINTEND(readability-identifier-naming)

namespace detail
{

// ============================================================================
// FP8 E5M2 FNUZ Bit Layout Constants
// ============================================================================
// See fp8_e5m2_fnuz struct documentation for full format specification.
//
// Bit layout: [S|EEEEE|MM]
//              7 6   2 1 0
// ============================================================================

/// Sign bit mask (bit 7)
constexpr uint8_t FP8_E5M2_FNUZ_SIGN_MASK = 0x80;

/// Absolute value mask (all bits except sign)
constexpr uint8_t FP8_E5M2_FNUZ_ABS_MASK = 0x7F;

/// Exponent field mask (bits 2-6)
constexpr uint8_t FP8_E5M2_FNUZ_EXP_MASK = 0x7C;

/// Mantissa field mask (bits 0-1)
constexpr uint8_t FP8_E5M2_FNUZ_MANT_MASK = 0x03;

/// Exponent bias
constexpr int FP8_E5M2_FNUZ_EXP_BIAS = 16;

/// Number of mantissa bits
constexpr int FP8_E5M2_FNUZ_MANT_BITS = 2;

/// Number of exponent bits
constexpr int FP8_E5M2_FNUZ_EXP_BITS = 5;

/// Maximum biased exponent representable in EXP_BITS bits
constexpr int FP8_E5M2_FNUZ_MAX_BIASED_EXP = (1 << FP8_E5M2_FNUZ_EXP_BITS) - 1;

// ============================================================================
// FP8 E5M2 FNUZ Special Values (bit patterns)
// ============================================================================

/// Single NaN encoding: sign=1, exp=0, mant=0 (0x80)
constexpr uint8_t FP8_E5M2_FNUZ_NAN = 0x80;

/// Positive zero (the only zero; no -0)
constexpr uint8_t FP8_E5M2_FNUZ_POS_ZERO = 0x00;

/// Maximum positive value: 0x7F = 57344.0
constexpr uint8_t FP8_E5M2_FNUZ_MAX = 0x7F;

/// Most negative value (lowest): 0xFF = -57344.0
constexpr uint8_t FP8_E5M2_FNUZ_LOWEST = 0xFF;

/// Minimum positive normal: 0x04 = 2^-15
constexpr uint8_t FP8_E5M2_FNUZ_MIN_NORMAL = 0x04;

/// Minimum positive subnormal: 0x01 = 1 x 2^-17
constexpr uint8_t FP8_E5M2_FNUZ_DENORM_MIN = 0x01;

/// Epsilon: smallest difference at 1.0 = 0.25
constexpr uint8_t FP8_E5M2_FNUZ_EPSILON = 0x38;

/// Round error (0.5) - maximum rounding error
constexpr uint8_t FP8_E5M2_FNUZ_ROUND_ERROR = 0x3C;

/// Rounding threshold for round-to-nearest-even (midpoint of 21-bit remainder)
constexpr uint32_t FP8_E5M2_FNUZ_ROUND_THRESHOLD = 0x100000;

// Convert float to FP8 E5M2 FNUZ bits
// NOLINTNEXTLINE(readability-identifier-naming)
inline uint8_t float_to_fp8_e5m2_fnuz_bits(float f) noexcept
{
    // Handle NaN: map to single FNUZ NaN encoding 0x80
    if(std::isnan(f))
    {
        return FP8_E5M2_FNUZ_NAN;
    }

    // Handle infinity: FNUZ has no infinity, saturate to MAX/LOWEST
    if(std::isinf(f))
    {
        return std::signbit(f) ? FP8_E5M2_FNUZ_LOWEST : FP8_E5M2_FNUZ_MAX;
    }

    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));

    const uint32_t sign = (bits >> 24) & 0x80; // Extract sign to bit 7
    const uint32_t fp32Exp = (bits >> 23) & 0xFF;
    // Rebias from float (127) to FNUZ E5M2 (16)
    int32_t exp = static_cast<int32_t>(fp32Exp) - 127 + FP8_E5M2_FNUZ_EXP_BIAS;
    const uint32_t mant = bits & 0x007FFFFF;

    // Handle +0.0 / -0.0 and fp32 subnormal inputs:
    // - FNUZ has no -0, both +0 and -0 map to 0x00
    // - fp32 subnormals (fp32Exp==0, mant!=0) are far smaller than any
    //   representable FNUZ value; flush to zero explicitly.
    if(fp32Exp == 0)
    {
        return FP8_E5M2_FNUZ_POS_ZERO;
    }

    // Handle overflow: saturate to MAX or LOWEST
    if(exp > FP8_E5M2_FNUZ_MAX_BIASED_EXP)
    {
        return static_cast<uint8_t>(sign | FP8_E5M2_FNUZ_ABS_MASK);
    }

    // Handle subnormal / underflow range
    if(exp <= 0)
    {
        // Add implicit leading 1 to the fp32 mantissa
        const uint32_t fullMant = mant | 0x00800000u;
        // Shift needed: normal encodes by shifting 23-2=21 bits; each step below exp=1
        // needs one more shift
        auto shift = static_cast<uint32_t>(1 - exp + 21);
        if(shift > 24)
        {
            // Value is smaller than half the minimum subnormal: underflow to zero.
            // FNUZ has no -0, so both positive and negative underflows return 0x00.
            return FP8_E5M2_FNUZ_POS_ZERO;
        }
        const uint32_t halfPoint = 1u << (shift - 1);
        const uint32_t remainder = fullMant & ((1u << shift) - 1u);
        uint32_t fp8Mant = fullMant >> shift;

        // Round to nearest even
        if(remainder > halfPoint || (remainder == halfPoint && ((fp8Mant & 1u) != 0u)))
        {
            fp8Mant++;
            if(fp8Mant > FP8_E5M2_FNUZ_MANT_MASK)
            {
                // Rounded up into the smallest normal (exp=1, mant=0)
                return static_cast<uint8_t>(sign | FP8_E5M2_FNUZ_MIN_NORMAL);
            }
        }
        // fp8Mant == 0 means round-to-zero: FNUZ has no -0, return positive zero.
        if(fp8Mant == 0u)
        {
            return FP8_E5M2_FNUZ_POS_ZERO;
        }
        return static_cast<uint8_t>(sign | (fp8Mant & FP8_E5M2_FNUZ_MANT_MASK));
    }

    // Normal case: shift mantissa from 23 bits to 2 bits with round-to-nearest-even
    uint32_t fp8Mant = (mant >> 21) & FP8_E5M2_FNUZ_MANT_MASK;
    const uint32_t remainder = mant & 0x001FFFFF;

    // Round to nearest even
    if(remainder > FP8_E5M2_FNUZ_ROUND_THRESHOLD
       || (remainder == FP8_E5M2_FNUZ_ROUND_THRESHOLD && ((fp8Mant & 1) != 0)))
    {
        fp8Mant++;
        if(fp8Mant > FP8_E5M2_FNUZ_MANT_MASK)
        {
            fp8Mant = 0;
            exp++;
            if(exp > FP8_E5M2_FNUZ_MAX_BIASED_EXP)
            {
                return static_cast<uint8_t>(sign | FP8_E5M2_FNUZ_ABS_MASK);
            }
        }
    }

    return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 2) | fp8Mant);
}

// Convert FP8 E5M2 FNUZ bits to float using a half-size lookup table.
// Only positive values (0x00..0x7F) are stored. For bits == 0x80 the function
// returns the single NaN value; for bits in 0x81..0xFF it negates the
// positive entry at (bits & 0x7F). This halves the table size from 1024 to
// 512 bytes.
// NOLINTNEXTLINE(readability-identifier-naming)
inline float fp8_e5m2_fnuz_bits_to_float(uint8_t bits) noexcept
{
    // clang-format off
    static constexpr std::array<float, 128> TABLE = {
        0.0f,                    7.62939453125e-06f,      1.52587890625e-05f,      2.288818359375e-05f,     // exp=0 (subnormal)
        3.0517578125e-05f,       3.814697265625e-05f,     4.57763671875e-05f,      5.340576171875e-05f,     // exp=1
        6.103515625e-05f,        7.62939453125e-05f,      9.1552734375e-05f,       0.0001068115234375f,     // exp=2
        0.0001220703125f,        0.000152587890625f,      0.00018310546875f,       0.000213623046875f,      // exp=3
        0.000244140625f,         0.00030517578125f,       0.0003662109375f,        0.00042724609375f,       // exp=4
        0.00048828125f,          0.0006103515625f,        0.000732421875f,         0.0008544921875f,        // exp=5
        0.0009765625f,           0.001220703125f,         0.00146484375f,          0.001708984375f,         // exp=6
        0.001953125f,            0.00244140625f,          0.0029296875f,           0.00341796875f,          // exp=7
        0.00390625f,             0.0048828125f,           0.005859375f,            0.0068359375f,           // exp=8
        0.0078125f,              0.009765625f,            0.01171875f,             0.013671875f,            // exp=9
        0.015625f,               0.01953125f,             0.0234375f,              0.02734375f,             // exp=10
        0.03125f,                0.0390625f,              0.046875f,               0.0546875f,              // exp=11
        0.0625f,                 0.078125f,               0.09375f,                0.109375f,               // exp=12
        0.125f,                  0.15625f,                0.1875f,                 0.21875f,                // exp=13
        0.25f,                   0.3125f,                 0.375f,                  0.4375f,                 // exp=14
        0.5f,                    0.625f,                  0.75f,                   0.875f,                  // exp=15
        1.0f,                    1.25f,                   1.5f,                    1.75f,                   // exp=16
        2.0f,                    2.5f,                    3.0f,                    3.5f,                    // exp=17
        4.0f,                    5.0f,                    6.0f,                    7.0f,                    // exp=18
        8.0f,                    10.0f,                   12.0f,                   14.0f,                   // exp=19
        16.0f,                   20.0f,                   24.0f,                   28.0f,                   // exp=20
        32.0f,                   40.0f,                   48.0f,                   56.0f,                   // exp=21
        64.0f,                   80.0f,                   96.0f,                   112.0f,                  // exp=22
        128.0f,                  160.0f,                  192.0f,                  224.0f,                  // exp=23
        256.0f,                  320.0f,                  384.0f,                  448.0f,                  // exp=24
        512.0f,                  640.0f,                  768.0f,                  896.0f,                  // exp=25
        1024.0f,                 1280.0f,                 1536.0f,                 1792.0f,                 // exp=26
        2048.0f,                 2560.0f,                 3072.0f,                 3584.0f,                 // exp=27
        4096.0f,                 5120.0f,                 6144.0f,                 7168.0f,                 // exp=28
        8192.0f,                 10240.0f,                12288.0f,                14336.0f,                // exp=29
        16384.0f,                20480.0f,                24576.0f,                28672.0f,                // exp=30
        32768.0f,                40960.0f,                49152.0f,                57344.0f,                // exp=31 (max)
    };
    // clang-format on
    if(bits < FP8_E5M2_FNUZ_NAN)
    {
        return TABLE[bits];
    }
    if(bits == FP8_E5M2_FNUZ_NAN)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return -TABLE[bits & FP8_E5M2_FNUZ_ABS_MASK];
}

} // namespace detail

/**
 * @brief Custom storage-only FP8 E5M2 FNUZ type for hipDNN
 *
 * This type provides a portable FP8 E5M2 FNUZ (1 sign, 5 exponent, 2 mantissa) implementation
 * that does not require the __HIPCC__ macro.
 *
 * Format properties (FNUZ differs from OCP in bias, NaN encoding, and lack of -0/inf):
 * - Bias = 16 (OCP uses 15)
 * - Single NaN encoding: 0x80 (sign=1, exp=0, mant=0)
 * - No negative zero; all zero inputs map to 0x00
 * - No infinity; overflow saturates to ±57344
 * - MAX = 57344.0 (0x7F), LOWEST = -57344.0 (0xFF)
 *
 * This is a STORAGE-ONLY type intended for data representation and conversion,
 * not direct computation. Arithmetic operations and comparisons are
 * intentionally not provided. For computation, explicitly convert to float.
 */
// NOLINTNEXTLINE(readability-identifier-naming) - lowercase for consistency
struct fp8_e5m2_fnuz
{
    /// Raw bit representation of the FP8 E5M2 FNUZ value.
    uint8_t data;

    // Default constructor - value-initialized to zero for constexpr support
    constexpr fp8_e5m2_fnuz() noexcept
        : data(0)
    {
    }

    // Copy/move constructors - implicit
    fp8_e5m2_fnuz(const fp8_e5m2_fnuz&) = default;
    fp8_e5m2_fnuz(fp8_e5m2_fnuz&&) noexcept = default;
    fp8_e5m2_fnuz& operator=(const fp8_e5m2_fnuz&) = default;
    fp8_e5m2_fnuz& operator=(fp8_e5m2_fnuz&&) noexcept = default;

    // EXPLICIT constructor from float
    explicit fp8_e5m2_fnuz(float f) noexcept
        : data(detail::float_to_fp8_e5m2_fnuz_bits(f))
    {
    }

    // EXPLICIT constructor from double (via float)
    explicit fp8_e5m2_fnuz(double d) noexcept
        : fp8_e5m2_fnuz(static_cast<float>(d))
    {
    }

    // EXPLICIT constructor from integral types
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    explicit fp8_e5m2_fnuz(T value) noexcept
        : fp8_e5m2_fnuz(static_cast<float>(value))
    {
    }

    // EXPLICIT constructors from other custom types (via float)
    // These are defined out-of-line in cross_types.hpp
    template <Bfloat16RoundingMode M>
    inline explicit fp8_e5m2_fnuz(bfloat16_t<M> b) noexcept;
    inline explicit fp8_e5m2_fnuz(half h) noexcept;
    inline explicit fp8_e5m2_fnuz(fp8_e4m3_fnuz f) noexcept;

    // Factory for raw bits
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr fp8_e5m2_fnuz from_bits(uint8_t bits) noexcept
    {
        fp8_e5m2_fnuz val;
        val.data = bits;
        return val;
    }

    // EXPLICIT conversion to float
    explicit operator float() const noexcept
    {
        return detail::fp8_e5m2_fnuz_bits_to_float(data);
    }

    // EXPLICIT conversion to double
    explicit operator double() const noexcept
    {
        return static_cast<double>(detail::fp8_e5m2_fnuz_bits_to_float(data));
    }

    // Unary negation - XOR sign bit
    // Note: applying to NaN (0x80) gives 0x00 (zero); operations on NaN are undefined.
    fp8_e5m2_fnuz operator-() const noexcept
    {
        return from_bits(data ^ detail::FP8_E5M2_FNUZ_SIGN_MASK);
    }

    // Unary plus
    fp8_e5m2_fnuz operator+() const noexcept
    {
        return *this;
    }

    // Stream output
    friend std::ostream& operator<<(std::ostream& os, fp8_e5m2_fnuz val)
    {
        return os << static_cast<float>(val);
    }
};

// Static assertions for binary compatibility
static_assert(sizeof(fp8_e5m2_fnuz) == sizeof(uint8_t), "fp8_e5m2_fnuz must be 1 byte");
static_assert(std::is_trivially_copyable_v<fp8_e5m2_fnuz>,
              "fp8_e5m2_fnuz must be trivially copyable");
static_assert(std::is_standard_layout_v<fp8_e5m2_fnuz>, "fp8_e5m2_fnuz must be standard layout");
static_assert(std::is_default_constructible_v<fp8_e5m2_fnuz>,
              "fp8_e5m2_fnuz must be default constructible");
static_assert(std::is_copy_constructible_v<fp8_e5m2_fnuz>,
              "fp8_e5m2_fnuz must be copy constructible");
static_assert(std::is_move_constructible_v<fp8_e5m2_fnuz>,
              "fp8_e5m2_fnuz must be move constructible");

// User-defined literal
// NOLINTNEXTLINE(readability-identifier-naming)
inline fp8_e5m2_fnuz operator""_e5m2_fnuz(long double val)
{
    return fp8_e5m2_fnuz(static_cast<float>(val));
}

// ============================================================================
// Math functions for fp8_e5m2_fnuz (in hipdnn_data_sdk::types namespace)
// ============================================================================
// These are defined in our namespace to enable ADL (Argument Dependent Lookup).
// Use unqualified calls like: isnan(x), isinf(x), etc.
// ============================================================================

inline bool isnan(fp8_e5m2_fnuz x)
{
    // FNUZ E5M2 has a single NaN encoding: 0x80
    return x.data == detail::FP8_E5M2_FNUZ_NAN;
}

inline bool isinf(fp8_e5m2_fnuz /*x*/)
{
    // FNUZ E5M2 has no infinity representation
    return false;
}

inline bool signbit(fp8_e5m2_fnuz x)
{
    return (x.data & detail::FP8_E5M2_FNUZ_SIGN_MASK) != 0;
}

inline bool isfinite(fp8_e5m2_fnuz x)
{
    return !isnan(x);
}

} // namespace hipdnn_data_sdk::types

// std::numeric_limits specialization
// NOLINTBEGIN(readability-identifier-naming) - standard library names must match exactly
template <>
class std::numeric_limits<hipdnn_data_sdk::types::fp8_e5m2_fnuz>
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
    static constexpr int digits = 3; // 2 mantissa + 1 implicit
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 2;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -14;
    static constexpr int min_exponent10 = -4;
    static constexpr int max_exponent = 16;
    static constexpr int max_exponent10 = 4;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;

    static constexpr hipdnn_data_sdk::types::fp8_e5m2_fnuz min() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2_fnuz::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E5M2_FNUZ_MIN_NORMAL);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2_fnuz lowest() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2_fnuz::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E5M2_FNUZ_LOWEST);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2_fnuz max() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2_fnuz::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E5M2_FNUZ_MAX);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2_fnuz epsilon() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2_fnuz::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E5M2_FNUZ_EPSILON);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2_fnuz round_error() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2_fnuz::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E5M2_FNUZ_ROUND_ERROR);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2_fnuz infinity() noexcept
    {
        // No infinity in FNUZ; return max instead
        return max();
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2_fnuz quiet_NaN() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2_fnuz::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E5M2_FNUZ_NAN);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2_fnuz signaling_NaN() noexcept
    {
        // has_signaling_NaN is false; by the standard, the return value is unspecified
        // when has_signaling_NaN is false. We return the single FNUZ NaN encoding for
        // consistency with quiet_NaN().
        return hipdnn_data_sdk::types::fp8_e5m2_fnuz::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E5M2_FNUZ_NAN);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2_fnuz denorm_min() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2_fnuz::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E5M2_FNUZ_DENORM_MIN);
    }
};
// NOLINTEND(readability-identifier-naming)
