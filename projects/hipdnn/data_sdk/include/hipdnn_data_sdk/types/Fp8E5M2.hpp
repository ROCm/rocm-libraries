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

/// Maximum biased exponent representable in EXP_BITS bits
constexpr int FP8_E5M2_MAX_BIASED_EXP = (1 << FP8_E5M2_EXP_BITS) - 1;

// ============================================================================
// FP8 E5M2 Special Values (bit patterns)
// ============================================================================

/// Positive infinity: exponent all 1s, mantissa 0 (0x7C)
constexpr uint8_t FP8_E5M2_POS_INF = 0x7C;

/// Negative infinity
constexpr uint8_t FP8_E5M2_NEG_INF = 0xFC;

/// NaN: exponent all 1s, mantissa non-zero (0x7F is)
/// Note: OCP E5M2 does not distinguish between signaling and quiet NaN.
/// All NaN encodings S.11111.{01, 10, 11} are treated as (quiet) NaN.
constexpr uint8_t FP8_E5M2_NAN = 0x7F;

/// Smallest absolute bit pattern that decodes to NaN (0x7D).
/// NaN range is [0x7D, 0x7F]; equivalently exp=31 and mant != 0.
constexpr uint8_t FP8_E5M2_NAN_MIN = 0x7D;

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
// NOTE: The `saturate=false` paths return Inf on overflow per the OCP spec and
// are reserved for future non-saturating-mode support.
// NOLINTNEXTLINE(readability-identifier-naming)
inline uint8_t float_to_fp8_e5m2_bits(float f, bool saturate = true) noexcept
{
    // Handle NaN: OCP E5M2 supports multiple NaN encodings; preserve sign
    if(std::isnan(f))
    {
        return std::signbit(f) ? static_cast<uint8_t>(FP8_E5M2_NAN | FP8_E5M2_SIGN_MASK)
                               : FP8_E5M2_NAN;
    }

    // Handle infinity: OCP E5M2 has infinity; saturate to MAX or pass through as Inf
    if(std::isinf(f))
    {
        if(saturate)
        {
            return std::signbit(f) ? FP8_E5M2_LOWEST : FP8_E5M2_MAX;
        }
        return std::signbit(f) ? FP8_E5M2_NEG_INF : FP8_E5M2_POS_INF;
    }

    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));

    const uint32_t sign = (bits >> 24) & FP8_E5M2_SIGN_MASK; // Extract sign to bit 7
    const uint32_t fp32Exp = (bits >> 23) & 0xFF;
    // Rebias from float (127) to OCP E5M2 (15)
    int32_t exp = static_cast<int32_t>(fp32Exp) - 127 + FP8_E5M2_EXP_BIAS;
    uint32_t mant = bits & 0x007FFFFF;

    // Handle +0.0 / -0.0 and fp32 subnormal inputs:
    // fp32 subnormals (fp32Exp==0, mant!=0) are far smaller than any representable
    // OCP E5M2 value; flush to signed zero explicitly.
    if(fp32Exp == 0)
    {
        return static_cast<uint8_t>(sign);
    }

    // Handle overflow: saturate to MAX or return Inf (E5M2 OCP supports infinity).
    // Note the asymmetry vs E4M3: in OCP E5M2, biased_exp == MAX_BIASED_EXP (31) is
    // reserved for Inf/NaN, so the highest valid normal biased_exp is 30 and the check
    // must reject exp == 31 too (hence >=). In OCP E4M3, biased_exp == 15 is a valid
    // normal exponent and the equivalent check there uses >.
    if(exp >= FP8_E5M2_MAX_BIASED_EXP)
    {
        if(saturate)
        {
            return static_cast<uint8_t>(sign | FP8_E5M2_MAX);
        }
        return static_cast<uint8_t>(sign | FP8_E5M2_POS_INF);
    }

    // Handle inputs in the subnormal/underflow range (may round up to normal)
    if(exp <= 0)
    {
        mant |= 0x00800000; // Add implicit 1
        auto shift = static_cast<uint32_t>(1 - exp + 21); // 23 - 2 = 21 bits to shift
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
            if(mant > FP8_E5M2_MANT_MASK)
            {
                // Rounded up into the smallest normal (exp=1, mant=0)
                return static_cast<uint8_t>(sign | FP8_E5M2_MIN_NORMAL);
            }
        }
        if(mant == 0u)
        {
            return static_cast<uint8_t>(sign); // Rounded down to zero
        }
        return static_cast<uint8_t>(sign | (mant & FP8_E5M2_MANT_MASK));
    }

    // Normal case: shift mantissa from 23 bits to 2 bits with round-to-nearest-even
    uint32_t fp8Mant = (mant >> 21) & FP8_E5M2_MANT_MASK;
    const uint32_t remainder = mant & 0x001FFFFF;

    // Round to nearest even
    if(remainder > FP8_E5M2_ROUND_THRESHOLD
       || (remainder == FP8_E5M2_ROUND_THRESHOLD && ((fp8Mant & 1) != 0)))
    {
        fp8Mant++;
        if(fp8Mant > FP8_E5M2_MANT_MASK)
        {
            fp8Mant = 0;
            exp++;
            if(exp >= FP8_E5M2_MAX_BIASED_EXP)
            {
                if(saturate)
                {
                    return static_cast<uint8_t>(sign | FP8_E5M2_MAX);
                }
                return static_cast<uint8_t>(sign | FP8_E5M2_POS_INF);
            }
        }
    }

    return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 2) | fp8Mant);
}

// Convert FP8 E5M2 bits to float using a half-size lookup table.
// Only positive values (0x00..0x7F) are stored. The function handles Inf at
// 0x7C and NaN at 0x7D..0x7F directly; for bits in 0x80..0xFF the function
// negates the positive entry at (bits & 0x7F), returning negative Inf and
// signed NaN for the corresponding negative patterns.
// NOLINTNEXTLINE(readability-identifier-naming)
inline float fp8_e5m2_bits_to_float(uint8_t bits) noexcept
{
    // clang-format off
    static constexpr std::array<float, 128> TABLE = {
        0.0f,                    1.52587890625e-05f,      3.0517578125e-05f,       4.57763671875e-05f,      // exp=0 (subnormal)
        6.103515625e-05f,        7.62939453125e-05f,      9.1552734375e-05f,       0.0001068115234375f,     // exp=1
        0.0001220703125f,        0.000152587890625f,      0.00018310546875f,       0.000213623046875f,      // exp=2
        0.000244140625f,         0.00030517578125f,       0.0003662109375f,        0.00042724609375f,       // exp=3
        0.00048828125f,          0.0006103515625f,        0.000732421875f,         0.0008544921875f,        // exp=4
        0.0009765625f,           0.001220703125f,         0.00146484375f,          0.001708984375f,         // exp=5
        0.001953125f,            0.00244140625f,          0.0029296875f,           0.00341796875f,          // exp=6
        0.00390625f,             0.0048828125f,           0.005859375f,            0.0068359375f,           // exp=7
        0.0078125f,              0.009765625f,            0.01171875f,             0.013671875f,            // exp=8
        0.015625f,               0.01953125f,             0.0234375f,              0.02734375f,             // exp=9
        0.03125f,                0.0390625f,              0.046875f,               0.0546875f,              // exp=10
        0.0625f,                 0.078125f,               0.09375f,                0.109375f,               // exp=11
        0.125f,                  0.15625f,                0.1875f,                 0.21875f,                // exp=12
        0.25f,                   0.3125f,                 0.375f,                  0.4375f,                 // exp=13
        0.5f,                    0.625f,                  0.75f,                   0.875f,                  // exp=14
        1.0f,                    1.25f,                   1.5f,                    1.75f,                   // exp=15
        2.0f,                    2.5f,                    3.0f,                    3.5f,                    // exp=16
        4.0f,                    5.0f,                    6.0f,                    7.0f,                    // exp=17
        8.0f,                    10.0f,                   12.0f,                   14.0f,                   // exp=18
        16.0f,                   20.0f,                   24.0f,                   28.0f,                   // exp=19
        32.0f,                   40.0f,                   48.0f,                   56.0f,                   // exp=20
        64.0f,                   80.0f,                   96.0f,                   112.0f,                  // exp=21
        128.0f,                  160.0f,                  192.0f,                  224.0f,                  // exp=22
        256.0f,                  320.0f,                  384.0f,                  448.0f,                  // exp=23
        512.0f,                  640.0f,                  768.0f,                  896.0f,                  // exp=24
        1024.0f,                 1280.0f,                 1536.0f,                 1792.0f,                 // exp=25
        2048.0f,                 2560.0f,                 3072.0f,                 3584.0f,                 // exp=26
        4096.0f,                 5120.0f,                 6144.0f,                 7168.0f,                 // exp=27
        8192.0f,                 10240.0f,                12288.0f,                14336.0f,                // exp=28
        16384.0f,                20480.0f,                24576.0f,                28672.0f,                // exp=29
        32768.0f,                40960.0f,                49152.0f,                57344.0f,                // exp=30: max finite
        // exp=31: Inf/NaN
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
    };
    // clang-format on
    const uint8_t absBits = bits & FP8_E5M2_ABS_MASK;
    if(bits <= FP8_E5M2_ABS_MASK)
    {
        // Positive values (including +Inf and positive NaN)
        return TABLE[bits];
    }
    // Negative values
    const float posVal = TABLE[absBits];
    // NaN requires copysign
    if(absBits >= FP8_E5M2_NAN_MIN) // 0x7D, 0x7E, 0x7F are NaN
    {
        return std::copysign(posVal, -1.0f);
    }
    // Finite values and infinity: negation works correctly
    return -posVal;
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
        : fp8_e5m2(static_cast<float>(d))
    {
    }

    // EXPLICIT constructor from integral types
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    explicit fp8_e5m2(T value) noexcept
        : fp8_e5m2(static_cast<float>(value))
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
static_assert(std::is_default_constructible_v<fp8_e5m2>, "fp8_e5m2 must be default constructible");
static_assert(std::is_copy_constructible_v<fp8_e5m2>, "fp8_e5m2 must be copy constructible");
static_assert(std::is_move_constructible_v<fp8_e5m2>, "fp8_e5m2 must be move constructible");

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
    static constexpr bool has_signaling_NaN = false; // OCP E5M2 does not distinguish signaling NaN
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
            hipdnn_data_sdk::types::detail::FP8_E5M2_NAN);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2 signaling_NaN() noexcept
    {
        // OCP E5M2 does not distinguish signaling NaN; return same as quiet_NaN()
        return hipdnn_data_sdk::types::fp8_e5m2::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E5M2_NAN);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2 denorm_min() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2::from_bits(
            hipdnn_data_sdk::types::detail::FP8_E5M2_DENORM_MIN);
    }
};
// NOLINTEND(readability-identifier-naming)
