// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <roc/hostnumerics/scalar_type.hpp>

namespace roc::hostnumerics::detail {
// FP16, BF16 and narrower formats with an explicit mantissa field (the stored
// fraction bits, excluding the implicit leading bit). For finite codes, a zero
// exponent field encodes zero or a subnormal; a nonzero exponent implies a
// leading one. Every finite value fits exactly in float. E8M0 scales
// have no mantissa field and are handled separately.
constexpr bool isSmallFloat(ScalarType type) {
    const ScalarTypeInfo& info = scalarTypeInfo(type);
    return info.category == ScalarCategory::FloatingPoint && info.mantissaBits > 0 &&
           info.storageBits <= 16;
}

enum class NonFiniteCodes : uint8_t {
    // Every code is finite (FP4 and FP6).
    None,
    // The all-ones exponent encodes infinity with a zero mantissa, otherwise NaN.
    Ieee,
    // Only the all-ones magnitude is NaN, and there is no infinity.
    AllOnesNaN,
    // FNUZ: the negative-zero code is the only NaN.
    NegativeZeroNaN,
};

// Masks and special-value codes used by the codec. ScalarTypeInfo supplies both
// the bit layout and numerical properties (bias, NaN and infinity support);
// smallFloatFormat derives these fields from it. NonFiniteCodes additionally
// distinguishes where each format stores its NaN and infinity codes.
struct SmallFloatFormat {
    uint32_t mantissaBits;
    int exponentBias;
    NonFiniteCodes nonFinite;
    // Zero for unsigned formats.
    uint32_t signMask;
    // Masks the exponent and mantissa bits, excluding any sign bit or unused storage bit.
    uint32_t magnitudeMask;
    uint32_t infinityMagnitude;
    uint32_t maximumFiniteMagnitude;
    // The NaN code that encoding produces, before any sign bit is added.
    uint32_t nanMagnitude;
};

template <ScalarType Type>
constexpr SmallFloatFormat smallFloatFormat() {
    static_assert(isSmallFloat(Type));
    constexpr ScalarTypeInfo info = scalarTypeInfo(Type);
    // E4M3 stores seven value bits in a byte, so the bit count can differ from storageBits.
    constexpr uint32_t magnitudeBits = info.exponentBits + info.mantissaBits;
    constexpr uint32_t magnitudeMask = (uint32_t{1} << magnitudeBits) - 1;
    constexpr uint32_t infinityMagnitude =
        magnitudeMask & ~((uint32_t{1} << info.mantissaBits) - 1);
    constexpr bool isFnuz =
        Type == ScalarType::Float8E4M3Fnuz || Type == ScalarType::Float8E5M2Fnuz;
    constexpr NonFiniteCodes nonFinite = info.supportsInfinity ? NonFiniteCodes::Ieee
                                         : !info.supportsNaN   ? NonFiniteCodes::None
                                         : isFnuz              ? NonFiniteCodes::NegativeZeroNaN
                                                               : NonFiniteCodes::AllOnesNaN;
    SmallFloatFormat format{};
    format.mantissaBits = info.mantissaBits;
    format.exponentBias = info.exponentBias;
    format.nonFinite = nonFinite;
    format.signMask = info.isSigned ? uint32_t{1} << magnitudeBits : 0;
    format.magnitudeMask = magnitudeMask;
    format.infinityMagnitude = infinityMagnitude;
    format.maximumFiniteMagnitude = nonFinite == NonFiniteCodes::Ieee ? infinityMagnitude - 1
                                    : nonFinite == NonFiniteCodes::AllOnesNaN ? magnitudeMask - 1
                                                                              : magnitudeMask;
    // FP16 and BF16 produce the quiet NaN with only the top mantissa bit set;
    // 8-bit formats produce the all-ones magnitude.
    format.nanMagnitude = info.storageBits == 16
                              ? infinityMagnitude | (uint32_t{1} << (info.mantissaBits - 1))
                              : magnitudeMask;
    return format;
}

// Returns 2^exponent, requiring exponent in [-149, 127]. Binary32 normal values
// have unbiased exponents -126..127 (stored as 1..254 with bias 127). Below
// -126, its 23 stored mantissa bits encode multiples of 2^(-126 - 23) = 2^-149;
// a single set mantissa bit therefore represents each remaining power of two.
constexpr float powerOfTwo(int exponent) {
    assert(exponent >= -149 && exponent <= 127);
    if (exponent >= -126) return std::bit_cast<float>(static_cast<uint32_t>(exponent + 127) << 23);
    return std::bit_cast<float>(uint32_t{1} << (exponent + 149));
}

// Interprets the encoded sign, exponent and mantissa and returns the numerical
// value as a native float. Finite values convert exactly: normal values are
// rebased to the binary32 exponent bias, while subnormals are multiples of the
// format's smallest positive value. Special codes become float NaN or infinity.
// Unused high bits of the code are ignored.
template <ScalarType Type>
float decodeSmallFloat(uint32_t code) noexcept {
    constexpr SmallFloatFormat format = smallFloatFormat<Type>();
    const bool negative = (code & format.signMask) != 0;
    const uint32_t magnitude = code & format.magnitudeMask;
    if constexpr (format.nonFinite == NonFiniteCodes::Ieee) {
        if (magnitude > format.infinityMagnitude) return std::numeric_limits<float>::quiet_NaN();
        if (magnitude == format.infinityMagnitude)
            return negative ? -std::numeric_limits<float>::infinity()
                            : std::numeric_limits<float>::infinity();
    } else if constexpr (format.nonFinite == NonFiniteCodes::AllOnesNaN) {
        if (magnitude == format.magnitudeMask) return std::numeric_limits<float>::quiet_NaN();
    } else if constexpr (format.nonFinite == NonFiniteCodes::NegativeZeroNaN) {
        if (negative && magnitude == 0) return std::numeric_limits<float>::quiet_NaN();
    }

    const uint32_t exponent = magnitude >> format.mantissaBits;
    const uint32_t mantissa = magnitude & ((uint32_t{1} << format.mantissaBits) - 1);
    float value;
    if (exponent == 0) {
        // The product is exact: mantissa has at most mantissaBits bits.
        constexpr float subnormalUnit =
            powerOfTwo(1 - format.exponentBias - static_cast<int>(format.mantissaBits));
        value = static_cast<float>(mantissa) * subnormalUnit;
    } else {
        const uint32_t floatExponent =
            static_cast<uint32_t>(static_cast<int>(exponent) - format.exponentBias + 127);
        value =
            std::bit_cast<float>((floatExponent << 23) | (mantissa << (23 - format.mantissaBits)));
    }
    return negative ? -value : value;
}

inline uint64_t roundRightShiftToNearestEven(uint64_t value, uint32_t shift) {
    if (shift == 0) return value;
    if (shift >= 64) return 0;
    const uint64_t rounded = value >> shift;
    const uint64_t remainder = value & ((uint64_t{1} << shift) - 1);
    const uint64_t halfway = uint64_t{1} << (shift - 1);
    return rounded + static_cast<uint64_t>(remainder > halfway ||
                                           (remainder == halfway && (rounded & 1) != 0));
}

// Rounds value to the nearest code with ties to even, or toward zero when
// truncate is set. Finite overflow becomes infinity for FP16 and BF16 and the
// largest finite value for narrower formats; infinity stays infinite when the
// format can encode it. Saturate maps all overflow and infinities to the
// largest finite value. A NaN encodes as NaN or, for formats without NaN, as
// the largest finite value. Underflow keeps the sign when the format has a
// negative zero. Unsigned formats require a value that is not negative.
template <ScalarType Type>
uint32_t encodeSmallFloat(double value, bool saturate, bool truncate) noexcept {
    constexpr SmallFloatFormat format = smallFloatFormat<Type>();
    constexpr bool hasSignedZero =
        format.signMask != 0 && format.nonFinite != NonFiniteCodes::NegativeZeroNaN;
    const uint64_t bits = std::bit_cast<uint64_t>(value);
    const bool negative = (bits >> 63) != 0;
    const uint32_t sign = negative ? format.signMask : 0;

    if (std::isnan(value)) {
        if constexpr (format.nonFinite == NonFiniteCodes::None)
            return sign | format.maximumFiniteMagnitude;
        else if constexpr (format.nonFinite == NonFiniteCodes::NegativeZeroNaN)
            return format.signMask;
        else
            return sign | format.nanMagnitude;
    }
    assert(format.signMask != 0 || !(value < 0));

    constexpr bool hasInfinity = format.nonFinite == NonFiniteCodes::Ieee;
    constexpr bool finiteOverflowToInfinity = hasInfinity && scalarTypeInfo(Type).storageBits == 16;
    const bool infinite = std::isinf(value);
    const uint32_t overflowMagnitude =
        hasInfinity && !saturate && (infinite || finiteOverflowToInfinity)
            ? format.infinityMagnitude
            : format.maximumFiniteMagnitude;
    if (infinite) return sign | overflowMagnitude;

    const auto sourceExponent = static_cast<int>((bits >> 52) & 0x7ff);
    // Binary64 zeros and subnormals are far below every format's smallest subnormal.
    if (sourceExponent == 0) return hasSignedZero ? sign : 0;

    int exponent = sourceExponent - 1023;
    const uint64_t significand = (uint64_t{1} << 52) | (bits & ((uint64_t{1} << 52) - 1));
    const int minimumExponent = 1 - format.exponentBias;
    const auto shift = static_cast<uint32_t>(52 - static_cast<int>(format.mantissaBits) +
                                             std::max(0, minimumExponent - exponent));
    uint64_t rounded = truncate ? (shift >= 64 ? 0 : significand >> shift)
                                : roundRightShiftToNearestEven(significand, shift);
    uint64_t magnitude;
    if (exponent < minimumExponent) {
        // A subnormal that rounds up to the smallest normal value produces its code.
        magnitude = rounded;
    } else {
        if (rounded == uint64_t{1} << (format.mantissaBits + 1)) {
            rounded >>= 1;
            ++exponent;
        }
        magnitude = (static_cast<uint64_t>(exponent + format.exponentBias) << format.mantissaBits) |
                    (rounded & ((uint64_t{1} << format.mantissaBits) - 1));
    }
    if (magnitude > format.maximumFiniteMagnitude) return sign | overflowMagnitude;
    if (magnitude == 0 && !hasSignedZero) return 0;
    return sign | static_cast<uint32_t>(magnitude);
}

// E8M0 code c encodes 2^(c - 127); 0xff is NaN.
inline float decodeE8M0(uint8_t code) noexcept {
    if (code == 0xff) return std::numeric_limits<float>::quiet_NaN();
    return powerOfTwo(static_cast<int>(code) - 127);
}

// As E8M0, except that code 0 encodes zero.
inline float decodeE8M0Zero(uint8_t code) noexcept {
    return code == 0 ? 0.0f : decodeE8M0(code);
}

// Rounds to the nearest power of two, ties choosing the even code. Values below
// the smallest code, 2^-127, or below 2^-126 when flushBelowFloat32Normal is set,
// encode as code 0. Infinity and large values encode as 2^127; NaN encodes as NaN.
// The value must not be negative.
inline uint8_t encodeE8M0(double value, bool flushBelowFloat32Normal) noexcept {
    if (std::isnan(value)) return 0xff;
    assert(!(value < 0));
    if (std::isinf(value)) return 0xfe;
    if (value < std::ldexp(1.0, flushBelowFloat32Normal ? -126 : -127)) return 0;
    int exponent = 0;
    const double significand = 2.0 * std::frexp(value, &exponent);
    int code = exponent - 1 + 127;
    if (significand > 1.5 || (significand == 1.5 && (code & 1) != 0)) ++code;
    return static_cast<uint8_t>(std::clamp(code, 0, 0xfe));
}

// Encodes the binary32 exponent field of |value|, ignoring the sign and
// truncating the mantissa: 1.75 and -1.0 both encode as 1.0. Values below 2^-126
// encode as zero; infinity, NaN and values from 2^128 encode as NaN.
inline uint8_t encodeE8M0Zero(double value) noexcept {
    const double magnitude = std::abs(value);
    if (!std::isfinite(magnitude)) return 0xff;
    if (magnitude < std::ldexp(1.0, -126)) return 0;
    int exponent = 0;
    (void)std::frexp(magnitude, &exponent);
    return static_cast<uint8_t>(std::clamp(exponent - 1 + 127, 0, 0xff));
}
}  // namespace roc::hostnumerics::detail
