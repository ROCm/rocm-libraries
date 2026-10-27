// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <roc/hostnumerics/scalar_conversion_options.hpp>
#include <roc/hostnumerics/scalar_traits.hpp>
#include <roc/hostnumerics/scalar_type.hpp>
#include <stdexcept>

namespace roc::hostnumerics::detail {
inline uint32_t roundRightShiftToNearestEven(uint32_t value, uint32_t shift) {
    if (shift == 0) return value;
    if (shift >= 32) return 0;
    const uint32_t rounded = value >> shift;
    const uint32_t remainder = value & ((uint32_t{1} << shift) - 1U);
    const uint32_t halfway = uint32_t{1} << (shift - 1U);
    return rounded + static_cast<uint32_t>(remainder > halfway ||
                                           (remainder == halfway && (rounded & 1U) != 0));
}

inline float decodeFloat16(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1fU;
    const uint32_t mantissa = bits & 0x3ffU;

    if (exponent == 0) {
        if (mantissa == 0) return std::bit_cast<float>(sign);
        const uint32_t shift = std::countl_zero(mantissa) - 21U;
        const uint32_t floatExponent = 113U - shift;
        const uint32_t floatMantissa = (mantissa << (13U + shift)) & 0x7fffffU;
        return std::bit_cast<float>(sign | (floatExponent << 23) | floatMantissa);
    }
    if (exponent == 0x1fU)
        return std::bit_cast<float>(sign | (mantissa == 0 ? 0x7f800000U : 0x7fc00000U));
    return std::bit_cast<float>(sign | ((exponent + 112U) << 23) | (mantissa << 13));
}

inline uint16_t encodeFloat16(float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
    const uint32_t exponent = (bits >> 23) & 0xffU;
    uint32_t mantissa = bits & 0x7fffffU;

    if (exponent == 0xffU) {
        if (mantissa == 0) return static_cast<uint16_t>(sign | 0x7c00U);
        return static_cast<uint16_t>(sign | 0x7e00U);
    }

    int32_t halfExponent = static_cast<int32_t>(exponent) - 127 + 15;
    if (halfExponent >= 0x1f) return static_cast<uint16_t>(sign | 0x7c00U);
    if (halfExponent <= 0) {
        if (halfExponent < -10) return sign;
        mantissa |= 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(14 - halfExponent);
        const uint32_t rounded = roundRightShiftToNearestEven(mantissa, shift);
        return static_cast<uint16_t>(sign | rounded);
    }

    uint32_t roundedMantissa = roundRightShiftToNearestEven(mantissa, 13);
    if (roundedMantissa == 0x400U) {
        roundedMantissa = 0;
        ++halfExponent;
        if (halfExponent >= 0x1f) return static_cast<uint16_t>(sign | 0x7c00U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(halfExponent) << 10) |
                                 roundedMantissa);
}

inline float decodeBFloat16(uint16_t bits) {
    return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
}

inline uint16_t encodeBFloat16(float value) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint32_t leastSignificantBit = (bits >> 16) & 1U;
    bits += 0x7fffU + leastSignificantBit;
    return static_cast<uint16_t>(bits >> 16);
}

inline uint16_t encodeBFloat16Truncated(float value) {
    if (std::isnan(value)) return 0xffc1U;
    return static_cast<uint16_t>(std::bit_cast<uint32_t>(value) >> 16U);
}

struct BinaryFloatFormat {
    uint8_t exponentBits;
    uint8_t mantissaBits;
    int16_t exponentBias;
    uint8_t totalBits;
    bool hasSign;
    bool hasSignedZero;
    bool hasInfinity;
    uint32_t maximumPositiveFiniteRaw;
    uint32_t positiveInfinityRaw;
    uint32_t canonicalNaNRaw;
};

template <ScalarType Type>
inline constexpr bool IsBinaryFloatTypeV =
    Type == ScalarType::Float4E2M1 || Type == ScalarType::Float6E2M3 ||
    Type == ScalarType::Float6E3M2 || Type == ScalarType::Float8E4M3 ||
    Type == ScalarType::Float8E5M2 || Type == ScalarType::Float8E4M3Fnuz ||
    Type == ScalarType::Float8E5M2Fnuz || Type == ScalarType::E5M3 || Type == ScalarType::E4M3;

template <ScalarType Type>
inline constexpr BinaryFloatFormat binaryFloatFormatKnown() {
    static_assert(IsBinaryFloatTypeV<Type>);
    if constexpr (Type == ScalarType::Float4E2M1)
        return {2, 1, 1, 4, true, true, false, 0x7, 0, 0};
    else if constexpr (Type == ScalarType::Float6E2M3)
        return {2, 3, 1, 6, true, true, false, 0x1f, 0, 0};
    else if constexpr (Type == ScalarType::Float6E3M2)
        return {3, 2, 3, 6, true, true, false, 0x1f, 0, 0};
    else if constexpr (Type == ScalarType::Float8E4M3)
        return {4, 3, 7, 8, true, true, false, 0x7e, 0, 0x7f};
    else if constexpr (Type == ScalarType::Float8E5M2)
        return {5, 2, 15, 8, true, true, true, 0x7b, 0x7c, 0x7f};
    else if constexpr (Type == ScalarType::Float8E4M3Fnuz)
        return {4, 3, 8, 8, true, false, false, 0x7f, 0, 0x80};
    else if constexpr (Type == ScalarType::Float8E5M2Fnuz)
        return {5, 2, 16, 8, true, false, false, 0x7f, 0, 0x80};
    else if constexpr (Type == ScalarType::E5M3)
        return {5, 3, 15, 8, false, false, false, 0xfe, 0, 0xff};
    else if constexpr (Type == ScalarType::E4M3)
        return {4, 3, 7, 7, false, false, false, 0x7e, 0, 0x7f};
    else
        static_assert(AlwaysFalseV<std::integral_constant<ScalarType, Type>>,
                      "Unhandled binary floating-point ScalarType.");
}

inline BinaryFloatFormat binaryFloatFormat(ScalarType type) {
    return visitScalarType(type, []<typename Tag>() -> BinaryFloatFormat {
        if constexpr (IsBinaryFloatTypeV<Tag::type>)
            return binaryFloatFormatKnown<Tag::type>();
        else
            throw std::runtime_error("ScalarType is not a supported binary floating-point format.");
    });
}

template <ScalarType Type>
inline constexpr bool isBinaryFloatNaNKnown(uint32_t raw) {
    static_assert(IsBinaryFloatTypeV<Type>);
    if constexpr (Type == ScalarType::Float8E4M3 || Type == ScalarType::E4M3)
        return (raw & 0x7fU) == 0x7fU;
    else if constexpr (Type == ScalarType::Float8E5M2)
        return (raw & 0x7fU) > 0x7cU;
    else if constexpr (Type == ScalarType::Float8E4M3Fnuz || Type == ScalarType::Float8E5M2Fnuz)
        return raw == 0x80U;
    else if constexpr (Type == ScalarType::E5M3)
        return raw == 0xffU;
    else if constexpr (Type == ScalarType::Float4E2M1 || Type == ScalarType::Float6E2M3 ||
                       Type == ScalarType::Float6E3M2)
        return false;
    else
        static_assert(AlwaysFalseV<std::integral_constant<ScalarType, Type>>,
                      "Unhandled binary floating-point ScalarType.");
}

inline bool isBinaryFloatNaN(ScalarType type, uint32_t raw) {
    return visitScalarType(type, [raw]<typename Tag>() {
        if constexpr (IsBinaryFloatTypeV<Tag::type>)
            return isBinaryFloatNaNKnown<Tag::type>(raw);
        else
            return false;
    });
}

inline bool isBinaryFloatInfinity(ScalarType type, uint32_t raw) {
    const BinaryFloatFormat format = binaryFloatFormat(type);
    if (!format.hasInfinity) return false;
    const uint32_t signMask = format.hasSign ? 1U << (format.totalBits - 1U) : 0U;
    const uint32_t payloadMask = (1U << format.totalBits) - 1U;
    const uint32_t magnitude = format.hasSign ? raw & (signMask - 1U) : raw & payloadMask;
    return magnitude == format.positiveInfinityRaw;
}

inline float decodeFiniteBinaryFloatMagnitude(uint32_t raw, const BinaryFloatFormat& format) {
    const uint32_t exponentMask = (1U << format.exponentBits) - 1U;
    const uint32_t mantissaMask = (1U << format.mantissaBits) - 1U;
    const uint32_t exponent = (raw >> format.mantissaBits) & exponentMask;
    const uint32_t mantissa = raw & mantissaMask;
    const float mantissaScale = 1.0f / static_cast<float>(1U << format.mantissaBits);

    if (exponent == 0)
        return std::ldexp(static_cast<float>(mantissa) * mantissaScale, 1 - format.exponentBias);
    return std::ldexp(1.0f + static_cast<float>(mantissa) * mantissaScale,
                      static_cast<int>(exponent) - format.exponentBias);
}

inline float decodeBinaryFloat(ScalarType type, uint32_t raw) {
    const auto format = binaryFloatFormat(type);
    if (isBinaryFloatNaN(type, raw)) return std::numeric_limits<float>::quiet_NaN();

    const uint32_t signMask = format.hasSign ? 1U << (format.totalBits - 1U) : 0U;
    const bool negative = format.hasSign && (raw & signMask) != 0;
    if (isBinaryFloatInfinity(type, raw))
        return negative ? -std::numeric_limits<float>::infinity()
                        : std::numeric_limits<float>::infinity();

    const uint32_t payloadMask = (1U << format.totalBits) - 1U;
    const uint32_t magnitude = format.hasSign ? raw & (signMask - 1U) : raw & payloadMask;
    const float value = decodeFiniteBinaryFloatMagnitude(magnitude, format);
    return negative ? -value : value;
}

template <ScalarType Type>
inline float decodeBinaryFloatKnown(uint32_t raw) {
    static const auto values = [] {
        constexpr BinaryFloatFormat format = binaryFloatFormatKnown<Type>();
        std::array<float, size_t{1} << format.totalBits> result{};
        for (uint32_t value = 0; value < result.size(); ++value)
            result[value] = decodeBinaryFloat(Type, value);
        return result;
    }();
    return values[raw & (values.size() - 1U)];
}

template <ScalarType Type>
inline uint32_t encodeBinaryFloatKnown(float value) {
    constexpr BinaryFloatFormat format = binaryFloatFormatKnown<Type>();
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint32_t sourceExponent = (bits >> 23) & 0xffU;
    const uint32_t sourceMantissa = bits & 0x7fffffU;
    const bool negative = (bits >> 31) != 0;

    if (sourceExponent == 0xffU && sourceMantissa != 0) {
        if (!scalarTypeInfo(Type).supportsNaN) {
            const uint32_t signMask = format.hasSign ? 1U << (format.totalBits - 1U) : 0U;
            const uint32_t sign = negative ? signMask : 0U;
            return sign | format.maximumPositiveFiniteRaw;
        }
        constexpr bool preserveSign =
            Type == ScalarType::Float8E4M3 || Type == ScalarType::Float8E5M2;
        const uint32_t sign = preserveSign && negative ? 0x80U : 0U;
        return sign | format.canonicalNaNRaw;
    }

    const uint32_t magnitudeBits = bits & 0x7fffffffU;
    if (!format.hasSign && magnitudeBits != 0 && negative)
        throw std::runtime_error("Unsigned scale formats cannot encode negative values.");

    const uint32_t signMask = format.hasSign ? 1U << (format.totalBits - 1U) : 0U;
    const uint32_t sign = negative ? signMask : 0U;
    if (sourceExponent == 0xffU) {
        if (format.hasInfinity) return sign | format.positiveInfinityRaw;
        return sign | format.maximumPositiveFiniteRaw;
    }
    if (magnitudeBits == 0) return format.hasSignedZero ? sign : 0U;
    if (sourceExponent == 0) return format.hasSignedZero ? sign : 0U;

    // Destination values are exact floats, so align the source significand to the
    // destination precision and round once instead of searching a value table.
    int exponent = static_cast<int>(sourceExponent) - 127;
    constexpr int minimumNormalExponent = 1 - format.exponentBias;
    const uint32_t significand = 0x800000U | sourceMantissa;
    uint32_t magnitude;
    if (exponent < minimumNormalExponent) {
        const int shift = 23 + minimumNormalExponent - format.mantissaBits - exponent;
        magnitude = shift >= 32
                        ? 0U
                        : roundRightShiftToNearestEven(significand, static_cast<uint32_t>(shift));
    } else {
        constexpr uint32_t shift = 23 - format.mantissaBits;
        uint32_t rounded = roundRightShiftToNearestEven(significand, shift);
        if (rounded == (uint32_t{1} << (format.mantissaBits + 1U))) {
            rounded >>= 1;
            ++exponent;
        }
        const int targetExponent = exponent + format.exponentBias;
        magnitude = targetExponent <= 0
                        ? 0U
                        : (static_cast<uint32_t>(targetExponent) << format.mantissaBits) |
                              (rounded & ((uint32_t{1} << format.mantissaBits) - 1U));
        magnitude = std::min(magnitude, format.maximumPositiveFiniteRaw);
    }
    if (magnitude == 0 && !format.hasSignedZero) return 0;
    return sign | magnitude;
}

inline uint32_t encodeBinaryFloat(ScalarType type, float value) {
    return visitScalarType(type, [value]<typename Tag>() -> uint32_t {
        if constexpr (IsBinaryFloatTypeV<Tag::type>)
            return encodeBinaryFloatKnown<Tag::type>(value);
        else
            throw std::runtime_error("ScalarType is not a supported binary floating-point format.");
    });
}

inline float decodeE8M0(uint8_t raw) {
    if (raw == 0xffU) return std::numeric_limits<float>::quiet_NaN();
    return std::ldexp(1.0f, static_cast<int>(raw) - 127);
}

inline float decodeE8M0Zero(uint8_t raw) {
    return raw == 0 ? 0.0f : decodeE8M0(raw);
}

inline uint8_t encodeE8M0(float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint32_t exponent = (bits >> 23) & 0xffU;
    const uint32_t mantissa = bits & 0x7fffffU;
    const uint32_t magnitude = bits & 0x7fffffffU;
    const bool negative = (bits >> 31) != 0;

    if (exponent == 0xffU && mantissa != 0) return 0xffU;
    if (magnitude != 0 && negative) throw std::runtime_error("E8M0 cannot encode negative values.");
    if (exponent == 0xffU || exponent >= 0xfeU) return 0xfeU;
    if (magnitude <= 0x00400000U) return 0;
    if (exponent == 0) return static_cast<uint8_t>(magnitude > 0x00600000U);

    // The midpoint between adjacent powers of two has only the top mantissa bit set.
    const bool roundUp = mantissa > 0x400000U || (mantissa == 0x400000U && (exponent & 1U) != 0);
    return static_cast<uint8_t>(exponent + static_cast<uint32_t>(roundUp));
}

inline uint8_t encodeE8M0Zero(float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    return static_cast<uint8_t>((bits & 0x7fffffffU) >> 23U);
}

inline uint64_t roundRightShiftToNearestEven(uint64_t value, uint32_t shift) {
    if (shift == 0) return value;
    if (shift >= 64) return 0;
    const uint64_t rounded = value >> shift;
    const uint64_t remainder = value & ((uint64_t{1} << shift) - 1U);
    const uint64_t halfway = uint64_t{1} << (shift - 1U);
    return rounded + static_cast<uint64_t>(remainder > halfway ||
                                           (remainder == halfway && (rounded & 1U) != 0));
}

// Keep the double significand until the destination rounding step.
inline uint32_t encodeBinaryFloatDirect(double value, const BinaryFloatFormat& format,
                                        bool saturate, bool truncate = false) {
    const uint64_t bits = std::bit_cast<uint64_t>(value);
    const uint32_t sourceExponent = static_cast<uint32_t>((bits >> 52) & 0x7ffU);
    const uint64_t sourceMantissa = bits & 0xfffffffffffffULL;
    const bool negative = (bits >> 63) != 0;
    const uint32_t sign = format.hasSign && negative ? 1U << (format.totalBits - 1U) : 0U;
    if (!format.hasSign && negative && value < 0.0)
        throw std::runtime_error("Unsigned scale formats cannot encode negative values.");
    if (sourceExponent == 0x7ffU) {
        if (sourceMantissa != 0) {
            if (format.canonicalNaNRaw == 0) return sign | format.maximumPositiveFiniteRaw;
            return format.canonicalNaNRaw == 0x80U ? 0x80U : sign | format.canonicalNaNRaw;
        }
        return sign | (format.hasInfinity && !saturate ? format.positiveInfinityRaw
                                                       : format.maximumPositiveFiniteRaw);
    }
    if (sourceExponent == 0) return format.hasSignedZero ? sign : 0U;
    int exponent = static_cast<int>(sourceExponent) - 1023;
    const int minimumExponent = 1 - format.exponentBias;
    const uint64_t significand = (uint64_t{1} << 52) | sourceMantissa;
    const int shift = 52 - format.mantissaBits + std::max(0, minimumExponent - exponent);
    uint64_t rounded =
        truncate ? (shift >= 64 ? 0 : significand >> shift)
                 : roundRightShiftToNearestEven(significand, static_cast<uint32_t>(shift));
    uint32_t magnitude;
    if (exponent < minimumExponent) {
        magnitude = static_cast<uint32_t>(rounded);
    } else {
        if (rounded == (uint64_t{1} << (format.mantissaBits + 1U))) {
            rounded >>= 1;
            ++exponent;
        }
        magnitude = (static_cast<uint32_t>(exponent + format.exponentBias) << format.mantissaBits) |
                    static_cast<uint32_t>(rounded & ((1U << format.mantissaBits) - 1U));
    }
    if (magnitude > format.maximumPositiveFiniteRaw)
        magnitude = format.hasInfinity && !saturate ? format.positiveInfinityRaw
                                                    : format.maximumPositiveFiniteRaw;
    if (magnitude == 0 && !format.hasSignedZero) return 0;
    return sign | magnitude;
}

inline uint16_t encodeFloat16(double value, FloatingOverflow overflow) {
    constexpr BinaryFloatFormat format{5, 10, 15, 16, true, true, true, 0x7bff, 0x7c00, 0x7e00};
    return static_cast<uint16_t>(
        encodeBinaryFloatDirect(value, format, overflow == FloatingOverflow::SaturateFinite));
}

inline uint16_t encodeBFloat16(double value, BFloat16Rounding rounding, FloatingOverflow overflow) {
    if (rounding == BFloat16Rounding::Truncate && std::isnan(value)) return 0xffc1U;
    constexpr BinaryFloatFormat format{8, 7, 127, 16, true, true, true, 0x7f7f, 0x7f80, 0x7fc0};
    return static_cast<uint16_t>(
        encodeBinaryFloatDirect(value, format, overflow == FloatingOverflow::SaturateFinite,
                                rounding == BFloat16Rounding::Truncate));
}

template <ScalarType Type>
inline uint32_t encodeBinaryFloatKnown(double value, FloatingOverflow overflow) {
    // Narrow formats saturate finite overflow even when they encode infinity.
    return encodeBinaryFloatDirect(
        value, binaryFloatFormatKnown<Type>(),
        std::isfinite(value) || overflow == FloatingOverflow::SaturateFinite);
}

inline uint8_t encodeE8M0(double value, E8M0Underflow underflow) {
    if (std::isnan(value)) return 0xffU;
    if (value < 0.0) throw std::runtime_error("E8M0 cannot encode negative values.");
    if (std::isinf(value)) return 0xfeU;
    if (value < std::ldexp(1.0, underflow == E8M0Underflow::FlushBelowFloat32Normal ? -126 : -127))
        return 0;
    int exponent;
    const double significand = 2.0 * std::frexp(value, &exponent);
    int biased = exponent - 1 + 127;
    if (significand > 1.5 || (significand == 1.5 && (biased & 1) != 0)) ++biased;
    return static_cast<uint8_t>(std::clamp(biased, 0, 254));
}

inline uint8_t encodeE8M0Zero(double value) {
    const double magnitude = std::abs(value);
    if (!std::isfinite(magnitude)) return 0xffU;
    if (magnitude < std::ldexp(1.0, -126)) return 0;
    int exponent;
    (void)std::frexp(magnitude, &exponent);
    return static_cast<uint8_t>(std::clamp(exponent - 1 + 127, 0, 255));
}
}  // namespace roc::hostnumerics::detail
