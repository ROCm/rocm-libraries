// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <roc/host_numerics/scalar_conversion.hpp>
#include <stdexcept>
#include <vector>

namespace roc::host_numerics::detail {
inline float decodeFloat16(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits >> 15);
    const uint32_t exponent = (bits >> 10) & 0x1fU;
    const uint32_t mantissa = bits & 0x3ffU;

    float value;
    if (exponent == 0) {
        value = mantissa == 0 ? 0.0f : std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exponent == 0x1fU) {
        value = mantissa == 0 ? std::numeric_limits<float>::infinity()
                              : std::numeric_limits<float>::quiet_NaN();
    } else {
        value = std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f,
                           static_cast<int>(exponent) - 15);
    }
    return sign ? -value : value;
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
        uint32_t rounded = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1U << shift) - 1U);
        const uint32_t halfway = 1U << (shift - 1U);
        if (remainder > halfway || (remainder == halfway && (rounded & 1U))) ++rounded;
        return static_cast<uint16_t>(sign | rounded);
    }

    uint32_t roundedMantissa = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffU;
    if (remainder > 0x1000U || (remainder == 0x1000U && (roundedMantissa & 1U))) {
        ++roundedMantissa;
        if (roundedMantissa == 0x400U) {
            roundedMantissa = 0;
            ++halfExponent;
            if (halfExponent >= 0x1f) return static_cast<uint16_t>(sign | 0x7c00U);
        }
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
            throw std::invalid_argument(
                "ScalarType is not a supported binary floating-point format.");
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

template <ScalarType Type>
std::vector<float> makePositiveFiniteBinaryFloatValues() {
    constexpr BinaryFloatFormat format = binaryFloatFormatKnown<Type>();
    std::vector<float> values(format.maximumPositiveFiniteRaw + 1U);
    for (uint32_t raw = 0; raw <= format.maximumPositiveFiniteRaw; ++raw)
        values[raw] = decodeFiniteBinaryFloatMagnitude(raw, format);
    return values;
}

inline const std::vector<float>& positiveFiniteBinaryFloatValues(ScalarType type) {
    return visitScalarType(type, []<typename Tag>() -> const std::vector<float>& {
        if constexpr (IsBinaryFloatTypeV<Tag::type>) {
            static const auto values = makePositiveFiniteBinaryFloatValues<Tag::type>();
            return values;
        } else {
            throw std::invalid_argument(
                "ScalarType is not a supported binary floating-point format.");
        }
    });
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

inline uint32_t nearestPositiveBinaryFloatRaw(ScalarType type, float value,
                                              const BinaryFloatFormat& format) {
    const std::vector<float>& values = positiveFiniteBinaryFloatValues(type);
    const auto upperIterator = std::lower_bound(values.begin(), values.end(), value);
    if (upperIterator == values.begin()) return 0;
    if (upperIterator == values.end()) return format.maximumPositiveFiniteRaw;

    const uint32_t upper = static_cast<uint32_t>(std::distance(values.begin(), upperIterator));
    const uint32_t lower = upper - 1U;
    const double lowerValue = static_cast<double>(values[lower]);
    const double upperValue = static_cast<double>(values[upper]);
    const double lowerDistance = static_cast<double>(value) - lowerValue;
    const double upperDistance = upperValue - static_cast<double>(value);
    if (lowerDistance < upperDistance) return lower;
    if (upperDistance < lowerDistance) return upper;
    return (lower & 1U) == 0 ? lower : upper;
}

inline uint32_t encodeBinaryFloat(ScalarType type, float value) {
    const auto format = binaryFloatFormat(type);
    if (std::isnan(value)) {
        if (!scalarTypeInfo(type).supportsNaN) {
            const uint32_t signMask = format.hasSign ? 1U << (format.totalBits - 1U) : 0U;
            const uint32_t sign = std::signbit(value) ? signMask : 0U;
            return sign | format.maximumPositiveFiniteRaw;
        }
        const bool preserveSign = type == ScalarType::Float8E4M3 || type == ScalarType::Float8E5M2;
        const uint32_t sign = preserveSign && std::signbit(value) ? 0x80U : 0U;
        return sign | format.canonicalNaNRaw;
    }

    if (!format.hasSign && value != 0.0f && std::signbit(value))
        throw std::domain_error("Unsigned scale formats cannot encode negative values.");

    const uint32_t signMask = format.hasSign ? 1U << (format.totalBits - 1U) : 0U;
    const uint32_t sign = std::signbit(value) ? signMask : 0U;
    value = std::fabs(value);

    if (std::isinf(value)) {
        if (format.hasInfinity) return sign | format.positiveInfinityRaw;
        return sign | format.maximumPositiveFiniteRaw;
    }
    if (value == 0.0f) return format.hasSignedZero ? sign : 0U;

    const float maximumValue =
        decodeFiniteBinaryFloatMagnitude(format.maximumPositiveFiniteRaw, format);
    if (value >= maximumValue) return sign | format.maximumPositiveFiniteRaw;

    const uint32_t magnitude = nearestPositiveBinaryFloatRaw(type, value, format);
    if (magnitude == 0 && !format.hasSignedZero) return 0;
    return sign | magnitude;
}

inline float decodeE8M0(uint8_t raw) {
    if (raw == 0xffU) return std::numeric_limits<float>::quiet_NaN();
    return std::ldexp(1.0f, static_cast<int>(raw) - 127);
}

inline float decodeE8M0Zero(uint8_t raw) {
    return raw == 0 ? 0.0f : decodeE8M0(raw);
}

inline uint8_t encodeE8M0(float value) {
    if (std::isnan(value)) return 0xffU;
    if (value != 0.0f && std::signbit(value))
        throw std::domain_error("E8M0 cannot encode negative values.");
    if (value <= std::ldexp(1.0f, -127)) return 0;
    if (std::isinf(value) || value >= std::ldexp(1.0f, 127)) return 0xfeU;

    uint32_t lower = 0;
    uint32_t upper = 0xfeU;
    while (lower + 1 < upper) {
        const uint32_t middle = lower + (upper - lower) / 2;
        if (decodeE8M0(static_cast<uint8_t>(middle)) < value)
            lower = middle;
        else
            upper = middle;
    }

    const double lowerDistance =
        static_cast<double>(value) - decodeE8M0(static_cast<uint8_t>(lower));
    const double upperDistance =
        decodeE8M0(static_cast<uint8_t>(upper)) - static_cast<double>(value);
    if (lowerDistance < upperDistance) return static_cast<uint8_t>(lower);
    if (upperDistance < lowerDistance) return static_cast<uint8_t>(upper);
    return static_cast<uint8_t>((lower & 1U) == 0 ? lower : upper);
}

inline uint8_t encodeE8M0Zero(float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    return static_cast<uint8_t>((bits & 0x7fffffffU) >> 23U);
}
}  // namespace roc::host_numerics::detail
