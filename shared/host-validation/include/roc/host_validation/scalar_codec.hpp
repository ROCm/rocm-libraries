// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Template definitions for scalar conversion and encoded storage. Public code includes scalar.hpp,
// which includes this file after declaring the scalar API.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <roc/host_validation/scalar.hpp>
#include <vector>

namespace roc::host_validation::detail {
template <typename T>
struct RuntimeIsComplex : std::false_type {};

template <typename T>
struct RuntimeIsComplex<std::complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool RuntimeIsComplexV = RuntimeIsComplex<T>::value;

template <typename>
inline constexpr bool AlwaysFalseV = false;

inline constexpr uint64_t integerMask(uint32_t bits) {
    return bits == 64 ? std::numeric_limits<uint64_t>::max() : (uint64_t{1} << bits) - 1;
}

inline constexpr uint64_t signedIntegerMinimumRaw(uint32_t bits) {
    return uint64_t{1} << (bits - 1);
}

inline constexpr uint64_t integerMaximumRaw(uint32_t bits, bool signedDestination) {
    return signedDestination ? signedIntegerMinimumRaw(bits) - 1 : integerMask(bits);
}

// No-options conversion to a native C++ integer truncates fractional values
// and defines out-of-range conversion by reducing modulo the destination width.
inline constexpr ScalarConversionOptions implicitNativeConversionOptions() {
    return {IntegerRounding::TowardZero, IntegerOverflow::ModuloWrap};
}

// No-options storage conversion follows the native policy for byte-addressable
// integers. Synthetic packed integers saturate because they have no native C++
// cast whose behavior can define the conversion.
inline constexpr ScalarConversionOptions implicitStorageConversionOptions(ScalarType destination) {
    const IntegerOverflow overflow =
        destination == ScalarType::Int4 || destination == ScalarType::Int12
            ? IntegerOverflow::Saturate
            : IntegerOverflow::ModuloWrap;
    return {IntegerRounding::TowardZero, overflow};
}

template <typename Floating>
Floating roundForIntegerConversion(Floating value, IntegerRounding rounding) {
    static_assert(std::is_floating_point_v<Floating>);
    switch (rounding) {
        case IntegerRounding::TowardZero:
            return std::trunc(value);
        case IntegerRounding::NearestEven: {
            Floating integral = 0;
            const Floating fraction = std::modf(value, &integral);
            const Floating magnitude = std::fabs(fraction);
            if (magnitude < Floating{0.5}) return integral;

            const Floating direction = std::signbit(fraction) ? Floating{-1} : Floating{1};
            if (magnitude > Floating{0.5}) return integral + direction;

            const Floating parity = std::fmod(std::fabs(integral), Floating{2});
            return parity == Floating{0} ? integral : integral + direction;
        }
    }
    throw std::invalid_argument("Invalid integer rounding policy.");
}

template <typename Integral>
bool integralFitsIntegerDestination(Integral value, uint32_t bits, bool signedDestination) {
    static_assert(std::is_integral_v<Integral>);
    if (signedDestination) {
        if (bits == 64) return std::in_range<int64_t>(value);
        const int64_t minimum = -(int64_t{1} << (bits - 1));
        const int64_t maximum = (int64_t{1} << (bits - 1)) - 1;
        return !std::cmp_less(value, minimum) && !std::cmp_greater(value, maximum);
    }

    if (std::cmp_less(value, 0)) return false;
    if (bits == 64) return std::in_range<uint64_t>(value);
    return !std::cmp_greater(value, integerMask(bits));
}

template <typename Integral>
bool integralIsBelowIntegerDestination(Integral value, uint32_t bits, bool signedDestination) {
    static_assert(std::is_integral_v<Integral>);
    if (!signedDestination) return std::cmp_less(value, 0);
    if (bits == 64) return std::cmp_less(value, std::numeric_limits<int64_t>::min());
    const int64_t minimum = -(int64_t{1} << (bits - 1));
    return std::cmp_less(value, minimum);
}

template <typename Integral>
uint64_t convertIntegralToIntegerBits(Integral value, uint32_t bits, bool signedDestination,
                                      const ScalarConversionOptions& options) {
    static_assert(std::is_integral_v<Integral>);
    const uint64_t mask = integerMask(bits);
    if (integralFitsIntegerDestination(value, bits, signedDestination)) {
        if (signedDestination) return static_cast<uint64_t>(static_cast<int64_t>(value)) & mask;
        return static_cast<uint64_t>(value) & mask;
    }

    switch (options.integerOverflow) {
        case IntegerOverflow::Reject:
            throw std::overflow_error("Integer value is outside the destination range.");
        case IntegerOverflow::Saturate:
            return integralIsBelowIntegerDestination(value, bits, signedDestination)
                       ? (signedDestination ? signedIntegerMinimumRaw(bits) : 0)
                       : integerMaximumRaw(bits, signedDestination);
        case IntegerOverflow::ModuloWrap:
            return static_cast<uint64_t>(value) & mask;
    }
    throw std::invalid_argument("Invalid integer overflow policy.");
}

template <typename Floating>
uint64_t convertFloatingToIntegerBits(Floating value, uint32_t bits, bool signedDestination,
                                      const ScalarConversionOptions& options) {
    static_assert(std::is_floating_point_v<Floating>);
    if (std::isnan(value))
        throw std::domain_error("NaN cannot be converted to an integer destination.");

    if (std::isinf(value)) {
        if (options.integerOverflow == IntegerOverflow::Saturate) {
            if (std::signbit(value)) return signedDestination ? signedIntegerMinimumRaw(bits) : 0;
            return integerMaximumRaw(bits, signedDestination);
        }
        throw std::overflow_error("Infinity cannot be converted to an integer destination.");
    }

    const Floating rounded = roundForIntegerConversion(value, options.integerRounding);
    const int rangeExponent =
        signedDestination ? static_cast<int>(bits) - 1 : static_cast<int>(bits);
    const Floating upperExclusive = std::ldexp(Floating{1}, rangeExponent);
    const Floating lowerInclusive = signedDestination ? -upperExclusive : Floating{0};
    if (rounded >= lowerInclusive && rounded < upperExclusive) {
        if (signedDestination)
            return static_cast<uint64_t>(static_cast<int64_t>(rounded)) & integerMask(bits);
        return static_cast<uint64_t>(rounded) & integerMask(bits);
    }

    switch (options.integerOverflow) {
        case IntegerOverflow::Reject:
            throw std::overflow_error("Rounded value is outside the destination integer range.");
        case IntegerOverflow::Saturate:
            return rounded < lowerInclusive
                       ? (signedDestination ? signedIntegerMinimumRaw(bits) : 0)
                       : integerMaximumRaw(bits, signedDestination);
        case IntegerOverflow::ModuloWrap: {
            const Floating modulus = std::ldexp(Floating{1}, static_cast<int>(bits));
            const Floating remainder = std::fmod(std::fabs(rounded), modulus);
            uint64_t raw = static_cast<uint64_t>(remainder);
            if (std::signbit(rounded)) raw = uint64_t{0} - raw;
            return raw & integerMask(bits);
        }
    }
    throw std::invalid_argument("Invalid integer overflow policy.");
}

template <typename Source>
uint64_t convertToIntegerBits(Source source, uint32_t bits, bool signedDestination,
                              const ScalarConversionOptions& options) {
    using Value = std::remove_cvref_t<Source>;
    if constexpr (RuntimeIsComplexV<Value>) {
        if (source.imag() != typename Value::value_type{0})
            throw std::domain_error(
                "A complex value with a nonzero imaginary component cannot be converted to real.");
        return convertToIntegerBits(source.real(), bits, signedDestination, options);
    } else if constexpr (std::is_same_v<Value, bool>) {
        return convertIntegralToIntegerBits(static_cast<uint8_t>(source), bits, signedDestination,
                                            options);
    } else if constexpr (std::is_enum_v<Value>) {
        return convertToIntegerBits(static_cast<std::underlying_type_t<Value>>(source), bits,
                                    signedDestination, options);
    } else if constexpr (std::is_integral_v<Value>) {
        return convertIntegralToIntegerBits(source, bits, signedDestination, options);
    } else if constexpr (std::is_floating_point_v<Value>) {
        return convertFloatingToIntegerBits(source, bits, signedDestination, options);
    } else {
        static_assert(AlwaysFalseV<Value>, "Scalar integer conversion requires a numeric source.");
    }
}

template <typename Target>
Target integerFromBits(uint64_t raw) {
    using Value = std::remove_cv_t<Target>;
    static_assert(std::is_integral_v<Value> && !std::is_same_v<Value, bool>);
    using Unsigned = std::make_unsigned_t<Value>;
    constexpr uint32_t bits = std::numeric_limits<Unsigned>::digits;
    static_assert(bits > 0 && bits <= 64);
    raw &= integerMask(bits);

    if constexpr (std::is_unsigned_v<Value>) {
        return static_cast<Value>(raw);
    } else {
        const uint64_t sign = signedIntegerMinimumRaw(bits);
        if ((raw & sign) == 0) return static_cast<Value>(raw);

        const uint64_t magnitude = (uint64_t{0} - raw) & integerMask(bits);
        if (magnitude == sign) return std::numeric_limits<Value>::min();
        return static_cast<Value>(-static_cast<int64_t>(magnitude));
    }
}

template <typename Target, typename Source>
Target convertToInteger(Source source, const ScalarConversionOptions& options) {
    using Value = std::remove_cv_t<Target>;
    static_assert(std::is_integral_v<Value> && !std::is_same_v<Value, bool>);
    using Unsigned = std::make_unsigned_t<Value>;
    constexpr uint32_t bits = std::numeric_limits<Unsigned>::digits;
    static_assert(bits > 0 && bits <= 64);
    const uint64_t raw =
        convertToIntegerBits(std::move(source), bits, std::is_signed_v<Value>, options);
    return integerFromBits<Value>(raw);
}

template <typename Target, typename Source>
Target convertScalarValue(Source source, const ScalarConversionOptions& options) {
    using Result = std::remove_cv_t<Target>;
    using Value = std::remove_cvref_t<Source>;
    static_assert(!std::is_reference_v<Target>);

    if constexpr (RuntimeIsComplexV<Result>) {
        using Component = typename Result::value_type;
        if constexpr (RuntimeIsComplexV<Value>)
            return Result(convertScalarValue<Component>(source.real(), options),
                          convertScalarValue<Component>(source.imag(), options));
        else
            return Result(convertScalarValue<Component>(source, options), Component{0});
    } else if constexpr (RuntimeIsComplexV<Value>) {
        if (source.imag() != typename Value::value_type{0})
            throw std::domain_error(
                "A complex value with a nonzero imaginary component cannot be converted to real.");
        return convertScalarValue<Result>(source.real(), options);
    } else if constexpr (std::is_same_v<Result, bool>) {
        return source != Value{0};
    } else if constexpr (std::is_integral_v<Result>) {
        return convertToInteger<Result>(std::move(source), options);
    } else {
        return static_cast<Result>(source);
    }
}

inline uint32_t readPackedBits(std::span<const std::byte> storage, uint64_t bitOffset,
                               uint32_t bitCount) {
    if (bitCount == 0 || bitCount > 32)
        throw std::invalid_argument("Packed scalar bit count must be in [1, 32].");

    uint32_t result = 0;
    for (uint32_t bit = 0; bit < bitCount; ++bit) {
        const uint64_t absoluteBit = bitOffset + bit;
        const size_t byteIndex = static_cast<size_t>(absoluteBit / 8);
        if (byteIndex >= storage.size())
            throw std::out_of_range("Packed scalar read exceeds tensor storage.");
        const uint32_t sourceBit = static_cast<uint32_t>(absoluteBit % 8);
        const uint32_t value = (std::to_integer<uint8_t>(storage[byteIndex]) >> sourceBit) & 1U;
        result |= value << bit;
    }
    return result;
}

inline void writePackedBits(std::span<std::byte> storage, uint64_t bitOffset, uint32_t bitCount,
                            uint32_t value) {
    if (bitCount == 0 || bitCount > 32)
        throw std::invalid_argument("Packed scalar bit count must be in [1, 32].");

    for (uint32_t bit = 0; bit < bitCount; ++bit) {
        const uint64_t absoluteBit = bitOffset + bit;
        const size_t byteIndex = static_cast<size_t>(absoluteBit / 8);
        if (byteIndex >= storage.size())
            throw std::out_of_range("Packed scalar write exceeds tensor storage.");
        const uint8_t mask = static_cast<uint8_t>(1U << (absoluteBit % 8));
        uint8_t byte = std::to_integer<uint8_t>(storage[byteIndex]);
        if ((value >> bit) & 1U)
            byte |= mask;
        else
            byte &= static_cast<uint8_t>(~mask);
        storage[byteIndex] = static_cast<std::byte>(byte);
    }
}

template <typename T>
T readNative(std::span<const std::byte> storage, size_t byteOffset) {
    if (byteOffset > storage.size() || sizeof(T) > storage.size() - byteOffset)
        throw std::out_of_range("Native scalar read exceeds tensor storage.");
    T value;
    std::memcpy(&value, storage.data() + byteOffset, sizeof(T));
    return value;
}

template <typename T>
void writeNative(std::span<std::byte> storage, size_t byteOffset, const T& value) {
    if (byteOffset > storage.size() || sizeof(T) > storage.size() - byteOffset)
        throw std::out_of_range("Native scalar write exceeds tensor storage.");
    std::memcpy(storage.data() + byteOffset, &value, sizeof(T));
}

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
    else
        return {4, 3, 7, 7, false, false, false, 0x7e, 0, 0x7f};
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
    else
        return false;
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

inline uint64_t bitOffset(ScalarType type, ptrdiff_t logicalOffset) {
    if (logicalOffset < 0) throw std::out_of_range("Tensor logical offset is negative.");
    const uint64_t bits = scalarTypeInfo(type).storageBits;
    const uint64_t offset = static_cast<uint64_t>(logicalOffset);
    if (offset > std::numeric_limits<uint64_t>::max() / bits)
        throw std::overflow_error("Tensor bit offset overflow.");
    return offset * bits;
}

inline void copyBitRange(std::span<const std::byte> source, uint64_t sourceBitOffset,
                         std::span<std::byte> destination, uint64_t destinationBitOffset,
                         uint16_t bitCount) {
    if (bitCount > 128) throw std::invalid_argument("Tensor scalar storage exceeds 128 bits.");
    std::array<bool, 128> bits{};
    for (uint16_t bit = 0; bit < bitCount; ++bit) {
        const uint64_t sourcePosition = sourceBitOffset + bit;
        const uint8_t sourceByte =
            std::to_integer<uint8_t>(source[static_cast<size_t>(sourcePosition / 8)]);
        bits[bit] = ((sourceByte >> (sourcePosition % 8)) & 1U) != 0;
    }
    for (uint16_t bit = 0; bit < bitCount; ++bit) {
        const uint64_t destinationPosition = destinationBitOffset + bit;
        std::byte& destinationByte = destination[static_cast<size_t>(destinationPosition / 8)];
        const uint8_t mask = static_cast<uint8_t>(1U << (destinationPosition % 8));
        uint8_t value = std::to_integer<uint8_t>(destinationByte);
        value = bits[bit] ? static_cast<uint8_t>(value | mask)
                          : static_cast<uint8_t>(value & static_cast<uint8_t>(~mask));
        destinationByte = static_cast<std::byte>(value);
    }
}

inline int64_t signExtend(uint32_t value, uint32_t bits) {
    const uint32_t sign = 1U << (bits - 1U);
    return static_cast<int32_t>((value ^ sign) - sign);
}

template <ScalarType Type, typename Target>
Target decodeScalarKnown(std::span<const std::byte> storage, ptrdiff_t logicalOffset,
                         const ScalarConversionOptions& options) {
    static_assert(isConcreteScalarType(Type));
    const uint64_t offsetBits = bitOffset(Type, logicalOffset);
    const size_t offsetBytes = static_cast<size_t>(offsetBits / 8);

    if constexpr (Type == ScalarType::Boolean)
        return convertScalarValue<Target>(readNative<uint8_t>(storage, offsetBytes) != 0, options);
    else if constexpr (Type == ScalarType::UInt8)
        return convertScalarValue<Target>(readNative<uint8_t>(storage, offsetBytes), options);
    else if constexpr (Type == ScalarType::Int8)
        return convertScalarValue<Target>(readNative<int8_t>(storage, offsetBytes), options);
    else if constexpr (Type == ScalarType::UInt16)
        return convertScalarValue<Target>(readNative<uint16_t>(storage, offsetBytes), options);
    else if constexpr (Type == ScalarType::Int16)
        return convertScalarValue<Target>(readNative<int16_t>(storage, offsetBytes), options);
    else if constexpr (Type == ScalarType::UInt32)
        return convertScalarValue<Target>(readNative<uint32_t>(storage, offsetBytes), options);
    else if constexpr (Type == ScalarType::Int32)
        return convertScalarValue<Target>(readNative<int32_t>(storage, offsetBytes), options);
    else if constexpr (Type == ScalarType::UInt64)
        return convertScalarValue<Target>(readNative<uint64_t>(storage, offsetBytes), options);
    else if constexpr (Type == ScalarType::Int64)
        return convertScalarValue<Target>(readNative<int64_t>(storage, offsetBytes), options);
    else if constexpr (Type == ScalarType::Float16)
        return convertScalarValue<Target>(decodeFloat16(readNative<uint16_t>(storage, offsetBytes)),
                                          options);
    else if constexpr (Type == ScalarType::BFloat16)
        return convertScalarValue<Target>(
            decodeBFloat16(readNative<uint16_t>(storage, offsetBytes)), options);
    else if constexpr (Type == ScalarType::Float32)
        return convertScalarValue<Target>(readNative<float>(storage, offsetBytes), options);
    else if constexpr (Type == ScalarType::Float64)
        return convertScalarValue<Target>(readNative<double>(storage, offsetBytes), options);
    else if constexpr (Type == ScalarType::ComplexFloat32)
        return convertScalarValue<Target>(readNative<std::complex<float>>(storage, offsetBytes),
                                          options);
    else if constexpr (Type == ScalarType::ComplexFloat64)
        return convertScalarValue<Target>(readNative<std::complex<double>>(storage, offsetBytes),
                                          options);
    else if constexpr (Type == ScalarType::Int4)
        return convertScalarValue<Target>(signExtend(readPackedBits(storage, offsetBits, 4), 4),
                                          options);
    else if constexpr (Type == ScalarType::Int12)
        return convertScalarValue<Target>(signExtend(readPackedBits(storage, offsetBits, 12), 12),
                                          options);
    else if constexpr (Type == ScalarType::Float4E2M1 || Type == ScalarType::Float6E2M3 ||
                       Type == ScalarType::Float6E3M2 || Type == ScalarType::Float8E4M3 ||
                       Type == ScalarType::Float8E5M2 || Type == ScalarType::Float8E4M3Fnuz ||
                       Type == ScalarType::Float8E5M2Fnuz || Type == ScalarType::E5M3 ||
                       Type == ScalarType::E4M3)
        return convertScalarValue<Target>(
            decodeBinaryFloat(
                Type, readPackedBits(storage, offsetBits, scalarTypeInfo(Type).storageBits)),
            options);
    else if constexpr (Type == ScalarType::E8M0)
        return convertScalarValue<Target>(decodeE8M0(readNative<uint8_t>(storage, offsetBytes)),
                                          options);
}

template <ScalarType Type, typename Target>
Target decodeScalarKnown(std::span<const std::byte> storage, ptrdiff_t logicalOffset) {
    return decodeScalarKnown<Type, Target>(storage, logicalOffset,
                                           implicitNativeConversionOptions());
}

template <typename Target>
Target decodeScalar(ScalarType type, std::span<const std::byte> storage, ptrdiff_t logicalOffset,
                    const ScalarConversionOptions& options) {
    return visitScalarType(type, [&]<typename Tag>() {
        return decodeScalarKnown<Tag::type, Target>(storage, logicalOffset, options);
    });
}

template <typename Target>
Target decodeScalar(ScalarType type, std::span<const std::byte> storage, ptrdiff_t logicalOffset) {
    return decodeScalar<Target>(type, storage, logicalOffset, implicitNativeConversionOptions());
}

template <ScalarType Type, typename Source>
void encodeScalarKnown(std::span<std::byte> storage, ptrdiff_t logicalOffset, Source source,
                       const ScalarConversionOptions& options) {
    static_assert(isConcreteScalarType(Type));
    const uint64_t offsetBits = bitOffset(Type, logicalOffset);
    const size_t offsetBytes = static_cast<size_t>(offsetBits / 8);

    if constexpr (Type == ScalarType::Boolean)
        writeNative<uint8_t>(storage, offsetBytes,
                             convertScalarValue<bool>(source, options) ? uint8_t{1} : uint8_t{0});
    else if constexpr (Type == ScalarType::UInt8)
        writeNative<uint8_t>(storage, offsetBytes, convertScalarValue<uint8_t>(source, options));
    else if constexpr (Type == ScalarType::Int8)
        writeNative<int8_t>(storage, offsetBytes, convertScalarValue<int8_t>(source, options));
    else if constexpr (Type == ScalarType::UInt16)
        writeNative<uint16_t>(storage, offsetBytes, convertScalarValue<uint16_t>(source, options));
    else if constexpr (Type == ScalarType::Int16)
        writeNative<int16_t>(storage, offsetBytes, convertScalarValue<int16_t>(source, options));
    else if constexpr (Type == ScalarType::UInt32)
        writeNative<uint32_t>(storage, offsetBytes, convertScalarValue<uint32_t>(source, options));
    else if constexpr (Type == ScalarType::Int32)
        writeNative<int32_t>(storage, offsetBytes, convertScalarValue<int32_t>(source, options));
    else if constexpr (Type == ScalarType::UInt64)
        writeNative<uint64_t>(storage, offsetBytes, convertScalarValue<uint64_t>(source, options));
    else if constexpr (Type == ScalarType::Int64)
        writeNative<int64_t>(storage, offsetBytes, convertScalarValue<int64_t>(source, options));
    else if constexpr (Type == ScalarType::Float16)
        writeNative<uint16_t>(storage, offsetBytes,
                              encodeFloat16(convertScalarValue<float>(source, options)));
    else if constexpr (Type == ScalarType::BFloat16)
        writeNative<uint16_t>(storage, offsetBytes,
                              encodeBFloat16(convertScalarValue<float>(source, options)));
    else if constexpr (Type == ScalarType::Float32)
        writeNative<float>(storage, offsetBytes, convertScalarValue<float>(source, options));
    else if constexpr (Type == ScalarType::Float64)
        writeNative<double>(storage, offsetBytes, convertScalarValue<double>(source, options));
    else if constexpr (Type == ScalarType::ComplexFloat32)
        writeNative<std::complex<float>>(storage, offsetBytes,
                                         convertScalarValue<std::complex<float>>(source, options));
    else if constexpr (Type == ScalarType::ComplexFloat64)
        writeNative<std::complex<double>>(
            storage, offsetBytes, convertScalarValue<std::complex<double>>(source, options));
    else if constexpr (Type == ScalarType::Int4)
        writePackedBits(storage, offsetBits, 4,
                        static_cast<uint32_t>(convertToIntegerBits(source, 4, true, options)));
    else if constexpr (Type == ScalarType::Int12)
        writePackedBits(storage, offsetBits, 12,
                        static_cast<uint32_t>(convertToIntegerBits(source, 12, true, options)));
    else if constexpr (Type == ScalarType::Float4E2M1 || Type == ScalarType::Float6E2M3 ||
                       Type == ScalarType::Float6E3M2 || Type == ScalarType::Float8E4M3 ||
                       Type == ScalarType::Float8E5M2 || Type == ScalarType::Float8E4M3Fnuz ||
                       Type == ScalarType::Float8E5M2Fnuz || Type == ScalarType::E5M3 ||
                       Type == ScalarType::E4M3)
        writePackedBits(storage, offsetBits, scalarTypeInfo(Type).storageBits,
                        encodeBinaryFloat(Type, convertScalarValue<float>(source, options)));
    else if constexpr (Type == ScalarType::E8M0)
        writeNative<uint8_t>(storage, offsetBytes,
                             encodeE8M0(convertScalarValue<float>(source, options)));
}

template <ScalarType Type, typename Source>
void encodeScalarKnown(std::span<std::byte> storage, ptrdiff_t logicalOffset, Source source) {
    encodeScalarKnown<Type>(storage, logicalOffset, std::move(source),
                            implicitStorageConversionOptions(Type));
}

template <typename Source>
void encodeScalar(ScalarType type, std::span<std::byte> storage, ptrdiff_t logicalOffset,
                  Source source, const ScalarConversionOptions& options) {
    visitScalarType(type, [&]<typename Tag>() {
        encodeScalarKnown<Tag::type>(storage, logicalOffset, std::move(source), options);
    });
}

template <typename Source>
void encodeScalar(ScalarType type, std::span<std::byte> storage, ptrdiff_t logicalOffset,
                  Source source) {
    encodeScalar(type, storage, logicalOffset, std::move(source),
                 implicitStorageConversionOptions(type));
}
}  // namespace roc::host_validation::detail
