// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <roc/host_numerics/scalar_type.hpp>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace roc::host_numerics::detail {
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
// integers. Packed Int4 saturates because it has no native C++ cast whose
// behavior can define the conversion.
inline constexpr ScalarConversionOptions implicitStorageConversionOptions(ScalarType destination) {
    const IntegerOverflow overflow =
        destination == ScalarType::Int4 ? IntegerOverflow::Saturate : IntegerOverflow::ModuloWrap;
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
    using Destination = std::remove_cv_t<Target>;
    using Value = std::remove_cvref_t<Source>;
    static_assert(!std::is_reference_v<Target>);

    if constexpr (RuntimeIsComplexV<Destination>) {
        using Component = typename Destination::value_type;
        if constexpr (RuntimeIsComplexV<Value>)
            return Destination(convertScalarValue<Component>(source.real(), options),
                               convertScalarValue<Component>(source.imag(), options));
        else
            return Destination(convertScalarValue<Component>(source, options), Component{0});
    } else if constexpr (RuntimeIsComplexV<Value>) {
        if (source.imag() != typename Value::value_type{0})
            throw std::domain_error(
                "A complex value with a nonzero imaginary component cannot be converted to real.");
        return convertScalarValue<Destination>(source.real(), options);
    } else if constexpr (std::is_same_v<Destination, bool>) {
        return source != Value{0};
    } else if constexpr (std::is_integral_v<Destination>) {
        return convertToInteger<Destination>(std::move(source), options);
    } else {
        return static_cast<Destination>(source);
    }
}
}  // namespace roc::host_numerics::detail
