// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <roc/hostnumerics/expected.hpp>
#include <roc/hostnumerics/scalar_conversion_options.hpp>
#include <string_view>
#include <type_traits>
#include <utility>

namespace roc::hostnumerics::detail {
template <typename T>
struct RuntimeIsComplex : std::false_type {};

template <typename T>
struct RuntimeIsComplex<std::complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool RuntimeIsComplexV = RuntimeIsComplex<T>::value;

template <typename>
inline constexpr bool AlwaysFalseV = false;

enum class ConversionError {
    InvalidRounding,
    IntegerOutOfRange,
    InvalidOverflow,
    NaN,
    Infinity,
    RoundedOutOfRange,
    Nonreal,
};

inline constexpr std::string_view message(ConversionError error) noexcept {
    switch (error) {
        case ConversionError::InvalidRounding:
            return "Invalid integer rounding policy.";
        case ConversionError::IntegerOutOfRange:
            return "Integer value is outside the destination range.";
        case ConversionError::InvalidOverflow:
            return "Invalid integer overflow policy.";
        case ConversionError::NaN:
            return "NaN cannot be converted to an integer destination.";
        case ConversionError::Infinity:
            return "Infinity cannot be converted to an integer destination.";
        case ConversionError::RoundedOutOfRange:
            return "Rounded value is outside the destination integer range.";
        case ConversionError::Nonreal:
            return "A complex value with a nonzero imaginary component cannot be converted to "
                   "real.";
    }
    return "Unknown ConversionError failure.";
}

struct ConversionFailure {
    ConversionError code;
    uint32_t destinationBits = 0;
    bool signedDestination = false;
};
// Callers retain the source value and element position if needed for reporting.
// Keep only the failure kind and destination format in the per-value result.
inline constexpr std::string_view message(const ConversionFailure& failure) noexcept {
    return message(failure.code);
}

// Low-bit mask for a width in [1, 64]: integerMask(8) is 0xff; integerMask(64)
// is UINT64_MAX. The 64-bit case avoids shifting by the type's width.
inline constexpr uint64_t integerMask(uint32_t bits) {
    return bits == 64 ? std::numeric_limits<uint64_t>::max() : (uint64_t{1} << bits) - 1;
}

// Two's-complement bits for the signed minimum of a width in [1, 64].
// signedIntegerMinimumRaw(8) is 0x80, representing -128.
inline constexpr uint64_t signedIntegerMinimumRaw(uint32_t bits) {
    return uint64_t{1} << (bits - 1);
}

// Bits for the maximum value of a width in [1, 64]. For eight bits, signed
// destinations return 0x7f (127), and unsigned destinations return 0xff (255).
inline constexpr uint64_t integerMaximumRaw(uint32_t bits, bool signedDestination) noexcept {
    return signedDestination ? signedIntegerMinimumRaw(bits) - 1 : integerMask(bits);
}

// Round without choosing an integer width. The rounding policy must be valid.
// TowardZero maps -3.75 to -3; NearestEven maps 2.5 to 2 and 3.5 to 4 regardless
// of the host rounding mode.
template <typename Floating>
Floating roundForIntegerConversion(Floating value, IntegerRounding rounding) noexcept {
    static_assert(std::is_floating_point_v<Floating>);
    assert((rounding == IntegerRounding::TowardZero || rounding == IntegerRounding::NearestEven) &&
           "Integer rounding policy must be valid.");
    if (rounding == IntegerRounding::TowardZero) return std::trunc(value);

    Floating integral = 0;
    const Floating fraction = std::modf(value, &integral);
    const Floating magnitude = std::fabs(fraction);
    if (magnitude < Floating{0.5}) return integral;

    const Floating direction = std::signbit(fraction) ? Floating{-1} : Floating{1};
    if (magnitude > Floating{0.5}) return integral + direction;

    const Floating parity = std::fmod(std::fabs(integral), Floating{2});
    return parity == Floating{0} ? integral : integral + direction;
}

// Checked rounding: an unsupported policy returns InvalidRounding.
template <typename Floating>
Expected<Floating, ConversionFailure> getRoundForIntegerConversion(
    Floating value, IntegerRounding rounding) noexcept {
    if (rounding != IntegerRounding::TowardZero && rounding != IntegerRounding::NearestEven)
        return Unexpected(ConversionFailure{ConversionError::InvalidRounding});
    return roundForIntegerConversion(value, rounding);
}

// Test representability before applying an overflow policy. Width is in [1, 64].
// For value 255 and eight bits, an unsigned destination fits and a signed one does not.
template <typename Integral>
bool integralFitsIntegerDestination(Integral value, uint32_t bits,
                                    bool signedDestination) noexcept {
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

// Test whether a value is below the destination's minimum. Width is in [1, 64].
// With eight bits, -1 is below an unsigned minimum; -128 equals a signed minimum.
template <typename Integral>
bool integralIsBelowIntegerDestination(Integral value, uint32_t bits,
                                       bool signedDestination) noexcept {
    static_assert(std::is_integral_v<Integral>);
    if (!signedDestination) return std::cmp_less(value, 0);
    if (bits == 64) return std::cmp_less(value, std::numeric_limits<int64_t>::min());
    const int64_t minimum = -(int64_t{1} << (bits - 1));
    return std::cmp_less(value, minimum);
}

// Convert an integral value to destination bits, with width in [1, 64]. Rounding
// is unnecessary. For signed eight-bit output, 128 fails with IntegerOutOfRange
// under Reject and produces 0x80 under ModuloWrap.
template <typename Integral>
Expected<uint64_t, ConversionFailure> getConvertIntegralToIntegerBits(
    Integral value, uint32_t bits, bool signedDestination,
    const ScalarConversionOptions& options) noexcept {
    static_assert(std::is_integral_v<Integral>);
    assert((bits > 0 && bits <= 64) && "Integer width must be between 1 and 64 bits.");
    if (options.integerOverflow != IntegerOverflow::Reject &&
        options.integerOverflow != IntegerOverflow::Saturate &&
        options.integerOverflow != IntegerOverflow::ModuloWrap)
        return Unexpected(
            ConversionFailure{ConversionError::InvalidOverflow, bits, signedDestination});
    const uint64_t mask = integerMask(bits);
    if (integralFitsIntegerDestination(value, bits, signedDestination)) {
        if (signedDestination) return static_cast<uint64_t>(static_cast<int64_t>(value)) & mask;
        return static_cast<uint64_t>(value) & mask;
    }

    switch (options.integerOverflow) {
        case IntegerOverflow::Reject:
            return Unexpected(
                ConversionFailure{ConversionError::IntegerOutOfRange, bits, signedDestination});
        case IntegerOverflow::Saturate:
            return integralIsBelowIntegerDestination(value, bits, signedDestination)
                       ? (signedDestination ? signedIntegerMinimumRaw(bits) : 0)
                       : integerMaximumRaw(bits, signedDestination);
        case IntegerOverflow::ModuloWrap:
            return static_cast<uint64_t>(value) & mask;
        case IntegerOverflow::Count:
            break;
    }
    return Unexpected(ConversionFailure{ConversionError::InvalidOverflow, bits, signedDestination});
}

// Round, then convert to destination bits, with width in [1, 64]. For signed
// eight-bit output, NearestEven maps 2.5 to 0x02; Saturate maps 128.0 to 0x7f.
// NaN always fails; infinity succeeds only with saturation.
template <typename Floating>
Expected<uint64_t, ConversionFailure> getConvertFloatingToIntegerBits(
    Floating value, uint32_t bits, bool signedDestination,
    const ScalarConversionOptions& options) noexcept {
    static_assert(std::is_floating_point_v<Floating>);
    assert((bits > 0 && bits <= 64) && "Integer width must be between 1 and 64 bits.");
    if (options.integerOverflow != IntegerOverflow::Reject &&
        options.integerOverflow != IntegerOverflow::Saturate &&
        options.integerOverflow != IntegerOverflow::ModuloWrap)
        return Unexpected(
            ConversionFailure{ConversionError::InvalidOverflow, bits, signedDestination});
    if (std::isnan(value)) {
        return Unexpected(ConversionFailure{ConversionError::NaN, bits, signedDestination});
    }

    if (std::isinf(value)) {
        if (options.integerOverflow == IntegerOverflow::Saturate) {
            if (std::signbit(value)) return signedDestination ? signedIntegerMinimumRaw(bits) : 0;
            return integerMaximumRaw(bits, signedDestination);
        }
        return Unexpected(ConversionFailure{ConversionError::Infinity, bits, signedDestination});
    }

    if (options.integerRounding != IntegerRounding::TowardZero &&
        options.integerRounding != IntegerRounding::NearestEven) {
        return Unexpected(
            ConversionFailure{ConversionError::InvalidRounding, bits, signedDestination});
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
            return Unexpected(
                ConversionFailure{ConversionError::RoundedOutOfRange, bits, signedDestination});
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
        case IntegerOverflow::Count:
            break;
    }
    return Unexpected(ConversionFailure{ConversionError::InvalidOverflow, bits, signedDestination});
}

// Select integral or floating conversion for a numeric source. Width is in [1, 64].
// Boolean and enum sources use their integer values. A complex source (2, 0)
// converts like 2; (2, 1) fails with Nonreal rather than discarding its imaginary part.
template <typename Source>
Expected<uint64_t, ConversionFailure> getConvertToIntegerBits(
    Source source, uint32_t bits, bool signedDestination,
    const ScalarConversionOptions& options) noexcept {
    using Value = std::remove_cvref_t<Source>;
    if constexpr (RuntimeIsComplexV<Value>) {
        if (source.imag() != typename Value::value_type{0})
            return Unexpected(ConversionFailure{ConversionError::Nonreal, bits, signedDestination});
        return getConvertToIntegerBits(source.real(), bits, signedDestination, options);
    } else if constexpr (std::is_same_v<Value, bool>) {
        return getConvertIntegralToIntegerBits(static_cast<uint8_t>(source), bits,
                                               signedDestination, options);
    } else if constexpr (std::is_enum_v<Value>) {
        return getConvertToIntegerBits(static_cast<std::underlying_type_t<Value>>(source), bits,
                                       signedDestination, options);
    } else if constexpr (std::is_integral_v<Value>) {
        return getConvertIntegralToIntegerBits(source, bits, signedDestination, options);
    } else if constexpr (std::is_floating_point_v<Value>) {
        return getConvertFloatingToIntegerBits(source, bits, signedDestination, options);
    } else {
        static_assert(AlwaysFalseV<Value>, "Scalar integer conversion requires a numeric source.");
    }
}

// Interpret the low destination-width bits as a native integer. For raw 0xff,
// integerFromBits<int8_t> returns -1 and integerFromBits<uint8_t> returns 255.
template <typename Target>
Target integerFromBits(uint64_t raw) noexcept {
    using Value = std::remove_cv_t<Target>;
    static_assert(std::is_integral_v<Value> && !std::is_same_v<Value, bool>);
    // C++20 integral conversion is modulo 2^N, including signed destinations.
    return static_cast<Value>(raw);
}

// Convert to bits without a failure result. Width and overflow policy must be
// valid; under Reject, the input must fit. For signed eight-bit output,
// ModuloWrap maps 128 to 0x80 and Saturate maps it to 0x7f.
template <typename Integral>
uint64_t convertIntegralToIntegerBits(Integral value, uint32_t bits, bool signedDestination,
                                      const ScalarConversionOptions& options) noexcept {
    static_assert(std::is_integral_v<Integral>);
    assert((bits > 0 && bits <= 64) && "Integer width must be between 1 and 64 bits.");
    assert((options.integerOverflow == IntegerOverflow::Reject ||
            options.integerOverflow == IntegerOverflow::Saturate ||
            options.integerOverflow == IntegerOverflow::ModuloWrap) &&
           "Integer overflow policy must be valid.");
    if (options.integerOverflow == IntegerOverflow::Saturate &&
        !integralFitsIntegerDestination(value, bits, signedDestination)) {
        return integralIsBelowIntegerDestination(value, bits, signedDestination)
                   ? (signedDestination ? signedIntegerMinimumRaw(bits) : 0)
                   : integerMaximumRaw(bits, signedDestination);
    }
    assert((options.integerOverflow != IntegerOverflow::Reject ||
            integralFitsIntegerDestination(value, bits, signedDestination)) &&
           "Reject conversion requires a representable value.");
    return static_cast<uint64_t>(value) & integerMask(bits);
}

// Round and convert without a failure result. Width and policies must be valid.
// NaN is outside the domain; infinity is permitted only under Saturate. Reject
// requires the rounded value to fit. NearestEven maps 2.5 to 2 and 3.5 to 4.
template <typename Floating>
uint64_t convertFloatingToIntegerBits(Floating value, uint32_t bits, bool signedDestination,
                                      const ScalarConversionOptions& options) noexcept {
    static_assert(std::is_floating_point_v<Floating>);
    assert((bits > 0 && bits <= 64) && "Integer width must be between 1 and 64 bits.");
    assert(!std::isnan(value) && "Integer conversion requires a non-NaN value.");
    assert((options.integerOverflow == IntegerOverflow::Reject ||
            options.integerOverflow == IntegerOverflow::Saturate ||
            options.integerOverflow == IntegerOverflow::ModuloWrap) &&
           "Integer overflow policy must be valid.");
    assert((std::isfinite(value) || options.integerOverflow == IntegerOverflow::Saturate) &&
           "Infinity requires saturating integer conversion.");
    const Floating rounded = roundForIntegerConversion(value, options.integerRounding);
    const int rangeExponent =
        signedDestination ? static_cast<int>(bits) - 1 : static_cast<int>(bits);
    const Floating upperExclusive = std::ldexp(Floating{1}, rangeExponent);
    const Floating lowerInclusive = signedDestination ? -upperExclusive : Floating{0};
    if (options.integerOverflow == IntegerOverflow::Saturate) {
        if (rounded < lowerInclusive) return signedDestination ? signedIntegerMinimumRaw(bits) : 0;
        if (rounded >= upperExclusive) return integerMaximumRaw(bits, signedDestination);
    } else if (options.integerOverflow == IntegerOverflow::ModuloWrap &&
               !(rounded >= lowerInclusive && rounded < upperExclusive)) {
        const Floating modulus = std::ldexp(Floating{1}, static_cast<int>(bits));
        const Floating remainder = std::fmod(std::fabs(rounded), modulus);
        uint64_t raw = static_cast<uint64_t>(remainder);
        if (std::signbit(rounded)) raw = uint64_t{0} - raw;
        return raw & integerMask(bits);
    }
    assert((rounded >= lowerInclusive && rounded < upperExclusive) &&
           "Reject conversion requires a representable rounded value.");
    if (signedDestination)
        return static_cast<uint64_t>(static_cast<int64_t>(rounded)) & integerMask(bits);
    return static_cast<uint64_t>(rounded) & integerMask(bits);
}

// Numeric-source adapter for value-returning integer conversion. Complex inputs
// must be real: (2, 0) converts like 2; (2, 1) violates the precondition.
template <typename Source>
uint64_t convertToIntegerBits(Source source, uint32_t bits, bool signedDestination,
                              const ScalarConversionOptions& options) noexcept {
    using Value = std::remove_cvref_t<Source>;
    if constexpr (RuntimeIsComplexV<Value>) {
        assert(source.imag() == typename Value::value_type{0} &&
               "Complex-to-real conversion requires a zero imaginary component.");
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

// Checked, allocation-free conversion to a non-Boolean native integer, returning
// Expected without throwing. getConvertToInteger<int8_t>(128, {}) returns
// IntegerOutOfRange; with Saturate it returns 127.
template <typename Target, typename Source>
Expected<Target, ConversionFailure> getConvertToInteger(
    Source source, const ScalarConversionOptions& options) noexcept {
    using Value = std::remove_cv_t<Target>;
    static_assert(std::is_integral_v<Value> && !std::is_same_v<Value, bool>);
    constexpr uint32_t bits = std::numeric_limits<std::make_unsigned_t<Value>>::digits;
    const auto result = getConvertToIntegerBits(source, bits, std::is_signed_v<Value>, options);
    if (!result) return Unexpected(result.error());
    return integerFromBits<Value>(*result);
}

// Checked native conversion, applying integer policies when needed. Converting
// 1.5 to complex<double> gives (1.5, 0); converting (2, 1) to double returns Nonreal.
template <typename Target, typename Source>
Expected<Target, ConversionFailure> getConvertScalarValue(
    Source source, const ScalarConversionOptions& options) noexcept {
    using Destination = std::remove_cv_t<Target>;
    using Value = std::remove_cvref_t<Source>;
    static_assert(!std::is_reference_v<Target>);
    if constexpr (RuntimeIsComplexV<Destination>) {
        using Component = typename Destination::value_type;
        if constexpr (RuntimeIsComplexV<Value>) {
            const auto real = getConvertScalarValue<Component>(source.real(), options);
            if (!real) return Unexpected(real.error());
            const auto imaginary = getConvertScalarValue<Component>(source.imag(), options);
            if (!imaginary) return Unexpected(imaginary.error());
            return Destination(*real, *imaginary);
        } else {
            const auto real = getConvertScalarValue<Component>(source, options);
            if (!real) return Unexpected(real.error());
            return Destination(*real, Component{0});
        }
    } else if constexpr (RuntimeIsComplexV<Value>) {
        if constexpr (std::is_integral_v<Destination> && !std::is_same_v<Destination, bool>) {
            return getConvertToInteger<Destination>(source, options);
        }
        if (source.imag() != typename Value::value_type{0})
            return Unexpected(ConversionFailure{ConversionError::Nonreal});
        return getConvertScalarValue<Destination>(source.real(), options);
    } else if constexpr (std::is_same_v<Destination, bool>) {
        static_assert(std::is_arithmetic_v<Value> || std::is_enum_v<Value>);
        return source != Value{0};
    } else if constexpr (std::is_integral_v<Destination>) {
        return getConvertToInteger<Destination>(source, options);
    } else {
        static_assert(std::is_floating_point_v<Destination> &&
                      (std::is_arithmetic_v<Value> || std::is_enum_v<Value>));
        return static_cast<Destination>(source);
    }
}
}  // namespace roc::hostnumerics::detail
