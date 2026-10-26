// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <roc/hostnumerics/scalar_conversion_options.hpp>
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
// valid; under Identity, the input must fit. For signed eight-bit output,
// ModuloWrap maps 128 to 0x80 and Saturate maps it to 0x7f.
template <typename Integral>
uint64_t convertIntegralToIntegerBits(Integral value, uint32_t bits, bool signedDestination,
                                      const ScalarConversionOptions& options) noexcept {
    static_assert(std::is_integral_v<Integral>);
    if (options.integerOverflow == IntegerOverflow::Saturate &&
        !integralFitsIntegerDestination(value, bits, signedDestination)) {
        return integralIsBelowIntegerDestination(value, bits, signedDestination)
                   ? (signedDestination ? signedIntegerMinimumRaw(bits) : 0)
                   : integerMaximumRaw(bits, signedDestination);
    }
    return static_cast<uint64_t>(value) & integerMask(bits);
}

// Round and convert without a failure result. Width and policies must be valid.
// NaN is outside the domain; infinity is permitted only under Saturate. Identity
// requires the rounded value to fit. NearestEven maps 2.5 to 2 and 3.5 to 4.
template <typename Floating>
uint64_t convertFloatingToIntegerBits(Floating value, uint32_t bits, bool signedDestination,
                                      const ScalarConversionOptions& options) noexcept {
    static_assert(std::is_floating_point_v<Floating>);
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
        return convertToIntegerBits(source.real(), bits, signedDestination, options);
    } else if constexpr (std::is_same_v<Value, bool>) {
        return convertIntegralToIntegerBits(static_cast<uint8_t>(source), bits, signedDestination,
                                            options);
    } else if constexpr (std::is_enum_v<Value>) {
        return convertToIntegerBits(static_cast<std::underlying_type_t<Value>>(source), bits,
                                    signedDestination, options);
    } else if constexpr (std::is_integral_v<Value>) {
        // Standard integer comparisons exclude character types. Widen integral
        // sources without changing their values or signedness.
        using Integer = std::conditional_t<std::is_signed_v<Value>, int64_t, uint64_t>;
        static_assert(std::numeric_limits<Value>::digits <= std::numeric_limits<Integer>::digits);
        return convertIntegralToIntegerBits(static_cast<Integer>(source), bits, signedDestination,
                                            options);
    } else if constexpr (std::is_floating_point_v<Value>) {
        return convertFloatingToIntegerBits(source, bits, signedDestination, options);
    } else {
        static_assert(AlwaysFalseV<Value>, "Scalar integer conversion requires a numeric source.");
    }
}

}  // namespace roc::hostnumerics::detail
