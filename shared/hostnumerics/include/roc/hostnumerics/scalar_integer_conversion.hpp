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

// Integer widths below must be in [1, 64]. The mask avoids shifting by 64.
inline constexpr uint64_t integerMask(uint32_t bits) {
    return bits == 64 ? std::numeric_limits<uint64_t>::max() : (uint64_t{1} << bits) - 1;
}

// Two's-complement bits for the signed minimum.
inline constexpr uint64_t signedIntegerMinimumRaw(uint32_t bits) {
    return uint64_t{1} << (bits - 1);
}

inline constexpr uint64_t integerMaximumRaw(uint32_t bits, bool signedDestination) noexcept {
    return signedDestination ? signedIntegerMinimumRaw(bits) - 1 : integerMask(bits);
}

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

template <typename Integral>
uint64_t convertIntegralToIntegerBits(Integral value, uint32_t bits, bool signedDestination,
                                      const ScalarConversionOptions& options) noexcept {
    static_assert(std::is_integral_v<Integral>);
    if (options.integerOverflow == IntegerOverflow::Saturate) {
        const uint64_t maximum = integerMaximumRaw(bits, signedDestination);
        if (signedDestination) {
            const int64_t minimum = -static_cast<int64_t>(maximum) - 1;
            if (std::cmp_less(value, minimum)) return signedIntegerMinimumRaw(bits);
        } else if (std::cmp_less(value, 0)) {
            return 0;
        }
        if (std::cmp_greater(value, maximum)) return maximum;
    }
    return static_cast<uint64_t>(value) & integerMask(bits);
}

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

// Complex sources must have a zero imaginary part.
template <typename Source>
uint64_t convertToIntegerBits(Source source, uint32_t bits, bool signedDestination,
                              const ScalarConversionOptions& options) noexcept {
    using Value = std::remove_cvref_t<Source>;
    if constexpr (RuntimeIsComplexV<Value>) {
        return convertToIntegerBits(source.real(), bits, signedDestination, options);
    } else if constexpr (std::is_integral_v<Value>) {
        // Standard integer comparisons exclude character types; widen with signedness preserved.
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
