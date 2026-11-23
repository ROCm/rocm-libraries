// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace roc::hostnumerics {
// Rounding of a floating value to an integer. Integral inputs are unchanged.
enum class IntegerRounding : uint8_t {
    // Discard the fractional part, moving toward zero: -3.75 becomes -3.
    TowardZero,
    // Round to the nearest integer; halfway values choose the even integer,
    // independently of the process rounding mode: 2.5 becomes 2 and 3.5 becomes 4.
    NearestEven,
};

// Handling of a rounded value outside the destination's range. NaN is outside
// the domain for every policy. N is the destination's bit width.
enum class IntegerOverflow : uint8_t {
    // Leave the rounded value unchanged; it must already fit in the destination.
    Identity,
    // Clamp the rounded value to the destination's range.
    // For int8_t, 128 and positive infinity become 127; negative infinity becomes -128.
    Saturate,
    // Reduce modulo 2^N, then interpret the N-bit result as two's complement when signed.
    // Input must be finite. For int8_t, 128 becomes -128.
    ModuloWrap,
};

// Policy for converting a value to an integer destination. Both choices are
// required; the constructor has no defaults.
struct IntegerConversionOptions {
    constexpr IntegerConversionOptions(IntegerRounding rounding, IntegerOverflow overflow) noexcept
        : rounding(rounding), overflow(overflow) {}

    IntegerRounding rounding;
    IntegerOverflow overflow;
};

namespace detail {
template <std::floating_point Floating>
Floating roundToIntegral(Floating value, IntegerRounding rounding) noexcept {
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

// Converts value to a signed or unsigned integer of the given width in [1, 64]
// and returns the low `bits` bits of its two's-complement representation.
template <typename Source>
    requires std::is_arithmetic_v<Source>
uint64_t convertToIntegerBits(Source value, uint32_t bits, bool isSigned, IntegerRounding rounding,
                              IntegerOverflow overflow) noexcept {
    // Avoids shifting by 64.
    const uint64_t mask = bits == 64 ? ~uint64_t{0} : (uint64_t{1} << bits) - 1;
    const uint64_t minimumBits = isSigned ? uint64_t{1} << (bits - 1) : 0;
    const uint64_t maximumBits = isSigned ? minimumBits - 1 : mask;

    if constexpr (std::is_integral_v<Source>) {
        // Standard integer comparisons exclude bool and character types; widen with
        // signedness preserved.
        using Wide = std::conditional_t<std::is_signed_v<Source>, int64_t, uint64_t>;
        const Wide wide{value};
        const int64_t minimum = isSigned ? -static_cast<int64_t>(maximumBits) - 1 : 0;
        const bool belowRange = std::cmp_less(wide, minimum);
        const bool aboveRange = std::cmp_greater(wide, maximumBits);
        assert(overflow != IntegerOverflow::Identity || !(belowRange || aboveRange));
        if (overflow == IntegerOverflow::Saturate) {
            if (belowRange) return minimumBits;
            if (aboveRange) return maximumBits;
        }
        return static_cast<uint64_t>(wide) & mask;
    } else {
        assert(!std::isnan(value));
        assert(overflow == IntegerOverflow::Saturate || std::isfinite(value));
        const Source rounded = roundToIntegral(value, rounding);
        const int rangeExponent = static_cast<int>(isSigned ? bits - 1 : bits);
        const Source upperExclusive = std::ldexp(Source{1}, rangeExponent);
        const Source lowerInclusive = isSigned ? -upperExclusive : Source{0};
        const bool inRange = rounded >= lowerInclusive && rounded < upperExclusive;
        assert(overflow != IntegerOverflow::Identity || inRange);
        if (overflow == IntegerOverflow::Saturate) {
            if (rounded < lowerInclusive) return minimumBits;
            if (rounded >= upperExclusive) return maximumBits;
        } else if (overflow == IntegerOverflow::ModuloWrap && !inRange) {
            const Source modulus = std::ldexp(Source{1}, static_cast<int>(bits));
            uint64_t remainder = static_cast<uint64_t>(std::fmod(std::fabs(rounded), modulus));
            if (std::signbit(rounded)) remainder = uint64_t{0} - remainder;
            return remainder & mask;
        }
        if (isSigned) return static_cast<uint64_t>(static_cast<int64_t>(rounded)) & mask;
        return static_cast<uint64_t>(rounded) & mask;
    }
}
}  // namespace detail

// Converts an integer or floating value to the integer type Target: rounds it,
// then handles values outside Target's range. Integer types are limited to 64
// bits. The caller must meet the requirements of the chosen IntegerOverflow;
// builds without NDEBUG assert them.
//
// Examples (types name the integer destination):
//   input   target    rounding     overflow    result
//   2.5     int8_t    NearestEven  Identity    2
//   3.5     int8_t    NearestEven  Identity    4
//   -3.75   int8_t    TowardZero   Identity    -3
//   300     int8_t    TowardZero   Saturate    127
//   300     int8_t    TowardZero   ModuloWrap  44
//   -1      uint8_t   TowardZero   Saturate    0
//   -1      uint8_t   TowardZero   ModuloWrap  255
//   1e300   int8_t    TowardZero   Saturate    127
//   true    int64_t   TowardZero   Identity    1
//   42      size_t    TowardZero   Identity    42
//   65.0    char      TowardZero   Identity    'A'
// See IntegerConversion.HeaderExamples for the C++ calls.
//
// Rejected at compile time:
//   Boolean or floating targets: Target must be a non-Boolean integer.
//   Complex sources: Source must be a real number.
//   Integers wider than 64 bits: neither Source nor Target supports them.
//
// Outside the domain, failing an assertion or giving an undefined result:
//   300 to int8_t with Identity: the result does not fit.
//   Infinity with ModuloWrap: the input is not finite.
//   NaN with any policy: NaN is never valid.
template <std::integral Target, typename Source>
    requires(!std::is_same_v<Target, bool> && sizeof(Target) <= sizeof(uint64_t) &&
             (std::floating_point<Source> ||
              (std::integral<Source> && sizeof(Source) <= sizeof(uint64_t))))
Target convertToInteger(Source value, IntegerRounding rounding, IntegerOverflow overflow) noexcept {
    constexpr uint32_t bits = std::numeric_limits<std::make_unsigned_t<Target>>::digits;
    // C++20 integer casts reduce modulo 2^N, including signed destinations.
    return static_cast<Target>(
        detail::convertToIntegerBits(value, bits, std::is_signed_v<Target>, rounding, overflow));
}

}  // namespace roc::hostnumerics
