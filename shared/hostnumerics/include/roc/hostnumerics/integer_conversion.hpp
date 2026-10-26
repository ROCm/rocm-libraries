// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

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
        if (overflow == IntegerOverflow::Saturate) {
            const int64_t minimum = isSigned ? -static_cast<int64_t>(maximumBits) - 1 : 0;
            if (std::cmp_less(wide, minimum)) return minimumBits;
            if (std::cmp_greater(wide, maximumBits)) return maximumBits;
        }
        return static_cast<uint64_t>(wide) & mask;
    } else {
        const Source rounded = roundToIntegral(value, rounding);
        if (overflow != IntegerOverflow::Identity) {
            const int rangeExponent = static_cast<int>(isSigned ? bits - 1 : bits);
            const Source upperExclusive = std::ldexp(Source{1}, rangeExponent);
            const Source lowerInclusive = isSigned ? -upperExclusive : Source{0};
            if (overflow == IntegerOverflow::Saturate) {
                if (rounded < lowerInclusive) return minimumBits;
                if (rounded >= upperExclusive) return maximumBits;
            } else if (!(rounded >= lowerInclusive && rounded < upperExclusive)) {
                const Source modulus = std::ldexp(Source{1}, static_cast<int>(bits));
                uint64_t remainder = static_cast<uint64_t>(std::fmod(std::fabs(rounded), modulus));
                if (std::signbit(rounded)) remainder = uint64_t{0} - remainder;
                return remainder & mask;
            }
        }
        if (isSigned) return static_cast<uint64_t>(static_cast<int64_t>(rounded)) & mask;
        return static_cast<uint64_t>(rounded) & mask;
    }
}
}  // namespace detail

// Converts an integer or floating value to the integer type Target: rounds it,
// then handles values outside Target's range. Conversion does not validate its
// input; the caller must meet the requirements of the chosen IntegerOverflow.
//
// Examples, after `using enum IntegerRounding; using enum IntegerOverflow;`:
//   convertToInteger<int8_t>(double{2.5}, NearestEven, Identity) == 2
//   convertToInteger<int8_t>(double{3.5}, NearestEven, Identity) == 4
//   convertToInteger<int8_t>(double{-3.75}, TowardZero, Identity) == -3
//   convertToInteger<int8_t>(int32_t{300}, TowardZero, Saturate) == 127
//   convertToInteger<int8_t>(int32_t{300}, TowardZero, ModuloWrap) == 44
//   convertToInteger<uint8_t>(int32_t{-1}, TowardZero, Saturate) == 0
//   convertToInteger<uint8_t>(int32_t{-1}, TowardZero, ModuloWrap) == 255
//   convertToInteger<int8_t>(double{1e300}, TowardZero, Saturate) == 127
//   convertToInteger<int64_t>(bool{true}, TowardZero, Identity) == 1
//   convertToInteger<size_t>(uint64_t{42}, TowardZero, Identity) == 42
//   convertToInteger<char>(double{65.0}, TowardZero, Identity) == 'A'
//
// Rejected at compile time:
//   convertToInteger<bool>(double{0.5}, ...)                  Target cannot be bool.
//   convertToInteger<float>(int32_t{1}, ...)                  Target must be an integer.
//   convertToInteger<char>(std::complex<double>{65, 0}, ...)  Source must be a real number.
//
// Outside the domain, with an undefined result:
//   convertToInteger<int8_t>(int32_t{300}, TowardZero, Identity)  Result does not fit.
//   convertToInteger<int8_t>(infinity, TowardZero, ModuloWrap)    Input is not finite.
//   convertToInteger<int8_t>(NaN, rounding, overflow)             NaN is never valid.
template <std::integral Target, typename Source>
    requires(!std::is_same_v<Target, bool> && std::is_arithmetic_v<Source>)
Target convertToInteger(Source value, IntegerRounding rounding, IntegerOverflow overflow) noexcept {
    constexpr uint32_t bits = std::numeric_limits<std::make_unsigned_t<Target>>::digits;
    // C++20 integer casts reduce modulo 2^N, including signed destinations.
    return static_cast<Target>(
        detail::convertToIntegerBits(value, bits, std::is_signed_v<Target>, rounding, overflow));
}

}  // namespace roc::hostnumerics
