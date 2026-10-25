// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/hostnumerics/scalar_integer_conversion.hpp>
#include <utility>

namespace roc::hostnumerics::detail {
// Value-returning native conversion. Callers must satisfy the integer conversion
// preconditions and supply real inputs for complex-to-real conversion.
template <typename Target, typename Source>
Target convertScalarValue(Source source, const ScalarConversionOptions& options) noexcept {
    using Destination = std::remove_cv_t<Target>;
    using Value = std::remove_cvref_t<Source>;
    static_assert(!std::is_reference_v<Target>);
    if constexpr (RuntimeIsComplexV<Destination>) {
        using Component = typename Destination::value_type;
        if constexpr (RuntimeIsComplexV<Value>) {
            return Destination(convertScalarValue<Component>(source.real(), options),
                               convertScalarValue<Component>(source.imag(), options));
        } else {
            return Destination(convertScalarValue<Component>(source, options), Component{0});
        }
    } else if constexpr (RuntimeIsComplexV<Value>) {
        return convertScalarValue<Destination>(source.real(), options);
    } else if constexpr (std::is_same_v<Destination, bool>) {
        static_assert(std::is_arithmetic_v<Value> || std::is_enum_v<Value>);
        return source != Value{0};
    } else if constexpr (std::is_integral_v<Destination>) {
        if constexpr (std::is_floating_point_v<Value>) {
            if (options.integerOverflow == IntegerOverflow::Identity) {
                // A native cast already truncates; only round explicitly for nearest-even.
                const Value rounded =
                    options.integerRounding == IntegerRounding::TowardZero
                        ? source
                        : roundForIntegerConversion(source, options.integerRounding);
                return static_cast<Destination>(rounded);
            }
        }
        constexpr uint32_t bits = std::numeric_limits<std::make_unsigned_t<Destination>>::digits;
        return integerFromBits<Destination>(
            convertToIntegerBits(source, bits, std::is_signed_v<Destination>, options));
    } else {
        static_assert(std::is_floating_point_v<Destination> &&
                      (std::is_arithmetic_v<Value> || std::is_enum_v<Value>));
        return static_cast<Destination>(source);
    }
}
}  // namespace roc::hostnumerics::detail

namespace roc::hostnumerics {
// Convert a native numeric value without encoding it. Callers must specify
// integer rounding and overflow policies. For a non-Boolean integer destination T,
// the result is cast<T>(f_T(g(x))): g rounds TowardZero or NearestEven, leaving
// integral inputs unchanged; f_T applies Identity, Saturate, or ModuloWrap for T.
// Identity leaves g(x) unchanged, Saturate clamps it to T's range, and ModuloWrap
// reduces it modulo 2^N, where N is T's bit width, with a two's-complement
// interpretation for signed T.
// Identity requires g(x) to fit in T before the final cast. Input must not be NaN,
// and infinity requires Saturate. Complex-to-real input must have a zero
// imaginary component. Floating destinations use native casts, and Boolean
// conversion tests for a nonzero value; both ignore the integer policies.
// Policies must be valid when used. This function does not validate inputs or
// report conversion failures.
// For example, convertScalar<int8_t>(128.75, {IntegerRounding::TowardZero,
// IntegerOverflow::ModuloWrap}) returns -128. NearestEven with Identity maps 2.5
// to 2 and 3.5 to 4.
template <typename Target, typename Source>
Target convertScalar(Source source, const ScalarConversionOptions& options) noexcept {
    return detail::convertScalarValue<Target>(std::move(source), options);
}

}  // namespace roc::hostnumerics
