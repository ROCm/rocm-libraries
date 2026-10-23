// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/hostnumerics/scalar_integer_conversion.hpp>
#include <utility>

namespace roc::hostnumerics::detail {
// No-options conversion to a native C++ integer truncates fractional values
// and defines out-of-range conversion by reducing modulo the destination width.
inline constexpr ScalarConversionOptions implicitNativeConversionOptions() {
    return {IntegerRounding::TowardZero, IntegerOverflow::ModuloWrap};
}

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
        assert(source.imag() == typename Value::value_type{0} &&
               "Complex-to-real conversion requires a zero imaginary component.");
        return convertScalarValue<Destination>(source.real(), options);
    } else if constexpr (std::is_same_v<Destination, bool>) {
        static_assert(std::is_arithmetic_v<Value> || std::is_enum_v<Value>);
        return source != Value{0};
    } else if constexpr (std::is_integral_v<Destination>) {
        if constexpr (std::is_floating_point_v<Value>) {
            if (options.integerOverflow == IntegerOverflow::Reject) {
                assert(!std::isnan(source) && "Integer conversion requires a non-NaN value.");
                assert(std::isfinite(source) && "Infinity requires saturating integer conversion.");
                assert(getConvertToInteger<Destination>(source, options).has_value() &&
                       "Reject conversion requires a representable rounded value and valid "
                       "rounding policy.");
                // A native cast already truncates; do not round toward zero twice.
                return static_cast<Destination>(
                    options.integerRounding == IntegerRounding::TowardZero
                        ? source
                        : roundForIntegerConversion(source, options.integerRounding));
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
// Convert a native numeric value without encoding it. Integer conversion
// truncates toward zero and wraps modulo the destination width when options
// are omitted. With explicit Reject options, the rounded value must fit.
// Complex-to-real input must have a zero imaginary component. For non-Boolean
// integer destinations, input must not be NaN, and infinity requires Saturate.
// Policies must be valid when used. Preconditions apply even when assertions
// are disabled; this function does not report conversion failures.
// For example, convertScalar<int8_t>(128.75) returns -128. NearestEven with
// Reject maps 2.5 to 2 and 3.5 to 4. Boolean conversion tests for a nonzero value.
template <typename Target, typename Source>
Target convertScalar(Source source) noexcept {
    return detail::convertScalarValue<Target>(std::move(source),
                                              detail::implicitNativeConversionOptions());
}

template <typename Target, typename Source>
Target convertScalar(Source source, const ScalarConversionOptions& options) noexcept {
    return detail::convertScalarValue<Target>(std::move(source), options);
}

using ScalarConversionError = detail::ConversionError;
using ScalarConversionFailure = detail::ConversionFailure;

// Opt-in checked conversion. Returns a value or a typed failure without throwing
// or allocating. Explicit Reject options are the default: getConvertScalar<int8_t>(128)
// returns IntegerOutOfRange, while getConvertScalar<int8_t>(2.5) contains 2.
// Floating-to-floating conversion follows static_cast rather than checking precision loss.
template <typename Target, typename Source>
detail::Expected<Target, ScalarConversionFailure> getConvertScalar(
    Source source, const ScalarConversionOptions& options = {}) noexcept {
    return detail::getConvertScalarValue<Target>(std::move(source), options);
}
}  // namespace roc::hostnumerics
