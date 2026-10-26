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
    static_assert(std::is_arithmetic_v<Value> || RuntimeIsComplexV<Value>,
                  "Scalar conversion requires an arithmetic or std::complex source.");
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
        static_assert(std::is_floating_point_v<Destination>);
        return static_cast<Destination>(source);
    }
}
}  // namespace roc::hostnumerics::detail

namespace roc::hostnumerics {
// Convert a native numeric value. Encoded FP8/BF16 storage requires its codecs.
// This function returns a value without numerical validation or failure reporting.
//
// Types:
// Source and Target are C++ arithmetic types (integers up to 64 bits) or
// std::complex with a float, double, or long double component. Target cannot be a reference.
//
// Integer destinations:
// Apply rounding, then range handling, then the destination cast. Both policies
// must be supplied explicitly. See IntegerRounding and IntegerOverflow in
// scalar_conversion_options.hpp for policy behavior and integer-input preconditions.
// Policy enumerators must be valid when used.
//
// Other destinations:
// Floating destinations use native casts; finite input must lie within the
// destination's finite range. Floating and Boolean destinations ignore the
// integer policies. Complex destinations convert their components separately;
// a real source supplies a zero imaginary component. Conversion from complex to
// any non-complex Target, including bool, requires a zero imaginary component.
//
// Examples:
// const ScalarConversionOptions options{IntegerRounding::NearestEven,
//                                       IntegerOverflow::Identity};
// convertScalar<int>(3.5, options) == 4;
// convertScalar<double>(3.5, options) == 3.5;  // Floating values are not rounded to integers.
// convertScalar<bool>(0.25, options) == true;  // Boolean conversion tests for nonzero.
// convertScalar<int>(std::complex<double>{3.5, 0.0}, options) == 4;
// std::complex<double>{3.5, 1.0} is outside the domain for a non-complex Target.
template <typename Target, typename Source>
Target convertScalar(Source source, const ScalarConversionOptions& options) noexcept {
    return detail::convertScalarValue<Target>(std::move(source), options);
}

}  // namespace roc::hostnumerics
