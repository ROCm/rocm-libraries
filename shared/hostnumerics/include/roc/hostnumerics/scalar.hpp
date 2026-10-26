// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/hostnumerics/scalar_integer_conversion.hpp>
#include <roc/hostnumerics/scalar_type.hpp>
#include <span>

namespace roc::hostnumerics {
// Converts a numeric value to Target. Source and Target may each be bool, an
// integer of at most 64 bits, float, double, long double, or std::complex of
// one of those floating types.
//
// An integer Target receives the value rounded by options.integerRounding, with
// the range handled by options.integerOverflow. Other targets ignore the options:
// floating and Boolean targets behave like static_cast, and complex values
// convert each component.
//
// Conversion does not validate its input. The caller must meet the requirements
// of the chosen IntegerOverflow, keep finite values within a floating Target's
// range, and convert a complex value to a real Target only when its imaginary
// part is zero.
//
// Examples with options{IntegerRounding::NearestEven, IntegerOverflow::Saturate}:
//   convertScalar<int8_t>(double{2.5}, options) == 2
//   convertScalar<int8_t>(double{3.5}, options) == 4
//   convertScalar<int8_t>(int32_t{300}, options) == 127
//   convertScalar<float>(int32_t{3}, options) == 3.0f
//   convertScalar<bool>(double{0.25}, options) == true
//   convertScalar<int32_t>(std::complex<double>{3.5, 0.0}, options) == 4
template <typename Target, typename Source>
Target convertScalar(Source source, const ScalarConversionOptions& options) noexcept {
    using Destination = std::remove_cv_t<Target>;
    using Value = std::remove_cvref_t<Source>;
    static_assert(!std::is_reference_v<Target>);
    static_assert(std::is_arithmetic_v<Value> || detail::RuntimeIsComplexV<Value>,
                  "Scalar conversion requires an arithmetic or floating std::complex source.");
    if constexpr (detail::RuntimeIsComplexV<Destination>) {
        using Component = typename Destination::value_type;
        if constexpr (detail::RuntimeIsComplexV<Value>) {
            return Destination(convertScalar<Component>(source.real(), options),
                               convertScalar<Component>(source.imag(), options));
        } else {
            return Destination(convertScalar<Component>(source, options), Component{0});
        }
    } else if constexpr (detail::RuntimeIsComplexV<Value>) {
        return convertScalar<Destination>(source.real(), options);
    } else if constexpr (std::is_same_v<Destination, bool>) {
        return source != Value{0};
    } else if constexpr (std::is_integral_v<Destination>) {
        if constexpr (std::is_floating_point_v<Value>) {
            if (options.integerOverflow == IntegerOverflow::Identity) {
                // A native cast already truncates; only round explicitly for nearest-even.
                const Value rounded =
                    options.integerRounding == IntegerRounding::TowardZero
                        ? source
                        : detail::roundForIntegerConversion(source, options.integerRounding);
                return static_cast<Destination>(rounded);
            }
        }
        constexpr uint32_t bits = std::numeric_limits<std::make_unsigned_t<Destination>>::digits;
        // C++20 integer casts reduce modulo 2^N, including signed destinations.
        return static_cast<Destination>(
            detail::convertToIntegerBits(source, bits, std::is_signed_v<Destination>, options));
    } else {
        static_assert(std::is_floating_point_v<Destination>);
        return static_cast<Destination>(source);
    }
}

namespace detail {
template <typename Target>
Target decodeScalar(ScalarType type, std::span<const std::byte> storage, ptrdiff_t logicalOffset);

template <typename Target>
Target decodeScalar(ScalarType type, std::span<const std::byte> storage, ptrdiff_t logicalOffset,
                    const ScalarConversionOptions& options);

template <typename Source>
void encodeScalar(ScalarType type, std::span<std::byte> storage, ptrdiff_t logicalOffset,
                  Source source);

template <typename Source>
void encodeScalar(ScalarType type, std::span<std::byte> storage, ptrdiff_t logicalOffset,
                  Source source, const ScalarConversionOptions& options);
}  // namespace detail

}  // namespace roc::hostnumerics

#include <roc/hostnumerics/scalar_storage.hpp>
