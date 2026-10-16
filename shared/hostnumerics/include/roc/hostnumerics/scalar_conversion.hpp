// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/hostnumerics/scalar_integer_conversion.hpp>
#include <roc/hostnumerics/scalar_type.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace roc::hostnumerics::detail {
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

template <typename Source>
uint64_t convertToIntegerBits(Source source, uint32_t bits, bool signedDestination,
                              const ScalarConversionOptions& options) {
    const auto result = tryConvertToIntegerBits(source, bits, signedDestination, options);
    if (!result) throw std::runtime_error(std::string(message(result.error())));
    return *result;
}

template <typename Target, typename Source>
Target convertScalarValue(Source source, const ScalarConversionOptions& options) {
    auto result = tryConvertScalarValue<Target>(std::move(source), options);
    if (!result) throw std::runtime_error(std::string(message(result.error())));
    return *std::move(result);
}
}  // namespace roc::hostnumerics::detail
