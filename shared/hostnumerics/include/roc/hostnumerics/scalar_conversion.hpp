// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/hostnumerics/scalar_integer_conversion.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace roc::hostnumerics::detail {
// No-options conversion to a native C++ integer truncates fractional values
// and defines out-of-range conversion by reducing modulo the destination width.
inline constexpr ScalarConversionOptions implicitNativeConversionOptions() {
    return {IntegerRounding::TowardZero, IntegerOverflow::ModuloWrap};
}

template <typename Target, typename Source>
Target convertScalarValue(Source source, const ScalarConversionOptions& options) {
    auto result = getConvertScalarValue<Target>(std::move(source), options);
    if (!result) throw std::runtime_error(std::string(message(result.error())));
    return *std::move(result);
}
}  // namespace roc::hostnumerics::detail
