// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/hostnumerics/scalar_conversion.hpp>
#include <utility>

namespace roc::hostnumerics {
// Convert a native numeric value without encoding it. Integer conversion
// truncates toward zero and wraps modulo the destination width when options
// are omitted. Explicit options reject overflow by default.
// Complex-to-real conversion rejects a nonzero imaginary component. Invalid
// conversions throw std::runtime_error. Boolean conversion tests for a nonzero
// real value. For other integer destinations, NaN always fails, and infinity
// is accepted only with saturation.
// For example, convertScalar<int8_t>(128.75) returns -128, while the same call
// with {} options throws. NearestEven with Reject maps 2.5 to 2 and 3.5 to 4.
template <typename Target, typename Source>
Target convertScalar(Source source) {
    return detail::convertScalarValue<Target>(std::move(source),
                                              detail::implicitNativeConversionOptions());
}

template <typename Target, typename Source>
Target convertScalar(Source source, const ScalarConversionOptions& options) {
    return detail::convertScalarValue<Target>(std::move(source), options);
}
}  // namespace roc::hostnumerics
