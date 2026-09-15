// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/host_numerics/scalar_type.hpp>
#include <span>
#include <utility>

namespace roc::host_numerics {
namespace detail {
inline constexpr ScalarConversionOptions implicitNativeConversionOptions();

template <typename Target, typename Source>
Target convertScalarValue(Source source, const ScalarConversionOptions& options);

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

// Convert one native numeric value without encoding it. The no-options overload
// uses the same deterministic implicit policy as Tensor::item(): integer
// conversion truncates toward zero and wraps modulo the destination width.
// The option-bearing overload applies its explicit policy after rejecting a
// nonzero imaginary component in complex-to-real conversion.
template <typename Target, typename Source>
Target convertScalar(Source source) {
    return detail::convertScalarValue<Target>(std::move(source),
                                              detail::implicitNativeConversionOptions());
}

template <typename Target, typename Source>
Target convertScalar(Source source, const ScalarConversionOptions& options) {
    return detail::convertScalarValue<Target>(std::move(source), options);
}

}  // namespace roc::host_numerics

// Scalar conversion and encoded storage are header-only.
#include <roc/host_numerics/scalar_storage.hpp>
