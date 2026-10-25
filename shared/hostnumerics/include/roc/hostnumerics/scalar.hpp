// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/hostnumerics/scalar_conversion.hpp>
#include <roc/hostnumerics/scalar_type.hpp>
#include <span>
#include <utility>

namespace roc::hostnumerics {
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

// Scalar conversion and encoded storage are header-only.
#include <roc/hostnumerics/scalar_storage.hpp>
