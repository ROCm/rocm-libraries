// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "detail/comparison_iteration.hpp"

#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>

namespace roc::host_numerics::detail {
ComparisonValue loadComparisonValue(const Tensor& view, ptrdiff_t logicalOffset) {
    const auto storage = view.rawEncodedBackingStorage();
    if (scalarTypeInfo(view.type()).category == ScalarCategory::Complex)
        return typedComparisonValue(
            decodeScalar<std::complex<double>>(view.type(), storage, logicalOffset));
    return typedComparisonValue(decodeScalar<double>(view.type(), storage, logicalOffset));
}

bool isUnwrittenSentinelValue(ScalarType type, const ComparisonValue& value) {
    const auto& info = scalarTypeInfo(type);
    switch (info.category) {
        case ScalarCategory::SignedInteger: {
            const double lowest = info.storageBits == 64
                                      ? static_cast<double>(std::numeric_limits<int64_t>::lowest())
                                      : -std::ldexp(1.0, info.storageBits - 1);
            return value.real == lowest;
        }
        case ScalarCategory::UnsignedInteger:
            return value.real == std::ldexp(1.0, info.storageBits) - 1.0;
        case ScalarCategory::FloatingPoint:
        case ScalarCategory::Scale:
            return info.supportsInfinity ? std::isinf(value.real) : std::isnan(value.real);
        case ScalarCategory::Complex:
            return std::isinf(value.real) && std::isinf(value.imaginary);
        case ScalarCategory::Boolean:
            return false;
    }
    return false;
}

void validateSentinelRange(ScalarType type, std::span<const std::byte> storage, size_t firstElement,
                           size_t elementCount) {
    const size_t storageBits = scalarTypeInfo(type).storageBits;
    if (storageBits == 0) throw std::invalid_argument("Sentinel scalar type has no storage.");
    if (firstElement > std::numeric_limits<size_t>::max() - elementCount)
        throw std::invalid_argument("Sentinel element range overflows.");

    const size_t endElement = firstElement + elementCount;
    if (endElement != 0 &&
        endElement - 1 > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
        throw std::invalid_argument("Sentinel element range exceeds supported element offsets.");
    if (endElement != 0 && storageBits > std::numeric_limits<size_t>::max() / endElement)
        throw std::invalid_argument("Sentinel element range overflows.");

    const size_t requiredBits = endElement * storageBits;
    const size_t requiredBytes = requiredBits / 8 + static_cast<size_t>(requiredBits % 8 != 0);
    if (requiredBytes > storage.size())
        throw std::invalid_argument(
            "Sentinel storage is too small for the requested element range.");
}
}  // namespace roc::host_numerics::detail
