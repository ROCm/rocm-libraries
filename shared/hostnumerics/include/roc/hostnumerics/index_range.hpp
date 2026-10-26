// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <roc/hostnumerics/detail/prime.hpp>
#include <stdexcept>
#include <vector>

namespace roc::hostnumerics {
// Flat indices first + ordinal * stride, with at most maxElements entries.
// Consumers supply an exclusive end bound; entries at or beyond it are omitted.
struct IndexRange {
    size_t first;
    size_t stride;
    size_t maxElements;

    IndexRange(size_t first, size_t stride, size_t maxElements)
        : first(first), stride(stride), maxElements(maxElements) {
        if (stride == 0) throw std::runtime_error("Index range stride must be nonzero.");
    }

    // A start at or beyond end yields no entries. Validate stride even for empty work.
    size_t count(size_t end) const {
        if (stride == 0) throw std::runtime_error("Index range stride must be nonzero.");
        if (first >= end) return 0;
        return std::min(1 + (end - 1 - first) / stride, maxElements);
    }

    // Allocate an explicit list only when requested; traversal can retain the range.
    std::vector<size_t> indices(size_t end) const {
        std::vector<size_t> result(count(end));
        for (size_t ordinal = 0; ordinal < result.size(); ++ordinal)
            result[ordinal] = first + ordinal * stride;
        return result;
    }

    static IndexRange all() {
        return {0, 1, std::numeric_limits<size_t>::max()};
    }
};

// Sampling retains the allocated-element quotient and approximate sample count.
inline IndexRange primeIndexRange(size_t logicalElements, size_t allocatedElements,
                                  size_t requestedElements) {
    if (requestedElements == 0 || requestedElements >= logicalElements) return IndexRange::all();
    const auto stride =
        detail::getNextPrime(std::max<size_t>(1, allocatedElements / requestedElements));
    if (!stride) throw std::runtime_error("Prime-stride index selection overflow.");
    return {0, *stride, std::numeric_limits<size_t>::max()};
}

}  // namespace roc::hostnumerics
