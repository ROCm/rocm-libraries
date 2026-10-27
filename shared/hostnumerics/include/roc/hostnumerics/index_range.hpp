// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <roc/hostnumerics/detail/prime.hpp>
#include <stdexcept>
#include <vector>

namespace roc::hostnumerics {
// Flat indices start + ordinal * stride, all less than the exclusive end.
struct IndexRange {
    size_t start;
    size_t end;
    size_t stride;

    IndexRange(size_t start, size_t end, size_t stride) : start(start), end(end), stride(stride) {
        if (stride == 0) throw std::runtime_error("Index range stride must be nonzero.");
    }

    // A start at or beyond end yields no entries. Validate stride even for empty work.
    size_t count() const {
        if (stride == 0) throw std::runtime_error("Index range stride must be nonzero.");
        if (start >= end) return 0;
        return 1 + (end - 1 - start) / stride;
    }

    // Allocate an explicit list only when requested; traversal can retain the range.
    std::vector<size_t> indices() const {
        std::vector<size_t> result(count());
        for (size_t ordinal = 0; ordinal < result.size(); ++ordinal)
            result[ordinal] = start + ordinal * stride;
        return result;
    }

    static IndexRange all(size_t end) {
        return {0, end, 1};
    }
};

// Sampling retains the allocated-element quotient and approximate sample count.
inline IndexRange primeIndexRange(size_t logicalElements, size_t allocatedElements,
                                  size_t requestedElements) {
    if (requestedElements == 0 || requestedElements >= logicalElements)
        return IndexRange::all(logicalElements);
    const auto stride = detail::getNextPrime(allocatedElements / requestedElements);
    if (!stride) throw std::runtime_error("Prime-stride index selection overflow.");
    return {0, logicalElements, *stride};
}

}  // namespace roc::hostnumerics
