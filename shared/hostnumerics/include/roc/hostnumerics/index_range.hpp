// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <roc/hostnumerics/detail/prime.hpp>
#include <stdexcept>

namespace roc::hostnumerics {
struct IndexRange {
    size_t first;
    size_t stride;
    size_t maxElements;

    IndexRange(size_t first, size_t stride, size_t maxElements)
        : first(first), stride(stride), maxElements(maxElements) {
        if (stride == 0) throw std::runtime_error("Index selection stride must be nonzero.");
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
