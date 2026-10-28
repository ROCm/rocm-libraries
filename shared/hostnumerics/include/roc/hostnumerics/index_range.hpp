// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <roc/hostnumerics/detail/prime.hpp>
#include <stdexcept>
#include <vector>

namespace roc::hostnumerics {
// Flat indices start + ordinal * stride, all less than the exclusive end. A start
// at or beyond end yields no entries.
class IndexRange {
   public:
    IndexRange(size_t start, size_t end, size_t stride)
        : m_start(start), m_end(end), m_stride(stride) {
        if (stride == 0) throw std::runtime_error("Index range stride must be nonzero.");
    }

    static IndexRange all(size_t end) {
        return {0, end, 1};
    }

    size_t start() const noexcept {
        return m_start;
    }

    size_t end() const noexcept {
        return m_end;
    }

    size_t stride() const noexcept {
        return m_stride;
    }

    size_t count() const noexcept {
        if (m_start >= m_end) return 0;
        return 1 + (m_end - 1 - m_start) / m_stride;
    }

    std::vector<size_t> indices() const {
        std::vector<size_t> result(count());
        for (size_t ordinal = 0; ordinal < result.size(); ++ordinal)
            result[ordinal] = m_start + ordinal * m_stride;
        return result;
    }

   private:
    size_t m_start;
    size_t m_end;
    size_t m_stride;
};

// Samples [0, logicalElements) with the smallest prime stride not less than
// allocatedElements / requestedElements, so the sample count is approximate.
// Zero requested elements selects no index; at least logicalElements selects
// every index. Throws if no such prime fits size_t.
inline IndexRange primeIndexRange(size_t logicalElements, size_t allocatedElements,
                                  size_t requestedElements) {
    if (requestedElements == 0) return IndexRange::all(0);
    if (requestedElements >= logicalElements) return IndexRange::all(logicalElements);
    const auto stride = detail::getNextPrime(allocatedElements / requestedElements);
    if (!stride) throw std::runtime_error("Prime-stride index selection overflow.");
    return {0, logicalElements, *stride};
}

}  // namespace roc::hostnumerics
