// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <iterator>
#include <roc/hostnumerics/detail/prime.hpp>
#include <stdexcept>
#include <vector>

namespace roc::hostnumerics {
// Flat indices start + ordinal * stride, all less than the exclusive stop, as in
// Python's range(start, stop, step). A start at or beyond stop yields no entries.
class IndexRange {
   public:
    // Tracks the ordinal rather than the index, so advancing past the last entry
    // never computes an index beyond size_t.
    class Iterator {
       public:
        using iterator_concept = std::forward_iterator_tag;
        using iterator_category = std::input_iterator_tag;
        using value_type = size_t;
        using difference_type = std::ptrdiff_t;

        Iterator() = default;

        size_t operator*() const noexcept {
            return m_start + m_ordinal * m_stride;
        }

        Iterator& operator++() noexcept {
            ++m_ordinal;
            return *this;
        }

        Iterator operator++(int) noexcept {
            Iterator previous = *this;
            ++m_ordinal;
            return previous;
        }

        bool operator==(const Iterator& other) const noexcept {
            return m_ordinal == other.m_ordinal;
        }

       private:
        friend class IndexRange;

        Iterator(size_t start, size_t stride, size_t ordinal) noexcept
            : m_start(start), m_stride(stride), m_ordinal(ordinal) {}

        size_t m_start = 0;
        size_t m_stride = 1;
        size_t m_ordinal = 0;
    };

    // Explicit, so a braced list of indices is never read as a range.
    explicit IndexRange(size_t start, size_t stop, size_t stride)
        : m_start(start), m_stop(stop), m_stride(stride) {
        if (stride == 0) throw std::runtime_error("Index range stride must be nonzero.");
    }

    static IndexRange all(size_t stop) {
        return IndexRange(0, stop, 1);
    }

    size_t start() const noexcept {
        return m_start;
    }

    size_t stop() const noexcept {
        return m_stop;
    }

    size_t stride() const noexcept {
        return m_stride;
    }

    size_t count() const noexcept {
        if (m_start >= m_stop) return 0;
        return 1 + (m_stop - 1 - m_start) / m_stride;
    }

    Iterator begin() const noexcept {
        return Iterator(m_start, m_stride, 0);
    }

    Iterator end() const noexcept {
        return Iterator(m_start, m_stride, count());
    }

    std::vector<size_t> indices() const {
        return std::vector<size_t>(begin(), end());
    }

    bool operator==(const IndexRange&) const noexcept = default;

   private:
    size_t m_start;
    size_t m_stop;
    size_t m_stride;
};

// Every field is required. Designated initializers name the two counts at the
// call site, since they are easy to transpose.
struct PrimeIndexRangeOptions {
    size_t logicalElements;
    size_t requestedElements;
};

// Samples [0, logicalElements) with the smallest prime stride not less than
// logicalElements / requestedElements, so the sample count is approximate.
// Zero requested elements selects no index; at least logicalElements selects
// every index. Throws if no such prime fits size_t.
//
// A stride that shares a factor with a dimension keeps landing on the same few
// positions along it: in a matrix with 64 columns, stride 32 only ever visits
// columns 0 and 32. A prime stride shares a factor with a dimension only when
// the dimension is a multiple of it, so the samples usually spread across rows
// and columns. hipBLASLt's reference validation also uses a prime stride.
inline IndexRange primeIndexRange(const PrimeIndexRangeOptions& options) {
    if (options.requestedElements == 0) return IndexRange::all(0);
    if (options.requestedElements >= options.logicalElements)
        return IndexRange::all(options.logicalElements);
    const auto stride = detail::getNextPrime(options.logicalElements / options.requestedElements);
    if (!stride) throw std::runtime_error("Prime-stride index selection overflow.");
    return IndexRange(0, options.logicalElements, *stride);
}

}  // namespace roc::hostnumerics
