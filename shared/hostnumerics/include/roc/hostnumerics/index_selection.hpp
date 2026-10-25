// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <roc/hostnumerics/detail/prime.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace roc::hostnumerics {
enum class IndexSelectionKind {
    All,
    Strided,
    Explicit,
};

class IndexSelection;

namespace detail {
enum class IndexSelectionError { IndexOutOfRange };
inline constexpr std::string_view message(IndexSelectionError) noexcept {
    return "Explicit index selection index exceeds logical element count.";
}
inline std::optional<IndexSelectionError> indexSelectionError(const IndexSelection& selection,
                                                              size_t logicalElements) noexcept;

}  // namespace detail

// Selects logical indices independently of tensor storage. Explicit indices are
// sorted and deduplicated; indices and selectedCount check their bounds.
// Orders list axes from slowest to fastest and must be permutations of [0, rank).
// Only the LastDimensionFastest factories choose an order without an argument.
class IndexSelection {
   public:
    static IndexSelection all(std::span<const size_t> indexOrder) {
        return IndexSelection(indexOrder);
    }

    static IndexSelection allLastDimensionFastest() {
        return IndexSelection();
    }

    static IndexSelection strided(size_t first, size_t stride, size_t maxElements,
                                  std::span<const size_t> indexOrder) {
        return makeStrided(first, stride, maxElements, all(indexOrder));
    }

    static IndexSelection stridedLastDimensionFastest(size_t first, size_t stride,
                                                      size_t maxElements) {
        return makeStrided(first, stride, maxElements, allLastDimensionFastest());
    }

    static IndexSelection explicitIndices(std::vector<size_t> indices,
                                          std::span<const size_t> indexOrder) {
        return makeExplicit(std::move(indices), all(indexOrder));
    }

    static IndexSelection explicitIndicesLastDimensionFastest(std::vector<size_t> indices) {
        return makeExplicit(std::move(indices), allLastDimensionFastest());
    }

    // Chooses the smallest prime stride at least allocatedElements /
    // requestedElements. The resulting sample size is approximate. Zero requests
    // and requests at least logicalElements select all indices.
    static IndexSelection primeStride(size_t logicalElements, size_t allocatedElements,
                                      size_t requestedElements,
                                      std::span<const size_t> indexOrder) {
        return makePrimeStride(logicalElements, allocatedElements, requestedElements,
                               all(indexOrder));
    }

    static IndexSelection primeStrideLastDimensionFastest(size_t logicalElements,
                                                          size_t allocatedElements,
                                                          size_t requestedElements) {
        return makePrimeStride(logicalElements, allocatedElements, requestedElements,
                               allLastDimensionFastest());
    }

    IndexSelectionKind kind() const {
        return m_kind;
    }

    bool selectsAll() const {
        return m_kind == IndexSelectionKind::All;
    }

    size_t first() const {
        return m_first;
    }

    size_t stride() const {
        return m_stride;
    }

    size_t maxElements() const {
        return m_maxElements;
    }

    // Returns the chosen order for this rank. An explicit empty order is valid
    // only for rank zero.
    std::vector<size_t> indexOrder(size_t rank) const {
        if (m_indexOrder) {
            if (m_indexOrder->size() != rank)
                throw std::runtime_error("Index selection order rank does not match tensor rank.");
            return *m_indexOrder;
        }
        std::vector<size_t> order(rank);
        std::iota(order.begin(), order.end(), size_t{0});
        return order;
    }

    std::vector<size_t> indices(size_t logicalElements) const {
        if (m_kind == IndexSelectionKind::All) {
            std::vector<size_t> result(logicalElements);
            for (size_t index = 0; index < logicalElements; ++index) result[index] = index;
            return result;
        }
        if (m_kind == IndexSelectionKind::Strided) {
            std::vector<size_t> result;
            if (m_first >= logicalElements || m_maxElements == 0) return result;
            const size_t available = 1 + (logicalElements - 1 - m_first) / m_stride;
            const size_t count = std::min(available, m_maxElements);
            result.reserve(count);
            for (size_t index = m_first; index < logicalElements && result.size() < count;) {
                result.push_back(index);
                if (index > std::numeric_limits<size_t>::max() - m_stride) break;
                index += m_stride;
            }
            return result;
        }
        validateExplicitIndices(logicalElements);
        return m_indices;
    }

    size_t selectedCount(size_t logicalElements) const {
        if (m_kind == IndexSelectionKind::All) return logicalElements;
        if (m_kind == IndexSelectionKind::Strided) {
            if (m_first >= logicalElements) return 0;
            return std::min(1 + (logicalElements - 1 - m_first) / m_stride, m_maxElements);
        }
        validateExplicitIndices(logicalElements);
        return m_indices.size();
    }

   private:
    friend std::optional<detail::IndexSelectionError> detail::indexSelectionError(
        const IndexSelection&, size_t) noexcept;

    IndexSelection() = default;

    explicit IndexSelection(std::span<const size_t> order)
        : m_indexOrder(std::vector<size_t>(order.begin(), order.end())) {
        std::vector<bool> seen(order.size(), false);
        for (size_t axis : order) {
            if (axis >= order.size() || seen[axis])
                throw std::runtime_error(
                    "Index selection order must be a permutation of [0, rank).");
            seen[axis] = true;
        }
    }

    static IndexSelection makeStrided(size_t first, size_t stride, size_t maxElements,
                                      IndexSelection result) {
        if (stride == 0) throw std::runtime_error("Index selection stride must be nonzero.");
        result.m_kind = IndexSelectionKind::Strided;
        result.m_first = first;
        result.m_stride = stride;
        result.m_maxElements = maxElements;
        return result;
    }

    static IndexSelection makeExplicit(std::vector<size_t> indices, IndexSelection result) {
        result.m_kind = IndexSelectionKind::Explicit;
        std::sort(indices.begin(), indices.end());
        indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
        result.m_indices = std::move(indices);
        return result;
    }

    static IndexSelection makePrimeStride(size_t logicalElements, size_t allocatedElements,
                                          size_t requestedElements, IndexSelection result) {
        if (requestedElements == 0 || requestedElements >= logicalElements) return result;
        const size_t candidate = std::max<size_t>(1, allocatedElements / requestedElements);
        const auto stride = detail::getNextPrime(candidate);
        if (!stride) throw std::runtime_error("Prime-stride index selection overflow.");
        return makeStrided(0, *stride, std::numeric_limits<size_t>::max(), std::move(result));
    }

    void validateExplicitIndices(size_t logicalElements) const {
        if (auto error = detail::indexSelectionError(*this, logicalElements))
            throw std::runtime_error(std::string(detail::message(*error)));
    }

    IndexSelectionKind m_kind = IndexSelectionKind::All;
    size_t m_first = 0;
    size_t m_stride = 1;
    size_t m_maxElements = std::numeric_limits<size_t>::max();
    // No stored permutation means the named LastDimensionFastest order.
    std::optional<std::vector<size_t>> m_indexOrder;
    std::vector<size_t> m_indices;
};
namespace detail {
// Checks normalized explicit indices without allocating or throwing.
inline std::optional<IndexSelectionError> indexSelectionError(const IndexSelection& selection,
                                                              size_t logicalElements) noexcept {
    if (selection.m_kind == IndexSelectionKind::Explicit && !selection.m_indices.empty() &&
        selection.m_indices.back() >= logicalElements)
        return IndexSelectionError::IndexOutOfRange;
    return std::nullopt;
}
}  // namespace detail
}  // namespace roc::hostnumerics
