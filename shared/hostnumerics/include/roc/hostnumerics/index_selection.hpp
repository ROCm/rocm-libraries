// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <roc/hostnumerics/detail/prime.hpp>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace roc::hostnumerics {
struct LastDimensionFastest {};

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

// Owns a flat range or sorted unique indices, and the order interpreting them.
// Shape supplies bounds, counting and coordinate visitation; Layout supplies storage offsets.
class IndexSelection {
   public:
    using FlatIndices = std::variant<IndexRange, std::vector<size_t>>;
    using Order = std::variant<LastDimensionFastest, std::vector<size_t>>;

    IndexSelection(FlatIndices indices, std::span<const size_t> order)
        : IndexSelection(std::move(indices), LastDimensionFastest{}) {
        std::vector<bool> seen(order.size(), false);
        for (size_t axis : order) {
            if (axis >= order.size() || seen[axis])
                throw std::runtime_error(
                    "Index selection order must be a permutation of [0, rank).");
            seen[axis] = true;
        }
        m_order = std::vector<size_t>(order.begin(), order.end());
    }

    IndexSelection(FlatIndices indices, LastDimensionFastest order)
        : m_indices(std::move(indices)), m_order(order) {
        if (auto* list = std::get_if<std::vector<size_t>>(&m_indices)) {
            std::sort(list->begin(), list->end());
            list->erase(std::unique(list->begin(), list->end()), list->end());
        } else if (std::get<IndexRange>(m_indices).stride == 0) {
            throw std::runtime_error("Index selection stride must be nonzero.");
        }
    }

    const FlatIndices& flatIndices() const noexcept {
        return m_indices;
    }
    const Order& order() const noexcept {
        return m_order;
    }

   private:
    FlatIndices m_indices;
    Order m_order;
};

namespace detail {
enum class IndexSelectionError { IndexOutOfRange };
inline constexpr std::string_view message(IndexSelectionError) noexcept {
    return "Explicit index selection index exceeds logical element count.";
}
inline std::optional<IndexSelectionError> indexSelectionError(const IndexSelection& selection,
                                                              size_t logicalElements) noexcept {
    const auto* list = std::get_if<std::vector<size_t>>(&selection.flatIndices());
    if (list && !list->empty() && list->back() >= logicalElements)
        return IndexSelectionError::IndexOutOfRange;
    return std::nullopt;
}
}  // namespace detail
}  // namespace roc::hostnumerics
