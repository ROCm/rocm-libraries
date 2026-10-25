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
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace roc::hostnumerics {
enum class OutputSelectionKind {
    All,
    Strided,
    Explicit,
};

class OutputSelection;

namespace detail {
enum class OutputSelectionError { IndexOutOfRange };
inline constexpr std::string_view message(OutputSelectionError) noexcept {
    return "Explicit output selection index exceeds output shape.";
}
inline std::optional<OutputSelectionError> outputSelectionError(const OutputSelection& selection,
                                                                size_t logicalElements) noexcept;

}  // namespace detail

// Describes which logical tensor elements an operation or comparison should
// visit. Explicit indices are sorted and deduplicated at construction; indices
// and selectedCount check them against the supplied logical element count.
// As in Shape and Layout, orders list axes from slowest to fastest. An empty
// order resolves to ascending axes (C order) when a consumer supplies the rank.
// A nonempty order must contain each axis exactly once; the consumer validates
// it against that rank. Factories copy the supplied order.
class OutputSelection {
   public:
    static OutputSelection all(std::span<const size_t> indexOrder) {
        OutputSelection result;
        result.m_indexOrder.assign(indexOrder.begin(), indexOrder.end());
        return result;
    }

    static OutputSelection strided(size_t first, size_t stride, size_t maxElements,
                                   std::span<const size_t> indexOrder) {
        if (stride == 0) throw std::runtime_error("Output selection stride must be nonzero.");
        OutputSelection result;
        result.m_kind = OutputSelectionKind::Strided;
        result.m_first = first;
        result.m_stride = stride;
        result.m_maxElements = maxElements;
        result.m_indexOrder.assign(indexOrder.begin(), indexOrder.end());
        return result;
    }

    static OutputSelection explicitIndices(std::vector<size_t> indices,
                                           std::span<const size_t> indexOrder) {
        OutputSelection result;
        result.m_kind = OutputSelectionKind::Explicit;
        std::sort(indices.begin(), indices.end());
        indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
        result.m_indices = std::move(indices);
        result.m_indexOrder.assign(indexOrder.begin(), indexOrder.end());
        return result;
    }

    // Chooses the smallest prime stride at least allocatedElements /
    // requestedElements. The resulting sample size is approximate. Zero requests
    // and requests at least logicalElements select all outputs.
    static OutputSelection primeStride(size_t logicalElements, size_t allocatedElements,
                                       size_t requestedElements,
                                       std::span<const size_t> indexOrder) {
        if (requestedElements == 0 || requestedElements >= logicalElements) return all(indexOrder);
        const size_t candidate = std::max<size_t>(1, allocatedElements / requestedElements);
        const auto stride = detail::getNextPrime(candidate);
        if (!stride) throw std::runtime_error("Prime-stride output selection overflow.");
        return strided(0, *stride, std::numeric_limits<size_t>::max(), indexOrder);
    }

    OutputSelectionKind kind() const {
        return m_kind;
    }

    bool selectsAll() const {
        return m_kind == OutputSelectionKind::All;
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

    // Borrows this selection's order; valid while it is alive and unmodified.
    std::span<const size_t> indexOrder() const {
        return m_indexOrder;
    }

    std::vector<size_t> indices(size_t logicalElements) const {
        if (m_kind == OutputSelectionKind::All) {
            std::vector<size_t> result(logicalElements);
            for (size_t index = 0; index < logicalElements; ++index) result[index] = index;
            return result;
        }
        if (m_kind == OutputSelectionKind::Strided) {
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
        if (m_kind == OutputSelectionKind::All) return logicalElements;
        if (m_kind == OutputSelectionKind::Strided) {
            if (m_first >= logicalElements) return 0;
            return std::min(1 + (logicalElements - 1 - m_first) / m_stride, m_maxElements);
        }
        validateExplicitIndices(logicalElements);
        return m_indices.size();
    }

   private:
    friend std::optional<detail::OutputSelectionError> detail::outputSelectionError(
        const OutputSelection&, size_t) noexcept;

    void validateExplicitIndices(size_t logicalElements) const {
        if (auto error = detail::outputSelectionError(*this, logicalElements))
            throw std::runtime_error(std::string(detail::message(*error)));
    }

    OutputSelectionKind m_kind = OutputSelectionKind::All;
    size_t m_first = 0;
    size_t m_stride = 1;
    size_t m_maxElements = std::numeric_limits<size_t>::max();
    std::vector<size_t> m_indexOrder;
    std::vector<size_t> m_indices;
};
namespace detail {
// Checks normalized explicit indices without allocating or throwing.
inline std::optional<OutputSelectionError> outputSelectionError(const OutputSelection& selection,
                                                                size_t logicalElements) noexcept {
    if (selection.m_kind == OutputSelectionKind::Explicit && !selection.m_indices.empty() &&
        selection.m_indices.back() >= logicalElements)
        return OutputSelectionError::IndexOutOfRange;
    return std::nullopt;
}
}  // namespace detail
}  // namespace roc::hostnumerics
