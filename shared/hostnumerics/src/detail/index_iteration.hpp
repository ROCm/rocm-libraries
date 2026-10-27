// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <functional>
#include <numeric>
#include <roc/hostnumerics/detail/shape_access.hpp>
#include <roc/hostnumerics/layout.hpp>
#include <set>
#include <type_traits>
#include <vector>

namespace roc::hostnumerics::detail {
inline std::vector<size_t> validatedIndexOrder(size_t rank, std::span<const size_t> order) {
    std::vector<size_t> axes(order.begin(), order.end());
    if (axes.size() != rank)
        throw std::runtime_error("Tensor index order rank does not match shape rank.");
    if (!detail::ShapeAccess::isValidIndexOrder(rank, axes))
        throw std::runtime_error("Tensor index order must contain each dimension once.");
    return axes;
}

inline std::vector<size_t> ascendingIndexOrder(size_t rank) {
    std::vector<size_t> axes(rank);
    std::iota(axes.begin(), axes.end(), size_t{0});
    return axes;
}

inline std::vector<size_t> reverseIndexOrder(size_t rank) {
    std::vector<size_t> axes(rank);
    for (size_t i = 0; i < rank; ++i) axes[i] = rank - 1 - i;
    return axes;
}

inline std::function<size_t(size_t)> indexFunction(const IndexRange& range) {
    return [first = range.start(), stride = range.stride()](size_t ordinal) {
        return first + ordinal * stride;
    };
}
inline std::function<size_t(size_t)> indexFunction(const std::set<size_t>& indices) {
    return [list = std::vector<size_t>(indices.begin(), indices.end())](size_t ordinal) {
        return list[ordinal];
    };
}
inline std::optional<size_t> contiguousFirst(const IndexRange& range) {
    return range.stride() == 1 ? std::optional<size_t>{range.start()} : std::nullopt;
}
inline std::optional<size_t> contiguousFirst(const std::set<size_t>& indices) {
    if (indices.empty()) return std::nullopt;
    return *indices.rbegin() - *indices.begin() == indices.size() - 1
               ? std::optional<size_t>{*indices.begin()}
               : std::nullopt;
}

template <typename Invocation>
void bindIndices(Invocation& invocation, const Shape& shape, const IndexRange& indices,
                 std::span<const size_t> order) {
    invocation.indicesOrder = validatedIndexOrder(shape.rank(), order);
    const IndexRange clipped{indices.start(), std::min(indices.end(), shape.elementCount()),
                             indices.stride()};
    invocation.indicesCount = clipped.count();
    invocation.indexAt = indexFunction(indices);
    if constexpr (requires { invocation.firstContiguousIndex; })
        invocation.firstContiguousIndex = contiguousFirst(indices);
}

template <typename Invocation>
void bindIndices(Invocation& invocation, const Shape& shape, const std::set<size_t>& indices,
                 std::span<const size_t> order) {
    invocation.indicesOrder = validatedIndexOrder(shape.rank(), order);
    if (!indices.empty() && *indices.rbegin() >= shape.elementCount())
        throw std::runtime_error("Explicit index exceeds logical element count.");
    invocation.indicesCount = indices.size();
    invocation.indexAt = indexFunction(indices);
    if constexpr (requires { invocation.firstContiguousIndex; })
        invocation.firstContiguousIndex = contiguousFirst(indices);
}

inline Shape projectWithoutAxis(const Shape& shape, size_t excluded) {
    std::vector<size_t> axes;
    for (size_t axis = 0; axis < shape.rank(); ++axis)
        if (axis != excluded) axes.push_back(axis);
    return shape.fromDimensions(axes);
}

// Built at an invocation boundary. Subsequent methods require a valid index or
// coordinate from this plan's range and correctly sized output spans. These
// preconditions come from the operation's validated loop bounds.
class IndexIteration {
   public:
    explicit IndexIteration(const Shape& shape, std::span<const size_t> order)
        : m_extents(shape.dimensions().begin(), shape.dimensions().end()),
          m_axes(validatedIndexOrder(shape.rank(), order)),
          m_count(shape.elementCount()) {}

    size_t count() const {
        return m_count;
    }
    std::span<const size_t> axes() const {
        return m_axes;
    }

    void coordinates(size_t index, std::span<size_t> result) const noexcept {
        assert(result.size() == m_extents.size());
        assert(index < m_count);
        for (auto axis = m_axes.rbegin(); axis != m_axes.rend(); ++axis) {
            result[*axis] = index % m_extents[*axis];
            index /= m_extents[*axis];
        }
    }

    size_t linearIndex(std::span<const size_t> indices) const noexcept {
        assert(validCoordinates(indices));
        size_t result = 0;
        for (size_t axis : m_axes) result = result * m_extents[axis] + indices[axis];
        return result;
    }

    bool advance(std::span<size_t> indices) const noexcept {
        assert(validCoordinates(indices));
        for (auto axis = m_axes.rbegin(); axis != m_axes.rend(); ++axis) {
            if (++indices[*axis] < m_extents[*axis]) return true;
            indices[*axis] = 0;
        }
        return false;
    }

   private:
    bool validCoordinates(std::span<const size_t> indices) const noexcept {
        if (indices.size() != m_extents.size() || m_count == 0) return false;
        for (size_t axis = 0; axis < indices.size(); ++axis)
            if (indices[axis] >= m_extents[axis]) return false;
        return true;
    }

    std::vector<size_t> m_extents;
    std::vector<size_t> m_axes;
    size_t m_count;
};

template <typename Invocation, typename Indices>
void bindIndices(Invocation& invocation, const Shape& shape, const Indices& indices,
                 LastDimensionFastest) {
    bindIndices(invocation, shape, indices, shape.indexOrderLastDimensionFastest());
}

template <typename Invocation, typename Function>
void forEachBoundCoordinate(const Shape& shape, const Invocation& invocation, Function&& function) {
    const IndexIteration iteration(shape, invocation.indicesOrder);
    std::vector<size_t> coordinates(shape.rank());
    for (size_t ordinal = 0; ordinal < invocation.indicesCount; ++ordinal) {
        const size_t index = invocation.indexAt(ordinal);
        iteration.coordinates(index, coordinates);
        function(index, std::span<const size_t>(coordinates));
    }
}

// Tensor construction validates elementBounds and storage once. Unsigned
// arithmetic avoids signed intermediate overflow, including zero-stride axes
// with extents greater than PTRDIFF_MAX. The resulting address is representable
// and nonnegative for coordinates in that validated tensor.
inline ptrdiff_t preparedElementOffset(const Layout& layout,
                                       std::span<const size_t> indices) noexcept {
    assert(indices.size() == layout.shape().rank());
    using Unsigned = std::make_unsigned_t<ptrdiff_t>;
    Unsigned offset = static_cast<Unsigned>(layout.offset());
    const auto strides = layout.strides();
    for (size_t axis = 0; axis < indices.size(); ++axis) {
        assert(indices[axis] < layout.shape().dimensions()[axis]);
        offset += static_cast<Unsigned>(indices[axis]) * static_cast<Unsigned>(strides[axis]);
    }
    assert(offset <= static_cast<Unsigned>(std::numeric_limits<ptrdiff_t>::max()));
    return static_cast<ptrdiff_t>(offset);
}
}  // namespace roc::hostnumerics::detail
