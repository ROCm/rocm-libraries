// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace roc::hostnumerics {
namespace detail {
inline std::optional<size_t> tryElementCount(std::span<const size_t> dimensions) noexcept {
    if (std::ranges::find(dimensions, size_t{0}) != dimensions.end()) return 0;
    size_t count = 1;
    for (size_t extent : dimensions) {
        if (count > std::numeric_limits<size_t>::max() / extent) return std::nullopt;
        count *= extent;
    }
    return count;
}

inline bool isValidIndexOrder(size_t rank, std::span<const size_t> order) {
    if (order.empty()) return true;
    if (order.size() != rank) return false;
    std::vector<bool> seen(rank, false);
    for (size_t axis : order) {
        if (axis >= rank || seen[axis]) return false;
        seen[axis] = true;
    }
    return true;
}

inline bool canBroadcastTo(std::span<const size_t> source,
                           std::span<const size_t> destination) noexcept {
    if (source.size() > destination.size()) return false;
    const size_t leading = destination.size() - source.size();
    for (size_t axis = 0; axis < source.size(); ++axis)
        if (source[axis] != 1 && source[axis] != destination[leading + axis]) return false;
    return true;
}
}  // namespace detail

// Owns logical extents independently of a datatype, strides, or storage. A
// rank-zero shape describes one scalar; any zero extent describes an empty shape.
// Construction requires the full element count to fit in size_t.
// Counts and index conversions perform checked, rank-dependent work. Reuse a
// computed count or an iteration plan in performance-critical element loops.
class Shape {
   public:
    Shape() = default;

    explicit Shape(std::vector<size_t> dimensions) : m_dimensions(std::move(dimensions)) {
        validateElementCount(m_dimensions);
    }

    Shape(std::initializer_list<size_t> dimensions) : Shape(std::vector<size_t>(dimensions)) {}

    explicit Shape(std::span<const size_t> dimensions)
        : Shape(std::vector<size_t>(dimensions.begin(), dimensions.end())) {}

    size_t rank() const {
        return m_dimensions.size();
    }

    size_t operator[](size_t dimension) const {
        return extent(dimension);
    }

    size_t extent(size_t dimension) const {
        validateDimension(dimension);
        return m_dimensions[dimension];
    }

    // The returned view remains valid while this Shape is alive and unmodified.
    std::span<const size_t> dimensions() const {
        return m_dimensions;
    }

    size_t elementCount() const {
        return computeElementCount(m_dimensions);
    }

    // Products use [firstDimension, onePastLastDimension). A zero extent makes
    // the product zero even if other extents would overflow size_t.
    size_t elementCount(size_t firstDimension, size_t onePastLastDimension) const {
        validateDimensionRange(firstDimension, onePastLastDimension);
        const auto extents =
            dimensions().subspan(firstDimension, onePastLastDimension - firstDimension);
        validateElementCount(extents);
        return computeElementCount(extents);
    }

    // Select dimensions in the supplied order.
    Shape project(std::span<const size_t> axes) const {
        validateProjection(axes);
        std::vector<size_t> dimensions;
        dimensions.reserve(axes.size());
        for (size_t axis : axes) dimensions.push_back(m_dimensions[axis]);
        return Shape(std::move(dimensions));
    }

    // Convert an n-dimensional coordinate to a logical linear index, independently
    // of physical strides. For shape (2, 3, 5), row-major strides are (15, 5, 1).
    // With order (1, 2, 0), strides are (1, 10, 2), so indices (1, 2, 3) give
    // 1 * 1 + 2 * 10 + 3 * 2 = 27.
    // \param indices One coordinate per dimension.
    // \param order A permutation of dimensions, slowest to fastest. Ascending
    // dimensions give row-major order (last fastest); empty selects this default.
    size_t linearIndex(std::span<const size_t> indices, std::span<const size_t> order = {}) const {
        validateLinearIndex(indices, order);
        size_t result = 0;
        for (size_t position = 0; position < rank(); ++position) {
            const size_t dimension = order.empty() ? position : order[position];
            result = result * m_dimensions[dimension] + indices[dimension];
        }
        return result;
    }

    // Invert linearIndex.
    // \param linearIndex Logical index in [0, elementCount()).
    // \param order A permutation of dimensions, slowest to fastest. Empty
    // selects row-major order (last fastest).
    // \param result Output with one entry per dimension.
    void coordinates(size_t linearIndex, std::span<const size_t> order,
                     std::span<size_t> result) const {
        validateCoordinates(linearIndex, order, result.size());
        computeCoordinates(linearIndex, order, result);
    }

    std::vector<size_t> coordinates(size_t linearIndex, std::span<const size_t> order = {}) const {
        validateCoordinates(linearIndex, order, rank());
        std::vector<size_t> result(rank(), 0);
        computeCoordinates(linearIndex, order, result);
        return result;
    }

    friend bool operator==(const Shape&, const Shape&) = default;

   private:
    void validateDimension(size_t dimension) const {
        if (dimension >= rank())
            throw std::runtime_error("Tensor shape dimension " + std::to_string(dimension) +
                                     " is outside [0, " + std::to_string(rank()) + ").");
    }

    void validateDimensionRange(size_t firstDimension, size_t onePastLastDimension) const {
        if (firstDimension > onePastLastDimension || onePastLastDimension > rank()) {
            const std::string reason = firstDimension > onePastLastDimension
                                           ? "the start exceeds the end"
                                           : "the end exceeds the shape rank";
            throw std::runtime_error(
                "Tensor shape dimension range [" + std::to_string(firstDimension) + ", " +
                std::to_string(onePastLastDimension) + ") is invalid for rank " +
                std::to_string(rank()) + ": " + reason + ".");
        }
    }

    void validateProjection(std::span<const size_t> axes) const {
        std::vector<bool> seen(rank(), false);
        for (size_t axis : axes) {
            validateDimension(axis);
            if (seen[axis])
                throw std::runtime_error("Tensor shape projection repeats dimension " +
                                         std::to_string(axis) + ".");
            seen[axis] = true;
        }
        for (size_t axis : axes)
            if (m_dimensions[axis] == 0) return;
        size_t count = 1;
        for (size_t axis : axes) {
            if (count > std::numeric_limits<size_t>::max() / m_dimensions[axis])
                throw std::runtime_error("Tensor shape element count exceeds size_t.");
            count *= m_dimensions[axis];
        }
    }

    void validateCoordinates(size_t linearIndex, std::span<const size_t> order,
                             size_t outputRank) const {
        validateIndexOrder(order);
        if (outputRank != rank())
            throw std::runtime_error("Tensor coordinate output rank " + std::to_string(outputRank) +
                                     " does not match shape rank " + std::to_string(rank()) + ".");
        const size_t count = elementCount();
        if (linearIndex >= count)
            throw std::runtime_error("Tensor logical index " + std::to_string(linearIndex) +
                                     " is outside [0, " + std::to_string(count) + ").");
    }

    void validateLinearIndex(std::span<const size_t> indices, std::span<const size_t> order) const {
        validateIndexOrder(order);
        if (indices.size() != rank())
            throw std::runtime_error("Tensor coordinate rank " + std::to_string(indices.size()) +
                                     " does not match shape rank " + std::to_string(rank()) + ".");
        for (size_t dimension = 0; dimension < rank(); ++dimension)
            if (indices[dimension] >= m_dimensions[dimension])
                throw std::runtime_error("Tensor coordinate " + std::to_string(indices[dimension]) +
                                         " in dimension " + std::to_string(dimension) +
                                         " is outside [0, " +
                                         std::to_string(m_dimensions[dimension]) + ").");
    }

    void computeCoordinates(size_t linearIndex, std::span<const size_t> order,
                            std::span<size_t> result) const noexcept {
        for (size_t position = rank(); position > 0; --position) {
            const size_t dimension = order.empty() ? position - 1 : order[position - 1];
            result[dimension] = linearIndex % m_dimensions[dimension];
            linearIndex /= m_dimensions[dimension];
        }
    }

    static void validateElementCount(std::span<const size_t> extents) {
        if (!detail::tryElementCount(extents))
            throw std::runtime_error("Tensor shape element count exceeds size_t.");
    }

    static size_t computeElementCount(std::span<const size_t> extents) noexcept {
        if (std::ranges::find(extents, size_t{0}) != extents.end()) return 0;
        return std::accumulate(extents.begin(), extents.end(), size_t{1},
                               [](size_t count, size_t extent) { return count * extent; });
    }

    void validateIndexOrder(std::span<const size_t> order) const {
        if (order.empty()) return;
        if (order.size() != rank())
            throw std::runtime_error("Tensor index order rank " + std::to_string(order.size()) +
                                     " does not match shape rank " + std::to_string(rank()) + ".");
        if (!detail::isValidIndexOrder(rank(), order))
            throw std::runtime_error("Tensor index order must be a permutation of [0, " +
                                     std::to_string(rank()) + ").");
    }

    std::vector<size_t> m_dimensions;
};

// Returns the NumPy-style broadcast shape of two inputs. Dimensions are
// aligned from the end and must be equal or have extent one. A zero extent
// broadcasts with one to zero, but not with any other extent.
inline Shape broadcastShapes(const Shape& left, const Shape& right) {
    const size_t resultRank = std::max(left.rank(), right.rank());
    std::vector<size_t> dimensions(resultRank, 1);
    for (size_t reverseDimension = 0; reverseDimension < resultRank; ++reverseDimension) {
        const size_t leftExtent =
            reverseDimension < left.rank() ? left[left.rank() - reverseDimension - 1] : 1;
        const size_t rightExtent =
            reverseDimension < right.rank() ? right[right.rank() - reverseDimension - 1] : 1;
        if (leftExtent != rightExtent && leftExtent != 1 && rightExtent != 1)
            throw std::runtime_error("Tensor shapes are not broadcast-compatible.");
        dimensions[resultRank - reverseDimension - 1] = leftExtent == 1 ? rightExtent : leftExtent;
    }
    return Shape(std::move(dimensions));
}

namespace detail {
template <typename Function>
void forEachIndex(const Shape& shape, Function&& function) {
    const size_t count = shape.elementCount();
    std::vector<size_t> indices(shape.rank(), 0);
    for (size_t linearIndex = 0; linearIndex < count; ++linearIndex) {
        function(std::span<const size_t>(indices), linearIndex);
        for (size_t dimension = shape.rank(); dimension > 0; --dimension) {
            const size_t index = dimension - 1;
            if (++indices[index] < shape.dimensions()[index]) break;
            indices[index] = 0;
        }
    }
}
}  // namespace detail
}  // namespace roc::hostnumerics
