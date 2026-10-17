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
// Validate raw extents against Shape's invariant before constructing a Shape.
inline std::optional<size_t> tryElementCount(std::span<const size_t> dimensions) noexcept {
    size_t count = 1;
    bool empty = false;
    for (size_t extent : dimensions) {
        if (extent == 0) {
            empty = true;
            continue;
        }
        if (count > std::numeric_limits<size_t>::max() / extent) return std::nullopt;
        count *= extent;
    }
    return empty ? 0 : count;
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

}  // namespace detail

// Owns logical extents independently of a datatype, strides, or storage. A
// rank-zero shape describes one scalar; any zero extent describes an empty shape.
// Construction requires the product of nonzero extents to fit in size_t;
// every subrange and projection therefore has a representable element count.
// Counts and index conversions perform checked, rank-dependent work. Reuse a
// computed count or an iteration plan in performance-critical element loops.
class Shape {
   public:
    Shape() = default;

    explicit Shape(std::vector<size_t> dimensions) : m_dimensions(std::move(dimensions)) {
        validateDimensions(m_dimensions);
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

    // Product over [firstDimension, onePastLastDimension); an empty product is one.
    size_t elementCount(size_t firstDimension, size_t onePastLastDimension) const {
        validateDimensionRange(firstDimension, onePastLastDimension);
        const auto extents =
            dimensions().subspan(firstDimension, onePastLastDimension - firstDimension);
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

    // Convert coordinates to a logical linear index, independently of physical
    // storage strides. Axis numbers are positions in dimensions(): for shape
    // (2, 3, 5), axes 0, 1, 2 have extents 2, 3, 5. Coordinates (i0, i1, i2)
    // always remain in that axis order, regardless of the traversal order.
    //
    // Order lists axis numbers from slowest to fastest (outermost to innermost
    // loop). For shape (2, 3, 5), order (1, 2, 0) enumerates coordinates as:
    //
    //   size_t index = 0;
    //   for (size_t i1 = 0; i1 < 3; ++i1)
    //       for (size_t i2 = 0; i2 < 5; ++i2)
    //           for (size_t i0 = 0; i0 < 2; ++i0)
    //               visit(index++, i0, i1, i2);
    //
    // The first coordinates are (0,0,0), (1,0,0), (0,0,1), (1,0,1), ... .
    // Axis 0 changes every step, axis 2 every 2 steps, and axis 1 every 10 steps.
    // Thus index = (i1 * 5 + i2) * 2 + i0 = i0 + 10 * i1 + 2 * i2;
    // logical strides in axis order are (1, 10, 2), and (1, 2, 3) maps to 27.
    //
    // Other orders for the same shape:
    //   (0, 1, 2): last axis fastest, strides (15, 5, 1); (1, 2, 3) maps to 28.
    //   (0, 2, 1): axis 1 fastest, strides (15, 1, 3); (1, 2, 3) maps to 26.
    //   (2, 1, 0): first axis fastest, strides (1, 2, 6); (1, 2, 3) maps to 23.
    //
    // For any three-axis order (a, b, c), the formula is
    // (indices[a] * dimensions[b] + indices[b]) * dimensions[c] + indices[c].
    // Empty order selects ascending axes, i.e. row-major with the last fastest.
    // \param indices One coordinate per axis, in the original axis order.
    // \param order A permutation of axis numbers, slowest to fastest.
    size_t linearIndex(std::span<const size_t> indices, std::span<const size_t> order = {}) const {
        validateLinearIndex(indices, order);
        size_t result = 0;
        for (size_t position = 0; position < rank(); ++position) {
            const size_t dimension = order.empty() ? position : order[position];
            result = result * m_dimensions[dimension] + indices[dimension];
        }
        return result;
    }

    // Invert linearIndex using the same traversal order. The output stays in
    // the original axis order: for shape (2, 3, 5), coordinates(27, (1, 2, 0))
    // returns (1, 2, 3). See linearIndex above for the nested-loop interpretation.
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

    static void validateDimensions(std::span<const size_t> extents) {
        if (!detail::tryElementCount(extents))
            throw std::runtime_error("Tensor shape product of nonzero extents exceeds size_t.");
    }

    static size_t computeElementCount(std::span<const size_t> extents) noexcept {
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

}  // namespace roc::hostnumerics
