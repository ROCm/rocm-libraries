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
struct ShapeAccess;

inline bool canBroadcastTo(std::span<const size_t> source,
                           std::span<const size_t> destination) noexcept {
    if (source.size() > destination.size()) return false;
    const size_t leading = destination.size() - source.size();
    for (size_t axis = 0; axis < source.size(); ++axis)
        if (source[axis] != 1 && source[axis] != destination[leading + axis]) return false;
    return true;
}

}  // namespace detail

// The logical shape of a tensor, independent of datatypes and storage strides.
// The element count must fit size_t.
class Shape {
   public:
    Shape() = default;

    explicit Shape(std::vector<size_t> dimensions) : m_dimensions(std::move(dimensions)) {
        const auto count = tryElementCount(m_dimensions);
        if (!count) throw std::runtime_error("Tensor shape element count exceeds size_t.");
        m_elementCount = *count;
    }

    Shape(std::initializer_list<size_t> dimensions) : Shape(std::vector<size_t>(dimensions)) {}

    explicit Shape(std::span<const size_t> dimensions)
        : Shape(std::vector<size_t>(dimensions.begin(), dimensions.end())) {}

    Shape(const Shape&) = default;
    Shape& operator=(const Shape&) = default;

    Shape(Shape&&) noexcept = default;
    Shape& operator=(Shape&&) noexcept = default;

    size_t rank() const noexcept {
        return m_dimensions.size();
    }

    // The number of elements along the specified axis.
    size_t extent(size_t dimension) const {
        validateDimension(dimension);
        return m_dimensions[dimension];
    }

    size_t operator[](size_t dimension) const noexcept {
        return m_dimensions[dimension];
    }

    // The returned view remains valid while this Shape is alive and unmodified.
    std::span<const size_t> dimensions() const noexcept {
        return m_dimensions;
    }

    size_t elementCount() const noexcept {
        return m_elementCount;
    }

    // Select dimensions in the supplied order; repetition is allowed.
    Shape fromDimensions(std::span<const size_t> axes) const {
        validateDimensions(axes);
        std::vector<size_t> dimensions;
        dimensions.reserve(axes.size());
        for (size_t axis : axes) dimensions.push_back(m_dimensions[axis]);
        return Shape(std::move(dimensions));
    }

    // Convert coordinates to a logical linear index.
    //
    // Parameter `order` lists axis numbers from slowest to fastest. For shape (2, 3, 5),
    // order (1, 2, 0) enumerates coordinates as:
    //
    //   size_t index = 0;
    //   for (size_t i1 = 0; i1 < 3; ++i1)
    //       for (size_t i2 = 0; i2 < 5; ++i2)
    //           for (size_t i0 = 0; i0 < 2; ++i0)
    //               visit(index++, i0, i1, i2);
    //
    // Axis 0 changes every step, axis 2 every 2 steps, and axis 1 every 2*5=10 steps.
    // logical strides are therefore (1, 10, 2) for the 3 dimensions.
    //
    // Other orders for the same shape:
    //   (0, 1, 2): last axis fastest, strides (15, 5, 1); (1, 2, 3) maps to 28.
    //   (2, 1, 0): first axis fastest, strides (1, 2, 6); (1, 2, 3) maps to 23.
    //   etc.
    //
    // The linear index is the inner product of `indices` and the logical strides.
    size_t linearIndex(std::span<const size_t> indices, std::span<const size_t> order) const {
        validateLinearIndex(indices, order);
        return computeLinearIndex(indices, order);
    }

    size_t linearIndex(std::span<const size_t> indices) const {
        std::vector<size_t> order(rank());
        std::iota(order.begin(), order.end(), 0);
        return linearIndex(indices, order);
    }

    // Invert linearIndex with the same traversal order. The result stays in the
    // original axis order: Shape{2, 3, 5}.coordinates(27, (1, 2, 0)) is (1, 2, 3).
    // The index must lie in [0, elementCount()).
    std::vector<size_t> coordinates(size_t index, std::span<const size_t> order) const {
        validateCoordinates(index, order);
        std::vector<size_t> result(rank());
        computeCoordinates(index, order, result);
        return result;
    }

    std::vector<size_t> coordinates(size_t index) const {
        std::vector<size_t> order(rank());
        std::iota(order.begin(), order.end(), 0);
        return coordinates(index, order);
    }

    bool operator==(const Shape& other) const noexcept {
        return m_dimensions == other.m_dimensions;
    }

   private:
    friend struct detail::ShapeAccess;

    void validateDimension(size_t dimension) const {
        if (dimension >= rank())
            throw std::runtime_error("Tensor shape dimension " + std::to_string(dimension) +
                                     " is outside [0, " + std::to_string(rank()) + ").");
    }

    void validateDimensions(std::span<const size_t> axes) const {
        for (size_t axis : axes) validateDimension(axis);
    }

    void validateOrder(std::span<const size_t> order) const {
        if (order.size() != rank())
            throw std::runtime_error("Tensor index order rank " + std::to_string(order.size()) +
                                     " does not match shape rank " + std::to_string(rank()) + ".");

        std::vector<bool> seen(rank(), false);
        for (size_t axis : order) {
            if (axis >= rank() || seen[axis])
                throw std::runtime_error("Tensor index order must be a permutation of [0, " +
                                         std::to_string(rank()) + ").");
            seen[axis] = true;
        }
    }

    void validateLinearIndex(std::span<const size_t> indices, std::span<const size_t> order) const {
        validateOrder(order);
        if (indices.size() != rank())
            throw std::runtime_error("Tensor coordinate rank " + std::to_string(indices.size()) +
                                     " does not match shape rank " + std::to_string(rank()) + ".");
        for (size_t axis = 0; axis < rank(); ++axis)
            if (indices[axis] >= m_dimensions[axis])
                throw std::runtime_error("Tensor coordinate " + std::to_string(indices[axis]) +
                                         " in dimension " + std::to_string(axis) +
                                         " is outside [0, " + std::to_string(m_dimensions[axis]) +
                                         ").");
    }

    void validateCoordinates(size_t index, std::span<const size_t> order) const {
        validateOrder(order);
        if (index >= elementCount())
            throw std::runtime_error("Tensor logical index " + std::to_string(index) +
                                     " is outside [0, " + std::to_string(elementCount()) + ").");
    }

    size_t computeLinearIndex(std::span<const size_t> indices,
                              std::span<const size_t> order) const noexcept {
        size_t result = 0;
        for (size_t position = 0; position < rank(); ++position) {
            const size_t axis = order[position];
            result = result * m_dimensions[axis] + indices[axis];
        }
        return result;
    }

    void computeCoordinates(size_t index, std::span<const size_t> order,
                            std::span<size_t> result) const noexcept {
        for (size_t position = rank(); position > 0; --position) {
            const size_t axis = order[position - 1];
            result[axis] = index % m_dimensions[axis];
            index /= m_dimensions[axis];
        }
    }

    static std::optional<size_t> tryElementCount(std::span<const size_t> dimensions) noexcept {
        if (std::find(dimensions.begin(), dimensions.end(), size_t{0}) != dimensions.end())
            return 0;
        size_t count = 1;
        for (size_t extent : dimensions) {
            if (count > std::numeric_limits<size_t>::max() / extent) return std::nullopt;
            count *= extent;
        }
        return count;
    }

    std::vector<size_t> m_dimensions;
    size_t m_elementCount = 1;
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
