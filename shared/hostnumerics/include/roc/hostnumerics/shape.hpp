// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <optional>
#include <ranges>
#include <roc/hostnumerics/index_range.hpp>
#include <roc/hostnumerics/permutation.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace roc::hostnumerics {
// A Shape traversal callback, taking (flat index, coordinates) or only
// (coordinates). The two-argument form is used when both are available.
template <typename Function>
concept CoordinateVisitor = std::is_invocable_v<Function&, size_t, std::span<const size_t>> ||
                            std::is_invocable_v<Function&, std::span<const size_t>>;

// The logical shape of a tensor, independent of datatypes and storage strides.
// The element count must fit size_t.
class Shape {
   public:
    Shape() = default;

    explicit Shape(std::vector<size_t> dimensions) : m_dimensions(std::move(dimensions)) {
        const auto count = getElementCount(m_dimensions);
        if (!count) throw std::runtime_error("Tensor shape element count exceeds size_t.");
        m_elementCount = *count;
    }

    Shape(std::initializer_list<size_t> dimensions) : Shape(std::vector<size_t>(dimensions)) {}

    explicit Shape(std::span<const size_t> dimensions)
        : Shape(std::vector<size_t>(dimensions.begin(), dimensions.end())) {}

    Shape(const Shape&) = default;
    Shape& operator=(const Shape&) = default;

    // Moving leaves the source as a rank-zero scalar.
    Shape(Shape&& other) noexcept
        : m_dimensions(std::move(other.m_dimensions)),
          m_elementCount(std::exchange(other.m_elementCount, 1)) {
        other.m_dimensions.clear();
    }

    Shape& operator=(Shape&& other) noexcept {
        if (this != &other) {
            m_dimensions = std::move(other.m_dimensions);
            m_elementCount = std::exchange(other.m_elementCount, 1);
            other.m_dimensions.clear();
        }
        return *this;
    }

    size_t rank() const noexcept {
        return m_dimensions.size();
    }

    // The number of elements along the specified axis.
    size_t extent(size_t dimension) const {
        validateDimension(dimension);
        return m_dimensions[dimension];
    }

    // Requires dimension < rank().
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
    size_t linearIndex(std::span<const size_t> indices, const Permutation& order) const {
        validateLinearIndex(indices, order);
        return computeLinearIndex(indices, order);
    }

    size_t linearIndexLastDimensionFastest(std::span<const size_t> indices) const {
        return linearIndex(indices, Permutation::identity(rank()));
    }

    // Invert linearIndex with the same traversal order. The result stays in the
    // original axis order: Shape{2, 3, 5}.coordinates(27, (1, 2, 0)) is (1, 2, 3).
    // The index must lie in [0, elementCount()).
    std::vector<size_t> coordinates(size_t index, const Permutation& order) const {
        validateCoordinates(index, order);
        std::vector<size_t> result(rank());
        computeCoordinates(index, order, result);
        return result;
    }

    std::vector<size_t> coordinatesLastDimensionFastest(size_t index) const {
        return coordinates(index, Permutation::identity(rank()));
    }

    bool operator==(const Shape& other) const noexcept {
        return m_dimensions == other.m_dimensions;
    }

    // Visit each selected flat index with its coordinates, which use the original
    // axis numbering. Order lists every axis once, from slowest to fastest,
    // and defines how flat indices map to coordinates. Every overload validates
    // its inputs before the first callback, including empty work. The callback
    // takes (index, coordinates) or only (coordinates); the coordinate span is
    // borrowed and reused, and remains valid only during the callback. No
    // flat-index list is allocated.
    //
    // A range's stop must not exceed elementCount(). Clip it first with
    // std::min to visit only the elements that exist.
    template <CoordinateVisitor Function>
    void forEachCoordinate(const IndexRange& range, const Permutation& order,
                           Function&& function) const {
        validateOrder(order);
        if (range.stop() > elementCount())
            throw std::runtime_error("Index range stop " + std::to_string(range.stop()) +
                                     " exceeds the element count " +
                                     std::to_string(elementCount()) + ".");
        const size_t count = range.count();
        if (count == 0) return;
        std::vector<size_t> coordinates(rank());
        computeCoordinates(range.start(), order, coordinates);
        for (size_t ordinal = 0; ordinal < count; ++ordinal) {
            const size_t index = range.start() + ordinal * range.stride();
            if (range.stride() != 1) computeCoordinates(index, order, coordinates);
            visitCoordinate(function, index, coordinates);
            if (ordinal + 1 == count) break;
            if (range.stride() == 1) {
                for (size_t position = rank(); position > 0; --position) {
                    const size_t axis = order[position - 1];
                    if (++coordinates[axis] < m_dimensions[axis]) break;
                    coordinates[axis] = 0;
                }
            }
        }
    }

    // Explicit flat indices are visited in the order given, including repeats.
    // Every index must be less than elementCount(); all are checked before the
    // first callback, so the range is read twice.
    template <typename Indices, CoordinateVisitor Function>
        requires std::ranges::forward_range<const Indices> &&
                 std::same_as<std::ranges::range_value_t<const Indices>, size_t>
    void forEachCoordinate(const Indices& indices, const Permutation& order,
                           Function&& function) const {
        validateOrder(order);
        for (size_t index : indices)
            if (index >= elementCount())
                throw std::runtime_error("Tensor logical index " + std::to_string(index) +
                                         " is outside [0, " + std::to_string(elementCount()) +
                                         ").");
        std::vector<size_t> coordinates(rank());
        for (size_t index : indices) {
            computeCoordinates(index, order, coordinates);
            visitCoordinate(function, index, coordinates);
        }
    }

    // Visit every element.
    template <CoordinateVisitor Function>
    void forEachCoordinate(const Permutation& order, Function&& function) const {
        forEachCoordinate(IndexRange::all(elementCount()), order, function);
    }

    template <typename Indices, CoordinateVisitor Function>
    void forEachCoordinateLastDimensionFastest(const Indices& indices, Function&& function) const {
        forEachCoordinate(indices, Permutation::identity(rank()), function);
    }

    // Visit every element with the last axis fastest.
    template <CoordinateVisitor Function>
    void forEachCoordinateLastDimensionFastest(Function&& function) const {
        forEachCoordinate(Permutation::identity(rank()), function);
    }

   private:
    template <typename Function>
    static void visitCoordinate(Function& function, size_t index,
                                std::span<const size_t> coordinates) {
        if constexpr (std::is_invocable_v<Function&, size_t, std::span<const size_t>>)
            function(index, coordinates);
        else
            function(coordinates);
    }

    void validateDimension(size_t dimension) const {
        if (dimension >= rank())
            throw std::runtime_error("Tensor shape dimension " + std::to_string(dimension) +
                                     " is outside [0, " + std::to_string(rank()) + ").");
    }

    void validateDimensions(std::span<const size_t> axes) const {
        for (size_t axis : axes) validateDimension(axis);
    }

    void validateOrder(const Permutation& order) const {
        if (order.size() != rank())
            throw std::runtime_error("Tensor index order rank " + std::to_string(order.size()) +
                                     " does not match shape rank " + std::to_string(rank()) + ".");
    }

    void validateLinearIndex(std::span<const size_t> indices, const Permutation& order) const {
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

    void validateCoordinates(size_t index, const Permutation& order) const {
        validateOrder(order);
        if (index >= elementCount())
            throw std::runtime_error("Tensor logical index " + std::to_string(index) +
                                     " is outside [0, " + std::to_string(elementCount()) + ").");
    }

    size_t computeLinearIndex(std::span<const size_t> indices,
                              const Permutation& order) const noexcept {
        size_t result = 0;
        for (size_t position = 0; position < rank(); ++position) {
            const size_t axis = order[position];
            result = result * m_dimensions[axis] + indices[axis];
        }
        return result;
    }

    void computeCoordinates(size_t index, const Permutation& order,
                            std::span<size_t> result) const noexcept {
        for (size_t position = rank(); position > 0; --position) {
            const size_t axis = order[position - 1];
            result[axis] = index % m_dimensions[axis];
            index /= m_dimensions[axis];
        }
    }

    static std::optional<size_t> getElementCount(std::span<const size_t> dimensions) noexcept {
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

}  // namespace roc::hostnumerics
