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
// The logical shape of a tensor, independent of datatypes and storage strides.
//
// The number of elements stored in a Shape -- product of dimension sizes -- must fit in 64-bits.
class Shape {
   public:
    Shape() = default;

    explicit Shape(std::vector<size_t> dimensions)
        : m_dimensions(std::move(dimensions)), m_rank(m_dimensions.size()), m_elementCount(0) {
        std::optional<int64_t> count = tryElementCount(m_dimensions);
        if (!count.has_value()) {
            throw std::runtime_error("Shape dimensions are too large to compute element count.");
        }
        m_elementCount = count.value();
    }

    Shape(std::initializer_list<size_t> dimensions) : Shape(std::vector<size_t>(dimensions)) {}

    explicit Shape(std::span<const size_t> dimensions)
        : Shape(std::vector<size_t>(dimensions.begin(), dimensions.end())) {}

    size_t rank() const {
        return m_dimensions.size();
    }

    // The number of elements along the specified dimension, i.e. the size of that axis.
    size_t extent(size_t dimension) const {
        return m_dimensions.at(dimension);
    }

    size_t operator[](size_t dimension) const {
        return m_dimensions[dimension];
    }

    // The returned view remains valid while this Shape is alive and unmodified.
    std::span<const size_t> dimensions() const {
        return m_dimensions;
    }

    // The total number of elements in the tensor, i.e. the product of all dimensions of the shape.
    size_t elementCount() const {
        return m_elementCount;
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
        verifyIsPermutation(order);
        return getLinearIndex(indices, order);
    }

    size_t linearIndex(std::span<const size_t> indices) const {
        std::vector<size_t> order(rank());
        std::iota(order.begin(), order.end(), 0);
        return getLinearIndex(indices, order);
    }

    // Inverse of the bijection linearIndex().
    std::vector<size_t> coordinates(size_t linearIndex, std::span<const size_t> order) const {
        verifyIsPermutation(order);
        // Verify that linearIndex < elementCount().
        verifyLinearIndex(linearIndex);
        std::vector<size_t> result(rank(), 0);
        computeCoordinates(linearIndex, order, result);
        return result;
    }

    std::vector<size_t> coordinates(size_t linearIndex) const {
        verifyLinearIndex(linearIndex);
        std::vector<size_t> order(rank());
        std::iota(order.begin(), order.end(), 0);
        std::vector<size_t> result(rank(), 0);
        computeCoordinates(linearIndex, order, result);
        return result;
    }

    const bool operator==(const Shape& other) const {
        return m_dimensions == other.m_dimensions;
    }

    const bool operator!=(const Shape& other) const {
        return !(*this == other);
    }

   private:
    void validateDimensionRange(size_t firstDimension, size_t onePastLastDimension) const {
        if (firstDimension > onePastLastDimension || onePastLastDimension > rank())
            throw std::runtime_error("Invalid dimension range [" + std::to_string(firstDimension) +
                                     ", " + std::to_string(onePastLastDimension) +
                                     ") for shape of rank " + std::to_string(rank()) + ".");
    }

    void verifyLinearIndex(size_t linearIndex) const {
        if (linearIndex >= elementCount())
            throw std::runtime_error("Linear index " + std::to_string(linearIndex) +
                                     " is out of bounds for shape of rank " +
                                     std::to_string(rank()) + " and element count " +
                                     std::to_string(elementCount()) + ".");
    }

    size_t getLinearIndex(std::span<const size_t> indices, std::span<const size_t> order) const noexcept {
        size_t linearIndex = 0;
        size_t stride = 1;
        for (size_t position = rank(); position > 0; --position) {
            const size_t dimension = order.empty() ? position - 1 : order[position - 1];
            linearIndex += indices[dimension] * stride;
            stride *= m_dimensions[dimension];
        }
        return linearIndex;
    }

    static bool isPermutation(std::span<const size_t> order) {
        std::vector<bool> seen(order.size(), false);
        for (size_t dimension : order) {
            if (dimension >= order.size() || seen[dimension]) return false;
            seen[dimension] = true;
        }
        return true;
    }

    static void verifyIsPermutation(std::span<const size_t> order) {
        if (!isPermutation(order))
            throw std::runtime_error("Tensor index order must be a permutation of [0, " +
                                     std::to_string(order.size()) + ").");
    }

    // Validate raw extents against Shape's invariant that the product of non-zero dimensions fits
    // in 64-bits.
    static std::optional<size_t> tryElementCount(std::span<const size_t> dimensions) noexcept {
        if (std::any_of(dimensions.begin(), dimensions.end(),
                        [](size_t extent) { return extent == 0; })) {
            return 0;
        }
        size_t count = 1;
        for (size_t extent : dimensions) {
            if (count > std::numeric_limits<size_t>::max() / extent) return std::nullopt;
            count *= extent;
        }
        return count;
    }

    void computeCoordinates(size_t linearIndex, std::span<const size_t> order,
                            std::span<size_t> result) const noexcept {
        for (size_t position = rank(); position > 0; --position) {
            const size_t dimension = order.empty() ? position - 1 : order[position - 1];
            result[dimension] = linearIndex % m_dimensions[dimension];
            linearIndex /= m_dimensions[dimension];
        }
    }

    static size_t computeElementCount(std::span<const size_t> extents) noexcept {
        return std::accumulate(extents.begin(), extents.end(), size_t{1},
                               [](size_t count, size_t extent) { return count * extent; });
    }

    std::vector<size_t> m_dimensions;
    size_t m_rank;
    size_t m_elementCount;
};

}  // namespace roc::hostnumerics
