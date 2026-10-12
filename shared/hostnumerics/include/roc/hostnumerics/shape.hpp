// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <initializer_list>
#include <limits>
#include <roc/hostnumerics/index_order.hpp>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace roc::hostnumerics {
// Owns logical extents independently of a datatype, strides, or storage. A
// rank-zero shape describes one scalar; any zero extent describes an empty shape.
class Shape {
   public:
    Shape() = default;

    explicit Shape(std::vector<size_t> dimensions) : m_dimensions(std::move(dimensions)) {}

    Shape(std::initializer_list<size_t> dimensions) : m_dimensions(dimensions) {}

    explicit Shape(std::span<const size_t> dimensions)
        : m_dimensions(dimensions.begin(), dimensions.end()) {}

    size_t rank() const {
        return m_dimensions.size();
    }

    size_t operator[](size_t dimension) const {
        return extent(dimension);
    }

    size_t extent(size_t dimension) const {
        return m_dimensions.at(dimension);
    }

    // The returned view remains valid while this Shape is alive and unmodified.
    std::span<const size_t> dimensions() const {
        return m_dimensions;
    }

    size_t elementCount() const {
        return elementCount(0, rank());
    }

    // Products use [firstDimension, onePastLastDimension). An empty range has
    // product one; a zero extent makes the product zero even if other extents
    // would overflow size_t. Nonzero products throw on overflow.
    size_t elementCount(size_t firstDimension, size_t onePastLastDimension) const {
        if (firstDimension > onePastLastDimension || onePastLastDimension > rank())
            throw std::out_of_range("Tensor shape dimension range is invalid.");

        for (size_t dimension = firstDimension; dimension < onePastLastDimension; ++dimension)
            if (extent(dimension) == 0) return 0;

        size_t count = 1;
        for (size_t dimension = firstDimension; dimension < onePastLastDimension; ++dimension)
            count = checkedElementProduct(count, extent(dimension));
        return count;
    }

    size_t elementCountExcluding(size_t excludedDimension) const {
        if (excludedDimension >= rank())
            throw std::out_of_range("Excluded tensor shape dimension is invalid.");

        for (size_t dimension = 0; dimension < rank(); ++dimension)
            if (dimension != excludedDimension && extent(dimension) == 0) return 0;

        size_t count = 1;
        for (size_t dimension = 0; dimension < rank(); ++dimension) {
            if (dimension != excludedDimension)
                count = checkedElementProduct(count, extent(dimension));
        }
        return count;
    }

    // Converts coordinates to a logical linear index in the selected order.
    // Throws std::invalid_argument for a rank mismatch and std::out_of_range
    // for any coordinate outside the shape. Invalid orders are rejected; a
    // representable index can be returned even when the total count overflows.
    size_t linearIndex(std::span<const size_t> indices, IndexOrder order) const {
        validateIndexOrder(order);
        if (indices.size() != rank())
            throw std::invalid_argument("Tensor coordinate rank does not match shape.");

        size_t result = 0;
        const auto appendDimension = [&](size_t dimension) {
            if (indices[dimension] >= extent(dimension))
                throw std::out_of_range("Tensor coordinate exceeds shape.");
            result = checkedElementProduct(result, extent(dimension));
            if (indices[dimension] > std::numeric_limits<size_t>::max() - result)
                throw std::overflow_error("Tensor linear index overflow.");
            result += indices[dimension];
        };

        if (order == IndexOrder::LastDimensionFastest) {
            for (size_t dimension = 0; dimension < rank(); ++dimension) appendDimension(dimension);
        } else {
            for (size_t dimension = rank(); dimension > 0; --dimension)
                appendDimension(dimension - 1);
        }
        return result;
    }

    // Writes the coordinates for one logical linear index. Throws
    // std::invalid_argument for a rank mismatch and std::out_of_range when
    // linearIndex is outside [0, elementCount()). Invalid orders are rejected;
    // computing the element count throws if the shape is too large for size_t.
    void coordinates(size_t linearIndex, IndexOrder order, std::span<size_t> result) const {
        validateIndexOrder(order);
        if (result.size() != rank())
            throw std::invalid_argument("Tensor coordinate output rank does not match shape.");
        if (linearIndex >= elementCount())
            throw std::out_of_range("Tensor logical index exceeds shape.");

        if (order == IndexOrder::FirstDimensionFastest) {
            for (size_t dimension = 0; dimension < rank(); ++dimension) {
                result[dimension] = linearIndex % extent(dimension);
                linearIndex /= extent(dimension);
            }
        } else {
            for (size_t dimension = rank(); dimension > 0; --dimension) {
                const size_t index = dimension - 1;
                result[index] = linearIndex % extent(index);
                linearIndex /= extent(index);
            }
        }
    }

    std::vector<size_t> coordinates(size_t linearIndex, IndexOrder order) const {
        std::vector<size_t> result(rank(), 0);
        coordinates(linearIndex, order, result);
        return result;
    }

    friend bool operator==(const Shape&, const Shape&) = default;

   private:
    static void validateIndexOrder(IndexOrder order) {
        switch (order) {
            case IndexOrder::FirstDimensionFastest:
            case IndexOrder::LastDimensionFastest:
                return;
            case IndexOrder::Count:
                break;
        }
        throw std::invalid_argument("Invalid tensor index order.");
    }

    static size_t checkedElementProduct(size_t count, size_t extent) {
        if (extent == 0) return 0;
        if (count > std::numeric_limits<size_t>::max() / extent)
            throw std::overflow_error("Tensor shape element count overflow.");
        return count * extent;
    }

    std::vector<size_t> m_dimensions;
};

}  // namespace roc::hostnumerics
