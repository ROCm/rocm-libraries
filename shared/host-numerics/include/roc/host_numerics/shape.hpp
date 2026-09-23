// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <roc/host_numerics/index_order.hpp>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace roc::host_numerics {
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

    std::span<const size_t> dimensions() const {
        return m_dimensions;
    }

    size_t elementCount() const {
        return elementCount(0, rank());
    }

    size_t elementCount(size_t firstDimension, size_t onePastLastDimension) const {
        if (firstDimension > onePastLastDimension || onePastLastDimension > rank())
            throw std::out_of_range("Tensor shape dimension range is invalid.");

        size_t count = 1;
        for (size_t dimension = firstDimension; dimension < onePastLastDimension; ++dimension)
            count = checkedElementProduct(count, extent(dimension));
        return count;
    }

    size_t elementCountExcluding(size_t excludedDimension) const {
        if (excludedDimension >= rank())
            throw std::out_of_range("Excluded tensor shape dimension is invalid.");

        size_t count = 1;
        for (size_t dimension = 0; dimension < rank(); ++dimension) {
            if (dimension != excludedDimension)
                count = checkedElementProduct(count, extent(dimension));
        }
        return count;
    }

    // Converts coordinates to a logical linear index in the selected order.
    // Throws std::invalid_argument for a rank mismatch and std::out_of_range
    // for any coordinate outside the shape.
    size_t linearIndex(std::span<const size_t> indices, IndexOrder order) const {
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
    // linearIndex is outside [0, elementCount()).
    void coordinates(size_t linearIndex, IndexOrder order, std::span<size_t> result) const {
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
    static size_t checkedElementProduct(size_t count, size_t extent) {
        if (extent == 0) return 0;
        if (count > std::numeric_limits<size_t>::max() / extent)
            throw std::overflow_error("Tensor shape element count overflow.");
        return count * extent;
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
            throw std::invalid_argument("Tensor shapes are not broadcast-compatible.");
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
            if (++indices[index] < shape.extent(index)) break;
            indices[index] = 0;
        }
    }
}
}  // namespace detail
}  // namespace roc::host_numerics
