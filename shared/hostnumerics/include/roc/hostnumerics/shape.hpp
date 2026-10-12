// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <initializer_list>
#include <limits>
#include <roc/hostnumerics/index_order.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace roc::hostnumerics {
// Owns logical extents independently of a datatype, strides, or storage. A
// rank-zero shape describes one scalar; any zero extent describes an empty shape.
// Counts and index conversions perform checked, rank-dependent work. Reuse a
// computed count or an iteration plan in performance-critical element loops.
class Shape {
   public:
    Shape() = default;

    // Lvalue vectors are copied; rvalue vectors transfer their storage. Lists
    // and spans are copied too, so no constructor borrows the supplied extents.
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
        if (dimension >= rank())
            throw std::out_of_range("Tensor shape dimension " + std::to_string(dimension) +
                                    " is outside [0, " + std::to_string(rank()) + ").");
        return m_dimensions[dimension];
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
        if (firstDimension > onePastLastDimension || onePastLastDimension > rank()) {
            const std::string reason = firstDimension > onePastLastDimension
                                           ? "the start exceeds the end"
                                           : "the end exceeds the shape rank";
            throw std::out_of_range(
                "Tensor shape dimension range [" + std::to_string(firstDimension) + ", " +
                std::to_string(onePastLastDimension) + ") is invalid for rank " +
                std::to_string(rank()) + ": " + reason + ".");
        }
        return checkedElementCount(firstDimension, onePastLastDimension);
    }

    // Select retained dimensions in the supplied order. Empty selection yields
    // a scalar; repeated dimensions are rejected. Projection also expresses a
    // permutation or removal of dimensions without special product APIs.
    Shape project(std::span<const size_t> axes) const {
        std::vector<size_t> dimensions;
        std::vector<bool> seen(rank(), false);
        dimensions.reserve(axes.size());
        for (size_t axis : axes) {
            const size_t value = extent(axis);
            if (seen[axis])
                throw std::invalid_argument("Tensor shape projection repeats dimension " +
                                            std::to_string(axis) + ".");
            seen[axis] = true;
            dimensions.push_back(value);
        }
        return Shape(std::move(dimensions));
    }

    // Converts coordinates to a logical linear index in the selected order.
    // Throws std::invalid_argument for a rank mismatch and std::out_of_range
    // for any coordinate outside the shape. Orders list dimensions from slowest to fastest; an
    // empty order is row-major. Invalid permutations are rejected; a representable index can be
    // returned even when the total count overflows.
    size_t linearIndex(std::span<const size_t> indices, std::span<const size_t> order = {}) const {
        validateIndexOrder(order);
        if (indices.size() != rank())
            throw std::invalid_argument("Tensor coordinate rank " + std::to_string(indices.size()) +
                                        " does not match shape rank " + std::to_string(rank()) +
                                        ".");

        size_t result = 0;
        const auto appendDimension = [&](size_t dimension) {
            if (indices[dimension] >= extent(dimension))
                throw std::out_of_range("Tensor coordinate " + std::to_string(indices[dimension]) +
                                        " in dimension " + std::to_string(dimension) +
                                        " is outside [0, " + std::to_string(extent(dimension)) +
                                        ").");
            result = checkedElementProduct(result, extent(dimension));
            if (indices[dimension] > std::numeric_limits<size_t>::max() - result)
                throw std::overflow_error("Tensor linear index overflow.");
            result += indices[dimension];
        };

        for (size_t position = 0; position < rank(); ++position)
            appendDimension(order.empty() ? position : order[position]);
        return result;
    }

    // Writes the coordinates for one logical linear index. Throws
    // std::invalid_argument for a rank mismatch and std::out_of_range when
    // linearIndex is outside [0, elementCount()). Orders list dimensions from slowest to fastest;
    // an empty order is row-major. Invalid permutations are rejected; computing the element count
    // throws if the shape is too large for size_t.
    void coordinates(size_t linearIndex, std::span<const size_t> order,
                     std::span<size_t> result) const {
        validateIndexOrder(order);
        if (result.size() != rank())
            throw std::invalid_argument(
                "Tensor coordinate output rank " + std::to_string(result.size()) +
                " does not match shape rank " + std::to_string(rank()) + ".");
        const size_t count = elementCount();
        if (linearIndex >= count)
            throw std::out_of_range("Tensor logical index " + std::to_string(linearIndex) +
                                    " is outside [0, " + std::to_string(count) + ").");

        for (size_t position = rank(); position > 0; --position) {
            const size_t dimension = order.empty() ? position - 1 : order[position - 1];
            result[dimension] = linearIndex % m_dimensions[dimension];
            linearIndex /= m_dimensions[dimension];
        }
    }

    std::vector<size_t> coordinates(size_t linearIndex, std::span<const size_t> order = {}) const {
        std::vector<size_t> result(rank(), 0);
        coordinates(linearIndex, order, result);
        return result;
    }

    friend bool operator==(const Shape&, const Shape&) = default;

   private:
    size_t checkedElementCount(size_t firstDimension, size_t onePastLastDimension) const {
        size_t count = 1;
        bool overflow = false;
        for (size_t dimension = firstDimension; dimension < onePastLastDimension; ++dimension) {
            const size_t nextExtent = m_dimensions[dimension];
            if (nextExtent == 0) return 0;
            overflow = overflow || count > std::numeric_limits<size_t>::max() / nextExtent;
            if (!overflow) count *= nextExtent;
        }
        // Defer the error until all extents have been seen: a later zero makes
        // the exact product zero even if a preceding partial product overflowed.
        if (overflow) throw std::overflow_error("Tensor shape element count exceeds size_t.");
        return count;
    }

    void validateIndexOrder(std::span<const size_t> order) const {
        if (order.empty()) return;
        if (order.size() != rank())
            throw std::invalid_argument("Tensor index order rank " + std::to_string(order.size()) +
                                        " does not match shape rank " + std::to_string(rank()) +
                                        ".");
        std::vector<bool> seen(rank(), false);
        for (size_t axis : order) {
            if (axis >= rank() || seen[axis])
                throw std::invalid_argument("Tensor index order must be a permutation of [0, " +
                                            std::to_string(rank()) + ").");
            seen[axis] = true;
        }
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
