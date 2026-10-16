// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <limits>
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

}  // namespace detail

// Owns logical extents independently of a datatype, strides, or storage. A
// rank-zero shape describes one scalar; any zero extent describes an empty shape.
// Counts and index conversions perform checked, rank-dependent work. Reuse a
// computed count or an iteration plan in performance-critical element loops.
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
        if (dimension >= rank())
            throw std::runtime_error("Tensor shape dimension " + std::to_string(dimension) +
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

    // Products use [firstDimension, onePastLastDimension). A zero extent makes
    // the product zero even if other extents would overflow size_t.
    size_t elementCount(size_t firstDimension, size_t onePastLastDimension) const {
        if (firstDimension > onePastLastDimension || onePastLastDimension > rank()) {
            const std::string reason = firstDimension > onePastLastDimension
                                           ? "the start exceeds the end"
                                           : "the end exceeds the shape rank";
            throw std::runtime_error(
                "Tensor shape dimension range [" + std::to_string(firstDimension) + ", " +
                std::to_string(onePastLastDimension) + ") is invalid for rank " +
                std::to_string(rank()) + ": " + reason + ".");
        }
        return checkedElementCount(firstDimension, onePastLastDimension);
    }

    // Select dimensions in the supplied order.
    Shape project(std::span<const size_t> axes) const {
        std::vector<size_t> dimensions;
        std::vector<bool> seen(rank(), false);
        dimensions.reserve(axes.size());
        for (size_t axis : axes) {
            const size_t value = extent(axis);
            if (seen[axis])
                throw std::runtime_error("Tensor shape projection repeats dimension " +
                                         std::to_string(axis) + ".");
            seen[axis] = true;
            dimensions.push_back(value);
        }
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
        validateIndexOrder(order);
        if (indices.size() != rank())
            throw std::runtime_error("Tensor coordinate rank " + std::to_string(indices.size()) +
                                     " does not match shape rank " + std::to_string(rank()) + ".");

        size_t result = 0;
        const auto appendDimension = [&](size_t dimension) {
            if (indices[dimension] >= extent(dimension))
                throw std::runtime_error("Tensor coordinate " + std::to_string(indices[dimension]) +
                                         " in dimension " + std::to_string(dimension) +
                                         " is outside [0, " + std::to_string(extent(dimension)) +
                                         ").");
            result =
                checkedMultiply(result, extent(dimension), "Tensor shape element count overflow.");
            if (indices[dimension] > std::numeric_limits<size_t>::max() - result)
                throw std::runtime_error("Tensor linear index overflow.");
            result += indices[dimension];
        };

        for (size_t position = 0; position < rank(); ++position)
            appendDimension(order.empty() ? position : order[position]);
        return result;
    }

    // Invert linearIndex.
    // \param linearIndex Logical index in [0, elementCount()).
    // \param order A permutation of dimensions, slowest to fastest. Empty
    // selects row-major order (last fastest).
    // \param result Output with one entry per dimension.
    void coordinates(size_t linearIndex, std::span<const size_t> order,
                     std::span<size_t> result) const {
        validateIndexOrder(order);
        if (result.size() != rank())
            throw std::runtime_error("Tensor coordinate output rank " +
                                     std::to_string(result.size()) + " does not match shape rank " +
                                     std::to_string(rank()) + ".");
        const size_t count = elementCount();
        if (linearIndex >= count)
            throw std::runtime_error("Tensor logical index " + std::to_string(linearIndex) +
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
        const auto count = detail::tryElementCount(
            dimensions().subspan(firstDimension, onePastLastDimension - firstDimension));
        if (!count) throw std::runtime_error("Tensor shape element count exceeds size_t.");
        return *count;
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

    static size_t checkedMultiply(size_t left, size_t right, const char* message) {
        if (right == 0) return 0;
        if (left > std::numeric_limits<size_t>::max() / right) throw std::runtime_error(message);
        return left * right;
    }

    std::vector<size_t> m_dimensions;
};

}  // namespace roc::hostnumerics
