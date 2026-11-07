// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <roc/hostnumerics/expected.hpp>
#include <roc/hostnumerics/permutation.hpp>
#include <roc/hostnumerics/shape.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace roc::hostnumerics {
namespace detail {
enum class LayoutError {
    ExtentOutOfRange,
    ContiguousStrideOverflow,
    OffsetMultiplyOverflow,
    OffsetAddOverflow,
    OrderRankMismatch,
    CoordinateRankMismatch,
    CoordinateOutOfRange,
};

// Rank failures carry the supplied and expected ranks in value and limit.
// Other failures identify an axis in dimension. Coordinate and extent failures
// also carry value and limit.
struct LayoutFailure {
    LayoutError error;
    size_t dimension = 0;
    size_t value = 0;
    size_t limit = 0;

    std::string message() const {
        switch (error) {
            case LayoutError::OrderRankMismatch:
                return "Tensor layout order rank " + std::to_string(value) +
                       " does not match shape rank " + std::to_string(limit) + ".";
            case LayoutError::ExtentOutOfRange:
                return "Tensor extent " + std::to_string(value) + " in dimension " +
                       std::to_string(dimension) + " exceeds ptrdiff_t.";
            case LayoutError::ContiguousStrideOverflow:
                return "Tensor contiguous stride overflow in dimension " +
                       std::to_string(dimension) + ".";
            case LayoutError::OffsetMultiplyOverflow:
                return "Tensor layout offset multiplication overflow in dimension " +
                       std::to_string(dimension) + ".";
            case LayoutError::OffsetAddOverflow:
                return "Tensor layout offset addition overflow in dimension " +
                       std::to_string(dimension) + ".";
            case LayoutError::CoordinateRankMismatch:
                return "Tensor coordinate rank " + std::to_string(value) +
                       " does not match shape rank " + std::to_string(limit) + ".";
            case LayoutError::CoordinateOutOfRange:
                return "Tensor coordinate " + std::to_string(value) + " in dimension " +
                       std::to_string(dimension) + " is outside [0, " + std::to_string(limit) +
                       ").";
            default:
                return "Unknown LayoutError failure.";
        }
    }
};

}  // namespace detail

// Maps logical coordinates to element offsets using a shape, strides and a
// starting offset. Offsets count elements. Strides may be negative or zero, so
// offsets can be negative and distinct coordinates can share one; Layout does
// not know the extent of the underlying storage.
class Layout {
   public:
    Layout() = default;
    Layout(const Layout&) = default;

    // Moving leaves the source as a scalar with zero offset.
    Layout(Layout&& other) noexcept
        : m_shape(std::move(other.m_shape)),
          m_strides(std::move(other.m_strides)),
          m_offset(std::exchange(other.m_offset, 0)) {
        other.m_strides.clear();
    }

    // Copy the metadata before replacing this value so an allocation failure
    // cannot leave shape and strides with different ranks.
    Layout& operator=(Layout other) noexcept {
        m_shape = std::move(other.m_shape);
        m_strides = std::move(other.m_strides);
        m_offset = other.m_offset;
        return *this;
    }

    // Throws unless there is one stride per shape dimension.
    Layout(Shape shape, std::vector<ptrdiff_t> strides, ptrdiff_t offset)
        : m_shape(std::move(shape)), m_strides(std::move(strides)), m_offset(offset) {
        if (m_shape.rank() != m_strides.size())
            throw std::runtime_error("Tensor stride count " + std::to_string(m_strides.size()) +
                                     " does not match shape rank " +
                                     std::to_string(m_shape.rank()) + ".");
    }

    Layout(Shape shape, std::initializer_list<ptrdiff_t> strides, ptrdiff_t offset)
        : Layout(std::move(shape), std::vector<ptrdiff_t>(strides), offset) {}

    Layout(Shape shape, std::span<const ptrdiff_t> strides, ptrdiff_t offset)
        : Layout(std::move(shape), std::vector<ptrdiff_t>(strides.begin(), strides.end()), offset) {
    }

    // The result has zero offset. Order lists axes from slowest to fastest, as in
    // Shape::linearIndex, so its last axis has unit stride. Every extent and
    // running stride product, including the final element count,
    // must fit ptrdiff_t. An empty shape can overflow before reaching its zero
    // extent, depending on order. Validation failures are returned; allocation
    // can still throw.
    static detail::Expected<Layout, detail::LayoutFailure> getContiguousLayout(
        const Shape& shape, const Permutation& order) {
        using namespace detail;
        if (order.size() != shape.rank())
            return Unexpected(
                LayoutFailure{LayoutError::OrderRankMismatch, 0, order.size(), shape.rank()});

        const auto dimensions = shape.dimensions();
        std::vector<ptrdiff_t> strides(shape.rank());
        ptrdiff_t stride = 1;
        for (size_t position = shape.rank(); position > 0; --position) {
            const size_t axis = order[position - 1];
            strides[axis] = stride;
            if (!std::in_range<ptrdiff_t>(dimensions[axis])) {
                return Unexpected(
                    LayoutFailure{LayoutError::ExtentOutOfRange, axis, dimensions[axis],
                                  static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())});
            }
            const auto next = getOffsetMultiply(dimensions[axis], stride);
            if (!next) {
                return Unexpected(LayoutFailure{LayoutError::ContiguousStrideOverflow, axis});
            }
            stride = *next;
        }
        return Layout(shape, std::move(strides), 0);
    }

    static detail::Expected<Layout, detail::LayoutFailure> getContiguousLayoutLastDimensionFastest(
        const Shape& shape) {
        return getContiguousLayout(shape, Permutation::identity(shape.rank()));
    }

    // Throwing forms of getContiguousLayout.
    static Layout contiguous(const Shape& shape, const Permutation& order) {
        auto result = getContiguousLayout(shape, order);
        if (!result) throw std::runtime_error(result.error().message());
        return std::move(*result);
    }

    // For a matrix this is row-major/C-order: the last dimension has unit stride.
    static Layout contiguousLastDimensionFastest(const Shape& shape) {
        return contiguous(shape, Permutation::identity(shape.rank()));
    }

    // For a matrix this is column-major/Fortran-order: the first dimension has unit stride.
    static Layout contiguousFirstDimensionFastest(const Shape& shape) {
        return contiguous(shape, Permutation::reverse(shape.rank()));
    }

    // Returned references/views remain valid while this Layout is alive and unmodified.
    const Shape& shape() const noexcept {
        return m_shape;
    }

    std::span<const ptrdiff_t> strides() const noexcept {
        return m_strides;
    }

    ptrdiff_t stride(size_t dimension) const {
        if (dimension >= m_shape.rank())
            throw std::runtime_error("Tensor layout dimension " + std::to_string(dimension) +
                                     " is outside [0, " + std::to_string(m_shape.rank()) + ").");
        return m_strides[dimension];
    }

    ptrdiff_t offset() const noexcept {
        return m_offset;
    }

    // Returns offset + sum(indices[d] * strides[d]). Fails if indices has the wrong
    // rank or a coordinate is out of range, or if any partial product or sum does
    // not fit ptrdiff_t, even when a later term would bring it back in range.
    detail::Expected<ptrdiff_t, detail::LayoutFailure> elementOffset(
        std::span<const size_t> indices) const noexcept {
        using namespace detail;
        if (indices.size() != m_shape.rank())
            return Unexpected(LayoutFailure{LayoutError::CoordinateRankMismatch, 0, indices.size(),
                                            m_shape.rank()});

        const auto dimensions = m_shape.dimensions();
        for (size_t dimension = 0; dimension < indices.size(); ++dimension) {
            if (indices[dimension] >= dimensions[dimension])
                return Unexpected(LayoutFailure{LayoutError::CoordinateOutOfRange, dimension,
                                                indices[dimension], dimensions[dimension]});
        }
        ptrdiff_t result = m_offset;
        for (size_t dimension = 0; dimension < indices.size(); ++dimension) {
            const auto delta = getOffsetMultiply(indices[dimension], m_strides[dimension]);
            if (!delta)
                return Unexpected(LayoutFailure{LayoutError::OffsetMultiplyOverflow, dimension});
            const auto sum = getOffsetAdd(result, *delta);
            if (!sum) return Unexpected(LayoutFailure{LayoutError::OffsetAddOverflow, dimension});
            result = *sum;
        }
        return result;
    }

    // Inclusive minimum and maximum reachable element offsets, for validating
    // storage once before an internal loop. Scalars return {offset, offset}; an
    // empty shape returns {0, -1}. Every intermediate product and sum must fit
    // ptrdiff_t. This query does not allocate or inspect a storage buffer.
    detail::Expected<std::pair<ptrdiff_t, ptrdiff_t>, detail::LayoutFailure> getElementBounds()
        const noexcept {
        using namespace detail;
        const auto dimensions = m_shape.dimensions();
        if (std::find(dimensions.begin(), dimensions.end(), size_t{0}) != dimensions.end())
            return std::pair<ptrdiff_t, ptrdiff_t>{0, -1};
        ptrdiff_t lower = m_offset;
        ptrdiff_t upper = lower;
        for (size_t axis = 0; axis < dimensions.size(); ++axis) {
            const auto delta = getOffsetMultiply(dimensions[axis] - 1, m_strides[axis]);
            if (!delta) return Unexpected(LayoutFailure{LayoutError::OffsetMultiplyOverflow, axis});
            const auto sum = getOffsetAdd(*delta < 0 ? lower : upper, *delta);
            if (!sum) return Unexpected(LayoutFailure{LayoutError::OffsetAddOverflow, axis});
            (*delta < 0 ? lower : upper) = *sum;
        }
        return std::pair{lower, upper};
    }

    bool operator==(const Layout&) const noexcept = default;

   private:
    // Returns value * factor if it fits ptrdiff_t. Plain multiplication would
    // convert factor to size_t and wrap silently, and a signed product can
    // overflow. Instead, compare unsigned magnitudes against the limit before
    // multiplying. The magnitude of a negative factor is computed as
    // -(factor + 1) + 1 because -PTRDIFF_MIN does not fit ptrdiff_t.
    static std::optional<ptrdiff_t> getOffsetMultiply(size_t value, ptrdiff_t factor) noexcept {
        if (value == 0 || factor == 0) return 0;
        const bool negative = factor < 0;
        const uintmax_t magnitude =
            negative ? static_cast<uintmax_t>(-(factor + 1)) + 1 : static_cast<uintmax_t>(factor);
        // A negative product can reach PTRDIFF_MIN, one past -PTRDIFF_MAX.
        const uintmax_t limit = static_cast<uintmax_t>(std::numeric_limits<ptrdiff_t>::max()) +
                                static_cast<uintmax_t>(negative);
        if (static_cast<uintmax_t>(value) > limit / magnitude) return std::nullopt;
        const uintmax_t product = static_cast<uintmax_t>(value) * magnitude;
        if (!negative) return static_cast<ptrdiff_t>(product);
        if (product == limit) return std::numeric_limits<ptrdiff_t>::min();
        return -static_cast<ptrdiff_t>(product);
    }

    static std::optional<ptrdiff_t> getOffsetAdd(ptrdiff_t left, ptrdiff_t right) noexcept {
        if ((right > 0 && left > std::numeric_limits<ptrdiff_t>::max() - right) ||
            (right < 0 && left < std::numeric_limits<ptrdiff_t>::min() - right))
            return std::nullopt;
        return left + right;
    }

    Shape m_shape;
    std::vector<ptrdiff_t> m_strides;
    ptrdiff_t m_offset = 0;
};

}  // namespace roc::hostnumerics
