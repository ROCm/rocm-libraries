// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <optional>
#include <roc/hostnumerics/expected.hpp>
#include <roc/hostnumerics/shape.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace roc::hostnumerics {
class Layout;

namespace detail {
enum class LayoutError {
    ExtentOutOfRange,
    ContiguousStrideOverflow,
    OffsetMultiplyOverflow,
    OffsetAddOverflow,
    OrderRankMismatch,
    InvalidOrder,
    CoordinateRankMismatch,
    CoordinateOutOfRange,
};

inline constexpr std::string_view message(LayoutError error) noexcept {
    switch (error) {
        case LayoutError::ExtentOutOfRange:
            return "Tensor extent exceeds ptrdiff_t.";
        case LayoutError::ContiguousStrideOverflow:
            return "Tensor contiguous stride overflow.";
        case LayoutError::OffsetMultiplyOverflow:
            return "Tensor layout offset multiplication overflow.";
        case LayoutError::OffsetAddOverflow:
            return "Tensor layout offset addition overflow.";
        case LayoutError::OrderRankMismatch:
            return "Tensor layout order rank does not match shape rank.";
        case LayoutError::InvalidOrder:
            return "Tensor layout order must contain each dimension once.";
        case LayoutError::CoordinateRankMismatch:
            return "Tensor coordinate rank does not match shape rank.";
        case LayoutError::CoordinateOutOfRange:
            return "Tensor coordinate is outside the shape.";
    }
    return "Unknown LayoutError failure.";
}

struct LayoutFailure {
    LayoutError error;
    size_t dimension = 0;
    size_t value = 0;
    size_t limit = 0;

    // Formatting a diagnostic can allocate, so this operation can throw.
    std::string message() const {
        switch (error) {
            case LayoutError::CoordinateRankMismatch:
                return "Tensor coordinate rank " + std::to_string(value) +
                       " does not match shape rank " + std::to_string(limit) + ".";
            case LayoutError::CoordinateOutOfRange:
                return "Tensor coordinate " + std::to_string(value) + " in dimension " +
                       std::to_string(dimension) + " is outside [0, " + std::to_string(limit) +
                       ").";
            default:
                return std::string(detail::message(error));
        }
    }
};

}  // namespace detail
// Maps logical coordinates to element offsets using a shape, strides and a
// starting offset. Offsets count elements. The caller supplies the element type
// and memory, and is responsible for keeping the resulting offsets in bounds.
class Layout {
   public:
    Layout() = default;
    Layout(const Layout&) = default;
    Layout& operator=(const Layout&) = default;

    Layout(Layout&& other) noexcept
        : m_shape(std::move(other.m_shape)),
          m_strides(std::move(other.m_strides)),
          m_offset(std::exchange(other.m_offset, 0)) {
        other.m_strides.clear();
    }

    Layout& operator=(Layout&& other) noexcept {
        if (this != &other) {
            m_shape = std::move(other.m_shape);
            m_strides = std::move(other.m_strides);
            m_offset = std::exchange(other.m_offset, 0);
            other.m_strides.clear();
        }
        return *this;
    }

    // Requires one stride per shape dimension.
    Layout(Shape shape, std::vector<ptrdiff_t> strides, ptrdiff_t offset = 0)
        : m_shape(std::move(shape)), m_strides(std::move(strides)), m_offset(offset) {
        if (m_shape.rank() != m_strides.size())
            throw std::runtime_error("Tensor stride count " + std::to_string(m_strides.size()) +
                                     " does not match shape rank " +
                                     std::to_string(m_shape.rank()) + ".");
    }

    Layout(Shape shape, std::initializer_list<ptrdiff_t> strides, ptrdiff_t offset = 0)
        : Layout(std::move(shape), std::vector<ptrdiff_t>(strides), offset) {}

    Layout(Shape shape, std::span<const ptrdiff_t> strides, ptrdiff_t offset = 0)
        : Layout(std::move(shape), std::vector<ptrdiff_t>(strides.begin(), strides.end()), offset) {
    }

    // Order lists axes from slowest to fastest, as in Shape::linearIndex.
    // Validation failures are returned; allocating the owned metadata can throw.
    static detail::Expected<Layout, detail::LayoutError> getContiguousLayout(
        const Shape& shape, std::span<const size_t> order) {
        using namespace detail;
        if (order.size() != shape.rank()) return Unexpected(LayoutError::OrderRankMismatch);

        const auto dimensions = shape.dimensions();
        std::vector<ptrdiff_t> strides(shape.rank(), -1);
        // Mark axes in the output vector before replacing the marks with strides.
        for (size_t axis : order) {
            if (axis >= shape.rank() || strides[axis] != -1)
                return Unexpected(LayoutError::InvalidOrder);
            strides[axis] = 0;
        }
        ptrdiff_t stride = 1;
        for (size_t position = shape.rank(); position > 0; --position) {
            const size_t axis = order[position - 1];
            strides[axis] = stride;
            if (!std::in_range<ptrdiff_t>(dimensions[axis])) {
                return Unexpected(LayoutError::ExtentOutOfRange);
            }
            const auto next = getOffsetMultiply(dimensions[axis], stride);
            if (!next) {
                return Unexpected(LayoutError::ContiguousStrideOverflow);
            }
            stride = *next;
        }
        return Layout(shape, std::move(strides));
    }

    static detail::Expected<Layout, detail::LayoutError> getContiguousLayout(const Shape& shape) {
        std::vector<size_t> order(shape.rank());
        std::iota(order.begin(), order.end(), size_t{0});
        return getContiguousLayout(shape, order);
    }

    static Layout contiguous(const Shape& shape, std::span<const size_t> order) {
        auto result = getContiguousLayout(shape, order);
        if (!result) throw std::runtime_error(std::string(message(result.error())));
        return std::move(*result);
    }

    static Layout contiguous(const Shape& shape) {
        auto result = getContiguousLayout(shape);
        if (!result) throw std::runtime_error(std::string(message(result.error())));
        return std::move(*result);
    }

    // For a matrix this is row-major/C-order: the last dimension has unit stride.
    static Layout contiguousLastDimensionFastest(const Shape& shape) {
        return contiguous(shape);
    }

    // For a matrix this is column-major/Fortran-order: the first dimension has unit stride.
    static Layout contiguousFirstDimensionFastest(const Shape& shape) {
        std::vector<size_t> order(shape.rank());
        std::iota(order.rbegin(), order.rend(), size_t{0});
        return contiguous(shape, order);
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

    // offset + sum(indices[d] * strides[d]). Coordinate rank/range and each
    // multiplication/addition are checked, including intermediate overflow.
    // Failures retain numeric context without allocating or formatting a message.
    // Negative results and overlapping mappings are permitted without storage.
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
            if (!delta) return Unexpected(LayoutFailure{LayoutError::OffsetMultiplyOverflow});
            const auto sum = getOffsetAdd(result, *delta);
            if (!sum) return Unexpected(LayoutFailure{LayoutError::OffsetAddOverflow});
            result = *sum;
        }
        return result;
    }

    bool operator==(const Layout&) const noexcept = default;

   private:
    static std::optional<ptrdiff_t> getOffsetMultiply(size_t value, ptrdiff_t factor) noexcept {
        if (value == 0 || factor == 0) return 0;
        const bool negative = factor < 0;
        // -(factor + 1) is representable even for PTRDIFF_MIN. Add the missing one
        // after converting to unsigned, where the full magnitude is representable.
        const uintmax_t magnitude =
            negative ? static_cast<uintmax_t>(-(factor + 1)) + 1 : static_cast<uintmax_t>(factor);
        // A negative result can have magnitude PTRDIFF_MAX + 1 (PTRDIFF_MIN).
        const uintmax_t limit = static_cast<uintmax_t>(std::numeric_limits<ptrdiff_t>::max()) +
                                static_cast<uintmax_t>(negative);
        // Test against the bound before multiplying, so the product cannot wrap.
        if (static_cast<uintmax_t>(value) > limit / magnitude) return std::nullopt;
        const uintmax_t product = static_cast<uintmax_t>(value) * magnitude;
        if (!negative) return static_cast<ptrdiff_t>(product);
        // This magnitude has no positive ptrdiff_t representation to negate.
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
