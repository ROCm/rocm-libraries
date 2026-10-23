// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <optional>
#include <roc/hostnumerics/detail/layout_offset.hpp>
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

inline std::string LayoutFailure::message() const {
    switch (error) {
        case LayoutError::CoordinateRankMismatch:
            return "Tensor coordinate rank " + std::to_string(value) +
                   " does not match shape rank " + std::to_string(limit) + ".";
        case LayoutError::CoordinateOutOfRange:
            return "Tensor coordinate " + std::to_string(value) + " in dimension " +
                   std::to_string(dimension) + " is outside [0, " + std::to_string(limit) + ").";
        default:
            return std::string(detail::message(error));
    }
}

inline std::pair<ptrdiff_t, ptrdiff_t> elementBounds(const Layout& layout);
inline Expected<std::pair<ptrdiff_t, ptrdiff_t>, LayoutError> getElementBounds(
    const Layout& layout) noexcept;
}  // namespace detail

// Maps logical coordinates to element offsets using a shape, strides and a
// starting offset. Offsets count elements.
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
            const auto next = LayoutOffset::getOffsetMultiply(dimensions[axis], stride);
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
        return detail::LayoutOffset::getElementOffset(m_shape.dimensions(), m_strides, m_offset,
                                                      indices);
    }

    bool operator==(const Layout&) const noexcept = default;

   private:
    std::pair<ptrdiff_t, ptrdiff_t> checkedElementBounds() const;
    friend std::pair<ptrdiff_t, ptrdiff_t> detail::elementBounds(const Layout& layout);

    Shape m_shape;
    std::vector<ptrdiff_t> m_strides;
    ptrdiff_t m_offset = 0;
};

inline std::pair<ptrdiff_t, ptrdiff_t> Layout::checkedElementBounds() const {
    const auto result = detail::getElementBounds(*this);
    if (!result) throw std::runtime_error(std::string(message(result.error())));
    return *result;
}

namespace detail {
inline Expected<std::pair<ptrdiff_t, ptrdiff_t>, LayoutError> getElementBounds(
    const Layout& layout) noexcept {
    const auto dimensions = layout.shape().dimensions();
    if (std::ranges::find(dimensions, size_t{0}) != dimensions.end())
        return std::pair<ptrdiff_t, ptrdiff_t>{0, -1};
    ptrdiff_t lower = layout.offset();
    ptrdiff_t upper = lower;
    for (size_t axis = 0; axis < dimensions.size(); ++axis) {
        const auto delta =
            LayoutOffset::getOffsetMultiply(dimensions[axis] - 1, layout.strides()[axis]);
        if (!delta) {
            return Unexpected(LayoutError::OffsetMultiplyOverflow);
        }
        auto sum = LayoutOffset::getOffsetAdd(*delta < 0 ? lower : upper, *delta);
        if (!sum) {
            return Unexpected(LayoutError::OffsetAddOverflow);
        }
        (*delta < 0 ? lower : upper) = *sum;
    }
    return std::pair{lower, upper};
}

inline std::pair<ptrdiff_t, ptrdiff_t> elementBounds(const Layout& layout) {
    return layout.checkedElementBounds();
}

inline uint64_t strideMagnitude(ptrdiff_t stride) {
    if (stride >= 0) return static_cast<uint64_t>(stride);
    return static_cast<uint64_t>(-(stride + 1)) + 1;
}

inline bool hasProvablyDistinctElementOffsets(const Layout& layout) {
    if (layout.shape().elementCount() == 0) return true;

    std::vector<std::pair<uint64_t, size_t>> dimensions;
    dimensions.reserve(layout.shape().rank());
    for (size_t dimension = 0; dimension < layout.shape().rank(); ++dimension) {
        const size_t extent = layout.shape()[dimension];
        if (extent <= 1) continue;

        const uint64_t stride = strideMagnitude(layout.strides()[dimension]);
        if (stride == 0) return false;
        dimensions.emplace_back(stride, extent);
    }
    std::ranges::sort(dimensions);

    uint64_t addressedSpan = 1;
    for (const auto& [stride, extent] : dimensions) {
        if (stride < addressedSpan) return false;
        const uint64_t additionalExtent = static_cast<uint64_t>(extent - 1);
        if (additionalExtent > (std::numeric_limits<uint64_t>::max() - addressedSpan) / stride)
            return false;
        addressedSpan += additionalExtent * stride;
    }
    return true;
}

}  // namespace detail
}  // namespace roc::hostnumerics
