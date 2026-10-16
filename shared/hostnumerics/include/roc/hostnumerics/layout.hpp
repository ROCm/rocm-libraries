// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <roc/hostnumerics/result.hpp>
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
    }
    return "Unknown LayoutError failure.";
}

inline std::pair<ptrdiff_t, ptrdiff_t> elementBounds(const Layout& layout);
inline Expected<Layout, LayoutError> tryContiguousLayout(const Shape& shape,
                                                         bool firstDimensionFastest);
inline std::optional<ptrdiff_t> tryOffsetMultiply(size_t value, ptrdiff_t factor) noexcept {
    if (value == 0 || factor == 0) return 0;
    const bool negative = factor < 0;
    const uintmax_t magnitude =
        negative ? static_cast<uintmax_t>(-(factor + 1)) + 1 : static_cast<uintmax_t>(factor);
    const uintmax_t limit = static_cast<uintmax_t>(std::numeric_limits<ptrdiff_t>::max()) +
                            static_cast<uintmax_t>(negative);
    if (static_cast<uintmax_t>(value) > limit / magnitude) return std::nullopt;
    const uintmax_t product = static_cast<uintmax_t>(value) * magnitude;
    if (!negative) return static_cast<ptrdiff_t>(product);
    if (product == limit) return std::numeric_limits<ptrdiff_t>::min();
    return -static_cast<ptrdiff_t>(product);
}
inline std::optional<ptrdiff_t> tryOffsetAdd(ptrdiff_t left, ptrdiff_t right) noexcept {
    if ((right > 0 && left > std::numeric_limits<ptrdiff_t>::max() - right) ||
        (right < 0 && left < std::numeric_limits<ptrdiff_t>::min() - right))
        return std::nullopt;
    return left + right;
}
inline Expected<std::pair<ptrdiff_t, ptrdiff_t>, LayoutError> tryElementBounds(
    const Layout& layout) noexcept;
}  // namespace detail

// Maps logical coordinates to signed element offsets, independently of datatype
// or storage. Strides and the starting offset count elements, not bytes. Zero
// strides represent repeated addresses; negative strides represent reversal.
// This value owns its Shape and strides and does not check allocation bounds.
// elementOffset is a checked user-facing operation; production element loops
// should reuse a prepared mapping after validating their inputs once.
class Layout {
   public:
    Layout() = default;

    // Lvalue vectors are copied; rvalue vectors transfer their storage. Lists
    // and spans are copied too, so no constructor borrows the supplied strides.
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

    // For a matrix this is row-major/C-order: the last dimension has unit stride.
    static Layout contiguousLastDimensionFastest(const Shape& shape) {
        return contiguous(shape, false);
    }

    // For a matrix this is column-major/Fortran-order: the first dimension has unit stride.
    static Layout contiguousFirstDimensionFastest(const Shape& shape) {
        return contiguous(shape, true);
    }

    // Returned references/views remain valid while this Layout is alive and unmodified.
    const Shape& shape() const {
        return m_shape;
    }

    std::span<const ptrdiff_t> strides() const {
        return m_strides;
    }

    ptrdiff_t stride(size_t dimension) const {
        if (dimension >= m_shape.rank())
            throw std::runtime_error("Tensor layout dimension " + std::to_string(dimension) +
                                     " is outside [0, " + std::to_string(m_shape.rank()) + ").");
        return m_strides[dimension];
    }

    ptrdiff_t offset() const {
        return m_offset;
    }

    // offset + sum(indices[d] * strides[d]). Coordinate rank/range and each
    // multiplication/addition are checked, including intermediate overflow.
    // Negative results and overlapping mappings are permitted without storage.
    ptrdiff_t elementOffset(std::span<const size_t> indices) const {
        if (indices.size() != m_shape.rank())
            throw std::runtime_error("Tensor coordinate rank " + std::to_string(indices.size()) +
                                     " does not match shape rank " +
                                     std::to_string(m_shape.rank()) + ".");

        const auto dimensions = m_shape.dimensions();
        ptrdiff_t result = m_offset;
        for (size_t dimension = 0; dimension < indices.size(); ++dimension) {
            if (indices[dimension] >= dimensions[dimension])
                throw std::runtime_error("Tensor coordinate " + std::to_string(indices[dimension]) +
                                         " in dimension " + std::to_string(dimension) +
                                         " is outside [0, " +
                                         std::to_string(dimensions[dimension]) + ").");
            const ptrdiff_t delta = checkedMultiply(indices[dimension], m_strides[dimension]);
            result = checkedAdd(result, delta);
        }
        return result;
    }

    friend bool operator==(const Layout&, const Layout&) = default;

   private:
    static Layout contiguous(const Shape& shape, bool firstDimensionFastest) {
        auto result = detail::tryContiguousLayout(shape, firstDimensionFastest);
        if (!result) throw std::runtime_error(std::string(message(result.error())));
        return std::move(*result);
    }

    static ptrdiff_t checkedMultiply(size_t value, ptrdiff_t factor) {
        const auto result = detail::tryOffsetMultiply(value, factor);
        if (!result) throw std::runtime_error("Tensor layout offset multiplication overflow.");
        return *result;
    }

    static ptrdiff_t checkedAdd(ptrdiff_t left, ptrdiff_t right) {
        const auto result = detail::tryOffsetAdd(left, right);
        if (!result) throw std::runtime_error("Tensor layout offset addition overflow.");
        return *result;
    }

    std::pair<ptrdiff_t, ptrdiff_t> checkedElementBounds() const {
        const auto result = detail::tryElementBounds(*this);
        if (!result) throw std::runtime_error(std::string(message(result.error())));
        return *result;
    }

    friend std::pair<ptrdiff_t, ptrdiff_t> detail::elementBounds(const Layout& layout);

    Shape m_shape;
    std::vector<ptrdiff_t> m_strides;
    ptrdiff_t m_offset = 0;
};

namespace detail {
inline Expected<Layout, LayoutError> tryContiguousLayout(const Shape& shape,
                                                         bool firstDimensionFastest) {
    const auto dimensions = shape.dimensions();
    std::vector<ptrdiff_t> strides(shape.rank(), 1);
    ptrdiff_t stride = 1;
    for (size_t position = 0; position < shape.rank(); ++position) {
        const size_t axis = firstDimensionFastest ? position : shape.rank() - 1 - position;
        strides[axis] = stride;
        if (!std::in_range<ptrdiff_t>(dimensions[axis])) {
            return Unexpected(LayoutError::ExtentOutOfRange);
        }
        const auto next = tryOffsetMultiply(dimensions[axis], stride);
        if (!next) {
            return Unexpected(LayoutError::ContiguousStrideOverflow);
        }
        stride = *next;
    }
    return Layout(shape, std::move(strides));
}

inline Expected<std::pair<ptrdiff_t, ptrdiff_t>, LayoutError> tryElementBounds(
    const Layout& layout) noexcept {
    const auto dimensions = layout.shape().dimensions();
    if (std::ranges::find(dimensions, size_t{0}) != dimensions.end())
        return std::pair<ptrdiff_t, ptrdiff_t>{0, -1};
    ptrdiff_t lower = layout.offset();
    ptrdiff_t upper = lower;
    for (size_t axis = 0; axis < dimensions.size(); ++axis) {
        const auto delta = tryOffsetMultiply(dimensions[axis] - 1, layout.strides()[axis]);
        if (!delta) {
            return Unexpected(LayoutError::OffsetMultiplyOverflow);
        }
        auto sum = tryOffsetAdd(*delta < 0 ? lower : upper, *delta);
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
