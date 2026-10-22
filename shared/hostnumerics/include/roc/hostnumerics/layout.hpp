// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
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
        for (size_t dimension = 0; dimension < indices.size(); ++dimension) {
            if (indices[dimension] >= dimensions[dimension])
                throw std::runtime_error("Tensor coordinate " + std::to_string(indices[dimension]) +
                                         " in dimension " + std::to_string(dimension) +
                                         " is outside [0, " +
                                         std::to_string(dimensions[dimension]) + ").");
        }
        ptrdiff_t result = m_offset;
        for (size_t dimension = 0; dimension < indices.size(); ++dimension) {
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

}  // namespace detail
}  // namespace roc::hostnumerics
