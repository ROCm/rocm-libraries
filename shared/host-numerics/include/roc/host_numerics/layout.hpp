// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <roc/host_numerics/shape.hpp>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace roc::host_numerics {
class Layout;

namespace detail {
inline std::pair<ptrdiff_t, ptrdiff_t> elementBounds(const Layout& layout);
}

class Layout {
   public:
    Layout() = default;

    Layout(Shape shape, std::vector<ptrdiff_t> strides, ptrdiff_t offset = 0)
        : m_shape(std::move(shape)), m_strides(std::move(strides)), m_offset(offset) {
        if (m_shape.rank() != m_strides.size())
            throw std::invalid_argument("Tensor layout rank and stride count differ.");
    }

    // For a matrix this is row-major/C-order: the last dimension has unit stride.
    static Layout contiguousLastDimensionFastest(const Shape& shape) {
        std::vector<ptrdiff_t> strides(shape.rank(), 1);
        ptrdiff_t stride = 1;
        for (size_t dimension = shape.rank(); dimension > 0; --dimension) {
            const size_t index = dimension - 1;
            strides[index] = stride;
            stride = checkedContiguousStride(stride, shape.extent(index));
        }
        return Layout(shape, std::move(strides));
    }

    // For a matrix this is column-major/Fortran-order: the first dimension has unit stride.
    static Layout contiguousFirstDimensionFastest(const Shape& shape) {
        std::vector<ptrdiff_t> strides(shape.rank(), 1);
        ptrdiff_t stride = 1;
        for (size_t dimension = 0; dimension < shape.rank(); ++dimension) {
            strides[dimension] = stride;
            stride = checkedContiguousStride(stride, shape.extent(dimension));
        }
        return Layout(shape, std::move(strides));
    }

    const Shape& shape() const {
        return m_shape;
    }

    std::span<const ptrdiff_t> strides() const {
        return m_strides;
    }

    ptrdiff_t stride(size_t dimension) const {
        return m_strides.at(dimension);
    }

    ptrdiff_t offset() const {
        return m_offset;
    }

    ptrdiff_t elementOffset(std::span<const size_t> indices) const {
        if (indices.size() != m_shape.rank())
            throw std::invalid_argument("Tensor index rank does not match layout rank.");

        ptrdiff_t result = m_offset;
        for (size_t dimension = 0; dimension < indices.size(); ++dimension) {
            if (indices[dimension] >= m_shape.extent(dimension))
                throw std::out_of_range("Tensor index exceeds shape.");
            const ptrdiff_t delta = checkedMultiply(indices[dimension], stride(dimension));
            result = checkedAdd(result, delta);
        }
        return result;
    }

    friend bool operator==(const Layout&, const Layout&) = default;

   private:
    static ptrdiff_t checkedContiguousStride(ptrdiff_t stride, size_t extent) {
        if (extent > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
            throw std::overflow_error("Tensor extent exceeds ptrdiff_t.");
        const ptrdiff_t signedExtent = static_cast<ptrdiff_t>(extent);
        if (signedExtent != 0 && stride > std::numeric_limits<ptrdiff_t>::max() / signedExtent)
            throw std::overflow_error("Tensor contiguous stride overflow.");
        return stride * signedExtent;
    }

    static ptrdiff_t checkedMultiply(size_t value, ptrdiff_t factor) {
        if (value == 0 || factor == 0) return 0;

        const bool negative = factor < 0;
        const uintmax_t factorMagnitude =
            negative ? static_cast<uintmax_t>(-(factor + 1)) + 1 : static_cast<uintmax_t>(factor);
        const uintmax_t limit =
            negative ? static_cast<uintmax_t>(std::numeric_limits<ptrdiff_t>::max()) + 1
                     : static_cast<uintmax_t>(std::numeric_limits<ptrdiff_t>::max());
        if (static_cast<uintmax_t>(value) > limit / factorMagnitude)
            throw std::overflow_error("Tensor layout offset multiplication overflow.");

        const uintmax_t magnitude = static_cast<uintmax_t>(value) * factorMagnitude;
        if (!negative) return static_cast<ptrdiff_t>(magnitude);
        if (magnitude == limit) return std::numeric_limits<ptrdiff_t>::min();
        return -static_cast<ptrdiff_t>(magnitude);
    }

    static ptrdiff_t checkedAdd(ptrdiff_t left, ptrdiff_t right) {
        if ((right > 0 && left > std::numeric_limits<ptrdiff_t>::max() - right) ||
            (right < 0 && left < std::numeric_limits<ptrdiff_t>::min() - right))
            throw std::overflow_error("Tensor layout offset addition overflow.");
        return left + right;
    }

    std::pair<ptrdiff_t, ptrdiff_t> checkedElementBounds() const {
        for (size_t dimension = 0; dimension < m_shape.rank(); ++dimension) {
            if (m_shape.extent(dimension) == 0) return {0, -1};
        }

        ptrdiff_t lower = m_offset;
        ptrdiff_t upper = m_offset;
        for (size_t dimension = 0; dimension < m_shape.rank(); ++dimension) {
            const ptrdiff_t delta =
                checkedMultiply(m_shape.extent(dimension) - 1, stride(dimension));
            if (delta < 0)
                lower = checkedAdd(lower, delta);
            else
                upper = checkedAdd(upper, delta);
        }
        return {lower, upper};
    }

    friend std::pair<ptrdiff_t, ptrdiff_t> detail::elementBounds(const Layout& layout);

    Shape m_shape;
    std::vector<ptrdiff_t> m_strides;
    ptrdiff_t m_offset = 0;
};

namespace detail {
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
}  // namespace roc::host_numerics
