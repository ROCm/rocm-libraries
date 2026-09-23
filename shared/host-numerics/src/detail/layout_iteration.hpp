// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <roc/host_numerics/layout.hpp>
#include <stdexcept>
#include <utility>

namespace roc::host_numerics::detail {
inline bool isContiguous(const Layout& layout, IndexOrder order) {
    uint64_t expectedStride = 1;
    const auto matchesDimension = [&](size_t dimension) {
        const size_t extent = layout.shape()[dimension];
        if (extent > 1 && (layout.stride(dimension) < 0 ||
                           static_cast<uint64_t>(layout.stride(dimension)) != expectedStride))
            return false;
        if (extent != 0 && expectedStride > std::numeric_limits<uint64_t>::max() / extent)
            return false;
        expectedStride *= extent;
        return true;
    };

    if (order == IndexOrder::FirstDimensionFastest) {
        for (size_t dimension = 0; dimension < layout.shape().rank(); ++dimension)
            if (!matchesDimension(dimension)) return false;
    } else {
        for (size_t dimension = layout.shape().rank(); dimension > 0; --dimension)
            if (!matchesDimension(dimension - 1)) return false;
    }
    return true;
}

template <typename Function>
void forEachLayoutOffsetPairRunRange(const Layout& firstLayout, const Layout& secondLayout,
                                     IndexOrder indexOrder, size_t first, size_t pastLast,
                                     Function&& function) {
    if (firstLayout.shape() != secondLayout.shape())
        throw std::invalid_argument("Tensor offset traversal shape mismatch.");
    const Shape& shape = firstLayout.shape();
    const size_t total = shape.elementCount();
    if (first > pastLast || pastLast > total)
        throw std::out_of_range("Tensor offset traversal range exceeds shape.");
    if (first == pastLast) return;
    if (shape.rank() == 0) {
        function(0, firstLayout.offset(), ptrdiff_t{0}, secondLayout.offset(), ptrdiff_t{0},
                 size_t{1});
        return;
    }

    const bool firstDimensionFastest = indexOrder == IndexOrder::FirstDimensionFastest;
    const size_t innerDimension = firstDimensionFastest ? 0 : shape.rank() - 1;
    const size_t innerSize = shape[innerDimension];

    while (first < pastLast) {
        const size_t outerIndex = first / innerSize;
        const size_t firstInner = first % innerSize;
        size_t remaining = outerIndex;
        ptrdiff_t firstBase = firstLayout.offset();
        ptrdiff_t secondBase = secondLayout.offset();

        if (firstDimensionFastest) {
            for (size_t dimension = 1; dimension < shape.rank(); ++dimension) {
                const size_t coordinate = remaining % shape[dimension];
                remaining /= shape[dimension];
                firstBase += static_cast<ptrdiff_t>(coordinate) * firstLayout.stride(dimension);
                secondBase += static_cast<ptrdiff_t>(coordinate) * secondLayout.stride(dimension);
            }
        } else {
            for (size_t dimension = shape.rank() - 1; dimension > 0; --dimension) {
                const size_t index = dimension - 1;
                const size_t coordinate = remaining % shape[index];
                remaining /= shape[index];
                firstBase += static_cast<ptrdiff_t>(coordinate) * firstLayout.stride(index);
                secondBase += static_cast<ptrdiff_t>(coordinate) * secondLayout.stride(index);
            }
        }

        const size_t count = std::min(innerSize - firstInner, pastLast - first);
        function(
            first,
            firstBase + static_cast<ptrdiff_t>(firstInner) * firstLayout.stride(innerDimension),
            firstLayout.stride(innerDimension),
            secondBase + static_cast<ptrdiff_t>(firstInner) * secondLayout.stride(innerDimension),
            secondLayout.stride(innerDimension), count);
        first += count;
    }
}

template <typename Function>
void forEachLayoutOffsetPairRange(const Layout& firstLayout, const Layout& secondLayout,
                                  IndexOrder indexOrder, size_t first, size_t pastLast,
                                  Function&& function) {
    forEachLayoutOffsetPairRunRange(
        firstLayout, secondLayout, indexOrder, first, pastLast,
        [&](size_t logicalIndex, ptrdiff_t firstOffset, ptrdiff_t firstStride,
            ptrdiff_t secondOffset, ptrdiff_t secondStride, size_t count) {
            for (size_t index = 0; index < count; ++index) {
                function(logicalIndex + index,
                         firstOffset + static_cast<ptrdiff_t>(index) * firstStride,
                         secondOffset + static_cast<ptrdiff_t>(index) * secondStride);
            }
        });
}
}  // namespace roc::host_numerics::detail
