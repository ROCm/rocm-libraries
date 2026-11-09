// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <roc/hostnumerics/layout.hpp>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "index_iteration.hpp"

namespace roc::hostnumerics::detail {
inline bool isContiguous(const Layout& layout, std::span<const size_t> order) {
    const auto axes = validatedIndexOrder(layout.shape().rank(), order);
    uint64_t expectedStride = 1;
    const auto matchesDimension = [&](size_t dimension) {
        const size_t extent = layout.shape().dimensions()[dimension];
        if (extent > 1 && (layout.strides()[dimension] < 0 ||
                           static_cast<uint64_t>(layout.strides()[dimension]) != expectedStride))
            return false;
        if (extent != 0 && expectedStride > std::numeric_limits<uint64_t>::max() / extent)
            return false;
        expectedStride *= extent;
        return true;
    };

    for (auto axis = axes.rbegin(); axis != axes.rend(); ++axis)
        if (!matchesDimension(*axis)) return false;
    return true;
}

template <typename Function>
void forEachLayoutOffsetPairRunRange(const Layout& firstLayout, const Layout& secondLayout,
                                     std::span<const size_t> indexOrder, size_t first,
                                     size_t pastLast, Function&& function) {
    if (firstLayout.shape() != secondLayout.shape())
        throw std::runtime_error("Tensor offset traversal shape mismatch.");
    const Shape& shape = firstLayout.shape();
    const size_t total = shape.elementCount();
    if (first > pastLast || pastLast > total)
        throw std::runtime_error("Tensor offset traversal range exceeds shape.");
    const auto axes = validatedIndexOrder(shape.rank(), indexOrder);
    if (first == pastLast) return;
    if (shape.rank() == 0) {
        function(0, firstLayout.offset(), ptrdiff_t{0}, secondLayout.offset(), ptrdiff_t{0},
                 size_t{1});
        return;
    }

    const size_t innerDimension = axes.back();
    const size_t innerSize = shape.dimensions()[innerDimension];

    while (first < pastLast) {
        const size_t outerIndex = first / innerSize;
        const size_t firstInner = first % innerSize;
        size_t remaining = outerIndex;
        using Unsigned = std::make_unsigned_t<ptrdiff_t>;
        Unsigned firstBase = static_cast<Unsigned>(firstLayout.offset());
        Unsigned secondBase = static_cast<Unsigned>(secondLayout.offset());

        for (size_t position = axes.size() - 1; position > 0; --position) {
            const size_t dimension = axes[position - 1];
            const size_t coordinate = remaining % shape.dimensions()[dimension];
            remaining /= shape.dimensions()[dimension];
            firstBase += static_cast<Unsigned>(coordinate) *
                         static_cast<Unsigned>(firstLayout.strides()[dimension]);
            secondBase += static_cast<Unsigned>(coordinate) *
                          static_cast<Unsigned>(secondLayout.strides()[dimension]);
        }

        const size_t count = std::min(innerSize - firstInner, pastLast - first);
        function(
            first,
            static_cast<ptrdiff_t>(
                firstBase + static_cast<Unsigned>(firstInner) *
                                static_cast<Unsigned>(firstLayout.strides()[innerDimension])),
            firstLayout.strides()[innerDimension],
            static_cast<ptrdiff_t>(
                secondBase + static_cast<Unsigned>(firstInner) *
                                 static_cast<Unsigned>(secondLayout.strides()[innerDimension])),
            secondLayout.strides()[innerDimension], count);
        first += count;
    }
}

template <typename Function>
void forEachLayoutOffsetPairRange(const Layout& firstLayout, const Layout& secondLayout,
                                  std::span<const size_t> indexOrder, size_t first, size_t pastLast,
                                  Function&& function) {
    forEachLayoutOffsetPairRunRange(
        firstLayout, secondLayout, indexOrder, first, pastLast,
        [&](size_t logicalIndex, ptrdiff_t firstOffset, ptrdiff_t firstStride,
            ptrdiff_t secondOffset, ptrdiff_t secondStride, size_t count) {
            for (size_t index = 0; index < count; ++index) {
                using Unsigned = std::make_unsigned_t<ptrdiff_t>;
                function(logicalIndex + index,
                         static_cast<ptrdiff_t>(static_cast<Unsigned>(firstOffset) +
                                                static_cast<Unsigned>(index) *
                                                    static_cast<Unsigned>(firstStride)),
                         static_cast<ptrdiff_t>(static_cast<Unsigned>(secondOffset) +
                                                static_cast<Unsigned>(index) *
                                                    static_cast<Unsigned>(secondStride)));
            }
        });
}
}  // namespace roc::hostnumerics::detail
