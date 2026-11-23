// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <roc/hostnumerics/layout.hpp>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace roc::hostnumerics::detail {
// Whether source extents broadcast to destination extents under NumPy rules.
inline bool canBroadcastTo(std::span<const size_t> source,
                           std::span<const size_t> destination) noexcept {
    if (source.size() > destination.size()) return false;
    const size_t leading = destination.size() - source.size();
    for (size_t axis = 0; axis < source.size(); ++axis)
        if (source[axis] != 1 && source[axis] != destination[leading + axis]) return false;
    return true;
}

// Throwing form of Layout::getElementBounds.
inline std::optional<ElementOffsetBounds> elementBounds(const Layout& layout) {
    const auto result = layout.getElementBounds();
    if (!result) throw std::runtime_error(result.error().message());
    return *result;
}

inline uint64_t strideMagnitude(ptrdiff_t stride) noexcept {
    if (stride >= 0) return static_cast<uint64_t>(stride);
    return static_cast<uint64_t>(-(stride + 1)) + 1;
}

// A sufficient condition for distinct coordinates to address distinct
// elements: ordered by stride magnitude, each stride exceeds the span the
// smaller strides can reach. False does not prove that offsets overlap.
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
}  // namespace roc::hostnumerics::detail
