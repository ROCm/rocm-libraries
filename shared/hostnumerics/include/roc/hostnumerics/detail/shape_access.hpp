// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/hostnumerics/shape.hpp>

namespace roc::hostnumerics::detail {
// Internal access for consumers that report invalid descriptors as results.
struct ShapeAccess {
    static std::optional<size_t> getElementCount(std::span<const size_t> dimensions) noexcept {
        return Shape::getElementCount(dimensions);
    }

    static bool isValidIndexOrder(size_t rank, std::span<const size_t> order) {
        if (order.size() != rank) return false;
        std::vector<bool> seen(rank, false);
        for (size_t axis : order) {
            if (axis >= rank || seen[axis]) return false;
            seen[axis] = true;
        }
        return true;
    }
};
}  // namespace roc::hostnumerics::detail
