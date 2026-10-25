// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace roc::hostnumerics::detail {
// Requires a matching rank. Returns the position of the first repeated or
// out-of-range axis; a valid permutation has no invalid position.
inline std::optional<size_t> getInvalidOrderPosition(std::span<const size_t> order, size_t rank) {
    assert(order.size() == rank);
    std::vector<bool> seen(rank, false);
    for (size_t position = 0; position < order.size(); ++position) {
        const size_t axis = order[position];
        if (axis >= rank || seen[axis]) return position;
        seen[axis] = true;
    }
    return std::nullopt;
}
}  // namespace roc::hostnumerics::detail
