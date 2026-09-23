// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace roc::host_numerics::detail {
inline size_t checkedMultiply(size_t left, size_t right, std::string_view overflowMessage) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left)
        throw std::overflow_error(std::string(overflowMessage));
    return left * right;
}

inline size_t checkedAdd(size_t left, size_t right, std::string_view overflowMessage) {
    if (right > std::numeric_limits<size_t>::max() - left)
        throw std::overflow_error(std::string(overflowMessage));
    return left + right;
}

inline size_t ceilDivide(size_t value, size_t divisor) {
    if (divisor == 0) throw std::invalid_argument("Ceiling division requires a nonzero divisor.");
    return value / divisor + static_cast<size_t>(value % divisor != 0);
}
}  // namespace roc::host_numerics::detail
