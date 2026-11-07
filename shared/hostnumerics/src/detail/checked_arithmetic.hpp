// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace roc::hostnumerics::detail {

inline size_t checkedMultiply(size_t left, size_t right, std::string_view overflowMessage) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left)
        throw std::runtime_error(std::string(overflowMessage));
    return left * right;
}

inline size_t checkedAdd(size_t left, size_t right, std::string_view overflowMessage) {
    if (right > std::numeric_limits<size_t>::max() - left)
        throw std::runtime_error(std::string(overflowMessage));
    return left + right;
}

}  // namespace roc::hostnumerics::detail
