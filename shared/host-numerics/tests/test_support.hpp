// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace roc::host_numerics::test {
inline void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

inline void requireNear(double observed, double expected, double tolerance,
                        std::string_view message) {
    if (std::abs(observed - expected) > tolerance) require(false, message);
}

template <typename Exception, typename Function>
void requireThrows(Function&& function, std::string_view message) {
    try {
        std::forward<Function>(function)();
    } catch (const Exception&) {
        return;
    }
    require(false, message);
}

template <typename Function>
void requireInvalidArgument(Function&& function, std::string_view message) {
    requireThrows<std::invalid_argument>(std::forward<Function>(function), message);
}

template <typename Function>
void requireOutOfRange(Function&& function, std::string_view message) {
    requireThrows<std::out_of_range>(std::forward<Function>(function), message);
}

template <typename Function>
void requireOverflow(Function&& function, std::string_view message) {
    requireThrows<std::overflow_error>(std::forward<Function>(function), message);
}
}  // namespace roc::host_numerics::test
