// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace roc::hostnumerics::test {
inline void require(bool condition, std::string_view message,
                    std::source_location location = std::source_location::current()) {
    if (!condition) {
        ADD_FAILURE_AT(location.file_name(), static_cast<int>(location.line())) << message;
        throw std::runtime_error(std::string(message));
    }
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
void requireError(Function&& function, std::string_view message) {
    requireThrows<std::runtime_error>(std::forward<Function>(function), message);
}
}  // namespace roc::hostnumerics::test
