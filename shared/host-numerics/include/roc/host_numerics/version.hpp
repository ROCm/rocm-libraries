// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>

namespace roc::host_numerics {
inline constexpr std::string_view version() noexcept {
    return "0.1.0";
}
}  // namespace roc::host_numerics
