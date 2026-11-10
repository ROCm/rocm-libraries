// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>

namespace stinkytofu {

/// WMMAs that issue before \p latencyCycles have passed when each takes \p wmmaLatencyCycles:
/// the whole WMMAs that fit, plus one.
inline int latencyBudgetWmmas(int latencyCycles, int wmmaLatencyCycles) {
    return std::max(0, latencyCycles) / std::max(1, wmmaLatencyCycles) + 1;
}

}  // namespace stinkytofu
