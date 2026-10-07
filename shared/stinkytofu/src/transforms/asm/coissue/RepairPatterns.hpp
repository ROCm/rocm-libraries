// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Pattern plug-ins (WAITCNT_COISSUE.md 7.9): optional RepairRules that propose moves for a
// known code shape, matched by instruction properties only. Each comes with the measurement
// that motivated it; the engine judges its moves like any other.
//
//   hoist-compare      an SCC producer right before the branch that reads it moves into an
//                      earlier window that has lead (compare to branch takes 9 cycles)
//   switch-after-load  a bank switch right after an LDS op, in front of a co-issued VALU:
//                      a scalar goes between them, or the VALU moves before the loads (that
//                      switch costs 3 cycles, after a scalar 0)

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "RepairRules.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"

namespace stinkytofu::coissue {

/// The patterns registered for `arch`, in registration order.
std::vector<std::string> registeredPatterns(GfxArchID arch);

/// The patterns CoissuePatterns enables: empty means every registered one, "none" none,
/// otherwise a comma-separated list of names. Returns an error for an unknown name.
std::optional<std::string> enabledPatterns(GfxArchID arch, const std::string& spec,
                                           std::vector<std::unique_ptr<RepairRule>>& out);

}  // namespace stinkytofu::coissue
