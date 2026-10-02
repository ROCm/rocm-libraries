// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// TDMInflightGuardPass -- caps how many tensor (TDM) operations one wave has in
// flight. On gfx1250 B0 a wave with more than 11 outstanding can deadlock the TDM
// (LCOMPILER-2572). The pass bounds this wave's tensor counter immediately before
// every TDM issue over all CFG paths and inserts `s_wait_tensorcnt (limit - 1)`
// only before the issues where that bound exceeds limit - 1. See
// docs/developer/tdm-inflight-guard.md.

#include <memory>
#include <vector>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;
class Function;

/// Most TDM operations one wave may have in flight on gfx1250 B0.
inline constexpr int kDefaultTDMInflightLimit = 11;

/// \p limit is the most TDM operations one wave may have in flight; a value <= 0
/// makes the pass a no-op. \p functions is the whole-kernel function list (entry
/// + callable functions, e.g. StinkyAsmModule::getFunctions). A call leaves the
/// bound unchanged only when that list is given and no function in it other than
/// the one being guarded issues a TDM operation; otherwise the bound after a call
/// is unknown and saturates.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createTDMInflightGuardPass(
    int limit = kDefaultTDMInflightLimit, std::vector<Function*> functions = {});

}  // namespace stinkytofu
