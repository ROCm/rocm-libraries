// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;
class ModulePass;

/// Insert gfx1250 assembly hazards that cannot be left to hardware.
///
/// The initial policy implements XNACK replay protection for FLAT and SMEM
/// source clobbers, atomics/RMW operations, existing s_prefetch instructions,
/// forever s_sleep, and non-adjacent s_set_vgpr_msb. Future gfx1250 hazards
/// belong here when they require a late whole-kernel view of the final
/// instruction order.
///
/// Runs only on arches with the `RequiresXCntForVolatileVMEM` capability, the
/// one that makes s_wait_xcnt drains necessary; a no-op everywhere else.
///
/// Requires a correctly built CFG. In particular, a replay group that crosses
/// a physical basic-block boundary must have the corresponding fall-through
/// edge.
///
/// An SMEM instruction that overwrites its own source register cannot be
/// repaired by a drain; the pass reports it and asserts.
///
/// Existing full s_wait_xcnt drains reset the pass's replay-group state.
///
/// Per-function pass; run over all functions via createGfx1250HazardModulePass.
/// \p enableXcntDrainProfile emits a per-function stderr drain summary.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createGfx1250HazardPass(
    bool enableXcntDrainProfile = false);

/// Whole-kernel driver: runs the hazard transform across entry + every callable
/// function with one shared profile. With \p enableXcntDrainProfile it emits an
/// aggregated (kernel body vs helper functions) stderr drain report; off by default.
STINKYTOFU_EXPORT std::unique_ptr<ModulePass> createGfx1250HazardModulePass(
    bool enableXcntDrainProfile = false);

}  // namespace stinkytofu
