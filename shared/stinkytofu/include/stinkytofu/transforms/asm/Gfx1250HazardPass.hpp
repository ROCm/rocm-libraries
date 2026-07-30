// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <memory>
#include <vector>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Function;
class Pass;
class StinkyAsmModule;

/// Insert gfx1250 assembly hazards that cannot be left to hardware.
///
/// The initial policy implements XNACK replay protection for FLAT and SMEM
/// source clobbers, atomics/RMW operations, existing s_prefetch instructions,
/// forever s_sleep, and non-adjacent s_set_vgpr_msb. Future gfx1250 hazards
/// belong here when they require a late whole-kernel view of the final
/// instruction order.
///
/// Requires a correctly built CFG. In particular, a replay group that crosses
/// a physical basic-block boundary must have the corresponding fall-through
/// edge.
///
/// Existing full s_wait_xcnt drains reset the pass's replay-group state. When
/// \p functions is non-empty, the pass walks the whole kernel (entry plus
/// callable functions); otherwise it processes the single Function given to
/// the pipeline.
///
/// Define @c STINKYTOFU_GFX1250_HAZARD_PROFILE=1 at build time to emit an
/// stderr summary of inserted drains by rule and source region.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createGfx1250HazardPass(
    std::vector<Function*> functions = {}, StinkyAsmModule* module = nullptr);

}  // namespace stinkytofu
