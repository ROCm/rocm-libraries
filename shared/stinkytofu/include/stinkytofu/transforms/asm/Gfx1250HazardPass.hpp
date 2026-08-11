// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <memory>
#include <vector>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Function;
class Pass;

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
/// Existing full s_wait_xcnt drains reset the pass's replay-group state. When
/// \p functions is non-empty, the pass walks the whole kernel (entry plus
/// callable functions); otherwise it processes the single Function given to
/// the pipeline.
///
/// With \p enableXcntDrainProfile the pass emits an stderr summary of inserted
/// drains by rule, and by whether each drain's block belongs to a loop and/or
/// holds a matrix instruction. The summary covers every walked function, and
/// when the kernel has callable functions it also reports the kernel body (the
/// entry function) and the helper functions separately. Off by default: the
/// summary costs a loop and matrix-instruction scan per function.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createGfx1250HazardPass(
    std::vector<Function*> functions = {}, bool enableXcntDrainProfile = false);

}  // namespace stinkytofu
