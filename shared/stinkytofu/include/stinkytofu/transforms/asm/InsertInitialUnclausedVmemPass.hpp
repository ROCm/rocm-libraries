// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Creates a pass that prepends the gfx1250 hardware-entrypoint prologue:
///
///     global_prefetch_b8 v0, [s0, s1] scope:SCOPE_SE th:TH_LOAD_RT
///     v_nop
///
/// Both instructions are required at a gfx1250 kernel entry:
///   - global_prefetch_b8 makes the kernel's first VMEM instruction one that
///     is not in a clause. global_prefetch_b8 is a VMEM operation that ignores
///     the EXEC mask, so making it the first VMEM op guarantees a non-clause
///     first VMEM instruction.
///   - v_nop provides a safe first VALU instruction for the wave.
///
/// The pass inserts the two instructions before the first "real" (non-pseudo)
/// instruction of the entry function so they are the first instructions
/// executed. It must run late in the pipeline — after scheduling and any pass
/// that inserts at kernel entry — so nothing is reordered ahead of the prologue.
///
/// No-op on non-gfx1250 targets (the opcodes are gfx1250-specific).
STINKYTOFU_EXPORT std::unique_ptr<Pass> createInsertInitialUnclausedVmemPass();

}  // namespace stinkytofu
