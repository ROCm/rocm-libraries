// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Creates a pass that prepends the gfx1250 hardware-entrypoint prologue:
///
///     s_mov_b64 s[64:65], 0
///     v_nop
///     global_prefetch_b8 v0, [s64, s65] scope:SCOPE_SE th:TH_LOAD_RT
///
/// All three instructions are required at a gfx1250 kernel entry:
///   - s_mov_b64 gives the prefetch a defined SADDR. s[64:65] is never
///     initialized by the hardware, so zeroing it at wave start clobbers
///     nothing. An SGPR pair is preferred over initializing a VGPR for the
///     null-SADDR form: the write-to-use delay before a global instruction is
///     shorter for a SALU write than for a VGPR write.
///   - v_nop provides a safe first VALU instruction for the wave and covers the
///     write-to-use delay between the s_mov_b64 and the prefetch that reads it.
///   - global_prefetch_b8 makes the kernel's first VMEM instruction one that
///     is not in a clause. global_prefetch_b8 is a VMEM operation that ignores
///     the EXEC mask, so making it the first VMEM op guarantees a non-clause
///     first VMEM instruction.
///
/// The pass inserts the three instructions before the first "real" (non-pseudo)
/// instruction of the entry function so they are the first instructions
/// executed. It must run late in the pipeline — after scheduling and any pass
/// that inserts at kernel entry — so nothing is reordered ahead of the prologue.
///
/// No-op on non-gfx1250 targets (the opcodes are gfx1250-specific).
STINKYTOFU_EXPORT std::unique_ptr<Pass> createInsertInitialUnclausedVmemPass();

}  // namespace stinkytofu
