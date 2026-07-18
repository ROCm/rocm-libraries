// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Epilogue store-sink pass (gfx1250).
///
/// Sinks each global `buffer_store` in the global-write epilogue as late as it
/// legally can within its basic block, so the subsequent InsertWaitAluPass emits
/// a graduated `s_wait_alu depctr_va_vdst(N)` (near-free) instead of a full
/// `va_vdst(0)` drain — letting the store overlap following VALU compute.
///
/// Movement only; the pass never writes wait counts or s_set_vgpr_msb. It must
/// run BEFORE InsertVgprMsbPass and InsertWaitAluPass so both regenerate for the
/// new order. Intended to run inside a ScopeAdaptor targeting the
/// "globalWriteEpilogue" region so it cannot move a store across the region
/// boundary.
///
/// A store stops sinking at the first of:
///   - a later instruction that writes any of the store's data registers
///     (RAW/WAR: e.g. a next-batch buffer_load reusing the reg, or a v_cvt_pk),
///   - a later instruction that writes an SGPR the store reads (the SRD row
///     advance, s_add_u32 sgprSrd*),
///   - any side-effecting / boundary instruction (branch, label, other store,
///     s_nop wait-state, waitcnt),
///   - `targetValu` VALU instructions passed (the sink-distance knob).
STINKYTOFU_EXPORT std::unique_ptr<Pass> createEpilogueStoreSinkPass(unsigned targetValu = 10);

}  // namespace stinkytofu
