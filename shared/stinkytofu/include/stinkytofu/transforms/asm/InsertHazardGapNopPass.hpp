// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Pads the enforced kCdna5HazardRules gaps (SALU SGPR write -> SMEM/tensor_load/
/// VMEM address read, 8 cycles) with s_nop. The DAG scheduler only models them as
/// a stall it may pay, so its output can place the consumer closer; this pass
/// inserts `s_nop N` for the shortfall, counting cycles the way
/// HazardGapAnalysisPass does. Within each basic block, like that analysis and
/// the scheduler.
///
/// Run after every pass that schedules or moves instructions.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createInsertHazardGapNopPass();

}  // namespace stinkytofu
