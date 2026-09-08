// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Creates a pass that rewrites the last `global_prefetch_b8` of a prefetch group to
/// `flat_prefetch_b8` when doing so lets InsertWaitAluPass drop a vm_vsrc wait.
///
/// The two instructions are functionally identical -- same operands, cache hint only, no
/// destination, no completion counter. They differ only in ordering: a FLAT op takes an
/// ordinal in *both* the LDS and TEX order FIFOs, where a global one takes only TEX.
/// Because both FIFOs are ordered, a FLAT op therefore acts as a synchronisation point:
/// proving it drained proves every older op in either FIFO drained. InsertWaitAluPass uses
/// that to skip waits it would otherwise emit (HWModel::WaitHide::vmVsrcBridge).
///
/// The last prefetch of a group is the one rewritten, since TEX is ordered and a bridge
/// there covers every prefetch ahead of it. It is rewritten only when enough LDS ops
/// already separate it from the first instruction overwriting one of the group's address
/// registers -- otherwise the bridge could not discharge anything and the substitution
/// would cost an LDS FIFO slot for nothing.
///
/// Runs immediately before InsertWaitAluPass so the counts it measures are final: passes
/// that reorder instructions (WaitAwareScheduleRepairPass, AsmMovePropagationPass) have
/// already run.
///
/// Correctness never depends on this pass. Substituting where it does not pay costs an LDS
/// FIFO slot; failing to substitute costs a skip. InsertWaitAluPass re-derives every wait
/// from whatever stream it is given.
///
/// No-op when the arch sets `waitHide.vmVsrcBridge` to 0, or lacks `flat_prefetch_b8`.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createPrefetchBridgeSubstitutionPass();

}  // namespace stinkytofu
