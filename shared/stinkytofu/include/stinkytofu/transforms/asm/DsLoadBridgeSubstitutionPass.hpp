// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Creates a pass that rewrites the FIRST `ds_load_b*` following a prefetch group to
/// `flat_load_b*` when doing so lets InsertWaitAluPass drop the group's vm_vsrc waits.
///
/// Same bridge idea as PrefetchBridgeSubstitutionPass, anchored from the other side. A
/// FLAT op takes an ordinal in both the LDS and TEX order FIFOs, so proving it drained
/// proves every older op in either FIFO drained. PrefetchBridgeSubstitutionPass builds
/// that anchor by adding an LDS ordinal to a TEX-only prefetch; this pass builds it by
/// adding a TEX ordinal to an LDS-only ds_load. The first ds_load after the group sits
/// behind every prefetch in TEX order, which is what makes it cover them.
///
/// Which side to prefer is a queue-pressure question: the prefetch form spends an LDS
/// slot, this form spends a TEX slot. The follower counts differ by one, since the
/// anchor here is itself the first member of the ds run.
///
/// \p apertureSgpr is the low register of an SGPR pair used as the FLAT saddr, so the
/// address stays a single VGPR (a 64-bit vaddr pair would cost an extra address register
/// per lane). The caller that owns register allocation must reserve the pair for the whole
/// kernel and pass its index; -1 means unavailable and the pass is a no-op. The pass writes
/// the pair itself, emitting `s_mov_b64 <pair>, src_shared_base` at kernel entry, and
/// substitutes nothing if that write cannot be placed.
///
/// Correctness never depends on this pass. InsertWaitAluPass re-derives every wait from
/// whatever stream it is given.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createDsLoadBridgeSubstitutionPass(int apertureSgpr = -1);

}  // namespace stinkytofu
