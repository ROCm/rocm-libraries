// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Epilogue store-sink pass (msb/xcnt-aware). Sinks each global-write buffer_store as
/// late as legal so InsertWaitAluPass emits a graduated va_vdst(N) instead of a
/// full va_vdst(0) drain. Movement only; runs before InsertVgprMsb/InsertWaitAlu.
/// Only pays off under expert schedule mode2.
///
/// targetValu (typical 4-16, 0 = disabled): max VALU a store sinks past. Bounded
/// by block size; out-of-range values just no-op, they do not misbehave.
///
/// maxStoreGroupSize (0/1 = off, else 2-8): pack up to N adjacent stores so one
/// s_wait_xcnt drain covers a group instead of one per store. An upper bound —
/// a group ends early when the next store cannot hop what separates it.
///
/// avoidMsbXcntDrain: stop a store before it straddles an s_set_vgpr_msb flip.
/// Only active when xnack replay is on; without it no drain is emitted.
struct EpilogueStoreSinkOptions {
    unsigned targetValu = 10;
    unsigned maxStoreGroupSize = 4;
    bool avoidMsbXcntDrain = true;
};

STINKYTOFU_EXPORT std::unique_ptr<Pass> createEpilogueStoreSinkPass(
    EpilogueStoreSinkOptions options = {});

}  // namespace stinkytofu
