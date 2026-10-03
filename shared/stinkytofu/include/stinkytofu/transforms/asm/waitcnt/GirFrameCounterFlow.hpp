/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */
#pragma once

#include <functional>

#include "stinkytofu/analysis/asm/GirFrameAnalysis.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitDataflow.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitPlan.hpp"

namespace stinkytofu {
class Function;

namespace waitcnt {

/// Compute a finite-frame wait plan directly on `(ST basic block, GIR frame)` states.
/// `covers` must be the same predicate the caller emits with: a wait planned for a block that is
/// never written is not merely wasted, it demands a barrier nothing was asked to place.
WaitInsertionPlan buildGirFrameWaitPlan(
    Function& function, const GirFrameAnalysis::Result& frames,
    const GirFrameHazardAnalysis::Result& hazards,
    const std::function<bool(const BasicBlock&)>& covers,
    const std::function<bool(const StinkyInstruction&)>& alsoFences = {});

/// Same-counter issues standing at or after `producerIndex` once control reaches `anchorIndex` in
/// `anchorNode`, `span` frame-graph steps on; the producer itself counts as 1, so the wait that
/// retires it is `n - 1`.  -1 when no path arrives with it outstanding.
///
/// Exported so the pass that PLACES a fence measures its wait over the same frame span as the pass
/// that later fills it in; counting within one block instead makes the two disagree about a
/// loop-carried producer, whose issues live on the trips the block itself cannot see.
STINKYTOFU_EXPORT int girFrameIssuesAcrossSpan(const GirFrameAnalysis::Result& frames,
                                               CounterKind counter,
                                               const GirFrameNode& producerNode,
                                               size_t producerIndex, int span,
                                               const GirFrameNode& anchorNode, size_t anchorIndex);

/// Frame-graph steps from `from` to `to`, -1 when unreachable and 0 when they are the same node.
STINKYTOFU_EXPORT int girFrameDistance(const GirFrameAnalysis::Result& frames,
                                       const GirFrameNode& from, const GirFrameNode& to);

/// Span walks that ended with a path neither reaching the anchor nor retiring -- a hazard the
/// counter flow did not model and then treated as discharged.  Should be 0; `ST_GIR_SPAN_STATS=1`
/// reports it per function.
STINKYTOFU_EXPORT extern unsigned girUnaccountedSpans;

}  // namespace waitcnt
}  // namespace stinkytofu
