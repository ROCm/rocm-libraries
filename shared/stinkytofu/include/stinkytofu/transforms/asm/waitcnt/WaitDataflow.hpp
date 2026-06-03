/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

// Forward dataflow that computes, for every basic block, the set of
// asynchronous memory ops still in flight at block exit, per hardware
// counter. Source-of-truth for the wait-count insertion pass.
//
// One queue per counter (ds / buffer / tensor), populated by walking each
// block in program order. Cross-block flow is handled by an ordered
// intersection at merges (drop ops not present on every pred). PHI nodes
// of pseudo-reg memtokens are summarised into a small per-PHI record so
// downstream consumers can look up a single representative wait per
// counter.
//
// The solver iterates until per-block exit states stop changing or until
// a hard iteration cap is hit; on cap-hit it falls back to s_wait_* 0
// for every emitting anchor and logs a warning.

#include <array>
#include <deque>
#include <unordered_map>
#include <vector>

#include "stinkytofu/transforms/asm/waitcnt/WaitPlan.hpp"

namespace stinkytofu {
class BasicBlock;
class Function;
struct DominanceInfo;
struct StinkyInstruction;

namespace waitcnt {

/// Hardware counters we track. Index matches arrays in DataflowState.
enum CounterKind { CK_DS = 0, CK_Buffer = 1, CK_Tensor = 2, CK_Count = 3 };

/// Per-counter queue of in-flight memops in issue order. The "wait value"
/// for an op @c o at queue index @c i with size @c n is @c n - i - 1.
struct CounterQueue {
    std::deque<StinkyInstruction*> ops;

    int countFrom(StinkyInstruction* op) const;
    bool operator==(const CounterQueue& other) const {
        return ops == other.ops;
    }
};

/// Summary of what a PHI of pseudo-reg memtokens implies for a consumer:
/// the strictest wait per counter across every incoming CFG path. A field
/// equal to WaitCountSpec::kUnused means the PHI imposes no constraint on
/// that counter (e.g. all incoming sources were VALU or already drained).
struct PhiSummary {
    int waits[CK_Count] = {WaitCountSpec::kUnused, WaitCountSpec::kUnused,
                           WaitCountSpec::kUnused};

    bool operator==(const PhiSummary& other) const {
        for (int c = 0; c < CK_Count; ++c) {
            if (waits[c] != other.waits[c]) return false;
        }
        return true;
    }
};

/// Per-block dataflow lattice element. Stored separately for entry and exit
/// so the solver can detect convergence on exit while merging into entry.
struct DataflowState {
    std::array<CounterQueue, CK_Count> queues;
    std::unordered_map<StinkyInstruction*, PhiSummary> phiSummaries;

    bool operator==(const DataflowState& other) const;
    void clear();
};

/// Final per-block exit state plus the conservative emission decisions
/// reached during the last transfer pass. The optimizer layer reads from
/// here to compute per-pred path lengths for shallow-pred promotion.
struct DataflowResult {
    std::unordered_map<const BasicBlock*, DataflowState> entryState;
    std::unordered_map<const BasicBlock*, DataflowState> exitState;
};

class WaitDataflow {
   public:
    WaitDataflow(Function& func, const DominanceInfo& domInfo,
                 const std::vector<BasicBlock*>& rpo);

    /// Solve to a fixed point. Returns true on convergence; false if the
    /// iteration cap was hit (in which case the conservative plan that
    /// materializePlan() returns forces s_wait_* 0 at every emitting
    /// anchor).
    bool solve();

    /// Materialise the conservative per-consumer wait plan from the
    /// converged dataflow state. Optimizers run after this and may rewrite
    /// the plan in place.
    WaitInsertionPlan materializePlan() const;

    const DataflowResult& result() const {
        return result_;
    }

    /// Iteration cap. Public so tests can override.
    void setIterationCap(unsigned cap) {
        iterationCap_ = cap;
    }

   private:
    Function& func_;
    const std::vector<BasicBlock*>& rpo_;
    DataflowResult result_;

    /// Plan recorded by the last transfer pass: each block's list of
    /// (anchor, WaitCountSpec) pairs in program order. Populated by
    /// transferBlock(); consumed by materializePlan().
    std::unordered_map<const BasicBlock*, std::vector<std::pair<StinkyInstruction*, WaitCountSpec>>>
        emitPlan_;

    bool capHit_ = false;
    unsigned iterationCap_ = 0;

    /// Compute @c InState for @p bb by intersecting predecessors' OutStates.
    DataflowState mergeFromPredecessors(BasicBlock& bb) const;

    /// Walk @p bb in program order, updating @p state and recording any
    /// emitted waits into emitPlan_[&bb].
    void transferBlock(BasicBlock& bb, DataflowState& state);
};

}  // namespace waitcnt
}  // namespace stinkytofu
