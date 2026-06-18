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
// asynchronous memory ops still in flight at exit, per hardware counter.
// Source-of-truth for the wait-count insertion pass.
//
// State shape: each block holds one queue per CFG predecessor per counter
// during its in-block walk. Per-pred queues let a consumer at a join
// compute the strictest required wait as max-over-preds, which is the
// only sound choice (a single union or intersection would either drop
// deps still in flight on one path or over-drain every path).
//
// Per-pred queues are KEPT at block exit (not collapsed) so the optimizer
// layer can read each predecessor's path length for shallow-pred
// promotion. Each queue is capped at the hardware counter window
// (kMaxInFlight) so a self-loop with an undrained counter still reaches a
// fixed point instead of growing the queue every iteration.
//
// PHIs of pseudo-reg memtokens are summarised into a PhiSummary so
// downstream consumers look up a single representative wait per counter.
//
// The solver iterates until per-block exit states stop changing or until
// a hard iteration cap is hit; on cap-hit it falls back to s_wait_* 0
// for every emitting anchor and logs a warning.

#include <array>
#include <deque>
#include <functional>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "stinkytofu/transforms/asm/waitcnt/WaitPlan.hpp"

namespace stinkytofu {
class BasicBlock;
class Function;
struct DominanceInfo;
struct StinkyInstruction;

namespace waitcnt {

/// Hardware counters we track. Index matches arrays in DataflowState.
enum CounterKind { CK_DS = 0, CK_Buffer = 1, CK_KM = 2, CK_Tensor = 3, CK_Count = 4 };

/// Map a tracked async memop to its hardware counter. Returns CK_Count when
/// `inst` is not tracked by the waitcnt pass.
CounterKind classifyMemOp(const StinkyInstruction& inst);

/// One queue of in-flight memops on a given counter, tagged by the CFG
/// predecessor it was seeded from. For an op OP at index I in a queue of
/// size N, the wait value is N - I - 1.
///
/// At block entry there is one entry per CFG predecessor; these are kept
/// (not collapsed) at block exit so a successor's mergeFromPredecessors can
/// recover each predecessor's path length. Queue length is capped at the
/// hardware counter window so the lattice has finite height.
struct PerPredQueue {
    BasicBlock* pred = nullptr;
    std::deque<StinkyInstruction*> ops;

    int countFrom(StinkyInstruction* op) const;
    bool operator==(const PerPredQueue& other) const {
        return pred == other.pred && ops == other.ops;
    }
};

/// Summary of what a PHI of pseudo-reg memtokens implies for a consumer:
/// the strictest wait per counter across every incoming CFG path. A field
/// equal to WaitCountSpec::kUnused means the PHI imposes no constraint on
/// that counter (e.g. all incoming sources were VALU).
struct PhiSummary {
    int waits[CK_Count] = {WaitCountSpec::kUnused, WaitCountSpec::kUnused, WaitCountSpec::kUnused,
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
///
/// queues[c] is a list of per-pred queues at block entry, mutated during
/// transferBlock, and kept per-pred (not collapsed) at block exit.
struct DataflowState {
    std::array<std::vector<PerPredQueue>, CK_Count> queues;
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
    /// Predicate deciding whether a RAW dependency carried on a given counter
    /// must be drained at consumer `inst`. This is the per-counter "constraint
    /// to emit a wait": return true to force the dataflow to consider draining
    /// this counter for `inst`, false to skip it.
    using RawWaitPredicate = std::function<bool(const StinkyInstruction& inst)>;

    WaitDataflow(Function& func, const DominanceInfo& domInfo, const std::vector<BasicBlock*>& rpo);

    /// Solve to a fixed point. Returns true on convergence; false if the
    /// iteration cap was hit (in which case the conservative plan that
    /// materializePlan() returns forces s_wait_* 0 at every emitting
    /// anchor).
    bool solve();

    /// Override the per-counter RAW-wait constraint used by transferBlock.
    /// Must be called before solve(). Passing an empty predicate restores
    /// counter `c` to its built-in default (DS/buffer drain at every
    /// consumer; tensor drains only at a barrier).
    ///
    /// Example -- make the tensor counter also drain at any DS op:
    ///   df.setRawNeedsWait(CK_Tensor, [](const StinkyInstruction& i) {
    ///       return isBarrier(i) || isDSRead(i) || isDSWrite(i);
    ///   });
    void setRawNeedsWait(CounterKind c, RawWaitPredicate pred);

    /// Enable or disable loop-carried tensor-token dependencies. Disabled by
    /// default: CK_Tensor dataflow is frozen after the first solver sweep, so
    /// tensor state does not propagate through back-edges. Enable this to
    /// restore the conservative behavior that iterates tensor queues normally.
    void setLoopCarriedTokenDepsEnabled(bool enabled) {
        loopCarriedTokenDepsEnabled = enabled;
    }

    /// Materialise the conservative per-consumer wait plan from the
    /// converged dataflow state. Run WaitPlanOptimizer(s) on the result,
    /// then finalizePlan() before emit.
    WaitInsertionPlan materializePlan() const;

    const DataflowResult& getResult() const {
        return result;
    }

    /// Replay blocks using the post-optimizer plan: apply tail drains to
    /// entry state, trim with final wait values, add missing anchors, and
    /// drop redundant waits. Must run after solve() and optimizers; uses
    /// getResult().entryState.
    void finalizePlan(WaitInsertionPlan& plan) const;

    /// Iteration cap. Public so tests can override.
    void setIterationCap(unsigned cap) {
        iterationCap = cap;
    }

   private:
    const std::vector<BasicBlock*>& rpo;
    DataflowResult result;

    /// Per-counter RAW-wait constraint, indexed by CounterKind. Seeded with
    /// the built-in defaults by the constructor; overridable via
    /// setRawNeedsWait(). Always non-empty during solve().
    std::array<RawWaitPredicate, CK_Count> rawNeedsWait;

    /// Plan recorded by the last transfer pass: each block's list of
    /// (anchor, WaitCountSpec) pairs in program order. Populated by
    /// transferBlock(); consumed by materializePlan().
    std::unordered_map<const BasicBlock*, std::vector<std::pair<StinkyInstruction*, WaitCountSpec>>>
        emitPlan;

    bool capHit = false;
    unsigned iterationCap = 0;
    bool loopCarriedTokenDepsEnabled = false;

    /// (block, counter) pairs whose per-pred queue overflowed the hardware
    /// in-flight window (kMaxInFlight) during a solver sweep -- i.e. the
    /// counter was issued past its window without draining and
    /// appendToAllPaths had to drop the oldest (provably-complete) ops.
    /// Cleared at the start of every sweep so that, at the fixed point, it
    /// holds exactly the steady-state overflows. Surfaced by solve() as a
    /// non-fatal warning; it never changes the emitted plan.
    std::set<std::pair<const BasicBlock*, CounterKind>> overflowSites;

    /// Emit a non-fatal diagnostic (in RPO order) for every (block, counter)
    /// in overflowSites. Called by solve() once the fixed point is reached.
    void reportCounterOverflow() const;

    /// Build entry state for BB by seeding one per-pred queue per CFG
    /// predecessor, including self-preds (back-edges): at the fixed point a
    /// back-edge's exit is the loop body's true exit, which is what the
    /// header must see. Identical (pred, ops) queues are deduplicated.
    ///
    /// The overload taking an explicit `exitState` map lets finalizePlan
    /// re-propagate against its own recomputed exit states instead of the
    /// solver's; the no-arg form merges from the solver's exit states.
    DataflowState mergeFromPredecessors(BasicBlock& bb) const;
    DataflowState mergeFromPredecessors(
        BasicBlock& bb,
        const std::unordered_map<const BasicBlock*, DataflowState>& exitState) const;

    /// Walk BB in program order, mutating STATE. Per-pred queues are kept
    /// (not collapsed) at exit; each is capped at kMaxInFlight.
    void transferBlock(BasicBlock& bb, DataflowState& state);
};

}  // namespace waitcnt
}  // namespace stinkytofu
