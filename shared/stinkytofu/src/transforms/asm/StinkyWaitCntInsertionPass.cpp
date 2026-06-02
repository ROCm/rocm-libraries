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

// ----------------------------------------------------------------------------
// StinkyWaitCntInsertionPass
//
// Inserts s_wait_* instructions so that asynchronous memory operations complete
// before their results are consumed. Dependencies are resolved by walking
// instruction-level def-use chains (not by tracking individual registers); PHI
// nodes are flattened to their incoming values at lookup time.
//
// The pass covers three independent hardware counters:
//   - DS      (dlcnt)  via s_wait_dscnt
//   - buffer  (vlcnt)  via s_wait_loadcnt
//   - tensor  (tlcnt)  via s_wait_tensorcnt
//
// Only basic blocks approved by PassContext::shouldProcessBasicBlock are
// analyzed and modified; the companion StinkyRemoveWaitCntPass is expected to
// have stripped stale waits beforehand so this pass owns every emitted wait.
//
// High-level flow (matches the "Pass Flow" diagram in the user doc):
//
//   buildUseDefChain          Setup: build instruction-level def-use chains
//                             with PHIs preserved for flattening.
//   Phase 1 buildBlockExitStates
//                             Pre-scan every processed block once; populate
//                             blockExitMemState with the DS / buffer / tensor
//                             ops in flight at each block's exit. The tensor
//                             state is stored as a single collapsed
//                             TensorPath (pred == self) for successor seeding.
//   Phase 2 computeRequiredWaits   (per block, reverse post-order)
//                             Seed ONE TensorPath per non-self CFG
//                             predecessor (preserving per-pred FIFO depth),
//                             walk the block in program order, and produce a
//                             list of (anchor, WaitCountSpec) pairs across
//                             all three counters. Tensor waits run through
//                             planTensorWait, which can record compensating
//                             drains for shallower predecessors. Trims the
//                             refined exit state back into blockExitMemState
//                             for successors (tensor paths collapsed to a
//                             single union view).
//   Phase 3 emitWaitInstructions   (per block, reverse post-order)
//                             Insert one s_wait_* IR node per non-kUnused
//                             field in each WaitCountSpec, immediately before
//                             the anchor.
//   Phase 3.5 emitTailTensorCompensations  (after the RPO walk)
//                             For every (predBlock, predRelativeWait)
//                             recorded by planTensorWait's correctives,
//                             insert an s_wait_tensorcnt at the
//                             predecessor's tail (before its terminator)
//                             so the shallower path drains its dep before
//                             the merge anchor sees it.
//   Phase 4 removePHIs        Strip PHI pseudo-instructions from processed
//                             blocks now that def-use chains have been
//                             consumed.
//
// For the full algorithm walk-through (per-instruction logic, redundancy
// elision, cross-block seeding, MemTokenData conservative fallbacks, etc.)
// see docs/user/stinky-waitcnt-insertion-pass.md.
// ----------------------------------------------------------------------------

#include "stinkytofu/transforms/asm/StinkyWaitCntInsertionPass.hpp"

#include <deque>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/BBIndexAnalysis.hpp"
#include "stinkytofu/analysis/controlflow/DominanceAnalysis.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/RegisterKey.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/CFGTraversal.hpp"
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"

#define DEBUG_TYPE "StinkyWaitCntInsertionPass"

namespace {
using namespace stinkytofu;

// ----------------------------------------------------------------------------
// Free-standing helpers
// ----------------------------------------------------------------------------

/// Recursively collect register sources of @p inst, flattening PHI operands so
/// the caller sees only the real memory ops behind each merge point. Non-PHI
/// sources are inserted directly; each PHI is replaced by its incoming values
/// (which are themselves recursively flattened). @p seenPhi tracks already-
/// visited PHIs to avoid infinite recursion on cyclic PHI webs such as the
/// loop-carried-dependency case described in the user doc's "Dependency
/// Resolution: collectSources" section.
static void collectSourcesRec(StinkyInstruction* inst,
                              std::unordered_set<StinkyInstruction*>& seenPhi,
                              std::unordered_set<StinkyInstruction*>& out,
                              const std::function<bool(StinkyInstruction*)>& filter = nullptr) {
    if (inst == nullptr) {
        return;
    }

    if (inst->getUnifiedOpcode() == GFX::PHI) {
        if (!seenPhi.insert(inst).second) {
            return;
        }
    }

    for (StinkyInstruction* src : inst->getSources()) {
        if (src == nullptr) {
            continue;
        }
        if (src->getUnifiedOpcode() == GFX::PHI) {
            collectSourcesRec(src, seenPhi, out, filter);
        } else {
            if (filter == nullptr || filter(src)) {
                out.insert(src);
            }
        }
    }
}

/// Entry point for source collection. PHIs are flattened to their incoming
/// values; an optional @p filter restricts which sources are kept (this pass
/// uses it to retain only DS / buffer memory ops).
std::unordered_set<StinkyInstruction*> collectSources(
    StinkyInstruction* inst, const std::function<bool(StinkyInstruction*)>& filter = nullptr) {
    std::unordered_set<StinkyInstruction*> sources;
    std::unordered_set<StinkyInstruction*> seenPhi;
    if (inst != nullptr) {
        collectSourcesRec(inst, seenPhi, sources, filter);
    }
    return sources;
}

bool isDSMemoryOp(const StinkyInstruction& inst) {
    return isDSRead(inst) || isDSWrite(inst);
}

bool isBufferMemoryOp(const StinkyInstruction& inst) {
    return isGlobalMemLoad(inst) || isGlobalMemStore(inst);
}

/// Returns true if any element of @p a is also present in @p b. The basic
/// MemTokenData primitive used by every token-based conflict check in this pass
/// (barrier-vs-DS, WAR-on-LDS, tensor anchor).
template <typename ContainerA, typename ContainerB>
bool hasTokenOverlap(const ContainerA& a, const ContainerB& b) {
    return std::any_of(a.begin(), a.end(),
                       [&b](int token) { return std::find(b.begin(), b.end(), token) != b.end(); });
}

/// True for any LDS writer anchor (currently @c tensor_load_to_lds or
/// @c ds_write). Identifies the writers structurally so the WAR-on-LDS scan in
/// @c collectLdsWarDependencies runs unconditionally; the writer's
/// @c MemTokenData (or lack thereof) only selects per-token vs. conservative
/// behaviour inside the helper.
bool isLdsWriterAnchor(const StinkyInstruction& inst) {
    return isTensorLoad(inst) || isDSWrite(inst);
}

/// True for instructions that may consume tensor-load output via LDS and
/// therefore need an @c s_wait_tensorcnt before they issue: barriers, DS reads,
/// DS writes, and DS atomics. Tensor loads themselves are deliberately excluded
/// because the hardware FIFO already orders them on the shared tlcnt counter.
/// Mirrors the anchor set tagged by @c StinkyBuildImplicitDependencyPass.
bool isTensorWaitAnchor(const StinkyInstruction& inst) {
    return isBarrier(inst) || isDSRead(inst) || isDSWrite(inst) || isDSAtomic(inst);
}

/// True when @p a and @p b share the same hardware memory pipeline, so the
/// hardware-side FIFO retirement orders them implicitly and no synthetic wait
/// is required between them.
///
/// Today this only covers the DS (dlcnt) pipeline: ds_read, ds_write and
/// ds_atomic all issue and retire FIFO on dlcnt. tensor_load_to_lds is NOT a
/// DS op (it lives on tlcnt), so a tensor_load_to_lds / ds_read pair is not
/// on the same pipeline.
bool isOnSameDSPipeline(const StinkyInstruction& a, const StinkyInstruction& b) {
    return isDSMemoryOp(a) && isDSMemoryOp(b);
}

// ----------------------------------------------------------------------------
// Data structures
// ----------------------------------------------------------------------------

/// Result of WAR-on-LDS dependency analysis for an LDS writer anchor.
///
/// When the writer carries MemTokenData and the per-token overlap path runs
/// normally, @c deps holds the conflicting prior DS reads/atomics and
/// @c forceDsDrain stays false. When the writer lacks MemTokenData, the pass
/// cannot prove disjointness against any pending DS op, so @c forceDsDrain is
/// set and @c deps is left empty: the caller emits @c s_wait_dscnt 0 instead
/// of running the per-pair @c min(count - 1) computation.
struct LdsWarResult {
    std::unordered_set<StinkyInstruction*> deps;
    bool forceDsDrain = false;
};

/// Result of tensor-wait dependency analysis for a tagged anchor (barrier / DS op).
///
/// When the anchor carries MemTokenData and the per-token overlap path runs
/// normally, @c deps holds the conflicting prior tensor loads and
/// @c forceTensorDrain stays false. When the anchor lacks MemTokenData, the
/// pass cannot prove disjointness against any pending tensor load, so
/// @c forceTensorDrain is set and @c deps is left empty: the caller emits
/// @c s_wait_tensorcnt 0 instead of running the per-pair @c min(count - 1)
/// computation.
struct TensorWaitDepResult {
    std::unordered_set<StinkyInstruction*> deps;
    bool forceTensorDrain = false;
};

/// Descriptor for the wait counter immediate(s) to emit before an anchor instruction.
/// Each field is either a non-negative count immediate or @c kUnused if not needed.
struct WaitCountSpec {
    /// Sentinel: this counter is not used for this wait record.
    static constexpr int kUnused = -1;

    /// Immediate for @c s_wait_dscnt (dlcnt); @c kUnused if omitted.
    int dsCount;
    /// Immediate for @c s_wait_loadcnt (vlcnt); @c kUnused if omitted.
    int bufferCount;
    /// Immediate for @c s_wait_tensorcnt (tlcnt); @c kUnused if omitted.
    int tensorCount;

    WaitCountSpec(int dsCount = kUnused, int bufferCount = kUnused, int tensorCount = kUnused)
        : dsCount(dsCount), bufferCount(bufferCount), tensorCount(tensorCount) {}

    bool isValid() const {
        return dsCount != kUnused || bufferCount != kUnused || tensorCount != kUnused;
    }
};

/// Output of @c planTensorWait: the wait immediate to emit AT the anchor plus
/// any compensating waits to insert at predecessors' tails to keep the anchor
/// wait as lenient as possible.
///
/// Why per-path correctives: when paths converging at @c bb have different
/// tensor FIFO depths, the strictest single anchor wait is
/// @c min(w_i) over constrained paths, which over-drains the deeper paths
/// and loses parallelism. Pre-draining the SHALLOWER paths at their pred's
/// tail isolates their dep from the anchor, letting the anchor relax to
/// @c max(w_i over correctable paths) -- the deepest path's strictest
/// wait. The deepest path drains exactly its dep; shallower paths arrive
/// already drained, so the anchor wait is a no-op for them.
struct TensorWaitPlan {
    /// True iff the anchor needs to emit @c s_wait_tensorcnt @c anchorWait.
    bool needsWait = false;
    /// Immediate for the anchor's @c s_wait_tensorcnt; valid iff @c needsWait.
    int anchorWait = 0;
    /// Pre-drains to insert at each predecessor's tail (before its terminator).
    /// Each pair is @c (predBlock, predRelativeWait) where @c predRelativeWait
    /// is the immediate to emit so that the pred's exit FIFO drains its
    /// portion of the strictest dep on that path. Empty when every
    /// constrained path already agrees on the same wait.
    std::vector<std::pair<BasicBlock*, int>> correctives;
    /// Per-path post-anchor "keep" size, aligned with
    /// @c tracker.pendingTensorLoadOps. Each entry is the number of NEWEST
    /// entries that remain in flight on that path after the anchor's wait
    /// AND any compensating predecessor drains take effect:
    ///   - correctable path with @c w_i < @c anchorWait: keep @c w_i
    ///     (pred-tail corrective drains the rest);
    ///   - any other constrained path: keep @c anchorWait (anchor's wait
    ///     drains down to that residual on this path);
    ///   - unconstrained or empty path: keep @c anchorWait (anchor's wait
    ///     still applies; trim is a no-op when @c queue.size() <= keep).
    /// The caller applies this per-path trim so bookkeeping reflects what
    /// hardware will actually see on each CFG path, preventing later anchors
    /// from emitting redundant drains for ops that the prior wait +
    /// correctives already retired.
    std::vector<int> perPathTrim;
};

/// Tracks in-flight (not yet waited-on) memory operations as ordered queues,
/// one queue per hardware counter (DS / buffer / tensor). Each processed basic
/// block has an associated tracker that represents its exit state; the
/// @c *CountFrom(inst) methods translate a queue position into the hardware
/// wait-count immediate. See the user doc's "PendingMemOpTracker" section for
/// the full field/method table.
struct PendingMemOpTracker {
    /// DS reads/writes in issue order.
    std::deque<StinkyInstruction*> pendingDSOps;
    /// MemTokenData tokens accumulated from pending DS ops (used for barrier conflict).
    std::unordered_set<int> activeDSTokens;
    /// Global loads/stores in issue order.
    std::deque<StinkyInstruction*> pendingBufferOps;

    /// Per-predecessor view of in-flight @c tensor_load_to_lds ops.
    ///
    /// Each @c TensorPath represents the tensor FIFO along ONE incoming CFG
    /// edge (or a synthetic "local-only" path when no predecessor state was
    /// available). @c pred identifies the predecessor whose exit FIFO seeded
    /// the entry; @c queue holds the pending loads on that path.
    ///
    /// During Phase 2 @c computeRequiredWaits the tracker holds one entry per
    /// CFG predecessor (skip-self), populated by
    /// @c seedTensorPathFromPredecessor. Local @c tensor_load_to_lds ops are
    /// appended to EVERY entry by @c recordTensorLoadOperation so each path's
    /// tail mirrors the in-block prefix.
    ///
    /// At block exit the tracker collapses to a single entry (the union of
    /// all paths, tagged with @c pred == self) for storage in
    /// @c blockExitMemState. Successors seed their own per-pred path from
    /// that single queue via @c tensorExitQueue().
    struct TensorPath {
        BasicBlock* pred = nullptr;
        std::deque<StinkyInstruction*> queue;
    };
    std::vector<TensorPath> pendingTensorLoadOps;

    int pendingDSCount() const {
        return static_cast<int>(pendingDSOps.size());
    }

    int pendingBufferCount() const {
        return static_cast<int>(pendingBufferOps.size());
    }

    /// Number of per-pred tensor path queues currently tracked.
    int pendingTensorLoadPathCount() const {
        return static_cast<int>(pendingTensorLoadOps.size());
    }

    /// True iff every per-path tensor queue is empty.
    bool noPendingTensorLoad() const {
        for (const auto& p : pendingTensorLoadOps) {
            if (!p.queue.empty()) {
                return false;
            }
        }
        return true;
    }

    /// Deduplicated union of all per-path tensor queues, preserving
    /// first-occurrence order. Used by @c collectTensorLoadDependencies to
    /// harvest the set of candidate tensor loads across every incoming path.
    std::deque<StinkyInstruction*> tensorLoadUnion() const {
        std::deque<StinkyInstruction*> u;
        std::unordered_set<StinkyInstruction*> seen;
        for (const auto& p : pendingTensorLoadOps) {
            for (StinkyInstruction* op : p.queue) {
                if (seen.insert(op).second) {
                    u.push_back(op);
                }
            }
        }
        return u;
    }

    /// Single-queue exit-FIFO view used by successors when seeding their own
    /// per-pred path from this block. Returns an empty deque if no exit
    /// state has been stored. Precondition: the tracker stores at most one
    /// path (collapsed exit form), enforced by
    /// @c collapseTensorPathsToExitView after Phase 2.
    const std::deque<StinkyInstruction*>& tensorExitQueue() const {
        static const std::deque<StinkyInstruction*> kEmpty;
        return pendingTensorLoadOps.empty() ? kEmpty : pendingTensorLoadOps.front().queue;
    }

    /// Number of DS ops from @p inst (inclusive) to the end of the queue. Returns 0 if
    /// @p inst is not in the queue. Maps directly to the hardware wait-count immediate.
    int pendingDSCountFrom(StinkyInstruction* inst) const {
        auto it = std::find(pendingDSOps.begin(), pendingDSOps.end(), inst);
        if (it != pendingDSOps.end()) {
            return static_cast<int>(std::distance(it, pendingDSOps.end()));
        }
        return 0;
    }

    /// Number of buffer ops from @p inst (inclusive) to the end of the queue. Returns 0
    /// if @p inst is not in the queue.
    int pendingBufferCountFrom(StinkyInstruction* inst) const {
        auto it = std::find(pendingBufferOps.begin(), pendingBufferOps.end(), inst);
        if (it != pendingBufferOps.end()) {
            return static_cast<int>(std::distance(it, pendingBufferOps.end()));
        }
        return 0;
    }

    /// True iff any in-flight DS op lacks MemTokenData. Used by the conservative
    /// barrier-vs-DS conflict path: an untagged pending DS op cannot be proven
    /// disjoint from a barrier, so a tagged barrier must still drain.
    bool hasUntaggedDSOp() const {
        return std::any_of(
            pendingDSOps.begin(), pendingDSOps.end(),
            [](const StinkyInstruction* op) { return op->getModifier<MemTokenData>() == nullptr; });
    }

    /// True iff any in-flight tensor load on any per-path queue lacks
    /// @c MemTokenData. Used by the conservative tensor-wait path: an
    /// untagged pending tensor load cannot be proven disjoint from a tagged
    /// anchor, so the anchor still drains.
    bool hasUntaggedTensorLoadOp() const {
        for (const auto& p : pendingTensorLoadOps) {
            for (StinkyInstruction* op : p.queue) {
                if (op->getModifier<MemTokenData>() == nullptr) {
                    return true;
                }
            }
        }
        return false;
    }

    void clear() {
        pendingDSOps.clear();
        pendingBufferOps.clear();
        pendingTensorLoadOps.clear();
        activeDSTokens.clear();
    }

    /// Append @p inst to the DS queue (and collect its tokens). Returns true iff appended.
    bool recordDSOperation(StinkyInstruction* inst) {
        if (inst == nullptr || !isDSMemoryOp(*inst)) {
            return false;
        }

        pendingDSOps.push_back(inst);
        if (auto* memTokenData = inst->getModifier<MemTokenData>()) {
            for (int token : memTokenData->tokens) {
                activeDSTokens.emplace(token);
            }
        }
        return true;
    }

    /// Append @p inst to the buffer queue. Returns true iff appended.
    bool recordBufferOperation(StinkyInstruction* inst) {
        if (inst == nullptr || !isBufferMemoryOp(*inst)) {
            return false;
        }

        pendingBufferOps.push_back(inst);
        return true;
    }

    /// Append @p inst to EVERY per-path tensor queue, ensuring at least one
    /// path entry exists (a synthetic local-only path with @c pred == nullptr
    /// is created if none was seeded). Returns true iff appended (i.e. when
    /// @p inst is a @c tensor_load_to_lds).
    ///
    /// Per the per-path model, a local tensor load is in flight on every
    /// incoming CFG path through this block, so it joins every path's tail.
    /// The pred-portion of each path is therefore the entries SEEDED at
    /// block entry; everything appended afterwards is the shared in-block
    /// prefix.
    bool recordTensorLoadOperation(StinkyInstruction* inst) {
        if (inst == nullptr || !isTensorLoad(*inst)) {
            return false;
        }
        if (pendingTensorLoadOps.empty()) {
            pendingTensorLoadOps.push_back({nullptr, {}});
        }
        for (auto& p : pendingTensorLoadOps) {
            p.queue.push_back(inst);
        }
        return true;
    }

    /// Seed a new per-predecessor tensor path with @p pred's exit FIFO. Used
    /// by @c computeRequiredWaits at block entry so the per-anchor planner
    /// can compute waits independently against each incoming CFG path.
    /// @c exitQueue is copied; the caller may continue to mutate @p pred's
    /// state without invalidating this seed.
    void seedTensorPathFromPredecessor(BasicBlock* pred,
                                       const std::deque<StinkyInstruction*>& exitQueue) {
        pendingTensorLoadOps.push_back({pred, exitQueue});
    }

    /// Clear every per-path tensor queue (after a full drain
    /// @c s_wait_tensorcnt 0).
    void clearAllTensorPaths() {
        for (auto& p : pendingTensorLoadOps) {
            p.queue.clear();
        }
    }

    /// Collapse all per-path queues into a single entry holding the union
    /// view, tagged with @p selfBlock. Called at block exit before storing
    /// the tracker into @c blockExitMemState[selfBlock]: successors only see
    /// this block as one CFG predecessor, so a single representative queue
    /// is sufficient.
    void collapseTensorPathsToExitView(BasicBlock* selfBlock) {
        auto u = tensorLoadUnion();
        pendingTensorLoadOps.clear();
        pendingTensorLoadOps.push_back({selfBlock, std::move(u)});
    }

    /// Trim @p queue so only the last @p lastWait ops remain. No-op when there is no
    /// prior wait recorded (lastWait < 0) or it was 0 (the wait-0 emission paths
    /// already clear the queue themselves).
    void trimQueueToLastWait(std::deque<StinkyInstruction*>& queue, int lastWait) {
        if (lastWait <= 0) {
            return;
        }
        if (lastWait >= static_cast<int>(queue.size())) {
            return;
        }
        queue.erase(queue.begin(), queue.end() - lastWait);
    }
};

/// Plan the tensor wait at an anchor given the current per-path tracker
/// state, the set of @p deps (token-overlapping prior tensor loads collected
/// by @c collectTensorLoadDependencies), the number of tensor loads issued
/// in the current block so far (@p inBlockTensorLoadCount), and the block
/// itself (@p selfBlock) for the self-loop guard.
///
/// Algorithm (see "Per-path tensor wait planner" in the user doc):
///   1. For each per-path queue, find the strictest dep (smallest
///      @c count - 1) and remember whether that dep sits in the pred-portion
///      of the path (i.e. it was seeded from the predecessor, not appended
///      locally during this block walk).
///   2. Partition constrained paths into CORRECTABLE (pred != nullptr,
///      pred != selfBlock, strictest dep is in pred-portion) and the rest.
///      Correctable paths can be drained at the pred's tail; the rest must
///      be satisfied by the anchor's wait alone.
///   3. @c anchorWait = @c min(@c max_w_correctable, @c min_w_uncorrectable)
///      so the anchor wait is as lenient as possible while still draining
///      every dep on every uncorrectable path.
///   4. For each correctable path with @c w_i < @c anchorWait, emit a
///      compensating @c s_wait_tensorcnt at the pred's tail with the
///      pred-relative immediate @c w_i - @c inBlockTensorLoadCount, so by
///      the time control reaches the anchor the pred has drained its share
///      and the anchor wait is a no-op for that path.
TensorWaitPlan planTensorWait(const PendingMemOpTracker& tracker,
                              const std::unordered_set<StinkyInstruction*>& deps,
                              int inBlockTensorLoadCount, BasicBlock* selfBlock) {
    TensorWaitPlan plan;
    // perPathTrim is always sized to match tracker.pendingTensorLoadOps so the
    // caller can index it without a length check.
    plan.perPathTrim.assign(tracker.pendingTensorLoadOps.size(), 0);
    if (deps.empty()) {
        return plan;
    }

    struct PathInfo {
        BasicBlock* pred;
        size_t pathIdx;
        int w;
        int wPred;
        bool correctable;
    };
    std::vector<PathInfo> infos;
    infos.reserve(tracker.pendingTensorLoadOps.size());

    for (size_t pathIdx = 0; pathIdx < tracker.pendingTensorLoadOps.size(); ++pathIdx) {
        const auto& path = tracker.pendingTensorLoadOps[pathIdx];
        const int queueSize = static_cast<int>(path.queue.size());
        if (queueSize == 0) {
            continue;
        }

        int bestW = std::numeric_limits<int>::max();
        int bestIndex = -1;
        for (StinkyInstruction* dep : deps) {
            auto it = std::find(path.queue.begin(), path.queue.end(), dep);
            if (it == path.queue.end()) {
                continue;
            }
            const int index = static_cast<int>(std::distance(path.queue.begin(), it));
            const int w = queueSize - index - 1;
            if (w < bestW) {
                bestW = w;
                bestIndex = index;
            }
        }
        if (bestIndex < 0) {
            continue;  // path has no dep in flight; unconstrained
        }

        // Local prefix = in-block tensor loads, appended uniformly to every
        // path. predEntrySize is the per-path pred FIFO depth at block entry.
        int predEntrySize = queueSize - inBlockTensorLoadCount;
        if (predEntrySize < 0) {
            predEntrySize = 0;
        }
        const bool strictestInPredPortion = bestIndex < predEntrySize;
        const bool correctable =
            (path.pred != nullptr && path.pred != selfBlock && strictestInPredPortion);

        PathInfo info{path.pred, pathIdx, bestW, 0, correctable};
        if (correctable) {
            int wPred = bestW - inBlockTensorLoadCount;
            if (wPred < 0) {
                wPred = 0;
            }
            info.wPred = wPred;
        }
        infos.push_back(info);
    }

    if (infos.empty()) {
        return plan;
    }

    int maxCorrectable = std::numeric_limits<int>::min();
    int minUncorrectable = std::numeric_limits<int>::max();
    for (const auto& info : infos) {
        if (info.correctable) {
            if (info.w > maxCorrectable) {
                maxCorrectable = info.w;
            }
        } else {
            if (info.w < minUncorrectable) {
                minUncorrectable = info.w;
            }
        }
    }

    int anchorWait;
    if (maxCorrectable == std::numeric_limits<int>::min()) {
        anchorWait = minUncorrectable;
    } else if (minUncorrectable == std::numeric_limits<int>::max()) {
        anchorWait = maxCorrectable;
    } else {
        anchorWait = std::min(maxCorrectable, minUncorrectable);
    }

    plan.needsWait = true;
    plan.anchorWait = anchorWait;

    // Default per-path trim: anchorWait. The anchor's wait drains every path
    // to at most anchorWait newest, so this is the conservative residual for
    // unconstrained / uncorrectable paths AND for correctable paths whose
    // strictest dep is deeper than anchorWait (where anchorWait was capped
    // by some uncorrectable path).
    std::fill(plan.perPathTrim.begin(), plan.perPathTrim.end(), anchorWait);
    for (const auto& info : infos) {
        if (info.correctable && info.w < anchorWait) {
            // Pre-drain at pred's tail leaves this path with w_i ≤ anchorWait
            // ops in flight at the anchor; the anchor wait is a no-op for it.
            plan.correctives.emplace_back(info.pred, info.wPred);
            plan.perPathTrim[info.pathIdx] = info.w;
        } else if (info.correctable && info.w == anchorWait) {
            // No corrective needed (path already matches anchor wait), but
            // mark its trim explicitly so bookkeeping stays in sync.
            plan.perPathTrim[info.pathIdx] = info.w;
        }
        // Otherwise leave perPathTrim[info.pathIdx] at the default anchorWait.
    }
    return plan;
}

// ----------------------------------------------------------------------------
// Pass class
// ----------------------------------------------------------------------------

class StinkyWaitCntInsertionPass : public StinkyInstPass {
   public:
    /// List of (anchor instruction, wait spec) pairs in emission order.
    using WaitInsertionList = std::vector<std::pair<StinkyInstruction*, WaitCountSpec>>;

    StinkyWaitCntInsertionPass() = default;

    static char ID;

    const char* getName() const override {
        return "StinkyWaitCntInsertionPass";
    }

    PassID getPassID() const override {
        return &StinkyWaitCntInsertionPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        GfxArchID arch =
            getGfxArchID(passCtx.getGemmTileConfig().arch[0], passCtx.getGemmTileConfig().arch[1],
                         passCtx.getGemmTileConfig().arch[2]);

        const auto& domInfo = AM.getResult<DominanceAnalysis>(func);
        buildUseDefChain(func, domInfo, true);
        const auto& rpo = AM.getResult<BBIndexAnalysis>(func).rpo;

        blockExitMemState.clear();
        tailTensorCompensations.clear();

        buildBlockExitStates(func, passCtx);

        for (auto* bb : rpo) {
            if (!passCtx.shouldProcessBasicBlock(*bb)) {
                continue;
            }
            WaitInsertionList waits = computeRequiredWaits(*bb);
            emitWaitInstructions(*bb, arch, waits);
        }

        // Per-path planner may have recorded compensating tensor drains at
        // shallower predecessors' tails to keep the merge anchor wait lenient.
        // Emit them now (after all per-block waits are in place) so the
        // pre-drain immediates target each pred's actual exit FIFO state.
        emitTailTensorCompensations(passCtx, arch);

        removePHIs(passCtx, rpo);
        return preserveCFGAnalyses();
    }

   private:
    /// Per-counter bookkeeping during a single block walk. Tracks the last emitted wait
    /// value and how many new ops have been issued since, enabling redundancy elision.
    struct CounterWaitState {
        int lastEmittedWait = WaitCountSpec::kUnused;
        int opsSinceLastWait = 0;

        void recordNewOp() {
            ++opsSinceLastWait;
        }

        bool needsNewWait(int required) const {
            return lastEmittedWait == WaitCountSpec::kUnused ||
                   lastEmittedWait + opsSinceLastWait > required;
        }

        void recordEmittedWait(int value) {
            lastEmittedWait = value;
            opsSinceLastWait = 0;
        }
    };

    /// Exit-state queues for each processed block (full block walk, program order).
    /// Pre-populated by buildBlockExitStates and refined per block by computeRequiredWaits.
    std::unordered_map<BasicBlock*, PendingMemOpTracker> blockExitMemState;

    /// Compensating @c s_wait_tensorcnt drains to emit at the tail (before
    /// the terminator) of selected predecessor blocks. Populated by
    /// @c recordTailTensorCompensation from per-anchor @c TensorWaitPlan
    /// correctives; consumed by @c emitTailTensorCompensations after the
    /// RPO walk completes. The stored immediate is the STRICTEST (smallest)
    /// pred-relative wait across all anchors that requested a drain for the
    /// same predecessor.
    std::unordered_map<BasicBlock*, int> tailTensorCompensations;

    /// Phase 1: pre-scan every processed block once, recording DS / buffer /
    /// tensor memory ops in program order. Populates @c blockExitMemState[bb]
    /// with the full set of in-flight ops at each block's exit before any
    /// waits are considered; Phase 2 later refines (trims) each entry.
    void buildBlockExitStates(Function& func, PassContext& passCtx) {
        for (BasicBlock& bb : func) {
            if (passCtx.shouldProcessBasicBlock(bb)) {
                blockExitMemState.emplace(&bb, PendingMemOpTracker());
                scanBlockMemOps(bb);
            }
        }
    }

    /// Helper for buildBlockExitStates: record DS, buffer, and tensor-load ops
    /// in @p bb in program order. Each @c record*Operation call is a no-op for
    /// instructions that do not match the corresponding queue.
    ///
    /// After the scan, the tensor path set is collapsed to a single entry
    /// tagged with @p &bb so successors can seed their own per-pred path from
    /// the resulting @c tensorExitQueue().
    void scanBlockMemOps(BasicBlock& bb) {
        PendingMemOpTracker& state = blockExitMemState[&bb];
        state.clear();
        for (IRBase& ir : bb) {
            StinkyInstruction* inst = dyn_cast<StinkyInstruction>(&ir);
            if (inst == nullptr) {
                continue;
            }

            state.recordDSOperation(inst);
            state.recordBufferOperation(inst);
            state.recordTensorLoadOperation(inst);
        }
        state.collapseTensorPathsToExitView(&bb);
    }

    /// Compute the minimum wait-count value for a single counter (DS or buffer) given
    /// the consumer's memory-op dependencies. The @p getCountFrom callback delegates to
    /// either pendingDSCountFrom or pendingBufferCountFrom so the same algorithm covers
    /// both counters. Returns (requiredWait, needsWait).
    ///
    /// See "Wait Value Computation: computeWaitValueForCounter" in the user docs.
    std::pair<int, bool> computeWaitValueForCounter(
        int initialPendingCount, const PendingMemOpTracker& localState,
        const std::unordered_set<StinkyInstruction*>& memOpDependencies,
        const CounterWaitState& counterState,
        const std::function<int(const PendingMemOpTracker&, StinkyInstruction*)>& getCountFrom) {
        int requiredWait = initialPendingCount;
        bool needsWait = false;
        std::vector<int> predecessorValues;

        for (StinkyInstruction* src : memOpDependencies) {
            int count = getCountFrom(localState, src);
            if (count > 0) {
                requiredWait = std::min(requiredWait, count - 1);
                needsWait = true;
            } else if (counterState.lastEmittedWait != 0) {
                // Unprocessed predecessor blocks insert an empty state via operator[].
                count = getCountFrom(blockExitMemState[src->getParent()], src);
                if (count > 0) {
                    predecessorValues.push_back(count - 1);
                    needsWait = true;
                }
            }
        }

        if (!predecessorValues.empty()) {
            requiredWait += *std::min_element(predecessorValues.begin(), predecessorValues.end());
        }

        return {requiredWait, needsWait};
    }

    /// Collect prior DS reads/atomics whose MemTokenData tokens overlap @p writer's.
    /// Scans @p localState (current block) and each predecessor's exit state to
    /// surface the WAR-on-LDS hazard that the SSA def-use chain does not encode.
    ///
    /// Per-pair pipeline filter: if a candidate reader is on the same hardware
    /// memory pipeline as @p writer (see isOnSameDSPipeline), it is skipped --
    /// the hardware FIFO-retires the pair on the shared counter, so no
    /// synthetic wait is required. In practice this excludes all candidates
    /// when @p writer is a ds_write (both writer and readers live on dlcnt),
    /// while leaving tensor_load_to_lds writers (on tlcnt) fully covered.
    ///
    /// Conservative fallback (hybrid policy for missing MemTokenData):
    ///   - Writer lacks MemTokenData: we cannot prove disjointness against any
    ///     pending DS read/atomic on a different pipeline. Returns with
    ///     @c forceDsDrain == true; the caller emits @c s_wait_dscnt 0.
    ///   - Candidate reader lacks MemTokenData (writer has tokens): conservatively
    ///     treat it as conflicting and add it to @c deps so the normal
    ///     @c min(count - 1) algorithm computes the wait value.
    LdsWarResult collectLdsWarDependencies(StinkyInstruction* writer, BasicBlock& bb,
                                           const PendingMemOpTracker& localState) {
        LdsWarResult result;
        const MemTokenData* writerTokens = writer->getModifier<MemTokenData>();
        const bool writerLacksTokens = (writerTokens == nullptr);

        auto isCandidate = [&](StinkyInstruction* op) {
            if (op == nullptr || op == writer) {
                return false;
            }
            if (!isDSRead(*op) && !isDSAtomic(*op)) {
                return false;
            }
            // Same-pipeline pairs are FIFO-ordered by hardware, so no synthetic
            // WAR wait is needed. Today this matches ds_write writer paired
            // with ds_read / ds_atomic readers on the DS (dlcnt) pipeline.
            return !isOnSameDSPipeline(*writer, *op);
        };

        if (writerLacksTokens) {
            // Conservative branch: check whether any non-same-pipeline candidate
            // exists. If so, force a full DS drain; otherwise nothing to do.
            auto anyCandidate = [&](const std::deque<StinkyInstruction*>& q) {
                return std::any_of(q.begin(), q.end(), isCandidate);
            };
            if (anyCandidate(localState.pendingDSOps)) {
                result.forceDsDrain = true;
                PASS_DEBUG(std::cerr << "[WaitCntInsertion] conservative DS drain: LDS writer "
                                     << "lacks MemTokenData (" << writer->getHwInstDesc()->mnemonic
                                     << ")\n");
                return result;
            }
            for (BasicBlock* pred : bb.getPredecessors()) {
                auto it = blockExitMemState.find(pred);
                if (it == blockExitMemState.end()) {
                    continue;
                }
                if (anyCandidate(it->second.pendingDSOps)) {
                    result.forceDsDrain = true;
                    PASS_DEBUG(std::cerr << "[WaitCntInsertion] conservative DS drain: LDS writer "
                                         << "lacks MemTokenData ("
                                         << writer->getHwInstDesc()->mnemonic
                                         << "), candidate in predecessor block\n");
                    return result;
                }
            }
            return result;
        }

        // Writer has tokens: per-token overlap path with conservative widening
        // for candidates that lack MemTokenData.
        auto isConflictingRead = [&](StinkyInstruction* op) {
            if (!isCandidate(op)) {
                return false;
            }
            const MemTokenData* mt = op->getModifier<MemTokenData>();
            if (mt == nullptr) {
                PASS_DEBUG(std::cerr << "[WaitCntInsertion] conservative WAR include: candidate "
                                     << "lacks MemTokenData (" << op->getHwInstDesc()->mnemonic
                                     << "), writer (" << writer->getHwInstDesc()->mnemonic
                                     << ")\n");
                return true;
            }
            return hasTokenOverlap(mt->tokens, writerTokens->tokens);
        };

        for (StinkyInstruction* op : localState.pendingDSOps) {
            if (isConflictingRead(op)) {
                result.deps.insert(op);
            }
        }

        for (BasicBlock* pred : bb.getPredecessors()) {
            auto it = blockExitMemState.find(pred);
            if (it == blockExitMemState.end()) {
                continue;
            }
            for (StinkyInstruction* op : it->second.pendingDSOps) {
                if (isConflictingRead(op)) {
                    result.deps.insert(op);
                }
            }
        }

        return result;
    }

    /// Collect prior tensor loads whose MemTokenData tokens overlap @p anchor's.
    /// Walks every per-path tensor queue in @p localState. @c computeRequiredWaits
    /// seeds one path per predecessor's exit FIFO at block entry (via
    /// @c PendingMemOpTracker::seedTensorPathFromPredecessor) and appends local
    /// loads to every path via @c recordTensorLoadOperation, so iterating each
    /// path covers cross-block + in-block in-flight tensor state. Surfaces
    /// dependencies that flow through LDS pseudo-registers rather than the SSA
    /// def-use chain.
    ///
    /// Anchors are barriers / DS reads / DS writes / DS atomics (see
    /// @c isTensorWaitAnchor). Tensor loads themselves are excluded because
    /// the hardware FIFO already orders them on the shared tlcnt counter, so
    /// no synthetic pipeline filter is needed here.
    ///
    /// Conservative fallback (hybrid policy for missing MemTokenData):
    ///   - Anchor lacks MemTokenData: we cannot prove disjointness against any
    ///     pending tensor load on any path. Returns with
    ///     @c forceTensorDrain == true; the caller emits
    ///     @c s_wait_tensorcnt 0.
    ///   - Candidate tensor load lacks MemTokenData (anchor has tokens):
    ///     conservatively treat it as conflicting and add it to @c deps so the
    ///     per-path @c planTensorWait planner computes the wait value.
    TensorWaitDepResult collectTensorLoadDependencies(StinkyInstruction* anchor, BasicBlock& bb,
                                                      const PendingMemOpTracker& localState) {
        (void)bb;
        TensorWaitDepResult result;
        const MemTokenData* anchorTokens = anchor->getModifier<MemTokenData>();
        const bool anchorLacksTokens = (anchorTokens == nullptr);

        auto isCandidate = [&](StinkyInstruction* op) {
            if (op == nullptr || op == anchor) {
                return false;
            }
            return isTensorLoad(*op);
        };

        if (anchorLacksTokens) {
            for (const auto& path : localState.pendingTensorLoadOps) {
                if (std::any_of(path.queue.begin(), path.queue.end(), isCandidate)) {
                    result.forceTensorDrain = true;
                    PASS_DEBUG(std::cerr << "[WaitCntInsertion] conservative tensor drain: anchor "
                                         << "lacks MemTokenData ("
                                         << anchor->getHwInstDesc()->mnemonic << ")\n");
                    break;
                }
            }
            return result;
        }

        auto isConflictingLoad = [&](StinkyInstruction* op) {
            if (!isCandidate(op)) {
                return false;
            }
            const MemTokenData* mt = op->getModifier<MemTokenData>();
            if (mt == nullptr) {
                PASS_DEBUG(std::cerr << "[WaitCntInsertion] conservative tensor include: candidate "
                                     << "lacks MemTokenData (" << op->getHwInstDesc()->mnemonic
                                     << "), anchor (" << anchor->getHwInstDesc()->mnemonic
                                     << ")\n");
                return true;
            }
            return hasTokenOverlap(mt->tokens, anchorTokens->tokens);
        };

        // Walk every per-path queue. Dedupe via the result set itself.
        for (const auto& path : localState.pendingTensorLoadOps) {
            for (StinkyInstruction* op : path.queue) {
                if (isConflictingLoad(op)) {
                    result.deps.insert(op);
                }
            }
        }

        return result;
    }

    /// Phase 2: walk @p bb in program order and return the list of
    /// @c (anchor, WaitCountSpec) pairs that the emit phase will materialise.
    ///
    /// Outline (full per-step contract lives in the user doc, "Core Algorithm:
    /// computeRequiredWaits"):
    ///   0. Block entry -- for each non-self predecessor with stored exit
    ///      state, seed a per-pred @c TensorPath into
    ///      @c localState.pendingTensorLoadOps via
    ///      @c seedTensorPathFromPredecessor. DS / buffer queues are
    ///      intentionally NOT seeded here; their helpers walk predecessors
    ///      explicitly.
    ///   1. Tensor-wait anchor (barrier / DS read / DS write / DS atomic):
    ///      run @c collectTensorLoadDependencies over every per-path queue,
    ///      then call @c planTensorWait to produce the anchor wait plus any
    ///      compensating pre-drains for shallower paths. Force a full drain
    ///      on @c forceTensorDrain.
    ///   2. Barrier vs. DS conflict: drain on token overlap, on a barrier that
    ///      lacks @c MemTokenData, or on an untagged pending DS op.
    ///   3. Collect this instruction's DS / buffer dependencies via
    ///      @c collectSources (PHIs flattened, filtered to memory ops).
    ///   4. WAR-on-LDS for @c isLdsWriterAnchor: run
    ///      @c collectLdsWarDependencies; either widen the dependency set or
    ///      set @c forceDsDrain when the writer lacks @c MemTokenData.
    ///   5-6. Per-counter wait values via @c computeWaitValueForCounter, with
    ///        redundancy elision via @c CounterWaitState::needsNewWait.
    ///   7. Tail recording: only AFTER any wait is emitted, push @c inst onto
    ///      the relevant queues so the wait's queue snapshot excludes its own
    ///      consumer (matches hardware semantics). For tensor loads, the
    ///      record fans out to EVERY per-pred path queue.
    ///   8-9. Trim DS / buffer exit-state queues using each counter's
    ///        @c lastEmittedWait; collapse the per-pred tensor paths into a
    ///        single union exit view; store the refined state into
    ///        @c blockExitMemState for successors.
    WaitInsertionList computeRequiredWaits(BasicBlock& bb) {
        PendingMemOpTracker localState;
        WaitInsertionList waits;
        CounterWaitState dsState;
        CounterWaitState bufferState;
        CounterWaitState tensorState;
        // Local tensor-load prefix size (number of tensor_load_to_lds recorded
        // since block entry). Used by planTensorWait to split each path's
        // queue into pred-portion vs. in-block prefix so correctives target
        // the pred's tail with the correct pred-relative immediate.
        int inBlockTensorLoadCount = 0;

        // Seed one per-pred TensorPath from each non-self predecessor's
        // collapsed exit FIFO. Self-pred back-edges are skipped: their
        // Phase-1 raw scan includes loads issued AFTER the consumer (in the
        // same body), and treating them as in-flight at the header would
        // over-count and produce an unsafely-large wait immediate.
        for (BasicBlock* pred : bb.getPredecessors()) {
            if (pred == &bb) {
                continue;
            }
            auto it = blockExitMemState.find(pred);
            if (it == blockExitMemState.end()) {
                continue;
            }
            localState.seedTensorPathFromPredecessor(pred, it->second.tensorExitQueue());
        }

        for (auto it = bb.begin(); it != bb.end(); ++it) {
            StinkyInstruction* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (inst == nullptr || inst->getUnifiedOpcode() == GFX::PHI) {
                continue;
            }

            // Tensor-wait check: per-path planner.
            if (isTensorWaitAnchor(*inst)) {
                auto twResult = collectTensorLoadDependencies(inst, bb, localState);
                if (twResult.forceTensorDrain) {
                    if (tensorState.needsNewWait(0)) {
                        waits.emplace_back(
                            inst, WaitCountSpec(WaitCountSpec::kUnused, WaitCountSpec::kUnused, 0));
                        tensorState.recordEmittedWait(0);
                        localState.clearAllTensorPaths();
                    }
                } else if (!twResult.deps.empty()) {
                    TensorWaitPlan plan =
                        planTensorWait(localState, twResult.deps, inBlockTensorLoadCount, &bb);
                    if (plan.needsWait && tensorState.needsNewWait(plan.anchorWait)) {
                        waits.emplace_back(
                            inst, WaitCountSpec(WaitCountSpec::kUnused, WaitCountSpec::kUnused,
                                                plan.anchorWait));
                        tensorState.recordEmittedWait(plan.anchorWait);
                        for (const auto& corrective : plan.correctives) {
                            recordTailTensorCompensation(corrective.first, corrective.second);
                        }
                        // Apply per-path trim so each path's bookkeeping matches what
                        // the hardware sees on that CFG path after the anchor's wait
                        // AND the pred-tail correctives take effect. Trimming uniformly
                        // to anchorWait would leave shallow correctable paths "still
                        // holding" their drained ops in bookkeeping, causing later
                        // anchors in the same block to emit redundant drains.
                        for (size_t pathIdx = 0; pathIdx < localState.pendingTensorLoadOps.size();
                             ++pathIdx) {
                            const int keep = (pathIdx < plan.perPathTrim.size())
                                                 ? plan.perPathTrim[pathIdx]
                                                 : plan.anchorWait;
                            auto& q = localState.pendingTensorLoadOps[pathIdx].queue;
                            if (keep <= 0) {
                                q.clear();
                            } else if (static_cast<int>(q.size()) > keep) {
                                q.erase(q.begin(), q.end() - keep);
                            }
                        }
                    }
                }
            }

            // Barrier-vs-DS conflict (see doc: section "Barrier token conflict handling").
            // The original trigger fires on a tagged barrier whose tokens overlap with
            // activeDSTokens. The conservative fallback adds two extra triggers for
            // the missing-MemTokenData case:
            //   * barrier lacks MemTokenData (anchor-missing-tokens branch)
            //   * any pending DS op lacks MemTokenData (candidate-missing-tokens branch)
            // In all cases the action is the same: drain via s_wait_dscnt 0 and clear
            // DS state. Gated on !pendingDSOps.empty() so we never emit a no-op drain.
            // Barriers are never DS / buffer ops, so the tail recording at the bottom
            // of this loop body is a no-op for them; the explicit continue keeps the
            // existing semantics.
            if (isBarrier(*inst)) {
                const auto* memTokenData = inst->getModifier<MemTokenData>();
                const bool barrierLacksTokens = (memTokenData == nullptr);
                const bool tokenOverlap =
                    !barrierLacksTokens &&
                    hasTokenOverlap(localState.activeDSTokens, memTokenData->tokens);
                const bool untaggedPendingDSOp = localState.hasUntaggedDSOp();
                const bool needsDrain = !localState.pendingDSOps.empty() &&
                                        (tokenOverlap || barrierLacksTokens || untaggedPendingDSOp);
                if (needsDrain) {
                    if (barrierLacksTokens) {
                        PASS_DEBUG(std::cerr << "[WaitCntInsertion] conservative barrier drain: "
                                             << "barrier lacks MemTokenData (" << bb.getLabel()
                                             << ")\n");
                    } else if (untaggedPendingDSOp && !tokenOverlap) {
                        PASS_DEBUG(std::cerr << "[WaitCntInsertion] conservative barrier drain: "
                                             << "pending DS op lacks MemTokenData ("
                                             << bb.getLabel() << ")\n");
                    }
                    waits.emplace_back(inst, WaitCountSpec(0, WaitCountSpec::kUnused));
                    dsState.recordEmittedWait(0);
                    localState.activeDSTokens.clear();
                    localState.pendingDSOps.clear();
                    continue;
                }
            }

            auto memOpDependencies = collectSources(inst, [](StinkyInstruction* src) {
                return isDSMemoryOp(*src) || isBufferMemoryOp(*src);
            });

            // WAR-on-LDS: an LDS writer (tensor_load_to_lds / ds_write) must wait
            // for prior token-overlapping DS reads to drain. Run this before the
            // tail recording so that the writer is not yet in pendingDSOps, which
            // makes the count-based wait value match hardware semantics: the wait
            // executes before the writer issues, so the queue at wait-time excludes
            // the writer.
            //
            // Conservative branch: when the writer lacks MemTokenData,
            // collectLdsWarDependencies returns @c forceDsDrain == true and we
            // emit @c s_wait_dscnt 0 instead of merging deps. The buffer counter
            // is still driven by the SSA dependencies the normal way.
            bool ldsForceDsDrain = false;
            if (isLdsWriterAnchor(*inst)) {
                auto warResult = collectLdsWarDependencies(inst, bb, localState);
                if (warResult.forceDsDrain) {
                    ldsForceDsDrain = true;
                } else {
                    memOpDependencies.insert(warResult.deps.begin(), warResult.deps.end());
                }
            }

            if (ldsForceDsDrain && dsState.needsNewWait(0)) {
                waits.emplace_back(inst, WaitCountSpec(0, WaitCountSpec::kUnused));
                dsState.recordEmittedWait(0);
                localState.activeDSTokens.clear();
                localState.pendingDSOps.clear();
            }

            if (!memOpDependencies.empty()) {
                // Compute waits against the pre-consumer queue. For DS-op / buffer-op
                // consumers, this gives the correct dlcnt / vlcnt because the wait
                // executes before the consumer is issued.
                if (!ldsForceDsDrain) {
                    auto dsResult = computeWaitValueForCounter(
                        localState.pendingDSCount(), localState, memOpDependencies, dsState,
                        [](const PendingMemOpTracker& tracker, StinkyInstruction* src) {
                            return tracker.pendingDSCountFrom(src);
                        });
                    if (dsResult.second && dsState.needsNewWait(dsResult.first)) {
                        waits.emplace_back(inst,
                                           WaitCountSpec(dsResult.first, WaitCountSpec::kUnused));
                        dsState.recordEmittedWait(dsResult.first);
                    }
                }

                auto bufferResult = computeWaitValueForCounter(
                    localState.pendingBufferCount(), localState, memOpDependencies, bufferState,
                    [](const PendingMemOpTracker& tracker, StinkyInstruction* src) {
                        return tracker.pendingBufferCountFrom(src);
                    });
                if (bufferResult.second && bufferState.needsNewWait(bufferResult.first)) {
                    waits.emplace_back(inst,
                                       WaitCountSpec(WaitCountSpec::kUnused, bufferResult.first));
                    bufferState.recordEmittedWait(bufferResult.first);
                }
            }

            // Tail recording: after any wait for this instruction has been emitted,
            // push inst onto the trackers so future iterations and the block exit
            // state see it as in-flight. Tensor loads fan out to EVERY per-pred
            // path queue inside recordTensorLoadOperation, so each path's tail
            // mirrors the in-block prefix.
            if (localState.recordDSOperation(inst)) {
                dsState.recordNewOp();
            }
            if (localState.recordBufferOperation(inst)) {
                bufferState.recordNewOp();
            }
            if (localState.recordTensorLoadOperation(inst)) {
                tensorState.recordNewOp();
                ++inBlockTensorLoadCount;
            }
        }

        // Trim DS / buffer exit-state queues using each counter's last emitted
        // wait so successor blocks see only ops still genuinely in flight.
        localState.trimQueueToLastWait(localState.pendingDSOps, dsState.lastEmittedWait);
        localState.trimQueueToLastWait(localState.pendingBufferOps, bufferState.lastEmittedWait);
        // Tensor: per-path queues were trimmed at each anchor; collapse the
        // multi-path working state into a single union exit view tagged with
        // &bb for successor seeding.
        localState.collapseTensorPathsToExitView(&bb);

        blockExitMemState[&bb] = localState;
        return waits;
    }

    /// Phase 3: insert wait IR nodes immediately before each anchor. For every
    /// non-@c kUnused field in a @c WaitCountSpec, one instruction is emitted
    /// per the mapping in the user doc's "Emit Phase: emitWaitInstructions"
    /// table:
    ///   dsCount     -> s_wait_dscnt      (SWaitCntData.dlcnt)
    ///   bufferCount -> s_wait_loadcnt    (SWaitCntData.vlcnt)
    ///   tensorCount -> s_wait_tensorcnt  (SWaitTensorCntData.tlcnt)
    void emitWaitInstructions(BasicBlock& bb, GfxArchID arch, const WaitInsertionList& waits) {
        auto irBuilder = AsmIRBuilder(bb, arch);

        for (const auto& entry : waits) {
            StinkyInstruction* inst = entry.first;
            const WaitCountSpec& waitSpec = entry.second;

            if (waitSpec.dsCount != WaitCountSpec::kUnused) {
                StinkyInstruction* waitInst =
                    irBuilder.create(getMCIDByUOp(GFX::s_wait_dscnt, arch), inst);
                waitInst->addSrcReg(StinkyRegister(waitSpec.dsCount));

                SWaitCntData waitCntData;
                waitCntData.dlcnt = waitSpec.dsCount;
                waitInst->addModifier<SWaitCntData>(waitCntData);
            }

            if (waitSpec.bufferCount != WaitCountSpec::kUnused) {
                StinkyInstruction* waitInst =
                    irBuilder.create(getMCIDByUOp(GFX::s_wait_loadcnt, arch), inst);
                waitInst->addSrcReg(StinkyRegister(waitSpec.bufferCount));

                SWaitCntData waitCntData;
                waitCntData.vlcnt = waitSpec.bufferCount;
                waitInst->addModifier<SWaitCntData>(waitCntData);
            }

            if (waitSpec.tensorCount != WaitCountSpec::kUnused) {
                StinkyInstruction* waitInst =
                    irBuilder.create(getMCIDByUOp(GFX::s_wait_tensorcnt, arch), inst);
                waitInst->addSrcReg(StinkyRegister(waitSpec.tensorCount));

                SWaitTensorCntData waitCntData;
                waitCntData.tlcnt = waitSpec.tensorCount;
                waitInst->addModifier<SWaitTensorCntData>(waitCntData);
            }
        }
    }

    /// Record a compensating @c s_wait_tensorcnt drain to insert at the tail
    /// (before the terminator) of @p predBB. Per-path @c planTensorWait
    /// emits these when a merge anchor has divergent per-pred waits, draining
    /// the shallower preds early so the anchor can use the deepest path's
    /// wait. If multiple anchors target the same predecessor with different
    /// pred-relative immediates, the STRICTEST (smallest) wins so every
    /// requesting anchor is satisfied.
    void recordTailTensorCompensation(BasicBlock* predBB, int predRelativeWait) {
        if (predBB == nullptr) {
            return;
        }
        auto [it, inserted] = tailTensorCompensations.emplace(predBB, predRelativeWait);
        if (!inserted && predRelativeWait < it->second) {
            it->second = predRelativeWait;
        }
    }

    /// Materialise the compensating @c s_wait_tensorcnt drains recorded by
    /// @c recordTailTensorCompensation. Each entry produces one
    /// @c s_wait_tensorcnt placed immediately before the predecessor's
    /// terminator (or appended if the block has no branch terminator). Skips
    /// preds that were never processed by this pass (no entry in
    /// @c blockExitMemState) and preds whose stored exit tensor FIFO is
    /// already shorter than the recorded immediate (in which case the drain
    /// would be a hardware no-op anyway).
    void emitTailTensorCompensations(PassContext& passCtx, GfxArchID arch) {
        for (const auto& entry : tailTensorCompensations) {
            BasicBlock* predBB = entry.first;
            int predRelativeWait = entry.second;
            if (predBB == nullptr || !passCtx.shouldProcessBasicBlock(*predBB)) {
                continue;
            }
            auto stateIt = blockExitMemState.find(predBB);
            if (stateIt == blockExitMemState.end()) {
                continue;
            }
            const auto& exitQueue = stateIt->second.tensorExitQueue();
            if (static_cast<int>(exitQueue.size()) <= predRelativeWait) {
                continue;  // no-op: pred has already drained below the target
            }

            AsmIRBuilder builder(*predBB, arch);
            IRBase* term = predBB->getTerminator();
            StinkyInstruction* termInst = term ? dyn_cast<StinkyInstruction>(term) : nullptr;
            StinkyInstruction* anchor =
                (termInst != nullptr && isBranch(*termInst)) ? termInst : nullptr;

            StinkyInstruction* waitInst =
                builder.create(getMCIDByUOp(GFX::s_wait_tensorcnt, arch), anchor);
            waitInst->addSrcReg(StinkyRegister(predRelativeWait));
            SWaitTensorCntData waitCntData;
            waitCntData.tlcnt = predRelativeWait;
            waitInst->addModifier<SWaitTensorCntData>(waitCntData);
        }
    }

    /// Phase 4: strip PHI pseudo-instructions from processed blocks now that
    /// def-use chains have been fully consumed by Phases 1-3.
    void removePHIs(PassContext& passCtx, const std::vector<BasicBlock*>& rpo) {
        for (auto* bb : rpo) {
            if (!passCtx.shouldProcessBasicBlock(*bb)) {
                continue;
            }

            for (auto it = bb->begin(); it != bb->end();) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst && inst->getUnifiedOpcode() == GFX::PHI) {
                    it = bb->eraseIR(it);
                } else {
                    ++it;
                }
            }
        }
    }
};

char StinkyWaitCntInsertionPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createStinkyWaitCntInsertionPass() {
    return std::make_unique<StinkyWaitCntInsertionPass>();
}
}  // namespace stinkytofu
