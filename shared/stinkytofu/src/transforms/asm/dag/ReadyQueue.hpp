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

#include <climits>
#include <cmath>
#include <iostream>  // TODO: don't use iostream.
#include <map>
#include <queue>
#include <utility>
#include <vector>

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/LoopDetection.hpp"
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"

namespace stinkytofu {
namespace dag {

// Defined in RegionDAG.hpp, which includes THIS header -- forward-declared to keep the
// dependency one-way. Region pre-scans in derived queues read the same graph the
// scheduler drains, rather than rebuilding their own view of it.
struct RegionDAG;

// REMOVED: Local buildUseDefChain() has been replaced by stinkytofu::buildUseDefChain()
// from BuildDefUseChain.hpp. All callers now use the shared implementation.

// One (rule, register) hazard this node's issue must stamp: this node is a
// producer under the arch's hazard rule table [ruleIdx] and writes the register at
// regKey (regDepKey — register type folded in). Filled by the pre-scan.
struct HazardFlag {
    int ruleIdx;
    int regKey;
};

struct DAGNode {
    StinkyInstruction* inst;
    unsigned inDegree;
    unsigned id;
    // Pre-computed pick order for ds_reads. Lower = pick first.
    // Assigned by the pre-scan in scheduleRegionWithMovableSideEffects
    // based on DsReadOrder config and WMMA consumer analysis.
    unsigned dsReadPriority = UINT_MAX;
    // Packed s_set_vgpr_msb immediate this op needs (computeRequiredMsb); -1 = no MSB
    // opinion. Filled by the pre-scan; drives the MSB-affinity tiebreak in pickFreeBest.
    int requiredMsb = -1;
    // Hardware hazard: the exact (rule, register) pairs this node writes that some
    // later consumer reads, per the arch's hazard rule table (a fixed producer->consumer cycle
    // gap keyed by register file). Filled by the pre-scan via the def-use user walk.
    // Non-empty means this node's issue must stamp the corresponding hazard gate(s)
    // (see CDNA5ReadyQueue::hazardGates_) so the consumer waits the gap out. That
    // gate correctly blocks the consumer whenever *real* intervening instructions are
    // available to cover the wait; when none are, the scheduler's existing "pay the
    // remaining wait via advanceTime, then issue anyway" fallback still applies (see
    // findSmallestPickableNonWmma/pickFreeBest) and advances the simulated clock_
    // without a matching instruction — so the gate is not a substitute for
    // hazardDeadline actually reserving enough real cycles; both need to be right.
    // hazardDeadline below drives a *throughput* heuristic on top of the gate.
    std::vector<HazardFlag> hazardFlags;
    // Set by the pre-scan (see scheduleRegionWithMovableSideEffects) only when this
    // node has hazardFlags: the latest CDNA5ReadyQueue::clock_ value at which this
    // producer may still be deferred. Computed as X - rule.cycles - (this producer's
    // own issue/latency cost), where X is the hazarded consumer's estimated absolute
    // cycle position (a forward prefix sum over the region in original program
    // order) — i.e. "this producer must FINISH by t = X - cycles" (the gate is
    // stamped only after the producer's own advanceTime has already run, so the
    // deadline must reserve that cost too), the tightest (minimum) such deadline over
    // every rule/consumer this node feeds. INT_MAX (the default) means no deadline
    // applies. Compared against the live clock_ in CDNA5ReadyQueue::decidePromote():
    // unlike a readiness check, clock_ only advances via cycles actually issued, so
    // this can't fire early just because some unrelated node happens to be structurally
    // ready sooner than it is actually scheduled. Still an estimate (X is computed
    // from original order, which real scheduling may depart from) — an inaccurate
    // deadline shifts when the mandatory force kicks in, and (per above) can leave the
    // gate short of real cycles to cover the gap with, falling back to a simulated
    // wait that has no matching instruction.
    int hazardDeadline = INT_MAX;
    // --- Cluster-barrier SCC (ClusterBarrier kernels; see cluster-barrier.md) ---
    bool handshakeBarrier = false;
    unsigned sccChainId = 0;
    unsigned sccChainReaders = 0;  // def node only
    bool sccChainDef = false;
    // kRule3CrossLoop false: INT_MIN. true: live-out SCC def lead floor.
    int earliestClock = INT_MIN;

    DAGNode(StinkyInstruction* inst, unsigned id) : inst(inst), inDegree(0), id(id) {}
};

// A scheduler-only hard ordering constraint: first must issue before second.
// It is kept separate from the register-dependency DAG.
using HardSchedulingConstraint = std::pair<StinkyInstruction*, StinkyInstruction*>;

// The ordering a region is scheduled under: the dependency edges that already exist,
// and the ones the queue would like added. Both members are edges of the same graph,
// which is what makes them one parameter rather than two -- it is not a bag for
// whatever a future hook happens to need.
struct RegionDependencies {
    // IN: register-dependency graph the scheduler is about to drain, handed over before
    // this region's policy edges are merged into it -- a pre-scan asking "which loads
    // does this WMMA wait on" wants the data dependencies, not the heuristic orderings.
    const RegionDAG& dag;

    // OUT: orderings the queue requests. The caller merges each into the DAG as an edge
    // and drops any that would close a cycle. Writable through a const RegionDependencies&:
    // the constness is the struct's, not the vector's.
    std::vector<HardSchedulingConstraint>& requestedConstraints;
};

// comparator: return true if a should come *after* b.
struct CompareByDAGid {
    bool operator()(const DAGNode* a, const DAGNode* b) const {
        return a->id < b->id;  // smaller id has higher priority
    }
};

// Cross-BB scheduling state: outstanding memory op latencies carried
// from one BB to the next via CFG predecessor lookup.
struct BBScheduleState {
    int gapCycles = 0;
    std::map<int, int> dsResiduals;
    // Cross-BB tensor_load_to_lds credit state (see CDNA5ReadyQueue). Carried to
    // successor BBs in a loop. Kept separate from dsResiduals.
    int globalReadInflightCount = 0;  // credits still in flight at BB end
    int globalReadResidual = 0;       // max remaining drain latency among them
};

// Cache for cross-BB scheduling state. Lives in the scheduler's run() scope
// and is shared across all ReadyQueue instances via pointer.
// When an AnalysisManager is added, this class moves there — same interface.
class ScheduleAnalysisCache {
    std::map<BasicBlock*, BBScheduleState> bbStates_;

   public:
    void store(BasicBlock* bb, const BBScheduleState& state) {
        bbStates_[bb] = state;
    }

    const BBScheduleState* lookup(BasicBlock* bb) const {
        auto it = bbStates_.find(bb);
        return it != bbStates_.end() ? &it->second : nullptr;
    }
};

class ReadyQueue {
   public:
    explicit ReadyQueue(const PassContext& passCtx) : passCtx_(passCtx) {}

    const PassContext& getPassContext() const {
        return passCtx_;
    }

    // Mirrors ModuleOptions::ClusterBarrier (wired in Gfx1250Backend as
    // PassFeatureConfig::dagFeatures::clusterBarrier). When false, the DAG
    // scheduler follows the pre-cluster-barrier path.
    bool clusterBarrierEnabled() const {
        return passCtx_.getPassFeatureConfig().dagFeatures.clusterBarrier;
    }

    virtual ~ReadyQueue() = default;

    // Pick one node from the ready queue based on some strategy.
    virtual DAGNode* pickOne() = 0;

    // Detached fillers to emit before the last picked node.
    virtual std::vector<StinkyInstruction*> takePendingFillerInsts() {
        return {};
    }

    // Push a node into the ready queue which is ready to be scheduled
    // (i.e. all its deps are satisfied).
    virtual void push(DAGNode* node) = 0;

    virtual bool empty() const = 0;

    // Hook for derived classes to do something when the first group of instructions are ready to
    // issue.
    virtual void onInit(IRList::iterator regionStart, IRList::iterator regionEnd) {}

    // Hook called before scheduling each region. \p blockBegin is the start of the basic block
    // (prefix [blockBegin, regionStart) is visible for cross-region / preloop state).
    // \p deps carries what the region depends on and what the queue may ask for; see
    // RegionDependencies.
    virtual void onInitRegion(IRList::iterator regionStart, IRList::iterator regionEnd,
                              IRList::iterator blockBegin, const RegionDependencies& deps) {
        (void)regionStart;
        (void)regionEnd;
        (void)blockBegin;
        (void)deps;
    }

    // Hook called after a basic block has been fully scheduled. When the queue is
    // reused across BBs in a loop, this lets derived classes snapshot scheduling
    // state that a successor BB's onInit can restore.
    virtual void onFinishBB() {}

    // Set the cross-BB scheduling state cache. Derived classes use this to
    // read/write cross-BB latency info in onInit/onFinishBB.
    void setAnalysisCache(ScheduleAnalysisCache* cache) {
        analysisCache_ = cache;
    }

    ScheduleAnalysisCache* getAnalysisCache() const {
        return analysisCache_;
    }

    // Set the loop context for the current BB being scheduled.
    // Called before onInit. \p loop is null if the BB is not part of any loop.
    void setLoopContext(const Loop* loop) {
        loop_ = loop;
    }

    const Loop* getLoop() const {
        return loop_;
    }

   private:
    const PassContext& passCtx_;
    const Loop* loop_ = nullptr;
    ScheduleAnalysisCache* analysisCache_ = nullptr;
};

using DAGidPriorityQueue = std::priority_queue<DAGNode*, std::vector<DAGNode*>, CompareByDAGid>;

// Ordered set of DAGNode* sorted by DAG id (smallest first).
// Same asymptotic cost as DAGidPriorityQueue for top/pop/push, but also
// supports iterating in priority order and erasing any element in O(log N).
class ReadySetByDAGid {
    std::set<DAGNode*, CompareByDAGid> set;

   public:
    DAGNode* top() const {
        return *set.begin();
    }

    void pop() {
        set.erase(set.begin());
    }

    void push(DAGNode* node) {
        set.insert(node);
    }

    bool empty() const {
        return set.empty();
    }

    size_t size() const {
        return set.size();
    }

    void erase(DAGNode* node) {
        set.erase(node);
    }

    using iterator = std::set<DAGNode*, CompareByDAGid>::iterator;
    using const_iterator = std::set<DAGNode*, CompareByDAGid>::const_iterator;

    iterator begin() {
        return set.begin();
    }

    iterator end() {
        return set.end();
    }

    const_iterator begin() const {
        return set.begin();
    }

    const_iterator end() const {
        return set.end();
    }
};

class ReadyQueueByDAGid : public ReadyQueue {
    DAGidPriorityQueue queue;

   public:
    explicit ReadyQueueByDAGid(const PassContext& passCtx) : ReadyQueue(passCtx) {}

    DAGNode* pickOne() override;

    void push(DAGNode* node) override {
        queue.push(node);
    }

    bool empty() const override {
        return queue.empty();
    }
};

inline DAGNode* ReadyQueueByDAGid::pickOne() {
    assert(!queue.empty() && "Ready queue must not be empty");
    DAGNode* node = queue.top();
    queue.pop();
    return node;
}

struct MFMAIssueConfig {
    int latency = 0;                // original mfma latency
    int avgIssueInterval = 0;       // average issue interval for mfma
    int totalIssuedCycles = 0;      // total issued cycles in the region
    int totalMfmaIssuedCycles = 0;  // total mfma issued cycles in the region
    int issuedCount = 0;            // total mfma issued count in the region
};

struct WMMAIssueConfig {
    int latency = 0;      // WMMA latencyCycles (for barrier threshold math)
    int issueCycles = 1;  // single-WMMA issue cycles
    int issuedCount = 0;  // WMMA count in region (for barrier threshold math)
};
}  // namespace dag
}  // namespace stinkytofu
