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

#include <cmath>
#include <iostream>  // TODO: don't use iostream.
#include <map>
#include <queue>

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/LoopDetection.hpp"
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"

namespace {
using namespace stinkytofu;

// REMOVED: Local buildUseDefChain() has been replaced by stinkytofu::buildUseDefChain()
// from BuildDefUseChain.hpp. All callers now use the shared implementation.

struct DAGNode {
    StinkyInstruction* inst;
    unsigned inDegree;
    unsigned id;
    // Pre-computed pick order for ds_reads. Lower = pick first.
    // Assigned by the pre-scan in scheduleRegionWithMovableSideEffects
    // based on DsReadOrder config and WMMA consumer analysis.
    unsigned dsReadPriority = UINT_MAX;

    DAGNode(StinkyInstruction* inst, unsigned id) : inst(inst), inDegree(0), id(id) {}
};

// comparator: return true if a should come *after* b.
struct CompareByDAGid {
    bool operator()(const DAGNode* a, const DAGNode* b) const {
        return a->id < b->id;  // smaller id has higher priority
    }
};

using DAGNodeList = std::vector<DAGNode>;

static void addEdgeById(DAGNode* from, DAGNode* to,
                        std::vector<std::unordered_set<unsigned>>& dagGraph) {
    // Don't add duplicate edges, or self-loops.
    if (from->id == to->id || dagGraph[from->id].count(to->id) > 0) return;

    // Add edge from 'from' to 'to'
    dagGraph[from->id].insert(to->id);
    to->inDegree++;
}

// Cross-BB scheduling state: outstanding memory op latencies carried
// from one BB to the next via CFG predecessor lookup.
struct BBScheduleState {
    int gapCycles = 0;
    std::map<int, int> dsResiduals;
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

    virtual ~ReadyQueue() = default;

    // Pick one node from the ready queue based on some strategy.
    virtual DAGNode* pickOne() = 0;

    // Push a node into the ready queue which is ready to be scheduled
    // (i.e. all its deps are satisfied).
    virtual void push(DAGNode* node) = 0;

    virtual bool empty() const = 0;

    // Hook for derived classes to do something when the first group of instructions are ready to
    // issue.
    virtual void onInit(IRList::iterator regionStart, IRList::iterator regionEnd) {}

    // Hook called before scheduling each region. \p blockBegin is the start of the basic block
    // (prefix [blockBegin, regionStart) is visible for cross-region / preloop state).
    virtual void onInitRegion(IRList::iterator regionStart, IRList::iterator regionEnd,
                              IRList::iterator blockBegin) {
        (void)regionStart;
        (void)regionEnd;
        (void)blockBegin;
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

DAGNode* ReadyQueueByDAGid::pickOne() {
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
}  // namespace
