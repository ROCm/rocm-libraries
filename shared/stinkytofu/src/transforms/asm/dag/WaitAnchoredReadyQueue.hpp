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

#include <cassert>
#include <iostream>
#include <set>
#include <vector>

#include "ReadyQueue.hpp"
#include "RegionDAG.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitDataflow.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitPlan.hpp"

namespace stinkytofu {
namespace dag {

struct WaitAnchorInfo {
    StinkyInstruction* anchor = nullptr;
    std::vector<StinkyInstruction*> waits;
    waitcnt::WaitCountSpec spec;
};

using WaitAnchorMap = std::unordered_map<StinkyInstruction*, WaitAnchorInfo>;

inline bool counterApplies(waitcnt::CounterKind kind, const waitcnt::WaitCountSpec& spec) {
    using waitcnt::WaitCountSpec;
    switch (kind) {
        case waitcnt::CK_DS:
            return spec.dsCount != WaitCountSpec::kUnused;
        case waitcnt::CK_Load:
            return spec.loadCount != WaitCountSpec::kUnused;
        case waitcnt::CK_KM:
            return spec.kmCount != WaitCountSpec::kUnused;
        case waitcnt::CK_Tensor:
            return spec.tensorCount != WaitCountSpec::kUnused;
        case waitcnt::CK_Async:
            return spec.asyncCount != WaitCountSpec::kUnused;
        default:
            return false;
    }
}

/// Preserve the meaning of final wait immediates by ordering each counter's
/// producers and wait anchors in their original sequence.
inline void addCounterOrderEdges(RegionDAG& dag,
                                 const std::vector<StinkyInstruction*>& instructions,
                                 const WaitAnchorMap& anchors) {
    using waitcnt::CK_Count;
    using waitcnt::CounterKind;

    for (int ck = 0; ck < CK_Count; ++ck) {
        const auto kind = static_cast<CounterKind>(ck);
        std::vector<unsigned> events;
        events.reserve(instructions.size());

        for (unsigned i = 0; i < instructions.size(); ++i) {
            StinkyInstruction* inst = instructions[i];
            if (waitcnt::classifyMemOp(*inst) == kind) {
                events.push_back(i);
                continue;
            }

            auto anchor = anchors.find(inst);
            if (anchor != anchors.end() && counterApplies(kind, anchor->second.spec))
                events.push_back(i);
        }

        for (size_t i = 1; i < events.size(); ++i) {
            addEdgeById(&dag.nodes[events[i - 1]], &dag.nodes[events[i]], dag.graph);
        }
    }
}

struct CompareDAGNodeByOriginalOrder {
    bool operator()(const DAGNode* a, const DAGNode* b) const {
        return a->id < b->id;
    }
};

using OrderedReadyNodeSet = std::set<DAGNode*, CompareDAGNodeByOriginalOrder>;

/// Build-time toggle for the order in which a window's budgeted slots are filled.
///
/// When true, ready memory producers are selected ahead of other work so loads
/// issue as early as the window allows. This only reorders within a window: the
/// set of instructions on each side of an anchor is unaffected either way. The
/// cost is that work carried in from an earlier window has a lower original ID,
/// so placing it after those loads moves it further from its original position.
/// Set to false to fill the budget in strict original order.
constexpr bool kPreferMemProducerFirst = true;

/// Selection policy for shortening the window that ends at a matrix anchor.
///
/// A window is repaired when its anchor carries a final wait, and also when an
/// earlier window pushed work into it: that carried work must keep moving, or it
/// piles up in the first anchor that has no wait and stops being distributed.
/// Only a wait anchor gives up slots of its own work; a wait-less anchor has no
/// wait to protect and merely forwards the carry it received.
///
/// Only work that issues no asynchronous memory operation is moved past an
/// anchor, so loads keep their original position relative to every anchor.
///
/// All tuning state and decisions live here so ReadyQueue mechanics remain
/// independent from the repair heuristic.
class WaitAnchoredPickPolicy {
   public:
    WaitAnchoredPickPolicy(const WaitAnchorMap& waitAnchors, RegionDAG& regionDAG,
                           unsigned slotsToMovePastAnchor)
        : waitAnchors_(waitAnchors),
          regionDAG_(regionDAG),
          slotsToMovePastAnchor_(slotsToMovePastAnchor) {}

    /// Return a policy-selected node, or nullptr to request stable baseline order.
    DAGNode* select(const OrderedReadyNodeSet& wmmaQueue,
                    const OrderedReadyNodeSet& otherQueue) const {
        if (!window_.active()) return nullptr;

        // Fill one fewer non-WMMA slot than the original window. Work carried
        // from preceding windows participates in picks, but does not increase
        // this window's budget.
        if (window_.otherPicks < window_.otherPickBudget) {
            if (DAGNode* node = findReadyOtherBeforeAnchor(otherQueue)) return node;
        }

        // Memory producers stay ahead of the anchor even once the budget is
        // spent. Past a wait they would change how many operations its immediate
        // leaves outstanding, and past any anchor they only delay issuing a load.
        if (DAGNode* node = findReadyMemProducerBeforeAnchor(otherQueue)) return node;

        auto readyAnchor = wmmaQueue.find(window_.anchor);
        if (readyAnchor != wmmaQueue.end()) return *readyAnchor;

        // Keep selecting only dependency-path work needed to unlock the anchor.
        return findReadyAnchorPredecessor(otherQueue);
    }

    /// Update policy state after the queue commits a selected node.
    void onPicked(DAGNode& node, bool isWmma) {
        if (isWmma)
            onWmmaPicked(node);
        else
            onOtherPicked();
    }

   private:
    /// State for the currently active interval ending at a matrix anchor.
    struct WindowState {
        DAGNode* anchor = nullptr;
        /// Null when the anchor carries no final wait. Such a window is repaired
        /// only to keep carried work moving, and has no mandatory producers.
        const WaitAnchorInfo* anchorInfo = nullptr;
        unsigned startId = 0;
        /// Nodes originally in this interval.
        unsigned originalOtherCount = 0;
        /// Nodes originally in this interval plus work carried in from earlier ones.
        unsigned availableOtherCount = 0;
        unsigned otherPickBudget = 0;
        unsigned otherPicks = 0;

        bool active() const {
            return anchor != nullptr;
        }

        void reset() {
            *this = {};
        }
    };

    const WaitAnchorMap& waitAnchors_;
    RegionDAG& regionDAG_;
    const unsigned slotsToMovePastAnchor_;
    WindowState window_;
    /// Non-WMMA work deferred past the previous anchor and not yet picked.
    unsigned pendingCarry_ = 0;

    /// Account for a non-WMMA pick within the active window.
    void onOtherPicked() {
        if (!window_.active()) return;

        ++window_.otherPicks;
        if (window_.otherPicks > window_.availableOtherCount) {
            PASS_DEBUG(std::cerr << "[WaitAnchoredReadyQueue onOtherPicked] anchor dagId="
                                 << window_.anchor->id
                                 << " not shortened: otherPicks=" << window_.otherPicks
                                 << " exceeds availableOtherCount=" << window_.availableOtherCount
                                 << '\n');
        }
    }

    /// Close the current window and arm the next matrix anchor when present.
    void onWmmaPicked(DAGNode& node) {
        if (window_.active()) {
            assert(&node == window_.anchor && "Only the active anchor may close a window");
            pendingCarry_ = window_.availableOtherCount > window_.otherPicks
                                ? window_.availableOtherCount - window_.otherPicks
                                : 0;
            window_.reset();
        }

        PASS_DEBUG(std::cerr << "[WaitAnchoredReadyQueue onWmmaPicked] picked WMMA dagId="
                             << node.id << " pendingCarry=" << pendingCarry_ << '\n');

        DAGNode* nextWmma = findNextWmmaInOriginalOrder(node.id);
        if (nextWmma == nullptr) return;

        auto waitAnchor = waitAnchors_.find(nextWmma->inst);
        const WaitAnchorInfo* anchorInfo =
            waitAnchor != waitAnchors_.end() ? &waitAnchor->second : nullptr;
        PASS_DEBUG(std::cerr << "[WaitAnchoredReadyQueue onWmmaPicked] next WMMA dagId="
                             << nextWmma->id
                             << " waitAnchored=" << (anchorInfo != nullptr ? "yes" : "no") << '\n');

        // An anchor without a wait still needs a window once earlier windows
        // pushed work into it; otherwise that work settles there permanently and
        // the anchors after it keep their original, unrepaired spacing.
        if (anchorInfo != nullptr || pendingCarry_ > 0) armWindow(node, *nextWmma, anchorInfo);
    }

    /// Initialize shortening state for a matrix-anchored interval.
    void armWindow(const DAGNode& currentWmma, DAGNode& anchor, const WaitAnchorInfo* anchorInfo) {
        const unsigned ownOtherCount = anchor.id - currentWmma.id - 1;

        window_.anchor = &anchor;
        window_.anchorInfo = anchorInfo;
        window_.startId = currentWmma.id;
        window_.originalOtherCount = ownOtherCount;
        window_.availableOtherCount = ownOtherCount + pendingCarry_;
        if (anchorInfo == nullptr) {
            // Nothing to shorten without a wait, so this window keeps its own
            // occupancy and only lets the carry pass through to the next anchor.
            window_.otherPickBudget = window_.originalOtherCount;
        } else {
            window_.otherPickBudget = window_.originalOtherCount > slotsToMovePastAnchor_
                                          ? window_.originalOtherCount - slotsToMovePastAnchor_
                                          : 0;
        }
        window_.otherPicks = 0;
        pendingCarry_ = 0;

        PASS_DEBUG(std::cerr << "[WaitAnchoredReadyQueue armWindow] anchor dagId=" << anchor.id
                             << " waitAnchored=" << (anchorInfo != nullptr ? "yes" : "no")
                             << " ownOtherCount=" << ownOtherCount
                             << " originalOtherCount=" << window_.originalOtherCount
                             << " availableOtherCount=" << window_.availableOtherCount
                             << " otherPickBudget=" << window_.otherPickBudget << '\n');
    }

    /// Test whether a node originally lies inside the active interval.
    bool isInsideActiveWindow(const DAGNode& node) const {
        return window_.active() && node.id > window_.startId && node.id < window_.anchor->id;
    }

    /// Test whether a node issues an asynchronous memory operation.
    static bool isMemProducer(const DAGNode& node) {
        return waitcnt::classifyMemOp(*node.inst) != waitcnt::CK_Count;
    }

    /// Test whether a DAG path connects a node to the active anchor.
    bool reachesActiveAnchor(const DAGNode& start) const {
        if (!window_.active()) return false;

        std::vector<unsigned> worklist{start.id};
        std::vector<bool> visited(regionDAG_.nodes.size(), false);
        visited[start.id] = true;

        while (!worklist.empty()) {
            const unsigned id = worklist.back();
            worklist.pop_back();
            for (unsigned succId : regionDAG_.graph[id]) {
                if (succId == window_.anchor->id) return true;
                if (!visited[succId] && succId < window_.anchor->id) {
                    visited[succId] = true;
                    worklist.push_back(succId);
                }
            }
        }
        return false;
    }

    /// Find the earliest ready memory producer that belongs before the anchor.
    DAGNode* findReadyMemProducerBeforeAnchor(const OrderedReadyNodeSet& otherQueue) const {
        for (DAGNode* node : otherQueue) {
            if (node->id < window_.anchor->id && isMemProducer(*node)) return node;
        }
        return nullptr;
    }

    /// Find ready dependency work needed to make the anchor ready.
    DAGNode* findReadyAnchorPredecessor(const OrderedReadyNodeSet& otherQueue) const {
        for (DAGNode* node : otherQueue) {
            if (isInsideActiveWindow(*node) && reachesActiveAnchor(*node)) return node;
        }
        return nullptr;
    }

    /// Find stable ready work before the anchor, including carried work.
    DAGNode* findReadyOtherBeforeAnchor(const OrderedReadyNodeSet& otherQueue) const {
        if (!window_.active()) return nullptr;
        if constexpr (kPreferMemProducerFirst) {
            if (DAGNode* node = findReadyMemProducerBeforeAnchor(otherQueue)) return node;
        }
        for (DAGNode* node : otherQueue) {
            // This includes carried-over work from the previous WMMA window.
            if (node->id < window_.anchor->id) return node;
        }
        return nullptr;
    }

    /// Find the next WMMA in original DAG order.
    DAGNode* findNextWmmaInOriginalOrder(unsigned currentId) const {
        for (unsigned id = currentId + 1; id < regionDAG_.nodes.size(); ++id) {
            DAGNode& candidate = regionDAG_.nodes[id];
            if (isMatrixInstruction(*candidate.inst)) return &candidate;
        }
        return nullptr;
    }
};

class WaitAnchoredReadyQueue : public ReadyQueue {
   public:
    /// Create a stable queue using final wait anchors and the region DAG.
    WaitAnchoredReadyQueue(const PassContext& passCtx, const WaitAnchorMap& waitAnchors,
                           RegionDAG& regionDAG, unsigned slotsToMovePastAnchor)
        : ReadyQueue(passCtx), policy_(waitAnchors, regionDAG, slotsToMovePastAnchor) {}

    /// Add a ready node to the matrix or non-matrix queue.
    void push(DAGNode* node) override {
        if (isMatrixInstruction(*node->inst))
            wmmaQueue_.insert(node);
        else
            otherQueue_.insert(node);
    }

    /// Select and remove the next node, delegating tuning to the pick policy.
    DAGNode* pickOne() override {
        assert(!empty());

        DAGNode* node = policy_.select(wmmaQueue_, otherQueue_);
        if (node == nullptr) node = peekBaseline();

        const bool pickedWmma = removeSelectedNode(node);
        policy_.onPicked(*node, pickedWmma);
        return node;
    }

    bool empty() const override {
        return wmmaQueue_.empty() && otherQueue_.empty();
    }

   private:
    OrderedReadyNodeSet wmmaQueue_;
    OrderedReadyNodeSet otherQueue_;
    WaitAnchoredPickPolicy policy_;

    /// Remove a node from its queue and report whether it is a WMMA.
    bool removeSelectedNode(DAGNode* node) {
        if (isMatrixInstruction(*node->inst)) {
            const size_t erased = wmmaQueue_.erase(node);
            assert(erased == 1 && "Selected WMMA must be in the WMMA ready queue");
            return true;
        }

        const size_t erased = otherQueue_.erase(node);
        assert(erased == 1 && "Selected non-WMMA must be in the other ready queue");
        return false;
    }

    /// Pick the smallest original ID across both ready queues.
    DAGNode* peekBaseline() const {
        if (wmmaQueue_.empty()) return *otherQueue_.begin();
        if (otherQueue_.empty()) return *wmmaQueue_.begin();
        DAGNode* w = *wmmaQueue_.begin();
        DAGNode* o = *otherQueue_.begin();
        return w->id < o->id ? w : o;
    }
};

inline std::vector<StinkyInstruction*> scheduleWithWaitAnchoredReadyQueue(
    RegionDAG& dag, WaitAnchoredReadyQueue& queue) {
    std::vector<StinkyInstruction*> scheduled;
    scheduled.reserve(dag.nodes.size());

    for (DAGNode& node : dag.nodes) {
        if (node.inDegree == 0) queue.push(&node);
    }

    while (!queue.empty()) {
        DAGNode* node = queue.pickOne();
        scheduled.push_back(node->inst);
        for (unsigned succId : dag.graph[node->id]) {
            DAGNode& succ = dag.nodes[succId];
            if (--succ.inDegree == 0) queue.push(&succ);
        }
    }

    return scheduled;
}

}  // namespace dag
}  // namespace stinkytofu
