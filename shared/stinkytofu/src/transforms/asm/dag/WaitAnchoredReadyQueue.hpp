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
#include <limits>
#include <optional>
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

/// For each instruction \p shouldPin accepts, look forward for the next
/// instruction \p isBarrier accepts and add an edge between them. The edge
/// forces the first instruction to be scheduled before the second.
///
/// Both of the orderings this pass adds by itself, on top of the ones it gets
/// from register dependencies, are built out of this one step.
///
/// Two useful things follow from only ever looking forward. Edges always run
/// from a lower index to a higher one, so they can never form a cycle. And an
/// instruction with no barrier after it gets no edge at all, which is right:
/// there is nothing left for it to stay ahead of.
///
/// The predicates take a pointer instead of a reference because one of them
/// looks the instruction up in a \ref WaitAnchorMap, which is keyed on pointers.
///
/// On cost: when both predicates accept the same instructions, each forward scan
/// stops at the next accepted one, so no two scans cover the same ground and the
/// total work is linear. When they accept different instructions, several scans
/// can cross the same stretch, so keep the pinned set small.
template <typename PinPred, typename BarrierPred>
inline void addEdgesToFirstFollowing(RegionDAG& dag,
                                     const std::vector<StinkyInstruction*>& instructions,
                                     PinPred shouldPin, BarrierPred isBarrier) {
    for (unsigned i = 0; i < instructions.size(); ++i) {
        if (!shouldPin(instructions[i])) continue;
        for (unsigned j = i + 1; j < instructions.size(); ++j) {
            if (!isBarrier(instructions[j])) continue;
            addEdgeById(&dag.nodes[i], &dag.nodes[j], dag.graph);
            break;
        }
    }
}

/// Preserve the meaning of final wait immediates by ordering each counter's
/// producers and wait anchors in their original sequence.
///
/// Every event of a counter is both pinned and a barrier for that counter, so
/// the edges chain each event to the next one and the whole chain stays in input
/// order.
inline void addCounterOrderEdges(RegionDAG& dag,
                                 const std::vector<StinkyInstruction*>& instructions,
                                 const WaitAnchorMap& anchors) {
    using waitcnt::CK_Count;
    using waitcnt::CounterKind;

    for (int ck = 0; ck < CK_Count; ++ck) {
        const auto kind = static_cast<CounterKind>(ck);
        auto isCounterEvent = [&](StinkyInstruction* inst) {
            if (waitcnt::classifyMemOp(*inst) == kind) return true;
            auto anchor = anchors.find(inst);
            return anchor != anchors.end() && counterApplies(kind, anchor->second.spec);
        };
        addEdgesToFirstFollowing(dag, instructions, isCounterEvent, isCounterEvent);
    }
}

/// Hints whose only value is the lead time they get: no waitcnt counter and no
/// destination register, so nothing else in the DAG orders them.
///
/// The null check matters here. StinkyInstruction::is() dereferences its
/// descriptor unchecked, which is why hasSideEffect() tests the pointer first.
/// Any predicate used in \ref kPinRules has to do the same.
inline bool isPrefetchHint(const StinkyInstruction& inst) {
    if (inst.getHwInstDesc() == nullptr) return false;
    return isGlobalPrefetch(inst);
}

/// Build-time toggle for holding prefetch hints ahead of the next matrix
/// instruction. Set to false to let them float like any other work.
constexpr bool kPinPrefetchToAnchor = true;

/// One class of work that must not drift past a later instruction.
///
/// \p shouldPin selects what to hold in place and \p isBarrier what it must stay
/// ahead of, so a rule reads as "keep every X ahead of the next Y".
struct PinRule {
    bool enabled;
    bool (*shouldPin)(const StinkyInstruction&);
    bool (*isBarrier)(const StinkyInstruction&);
};

/// Work this pass pins, applied in table order.
///
/// Adding a class of pinned work is an entry here plus its predicates. Nothing
/// else changes: the edge building is shared, and the selection policy needs no
/// knowledge of any individual rule.
///
/// Prefetch hints are pinned to the next matrix instruction because a prefetch
/// is not a segment boundary, carries no counter and writes no register, so
/// without an edge the shortening budget is free to defer it and it slides past
/// one anchor after another, losing the lead time it exists for.
///
/// An edge is what makes this work, rather than a preference inside the
/// selection policy. A merely preferred prefetch still loses to the anchor
/// whenever its address operands are not ready yet. As a predecessor of the
/// anchor it instead blocks it, and the existing dependency-path step then pulls
/// the address chain in to unblock it.
inline constexpr PinRule kPinRules[] = {
    {kPinPrefetchToAnchor, isPrefetchHint, isMatrixInstruction},
};

/// Hold each rule's pinned work ahead of the barrier that follows it.
inline void addPinEdges(RegionDAG& dag, const std::vector<StinkyInstruction*>& instructions) {
    for (const PinRule& rule : kPinRules) {
        if (!rule.enabled) continue;
        addEdgesToFirstFollowing(
            dag, instructions, [&](StinkyInstruction* inst) { return rule.shouldPin(*inst); },
            [&](StinkyInstruction* inst) { return rule.isBarrier(*inst); });
    }
}

/// All ordering this pass adds that register dependencies did not give it.
///
/// The two rules stay separate because they answer different questions. Counter
/// order is a correctness constraint: it keeps a wait immediate counting the
/// operations it was computed for. Pinning is a performance constraint: it keeps
/// work whose value is its position from being deferred away from it.
inline void addSyntheticOrderEdges(RegionDAG& dag,
                                   const std::vector<StinkyInstruction*>& instructions,
                                   const WaitAnchorMap& anchors) {
    addCounterOrderEdges(dag, instructions, anchors);
    addPinEdges(dag, instructions);
}

struct CompareDAGNodeByOriginalOrder {
    bool operator()(const DAGNode* a, const DAGNode* b) const {
        return a->id < b->id;
    }
};

/// Keep each prefetch, and the VALU chain computing its address, ahead of the next matrix
/// instruction. Neither carries a counter, so without this edge the shortening budget defers
/// them from window to window until the segment ends; the DAG scheduler places them on
/// purpose (PrefetchLeadWmmas, address VALU well ahead to hide va_vdst), and this keeps the
/// repair from moving them away.
inline void addPrefetchPinEdges(RegionDAG& dag,
                                const std::vector<StinkyInstruction*>& instructions) {
    std::vector<bool> pinned(instructions.size(), false);
    for (unsigned i = instructions.size(); i-- > 0;) {
        const StinkyInstruction& inst = *instructions[i];
        if (inst.getHwInstDesc() == nullptr) continue;
        if (isGlobalPrefetch(inst)) {
            pinned[i] = true;
        } else if (isVectorALU(inst)) {
            for (unsigned succ : dag.graph[i])
                if (pinned[succ]) pinned[i] = true;
        }
        if (!pinned[i]) continue;
        for (unsigned j = i + 1; j < instructions.size(); ++j) {
            if (instructions[j]->getHwInstDesc() != nullptr &&
                isMatrixInstruction(*instructions[j])) {
                addEdgeById(&dag.nodes[i], &dag.nodes[j], dag.graph);
                break;
            }
        }
    }
}

using OrderedReadyNodeSet = std::set<DAGNode*, CompareDAGNodeByOriginalOrder>;

/// Build-time toggle for the order in which a window's budgeted slots are filled.
///
/// When true, ready memory producers are selected ahead of other work so loads
/// issue as early as the window allows. This only reorders within a window: the
/// set of instructions on each side of an anchor is unaffected either way. The
/// cost is that work carried in from an earlier window has a lower original ID,
/// so placing it after those loads moves it further from its original position.
/// Set to false to fill the budget in strict original order.
constexpr bool kPreferMemProducerFirst = false;

/// Build-time toggle for preserving the distance a DS load sits behind the
/// matrix instruction preceding it.
///
/// The scheduler that runs before the repair leaves a deliberate gap between a
/// matrix instruction and the DS loads that follow it. Selecting a load as early
/// as the window allows would close that gap, so each load is held back until the
/// schedule has moved at least as far past the matrix instruction as it had in
/// the input. This is a lower bound: a load may end up further away, never
/// nearer. Set to false to let loads issue as early as the window allows.
constexpr bool kPreserveMatrixToDsGap = true;

/// Whether the gap rule can actually hold a load back, which is the condition
/// for running any of its machinery.
///
/// The rule only ever acts inside the budgeted step, and that step is the only
/// thing that pulls a load forward — but only while memory-first is on. With
/// memory-first off the budget is filled in strict original order, so by the
/// time a load is the lowest-ID candidate, every lower-ID instruction of its
/// interval has already been emitted. That count is exactly the load's release
/// distance, so the distance is always already satisfied and the rule can never
/// block anything. Verified: with memory-first off, toggling the rule leaves the
/// filecheck suite passing and real kernel output byte-identical at every slot
/// count from 1 to 8.
///
/// So this is dead-code elimination, not a policy decision. If a new selection
/// path is ever added that can pull a load forward on its own, that path breaks
/// the reasoning above and belongs in this condition.
constexpr bool kGapRuleActive = kPreserveMatrixToDsGap && kPreferMemProducerFirst;

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
          slotsToMovePastAnchor_(slotsToMovePastAnchor) {
        // Every reader of dsReleaseDistance_ tests this same toggle before
        // indexing it, so leaving the vector empty here is safe.
        if constexpr (kGapRuleActive) buildDsReleaseDistances();
    }

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
    ///
    /// sinceLastMatrix_ is maintained here rather than in the per-window
    /// handlers because it spans windows: it must keep advancing while no window
    /// is active, so that a load in the next window still measures its distance
    /// from the matrix instruction that actually precedes it.
    void onPicked(DAGNode& node, bool isWmma) {
        if (isWmma) {
            sinceLastMatrix_ = 0;
            onWmmaPicked(node);
        } else {
            reportShortGapIfAny(node);
            ++sinceLastMatrix_;
            onOtherPicked();
        }
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

    /// Marks a node the gap rule says nothing about.
    static constexpr unsigned kNoGapConstraint = std::numeric_limits<unsigned>::max();

    const WaitAnchorMap& waitAnchors_;
    RegionDAG& regionDAG_;
    const unsigned slotsToMovePastAnchor_;
    WindowState window_;
    /// Non-WMMA work deferred past the previous anchor and not yet picked.
    unsigned pendingCarry_ = 0;
    /// Per DS load, how far past the preceding matrix instruction the schedule
    /// must be before it may be selected. kNoGapConstraint for everything else.
    std::vector<unsigned> dsReleaseDistance_;
    /// Instructions emitted since the last matrix instruction was selected.
    unsigned sinceLastMatrix_ = 0;

    /// Give every DS load in one matrix interval the input distance of the
    /// *first* load in that interval, so the whole run is gated on one release
    /// point. Distances are a difference of DAG IDs, which are dense and follow
    /// input order.
    ///
    /// Per-load distances would be wrong: they reconstruct the input's internal
    /// spacing between loads and so drive ordinary work between loads the
    /// scheduler had placed back to back.
    void buildDsReleaseDistances() {
        dsReleaseDistance_.assign(regionDAG_.nodes.size(), kNoGapConstraint);

        std::optional<unsigned> lastMatrixId;
        std::optional<unsigned> intervalDistance;
        for (unsigned id = 0; id < regionDAG_.nodes.size(); ++id) {
            const DAGNode& node = regionDAG_.nodes[id];
            if (isMatrixInstruction(*node.inst)) {
                lastMatrixId = id;
                intervalDistance.reset();
                continue;
            }
            if (!lastMatrixId.has_value()) continue;
            if (waitcnt::classifyMemOp(*node.inst) != waitcnt::CK_DS) continue;
            if (!intervalDistance.has_value()) intervalDistance = id - *lastMatrixId - 1;
            dsReleaseDistance_[id] = *intervalDistance;
        }
    }

    /// Report a DS load committed closer to its matrix instruction than the
    /// input had it. Legal, so this is a diagnostic rather than an assert: only
    /// the spacing is lost. Reporting at the commit point also covers the
    /// mandatory-producer step, which takes a load whatever its release distance.
    void reportShortGapIfAny(const DAGNode& node) const {
        if (matrixGapSatisfied(node)) return;
        PASS_DEBUG(std::cerr << "[WaitAnchoredReadyQueue onPicked] gap shortened for dagId="
                             << node.id << " required=" << dsReleaseDistance_[node.id]
                             << " actual=" << sinceLastMatrix_ << '\n');
    }

    /// Test whether the schedule has moved far enough past the last matrix
    /// instruction for this node to be selected.
    bool matrixGapSatisfied(const DAGNode& node) const {
        if constexpr (!kGapRuleActive) return true;
        const unsigned required = dsReleaseDistance_[node.id];
        return required == kNoGapConstraint || sinceLastMatrix_ >= required;
    }

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
                             << node.id << " pendingCarry=" << pendingCarry_
                             << " sinceLastMatrix reset\n");

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

    /// Find the earliest ready node that belongs before the anchor and matches.
    template <typename Predicate>
    DAGNode* findReadyBeforeAnchor(const OrderedReadyNodeSet& otherQueue, Predicate match) const {
        for (DAGNode* node : otherQueue) {
            if (node->id < window_.anchor->id && match(*node)) return node;
        }
        return nullptr;
    }

    /// Find the earliest ready memory producer that belongs before the anchor.
    DAGNode* findReadyMemProducerBeforeAnchor(const OrderedReadyNodeSet& otherQueue) const {
        return findReadyBeforeAnchor(otherQueue,
                                     [](const DAGNode& node) { return isMemProducer(node); });
    }

    /// Same, restricted to producers the gap rule has already released.
    DAGNode* findReadyReleasedMemProducer(const OrderedReadyNodeSet& otherQueue) const {
        return findReadyBeforeAnchor(otherQueue, [this](const DAGNode& node) {
            return isMemProducer(node) && matrixGapSatisfied(node);
        });
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
            if (DAGNode* node = findReadyReleasedMemProducer(otherQueue)) return node;
        }

        DAGNode* gapBlocked = nullptr;
        for (DAGNode* node : otherQueue) {
            // This includes carried-over work from the previous WMMA window.
            if (node->id >= window_.anchor->id) continue;
            if (!matrixGapSatisfied(*node)) {
                if (gapBlocked == nullptr) gapBlocked = node;
                continue;
            }
            return node;
        }

        // Taking a load before its gap is filled, rather than stalling. Believed
        // unreachable: the work that forms a gap sits between the load and the
        // matrix instruction, so it has lower IDs and is selected first. Kept
        // because the queue must always make progress if it is ever wrong.
        return gapBlocked;
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
