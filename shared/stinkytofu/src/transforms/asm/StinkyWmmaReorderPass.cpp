// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ----------------------------------------------------------------------------
// StinkyWmmaReorderPass
//
// Rewrites an unrolled-loop body to a new wmma issue order and re-places the
// ds_loads that feed it.
//
// The wmma order comes from a swappable mode (IWmmaOrderProvider): the
// VGPR-pressure analysis, an externally tuned permutation, or a future cost
// model. This file owns everything that happens *after* the order is chosen.
//
// Two invariants keep the rewrite honest:
//
//   1. The loop body is only permuted. No instruction is created or destroyed
//      there, so the kernel still issues exactly the same work per iteration.
//
//   2. All add/remove happens in the preheader, which is where the software
//      pipeline's iteration-0 ds_load prefetches live. A ds_load whose new slot
//      falls before the top of the body has to be issued one iteration early;
//      it gets a preheader clone and its own body copy is retuned to load the
//      *next* iteration. A ds_load that no longer needs the head start loses
//      its preheader copy and its body copy is de-rotated. That is the "some
//      prefetch ds_loads appear, some disappear" half of a pattern change.
//
// The per-iteration LDS distance is never invented: it is read off the
// preheader/body ds_load pairs the kernel already has. With no such pair the
// pass cannot know how to rotate an address, so cross-iteration migration is
// disabled for that loop and the body is permuted in place.
// ----------------------------------------------------------------------------

#include "stinkytofu/transforms/asm/StinkyWmmaReorderPass.hpp"

#include <algorithm>
#include <climits>
#include <iostream>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define DEBUG_TYPE "StinkyWmmaReorderPass"

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/StinkyWmmaVgprReorderPass.hpp"

namespace stinkytofu {
namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Small helpers
// ─────────────────────────────────────────────────────────────────────────────

bool isVgpr(const StinkyRegister& r) {
    return r.isRegister() && r.reg.type == RegType::V;
}

RegGroup toRegGroup(const StinkyRegister& r) {
    return RegGroup{r.reg.idx, r.reg.num};
}

bool regOverlapsGroup(const StinkyRegister& r, const RegGroup& g) {
    if (!isVgpr(r)) return false;
    return r.reg.idx < g.base + g.size && g.base < r.reg.idx + r.reg.num;
}

/// A ds_load the pass may move: it writes one VGPR group and at least one wmma
/// in the same segment reads it.
bool isMovableDsLoad(const StinkyInstruction& inst) {
    return isDSRead(inst) && inst.getNumDestRegs() == 1 && isVgpr(inst.getDestReg(0));
}

/// Instructions the pass refuses to move anything across: LDS synchronization
/// and wait counts both establish ordering the DAG cannot see.
bool isMovementBarrier(const StinkyInstruction& inst) {
    return isBarrier(inst) || inst.is(InstFlag::IF_WaitCnt);
}

int dsOffsetOf(const StinkyInstruction& inst) {
    const auto* ds = inst.getModifier<DSModifiers>();
    return ds ? ds->offset : 0;
}

void setDsOffset(StinkyInstruction& inst, int offset) {
    if (auto* ds = inst.getModifier<DSModifiers>()) ds->offset = offset;
}

/// The address operands of a ds_load — everything but its data. Two loads that
/// differ here rotate their buffer through the address register rather than
/// through the immediate offset.
bool sameAddressOperands(const StinkyInstruction& a, const StinkyInstruction& b) {
    if (a.getNumSrcRegs() != b.getNumSrcRegs()) return false;
    for (size_t i = 0; i < a.getNumSrcRegs(); ++i)
        if (!(a.getSrcReg(i) == b.getSrcReg(i))) return false;
    return true;
}

BasicBlock::iterator iteratorOf(BasicBlock& bb, const IRBase* node) {
    for (auto it = bb.begin(); it != bb.end(); ++it)
        if (it.getNodePtr() == node) return it;
    return bb.end();
}

/// Insertion point for a new preheader prefetch: after everything, but ahead of
/// the block's branch so the CFG edge stays last.
BasicBlock::iterator preheaderInsertPoint(BasicBlock& bb) {
    auto* term = dyn_cast<StinkyInstruction>(bb.getTerminator());
    if (term && term->is(InstFlag::IF_Branch)) return iteratorOf(bb, term);
    return bb.end();
}

// ─────────────────────────────────────────────────────────────────────────────
// Loop discovery
// ─────────────────────────────────────────────────────────────────────────────

/// The block that falls into the loop, i.e. the one holding its iteration-0
/// prefetches. When the CFG has been built the nearest preceding real
/// predecessor wins; otherwise — pipelines that skip CFGBuilderPass — the
/// nearest preceding non-empty block stands in for it.
BasicBlock* findPreheader(BasicBlock* body) {
    const auto& preds = body->getPredecessors();
    BasicBlock* physical = nullptr;
    for (BasicBlock* bb = body->getPrev(); bb; bb = bb->getPrev()) {
        if (bb->empty()) continue;
        if (!physical) physical = bb;
        if (std::find(preds.begin(), preds.end(), bb) != preds.end()) return bb;
    }
    return physical;
}

std::vector<WmmaLoopRegion> findLoopRegions(Function& func, const std::string& loopLabel) {
    std::vector<WmmaLoopRegion> out;
    for (BasicBlock& bb : func) {
        if (bb.getLabel() != loopLabel) continue;

        WmmaLoopRegion region;
        region.body = &bb;
        region.preheader = findPreheader(&bb);
        for (IRBase& node : bb) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (inst && isXDLWMMA(*inst)) region.wmma.push_back(inst);
        }
        if (!region.wmma.empty()) out.push_back(std::move(region));
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Rewrite
// ─────────────────────────────────────────────────────────────────────────────

/// A ds_load under consideration, with everything the placement rule needs.
struct DsLoadPlan {
    StinkyInstruction* inst = nullptr;
    RegGroup dest{};
    unsigned firstConsumer = 0;         ///< earliest reading wmma, as a rank in the new order
    unsigned lastConsumer = 0;          ///< latest reading wmma
    unsigned slot = 0;                  ///< new position: issue just before wmma[slot]
    bool crossIter = false;             ///< issues for the *next* iteration
    StinkyInstruction* twin = nullptr;  ///< existing preheader copy, if any
};

/// wmma slots needed to cover a ds_load's latency at the loop's wmma issue rate.
unsigned derivePrefetchDistance(const std::vector<DsLoadPlan>& loads,
                                const std::vector<StinkyInstruction*>& wmma) {
    int latency = 0;
    for (const DsLoadPlan& d : loads) latency = std::max(latency, d.inst->latencyCycles);
    int issue = 0;
    for (StinkyInstruction* w : wmma) issue = std::max(issue, w->issueCycles);
    if (latency <= 0 || issue <= 0) return 1;
    return static_cast<unsigned>((latency + issue - 1) / issue);
}

class WmmaReorder {
   public:
    WmmaReorder(const IWmmaOrderProvider& mode, const WmmaReorderOptions& options)
        : mode_(mode), options_(options) {}

    WmmaReorderOutcome run(WmmaLoopRegion& region) {
        WmmaReorderOutcome res;

        std::vector<StinkyInstruction*> order = mode_.desiredOrder(region);
        if (order.empty()) {
            res.skipReason = "mode supplied no order";
            return res;
        }
        if (!isPermutationOf(order, region.wmma)) {
            res.skipReason = "mode returned a non-permutation";
            return res;
        }

        // Body node list plus the movement window the wmma sequence lives in.
        std::vector<IRBase*> nodes;
        for (IRBase& node : *region.body) nodes.push_back(&node);

        size_t segBegin = 0, segEnd = nodes.size();
        if (!findSegment(nodes, segBegin, segEnd)) {
            res.skipReason = "wmma sequence spans a barrier or wait count";
            return res;
        }

        std::vector<DsLoadPlan> loads = collectDsLoads(nodes, segBegin, segEnd, order);

        // A loop with no preheader pairing is not skipped: it simply cannot have
        // ds_loads migrate across the back-edge, so the body is permuted in place.
        std::string pairReason;
        const int delta = pairWithPreheader(region, loads, pairReason);
        res.iterOffsetDelta = delta;
        const bool crossOk = options_.allowCrossIteration && pairReason.empty();

        res.prefetchDistance = options_.prefetchDistance >= 0
                                   ? static_cast<unsigned>(options_.prefetchDistance)
                                   : derivePrefetchDistance(loads, order);

        if (!assignSlots(loads, order.size(), res.prefetchDistance, crossOk)) {
            res.skipReason =
                "an already-prefetched ds_load has no safe slot without "
                "cross-iteration edits";
            if (!pairReason.empty()) res.skipReason += " (" + pairReason + ")";
            return res;
        }

        res.wmmaMoved = rewriteBody(*region.body, nodes, segBegin, segEnd, order, loads);
        res.dsLoadMoved = dsLoadMoved_;
        applyPreheaderEdits(region, loads, delta, crossOk, res);

        res.applied = true;
        return res;
    }

   private:
    const IWmmaOrderProvider& mode_;
    const WmmaReorderOptions& options_;
    unsigned dsLoadMoved_ = 0;

    static bool isPermutationOf(const std::vector<StinkyInstruction*>& a,
                                const std::vector<StinkyInstruction*>& b) {
        if (a.size() != b.size()) return false;
        std::unordered_set<const StinkyInstruction*> seen(a.begin(), a.end());
        if (seen.size() != a.size()) return false;
        for (StinkyInstruction* inst : b)
            if (!seen.count(inst)) return false;
        return true;
    }

    /// The movement window: from just after the last barrier/wait preceding the
    /// first wmma, to just before the first barrier/wait at or after it. Every
    /// wmma must fall inside — an LDS barrier in the middle of the sequence
    /// means the kernel is not a plain unrolled body and the pass backs off.
    static bool findSegment(const std::vector<IRBase*>& nodes, size_t& segBegin, size_t& segEnd) {
        size_t firstWmma = nodes.size(), lastWmma = 0;
        for (size_t i = 0; i < nodes.size(); ++i) {
            auto* inst = dyn_cast<StinkyInstruction>(nodes[i]);
            if (inst && isXDLWMMA(*inst)) {
                firstWmma = std::min(firstWmma, i);
                lastWmma = i;
            }
        }
        if (firstWmma == nodes.size()) return false;

        segBegin = 0;
        for (size_t i = 0; i < firstWmma; ++i) {
            auto* inst = dyn_cast<StinkyInstruction>(nodes[i]);
            if (inst && isMovementBarrier(*inst)) segBegin = i + 1;
        }
        segEnd = nodes.size();
        for (size_t i = firstWmma; i < nodes.size(); ++i) {
            auto* inst = dyn_cast<StinkyInstruction>(nodes[i]);
            if (inst && isMovementBarrier(*inst)) {
                segEnd = i;
                break;
            }
        }
        return lastWmma < segEnd;
    }

    /// ds_loads in the window that feed a wmma, with their consumer ranks taken
    /// from the *new* order.
    static std::vector<DsLoadPlan> collectDsLoads(const std::vector<IRBase*>& nodes,
                                                  size_t segBegin, size_t segEnd,
                                                  const std::vector<StinkyInstruction*>& order) {
        std::vector<DsLoadPlan> loads;
        for (size_t i = segBegin; i < segEnd; ++i) {
            auto* inst = dyn_cast<StinkyInstruction>(nodes[i]);
            if (!inst || !isMovableDsLoad(*inst)) continue;

            DsLoadPlan plan;
            plan.inst = inst;
            plan.dest = toRegGroup(inst->getDestReg(0));
            plan.firstConsumer = UINT_MAX;

            for (unsigned k = 0; k < order.size(); ++k) {
                bool reads = false;
                for (size_t s = 0; s < order[k]->getNumSrcRegs() && !reads; ++s)
                    reads = regOverlapsGroup(order[k]->getSrcReg(s), plan.dest);
                if (!reads) continue;
                plan.firstConsumer = std::min(plan.firstConsumer, k);
                plan.lastConsumer = std::max(plan.lastConsumer, k);
            }
            if (plan.firstConsumer != UINT_MAX) loads.push_back(plan);
        }
        return loads;
    }

    /// Match body ds_loads against preheader ds_loads writing the same VGPR
    /// group. Those pairs are the kernel's existing software pipeline; their
    /// offset difference is the per-iteration LDS stride. Sets @p reason when no
    /// usable pair exists, which disables cross-iteration migration.
    static int pairWithPreheader(WmmaLoopRegion& region, std::vector<DsLoadPlan>& loads,
                                 std::string& reason) {
        if (!region.preheader) {
            reason = "no preheader";
            return 0;
        }

        std::map<RegGroup, StinkyInstruction*> preLoads;
        for (IRBase& node : *region.preheader) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (!inst || !isMovableDsLoad(*inst)) continue;
            preLoads.emplace(toRegGroup(inst->getDestReg(0)), inst);
        }

        bool haveDelta = false;
        int delta = 0;
        for (DsLoadPlan& plan : loads) {
            auto it = preLoads.find(plan.dest);
            if (it == preLoads.end()) continue;
            if (!sameAddressOperands(*plan.inst, *it->second)) {
                reason = "preheader pair rotates through the address register";
                return 0;
            }
            const int d = dsOffsetOf(*plan.inst) - dsOffsetOf(*it->second);
            if (haveDelta && d != delta) {
                reason = "preheader pairs disagree on the per-iteration stride";
                return 0;
            }
            plan.twin = it->second;
            delta = d;
            haveDelta = true;
        }
        if (!haveDelta) reason = "no preheader/body ds_load pair to derive the stride from";
        return delta;
    }

    /// Place every ds_load `distance` wmma slots ahead of its first consumer,
    /// wrapping around the back-edge when that lands before the top of the body.
    ///
    /// A wrapped load issues for the *next* iteration, so it must sit after
    /// every wmma that reads its registers this iteration — not just the first.
    /// That is the one hard rule here; the requested distance yields to it and
    /// the load simply gets a shorter lead.
    ///
    /// A load that is already prefetched cannot be demoted to same-iteration
    /// unless its offset can be de-rotated, so with cross-iteration edits off it
    /// stays wrapped. Returns false when that leaves it nowhere legal to go, in
    /// which case the caller must leave the whole loop alone.
    static bool assignSlots(std::vector<DsLoadPlan>& loads, size_t n, unsigned distance,
                            bool crossOk) {
        const int lastSlot = static_cast<int>(n) - 1;
        for (DsLoadPlan& plan : loads) {
            const bool wasCross = plan.twin != nullptr;
            const int raw = static_cast<int>(plan.firstConsumer) - static_cast<int>(distance);

            // Both introducing and undoing a wrap need the preheader edit.
            const bool mustCross = wasCross && !crossOk;
            if (raw >= 0 && !mustCross) {
                plan.slot = static_cast<unsigned>(raw);
                plan.crossIter = false;
                continue;
            }
            if (!crossOk && !wasCross) {
                plan.slot = 0;  // as early as the body allows; always safe
                plan.crossIter = false;
                continue;
            }

            const int wrapped =
                raw < 0 ? (raw % static_cast<int>(n) + static_cast<int>(n)) % static_cast<int>(n)
                        : static_cast<int>(plan.lastConsumer) + 1;
            const int slot = std::max(wrapped, static_cast<int>(plan.lastConsumer) + 1);
            if (slot > lastSlot) {
                if (mustCross) return false;
                plan.slot = 0;
                plan.crossIter = false;
                continue;
            }
            plan.slot = static_cast<unsigned>(slot);
            plan.crossIter = true;
        }
        return true;
    }

    /// Splice the new sequence back into the body. Only the movable nodes change
    /// position; every other instruction keeps the slot it already occupied, so
    /// barriers, ds_stores and address arithmetic stay put.
    unsigned rewriteBody(BasicBlock& body, const std::vector<IRBase*>& nodes, size_t segBegin,
                         size_t segEnd, const std::vector<StinkyInstruction*>& order,
                         const std::vector<DsLoadPlan>& loads) {
        std::unordered_set<const IRBase*> movable(order.begin(), order.end());
        for (const DsLoadPlan& plan : loads) movable.insert(plan.inst);

        // Desired sequence of movable instructions: each slot's ds_loads, then
        // the wmma that opens the slot.
        std::vector<IRBase*> sequence;
        sequence.reserve(movable.size());
        for (unsigned k = 0; k < order.size(); ++k) {
            for (const DsLoadPlan& plan : loads)
                if (plan.slot == k) sequence.push_back(plan.inst);
            sequence.push_back(order[k]);
        }

        std::vector<IRBase*> newOrder;
        newOrder.reserve(nodes.size());
        size_t next = 0;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (i >= segBegin && i < segEnd && movable.count(nodes[i])) {
                newOrder.push_back(sequence[next++]);
            } else {
                newOrder.push_back(nodes[i]);
            }
        }

        unsigned wmmaMoved = 0;
        dsLoadMoved_ = 0;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i] == newOrder[i]) continue;
            auto* inst = dyn_cast<StinkyInstruction>(newOrder[i]);
            if (inst && isXDLWMMA(*inst))
                ++wmmaMoved;
            else if (inst && isDSRead(*inst))
                ++dsLoadMoved_;
        }

        for (IRBase* node : nodes) body.removeIR(node);
        for (IRBase* node : newOrder) body.appendIR(node);
        return wmmaMoved;
    }

    /// Materialize the placement decisions that cross the back-edge: clone into
    /// the preheader and push the body copy one iteration ahead, or drop the
    /// preheader copy and pull the body copy back.
    static void applyPreheaderEdits(WmmaLoopRegion& region, const std::vector<DsLoadPlan>& loads,
                                    int delta, bool crossOk, WmmaReorderOutcome& res) {
        if (!crossOk || !region.preheader) return;

        for (const DsLoadPlan& plan : loads) {
            const bool wasCross = plan.twin != nullptr;
            if (plan.crossIter == wasCross) continue;

            if (plan.crossIter) {
                // The body copy now loads for the next iteration; iteration 0
                // needs its own copy at the original offset.
                auto* clone = dyn_cast<StinkyInstruction>(plan.inst->clone());
                if (!clone) continue;
                region.preheader->insertIR(preheaderInsertPoint(*region.preheader), clone);
                setDsOffset(*plan.inst, dsOffsetOf(*plan.inst) + delta);
                ++res.prefetchAdded;
            } else {
                // The head start is gone; the body copy serves its own iteration
                // again and the preheader copy is dead.
                auto it = iteratorOf(*region.preheader, plan.twin);
                if (it == region.preheader->end()) continue;
                region.preheader->eraseIR(it);
                setDsOffset(*plan.inst, dsOffsetOf(*plan.inst) - delta);
                ++res.prefetchRemoved;
            }
        }
    }
};

// Per-loop-body results for getWmmaReorderOutcome().
// NOTE: global state, matching StinkyWmmaVgprReorderPass; not thread-safe.
std::unordered_map<const BasicBlock*, WmmaReorderOutcome> gResults;

class StinkyWmmaReorderPassImpl : public StinkyInstPass {
   public:
    StinkyWmmaReorderPassImpl(std::unique_ptr<IWmmaOrderProvider> mode, WmmaReorderOptions options)
        : mode_(std::move(mode)), options_(std::move(options)) {}

    static char ID;
    const char* getName() const override {
        return "StinkyWmmaReorderPass";
    }
    PassID getPassID() const override {
        return &ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager&) override {
        gResults.clear();
        auto regions = findLoopRegions(func, options_.loopLabel);

        bool changed = false;
        for (WmmaLoopRegion& region : regions) {
            if (!passCtx.shouldProcessBasicBlock(*region.body)) continue;

            WmmaReorder rewriter(*mode_, options_);
            WmmaReorderOutcome res = rewriter.run(region);
            changed |= res.applied;

            PASS_DEBUG({
                std::cerr << "[WmmaReorder] " << region.body->getLabel()
                          << " mode=" << mode_->name() << (res.applied ? " applied" : " skipped");
                if (res.applied)
                    std::cerr << ": distance=" << res.prefetchDistance
                              << " wmmaMoved=" << res.wmmaMoved
                              << " dsLoadMoved=" << res.dsLoadMoved << " prefetch+"
                              << res.prefetchAdded << "/-" << res.prefetchRemoved
                              << " iterOffsetDelta=" << res.iterOffsetDelta;
                else
                    std::cerr << ": " << res.skipReason;
                std::cerr << "\n";
            });
            gResults[region.body] = std::move(res);
        }
        // Instruction order and ds offsets change; the CFG does not.
        return changed ? preserveCFGAnalyses() : PreservedAnalyses::all();
    }

   private:
    std::unique_ptr<IWmmaOrderProvider> mode_;
    WmmaReorderOptions options_;
};

char StinkyWmmaReorderPassImpl::ID = 0;

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Modes
// ─────────────────────────────────────────────────────────────────────────────

std::vector<StinkyInstruction*> VgprAnalysisOrderProvider::desiredOrder(
    const WmmaLoopRegion& region) const {
    const WmmaReorderAnalysisResult analysis = analyzeWmmaVgprReorder(*region.body);
    if (!analysis.applicable) return {};
    return analysis.desiredWmmaOrder;
}

std::vector<StinkyInstruction*> ExplicitOrderProvider::desiredOrder(
    const WmmaLoopRegion& region) const {
    if (perm_.size() != region.wmma.size()) return {};
    std::vector<StinkyInstruction*> out;
    out.reserve(perm_.size());
    for (unsigned idx : perm_) {
        if (idx >= region.wmma.size()) return {};
        out.push_back(region.wmma[idx]);
    }
    return out;
}

std::vector<StinkyInstruction*> ReverseOrderProvider::desiredOrder(
    const WmmaLoopRegion& region) const {
    return {region.wmma.rbegin(), region.wmma.rend()};
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

std::unique_ptr<Pass> createStinkyWmmaReorderPass(std::unique_ptr<IWmmaOrderProvider> mode,
                                                  WmmaReorderOptions options) {
    if (!mode) mode = std::make_unique<VgprAnalysisOrderProvider>();
    return std::make_unique<StinkyWmmaReorderPassImpl>(std::move(mode), std::move(options));
}

const WmmaReorderOutcome* getWmmaReorderOutcome(const BasicBlock& bb) {
    auto it = gResults.find(&bb);
    return (it != gResults.end()) ? &it->second : nullptr;
}

}  // namespace stinkytofu
