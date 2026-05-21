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
//
// CDNA5 (Gfx1250) ready-queue for StinkyDAGSchedulerPass.
//
// StinkyDAGSchedulerPass splits each basic block into regions at non-movable side effects
// (waits, stores, branches, etc.), builds a per-region dependency DAG from physical registers,
// then drains ready nodes via this queue. CDNA5 models the WMMA–VALU co-issue timeline:
// WMMA issues in 1 cycle; VALU is only gated by the co-issue window.
// Memory ops and SALU use independent pipelines.
//
#include <algorithm>
#include <cassert>
#include <map>

#include "ReadyQueue.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"

namespace {
using namespace stinkytofu;

// -------------------------------------------------------------------------
// Prefix / loop analysis (free functions; no CDNA5ReadyQueue state)
// -------------------------------------------------------------------------

// Scheduling rule (2): simulate ds_load completion over [blockBegin, regionStart) — outstanding
// VGPR latencies decrease by each instruction's issueCycles; each ds_load overwrites dest VGPRs
// with that op's latencyCycles (WAW). Remaining counts seed wmmaRegisterLatencyCounters so the
// first WMMA in a region sees preloop / in-BB loads the register DAG may not edge to that WMMA
// (double-buffer: WMMA on X0 while in-loop ds fills X1). Caller: onInitRegion.
//
// crossBBResiduals: DS latency residuals from predecessor BBs (merged in onInit).
// They decay through the prefix alongside within-BB DS loads so cross-BB ds_loads
// properly gate WMMA issue.
static void seedWmmaDsLatencyFromPrefix(IRList::iterator blockBegin, IRList::iterator regionStart,
                                        std::map<int, int>& wmmaRegisterLatencyCounters,
                                        const std::map<int, int>& crossBBResiduals) {
    wmmaRegisterLatencyCounters.clear();
    std::map<int, int> pending(crossBBResiduals);

    for (IRList::iterator it = blockBegin; it != regionStart; ++it) {
        auto* instPtr = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!instPtr) continue;
        StinkyInstruction& inst = *instPtr;
        const int iss = inst.issueCycles;

        for (auto pit = pending.begin(); pit != pending.end();) {
            pit->second -= iss;
            if (pit->second <= 0)
                pit = pending.erase(pit);
            else
                ++pit;
        }

        if (isDSRead(inst)) {
            for (const StinkyRegister& dstReg : inst.getDestRegs()) {
                if (!dstReg.isRegister()) continue;
                for (unsigned off = 0; off < dstReg.reg.num; ++off)
                    pending[dstReg.reg.idx + off] = inst.latencyCycles;
            }
        }
    }

    for (const auto& [regIdx, rem] : pending) {
        if (rem > 0) wmmaRegisterLatencyCounters[regIdx] = rem;
    }
}

// Scheduling rule (5) helper: walk backward from the end of a BB, skipping LABEL and
// branch ops; true if the first real instruction found is WMMA/SWMMA.
// Used to detect “tail WMMA” in the latch BB of a loop (cross-BB aware).
static bool latchBBTailIsWmma(BasicBlock& latchBB) {
    if (latchBB.empty()) return false;
    // Walk from tail toward head. Do not use std::prev(end()) on IRList::iterator:
    // end() is nullptr and IntrusiveListIterator::operator-- does not step to tail_.
    for (auto it = latchBB.rbegin(); it != latchBB.rend(); ++it) {
        auto* instPtr = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!instPtr) continue;
        if (isLabel(*instPtr) || isBranch(*instPtr)) continue;
        return isMatrixInstruction(*instPtr);
    }
    return false;
}

// Rule (5) — loop tail WMMA / head WMMA deferral (cross-BB aware via LoopDetection).
//
// Uses the Loop* from setLoopContext() to detect whether the loop’s latch BB ends with
// WMMA before its back-edge branch. If so, the header BB’s first WMMA should be deferred
// to avoid back-to-back WMMA across iterations.
//
// When the loop is split across multiple BBs (e.g. unrolled loops), the latch BB
// (containing the back-edge branch) may be different from the header BB.
//
// Steps (unchanged from the old pipeline, but loop detection is now cross-BB):
//   Step 1 — onInit: check latchBB tail for WMMA via latchBBTailIsWmma().
//   Step 2 — onInitRegion: deferHeadBalanceThisRegion_ if this BB is the loop header.
//   Step 3 — pickOne Phase B: block WMMA while non-WMMA queues have work.
//   Step 4 — pickOneFromWMMA / popNonWmmaByKind: clear deferral after first pick.

// Collect non-pseudo VGPR destination register indices from an instruction.
static std::unordered_set<uint32_t> collectDestVGPRs(const StinkyInstruction& inst) {
    std::unordered_set<uint32_t> vgprs;
    for (const StinkyRegister& dst : inst.getDestRegs()) {
        if (!dst.isRegister() || isPseudoReg(dst)) continue;
        for (uint32_t off = 0; off < dst.reg.num; ++off) vgprs.insert(dst.reg.idx + off);
    }
    return vgprs;
}

// True if any non-pseudo src VGPR of inst overlaps the given VGPR index set.
static bool srcVGPRsOverlap(const StinkyInstruction& inst,
                            const std::unordered_set<uint32_t>& vgprs) {
    for (const StinkyRegister& src : inst.getSrcRegs()) {
        if (!src.isRegister() || isPseudoReg(src)) continue;
        for (uint32_t off = 0; off < src.reg.num; ++off) {
            if (vgprs.count(src.reg.idx + off)) return true;
        }
    }
    return false;
}

struct BarrierTokenEntry {
    StinkyInstruction* barrier;
    std::unordered_set<uint32_t> tokens;
};

// Collect all movable barriers in [regionStart, regionEnd) with their PSEUDO token sets.
// useSrc: true → collect from getSrcRegs(), false → collect from getDestRegs().
static std::vector<BarrierTokenEntry> collectBarrierTokens(IRList::iterator regionStart,
                                                           IRList::iterator regionEnd,
                                                           bool useSrc) {
    std::vector<BarrierTokenEntry> barriers;
    for (IRList::iterator it = regionStart; it != regionEnd; ++it) {
        StinkyInstruction& inst = getStinkyInst(it);
        if (!isBarrier(inst) || inst.getDestRegs().empty()) continue;
        BarrierTokenEntry entry;
        entry.barrier = &inst;
        const auto& regs = useSrc ? inst.getSrcRegs() : inst.getDestRegs();
        for (const StinkyRegister& r : regs) {
            if (isPseudoReg(r)) entry.tokens.insert(r.reg.idx);
        }
        if (!entry.tokens.empty()) barriers.push_back(std::move(entry));
    }
    return barriers;
}

// -------------------------------------------------------------------------
// CDNA5ReadyQueue — WMMA scheduling policy (Gfx1250)
// -------------------------------------------------------------------------
//
// Scheduling model: WMMA issues in 1 cycle; its latency defines a co-issue
// timeline during which VALU can only execute in specific cycle slots given
// by HwInstDesc::coIssueWindow.  Memory ops (ds_load, global_read, tensor_load)
// and SALU use independent pipelines and have no co-issue constraint with WMMA.
//
// Scheduling rules (every pick still respects the DAG: in-degree 0 only):
//
//  (1) Program order — prefer WMMA in Phase B, but not before a pickable
//      non-WMMA node with a smaller DAG id (preload / double-buffer).
//  (2) DS / VGPR latency — block WMMA until modeled ds_load latency for WMMA
//      src VGPRs has decayed; seed from the BB prefix before each region.
//  (3) VALU is only gated by the co-issue window.
//  (4) Adaptive DS distribution — soft hint to spread ds_loads across WMMAs;
//      prefer issuing ds_loads before WMMA when under the per-WMMA milestone.
//  (5) Loop tail vs head — defer first WMMA in the loop header BB until
//      non-WMMA queues drain once. Cross-BB via LoopDetection.
//
class CDNA5ReadyQueue : public ReadyQueue {
    // --- Priority buckets (DAG ids compare smaller = earlier in source) ---
    ReadySetByDAGid wmmaQueue;
    ReadySetByDAGid globalReadQueue;  // tensor_load_to_lds when distributeGlobalRead
    ReadySetByDAGid localReadQueue;   // ds_load
    ReadySetByDAGid valuQueue;        // VALU and transcendental instructions
    ReadySetByDAGid barrierQueue;
    ReadySetByDAGid otherQueue;  // scalars, waits in region, etc.

    // Throttle tensor issues vs other work.
    int globalReadCounter = 0;
    int globalReadPerWMMA = 1;

    // --- VALU co-issue timeline tracker ---
    uint16_t activeCoIssueWindow_ = 0;
    int coIssueCyclePos_ = 0;
    int activeWmmaLatency_ = 0;

    // --- Adaptive DS distribution ---
    int totalDsRemaining_ = 0;
    int totalWmmaRemaining_ = 0;
    int minDsPerWmma_ = 0;
    int dsInsertedSinceLastWmma_ = 0;

    // --- DS burst limiter ---
    static constexpr int kMaxConsecutiveDs = 4;
    int consecutiveDsCount_ = 0;

    // Per VGPR index: remaining modeled latency until ds_load result is safe for WMMA src.
    std::map<int, int> wmmaRegisterLatencyCounters;

    WMMAIssueConfig wmmaIssueConfig;

    bool hasWMMAInRegion_ = false;

    // --- Loop head balancing ---
    bool deferFirstHeadWmmaActive_ = false;
    bool deferHeadBalanceThisRegion_ = false;

    // Per-barrier forced-issue threshold: maps StinkyInstruction* -> N.
    std::unordered_map<StinkyInstruction*, int> barrierWmmaThresholds_;

    int wmmaIssuedCountThisRegion_ = 0;

    BasicBlock* currentBB_ = nullptr;

    DAGNode* lastPickedNode_ = nullptr;

    std::map<int, int> crossBBDsResiduals_;

    void advanceTime(int cycles);
    int computeValuAdvanceCycles(int issueCycles) const;
    void updateWMMAStatus(DAGNode* node);
    int getMaxDsLatency(DAGNode* node);
    std::pair<DAGNode*, int> findMostReadyWMMA();
    DAGNode* pickOneFromWMMA(DAGNode* pick = nullptr);
    bool findSmallestPickableNonWmma(DAGNode** outNode, int* kindOut) const;
    bool findOldestFallbackNonWmma(DAGNode** outNode, int* kindOut) const;
    DAGNode* popFallbackNonWmmaByKind(DAGNode* node, int pickKind);
    DAGNode* extractForcedBarrier();
    void computeBarrierAfterThresholds(IRList::iterator regionStart, IRList::iterator regionEnd);
    std::unordered_map<StinkyInstruction*, int> computeBarrierBeforeThresholds(
        IRList::iterator regionStart, IRList::iterator regionEnd);
    bool isValuPickable() const;
    DAGNode* popNonWmmaByKind(int pickKind);

    void restoreCrossBBStateFromLoop();

   public:
    explicit CDNA5ReadyQueue(const PassContext& passCtx) : ReadyQueue(passCtx) {}

    DAGNode* pickOne() override;
    void push(DAGNode* node) override;
    bool empty() const override;

    void onInit(IRList::iterator regionStart, IRList::iterator regionEnd) override;

    void onInitRegion(IRList::iterator regionStart, IRList::iterator regionEnd,
                      IRList::iterator blockBegin) override;

    void onFinishBB() override;
};

// Advance the co-issue timeline and decay DS latency counters by \p cycles.
void CDNA5ReadyQueue::advanceTime(int cycles) {
    coIssueCyclePos_ += cycles;
    for (auto it = wmmaRegisterLatencyCounters.begin(); it != wmmaRegisterLatencyCounters.end();) {
        it->second -= cycles;
        if (it->second <= 0)
            it = wmmaRegisterLatencyCounters.erase(it);
        else
            ++it;
    }
}

// Compute elapsed time needed to dispatch a VALU/transcendental op.
// During an active WMMA window, only allowed positions contribute to VALU progress.
int CDNA5ReadyQueue::computeValuAdvanceCycles(int issueCycles) const {
    if (issueCycles <= 0) return 0;
    if (coIssueCyclePos_ >= activeWmmaLatency_) return issueCycles;

    int elapsed = 0;
    int issued = 0;
    constexpr int kCoIssueBits = (int)(sizeof(activeCoIssueWindow_) * 8);

    while (issued < issueCycles) {
        const int pos = coIssueCyclePos_ + elapsed;
        bool canIssue = true;
        if (pos < activeWmmaLatency_) {
            canIssue = (pos < kCoIssueBits) && (((activeCoIssueWindow_ >> pos) & 1u) != 0u);
        }
        if (canIssue) issued++;
        elapsed++;
    }
    return elapsed;
}

// After any picked instruction (including barriers): advance by issueCycles.
void CDNA5ReadyQueue::updateWMMAStatus(DAGNode* node) {
    int elapsedCycles = node->inst->issueCycles;
    if (isVectorALU(*node->inst) || isTranscendental(*node->inst))
        elapsedCycles = computeValuAdvanceCycles(node->inst->issueCycles);
    advanceTime(elapsedCycles);
}

// True if VALU can be picked in the current co-issue timeline position.
bool CDNA5ReadyQueue::isValuPickable() const {
    if (coIssueCyclePos_ >= activeWmmaLatency_) return true;
    return (activeCoIssueWindow_ >> coIssueCyclePos_) & 1;
}

// Pop a non-WMMA node by kind (0=global, 1=local, 2=other, 3=valu),
// update state, and return the node.
DAGNode* CDNA5ReadyQueue::popNonWmmaByKind(int pickKind) {
    DAGNode* node = nullptr;
    if (pickKind == 0) {
        node = globalReadQueue.top();
        globalReadQueue.pop();
        globalReadCounter++;
        consecutiveDsCount_ = 0;
    } else if (pickKind == 1) {
        DAGNode* best = nullptr;
        for (DAGNode* n : localReadQueue) {
            if (!best || n->dsReadPriority < best->dsReadPriority) best = n;
        }
        node = best;
        localReadQueue.erase(best);
        for (const StinkyRegister& dstReg : node->inst->getDestRegs()) {
            if (!dstReg.isRegister() || isPseudoReg(dstReg)) continue;
            for (unsigned off = 0; off < dstReg.reg.num; ++off)
                wmmaRegisterLatencyCounters[dstReg.reg.idx + off] = node->inst->latencyCycles;
        }
        dsInsertedSinceLastWmma_++;
        totalDsRemaining_--;
        minDsPerWmma_ = (totalWmmaRemaining_ > 0)
                            ? (int)std::ceil((float)totalDsRemaining_ / totalWmmaRemaining_)
                            : 0;
        consecutiveDsCount_++;
    } else if (pickKind == 2) {
        node = otherQueue.top();
        otherQueue.pop();
        consecutiveDsCount_ = 0;
    } else {
        assert(pickKind == 3);
        node = valuQueue.top();
        valuQueue.pop();
        consecutiveDsCount_ = 0;
    }
    updateWMMAStatus(node);
    if (deferHeadBalanceThisRegion_) deferFirstHeadWmmaActive_ = false;
    return node;
}

// Scheduling rule (2): compute the maximum outstanding DS latency for a WMMA's src VGPRs.
// Returns 0 if all src data is ready (latency-free), >0 if hardware would stall.
int CDNA5ReadyQueue::getMaxDsLatency(DAGNode* node) {
    int maxLat = 0;
    for (const StinkyRegister& srcReg : node->inst->getSrcRegs()) {
        if (!srcReg.isRegister()) continue;
        for (unsigned off = 0; off < srcReg.reg.num; ++off) {
            auto it = wmmaRegisterLatencyCounters.find(srcReg.reg.idx + off);
            if (it != wmmaRegisterLatencyCounters.end() && it->second > maxLat) maxLat = it->second;
        }
    }
    return maxLat;
}

// Find the WMMA in wmmaQueue with the smallest max DS latency (most ready).
// Returns the node and its latency. Ties broken by DAG id (program order).
std::pair<DAGNode*, int> CDNA5ReadyQueue::findMostReadyWMMA() {
    DAGNode* best = nullptr;
    int bestLatency = INT_MAX;
    for (DAGNode* n : wmmaQueue) {
        int lat = getMaxDsLatency(n);
        if (lat < bestLatency || (lat == bestLatency && (!best || n->id < best->id))) {
            best = n;
            bestLatency = lat;
        }
    }
    return {best, bestLatency};
}

// Pick a WMMA: start a new co-issue timeline from its coIssueWindow,
// update DS distribution counters, clear loop-head deferral.
DAGNode* CDNA5ReadyQueue::pickOneFromWMMA(DAGNode* pick) {
    assert(!wmmaQueue.empty() && "The WMMA queue must not be empty");
    DAGNode* node;
    if (pick) {
        node = pick;
        wmmaQueue.erase(pick);
    } else {
        node = wmmaQueue.top();
        wmmaQueue.pop();
    }

    // consume the time that is not used by the WMMA
    if (coIssueCyclePos_ < activeWmmaLatency_) advanceTime(activeWmmaLatency_ - coIssueCyclePos_);

    activeCoIssueWindow_ = node->inst->hwInstDesc->coIssueWindow;
    coIssueCyclePos_ = 0;
    activeWmmaLatency_ = node->inst->latencyCycles;
    // Advance by WMMA issue cycles after opening a new timeline window.
    // This keeps coIssueCyclePos_ aligned with elapsed cycles right after WMMA issue.
    advanceTime(node->inst->issueCycles);
    wmmaIssueConfig.issuedCount--;

    if (deferHeadBalanceThisRegion_) deferFirstHeadWmmaActive_ = false;
    wmmaIssuedCountThisRegion_++;

    dsInsertedSinceLastWmma_ = 0;
    consecutiveDsCount_ = 0;
    totalWmmaRemaining_--;
    minDsPerWmma_ = (totalWmmaRemaining_ > 0)
                        ? (int)std::ceil((float)totalDsRemaining_ / totalWmmaRemaining_)
                        : 0;

    globalReadCounter = 0;
    return node;
}

// Pick minimum DAG id among ready non-WMMA nodes.
// Queues: globalReadQueue (throttled), localReadQueue, valuQueue (co-issue gated), otherQueue.
// kind: 0=global, 1=local, 2=other, 3=valu.
bool CDNA5ReadyQueue::findSmallestPickableNonWmma(DAGNode** outNode, int* kindOut) const {
    *outNode = nullptr;
    *kindOut = -1;
    DAGNode* best = nullptr;
    int kind = -1;

    bool dsBurstOk = consecutiveDsCount_ < kMaxConsecutiveDs;

    if (!globalReadQueue.empty() && (globalReadCounter < globalReadPerWMMA || otherQueue.empty())) {
        best = globalReadQueue.top();
        kind = 0;
    }
    if (!localReadQueue.empty() && dsBurstOk) {
        DAGNode* t = localReadQueue.top();
        if (!best || t->id < best->id) {
            best = t;
            kind = 1;
        }
    }
    if (!otherQueue.empty()) {
        DAGNode* t = otherQueue.top();
        if (!best || t->id < best->id) {
            best = t;
            kind = 2;
        }
    }
    if (!valuQueue.empty() && isValuPickable()) {
        DAGNode* t = valuQueue.top();
        // DS reads have higher priority than VALU: only pick VALU when no
        // ds_load is available (or burst limit reached).
        bool dsAvailable = !localReadQueue.empty() && dsBurstOk;
        if (!dsAvailable && (!best || t->id < best->id)) {
            best = t;
            kind = 3;
        }
    }

    if (!best) return false;
    *outNode = best;
    *kindOut = kind;
    return true;
}

// Final-fallback candidate search across non-WMMA queues.
// kind: 0=global, 1=local, 2=other, 3=valu.
bool CDNA5ReadyQueue::findOldestFallbackNonWmma(DAGNode** outNode, int* kindOut) const {
    *outNode = nullptr;
    *kindOut = -1;
    DAGNode* best = nullptr;
    int kind = -1;

    auto consider = [&](DAGNode* cand, int candKind) {
        if (cand == nullptr) return;
        if (best == nullptr || cand->id < best->id) {
            best = cand;
            kind = candKind;
        }
    };

    if (!globalReadQueue.empty()) consider(globalReadQueue.top(), 0);
    if (!localReadQueue.empty()) consider(localReadQueue.top(), 1);
    if (!otherQueue.empty()) consider(otherQueue.top(), 2);
    if (!valuQueue.empty()) consider(valuQueue.top(), 3);

    if (best == nullptr) return false;
    *outNode = best;
    *kindOut = kind;
    return true;
}

DAGNode* CDNA5ReadyQueue::popFallbackNonWmmaByKind(DAGNode* node, int pickKind) {
    assert(node != nullptr && "fallback node must not be null");
    if (pickKind == 0) {
        globalReadQueue.erase(node);
        globalReadCounter++;
    } else if (pickKind == 1) {
        localReadQueue.erase(node);
        for (const StinkyRegister& dstReg : node->inst->getDestRegs()) {
            if (!dstReg.isRegister() || isPseudoReg(dstReg)) continue;
            for (unsigned off = 0; off < dstReg.reg.num; ++off)
                wmmaRegisterLatencyCounters[dstReg.reg.idx + off] = node->inst->latencyCycles;
        }
        dsInsertedSinceLastWmma_++;
        totalDsRemaining_--;
        minDsPerWmma_ = (totalWmmaRemaining_ > 0)
                            ? (int)std::ceil((float)totalDsRemaining_ / totalWmmaRemaining_)
                            : 0;
    } else if (pickKind == 2) {
        otherQueue.erase(node);
    } else {
        assert(pickKind == 3 && "fallback kind must be global/local/other/valu");
        valuQueue.erase(node);
    }

    updateWMMAStatus(node);
    if (deferHeadBalanceThisRegion_) deferFirstHeadWmmaActive_ = false;
    return node;
}

// Drain barrierQueue to find the lowest-id barrier whose WMMA threshold is met,
// remove it, and push the rest back. Returns nullptr if no barrier qualifies.
DAGNode* CDNA5ReadyQueue::extractForcedBarrier() {
    if (barrierQueue.empty() || barrierWmmaThresholds_.empty()) return nullptr;

    DAGNode* forced = nullptr;
    for (DAGNode* node : barrierQueue) {
        auto thIt = barrierWmmaThresholds_.find(node->inst);
        if (thIt != barrierWmmaThresholds_.end() && wmmaIssuedCountThisRegion_ >= thIt->second) {
            forced = node;
            break;
        }
    }
    if (forced) barrierQueue.erase(forced);
    return forced;
}

// Compute forceBarrierAfterNthWmma_ for this region from register dependencies.
//
//  Step 1a — collect all movable barriers with their PSEUDO src token sets.
//  Step 1b — for each barrier, find the latest ds_read whose dest PSEUDO token matches.
//  Step 2 & 3 — find the last WMMA in [regionStart, that ds_read] whose src VGPRs
//               overlap the ds_read's dest VGPRs; record its 1-based index (wmmaIdx).
//  Step 4 — threshold N = wmmaIdx + (targetDSLoadLatency / wmmaIssueConfig.latency) + 1.
void CDNA5ReadyQueue::computeBarrierAfterThresholds(IRList::iterator regionStart,
                                                    IRList::iterator regionEnd) {
    // Step 1a: collect all movable barriers with their PSEUDO src token sets.
    auto barriers = collectBarrierTokens(regionStart, regionEnd, /*useSrc=*/true);

    for (const BarrierTokenEntry& be : barriers) {
        // Step 1b: scan [regionStart, barrier) — find the latest ds_read whose
        //          dest PSEUDO token matches a src token of this barrier.
        StinkyInstruction* targetDSLoad = nullptr;
        IRList::iterator targetDSLoadIt = regionEnd;
        uint32_t targetDSLoadLatency = 0;
        for (IRList::iterator it = regionStart; it != regionEnd; ++it) {
            StinkyInstruction& inst = getStinkyInst(it);
            if (&inst == be.barrier) break;
            if (!isDSRead(inst)) continue;
            for (const StinkyRegister& src : inst.getSrcRegs()) {
                if (isPseudoReg(src) && be.tokens.count(src.reg.idx)) {
                    targetDSLoad = &inst;
                    targetDSLoadIt = it;  // keep updating → ends up as latest
                    targetDSLoadLatency = inst.latencyCycles;
                    break;
                }
            }
        }
        if (!targetDSLoad) continue;

        // Step 2 & 3: collect VGPR dest regs of the latest ds_read, then scan
        //             [regionStart, targetDSLoad] (inclusive) for WMMAs — keep
        //             updating lastOverlap so the last matching WMMA is recorded.
        auto loadDestVGPRs = collectDestVGPRs(*targetDSLoad);
        int wmmaIdx = 0;
        int lastOverlap = 0;
        IRList::iterator wmmaEnd = std::next(targetDSLoadIt);
        for (IRList::iterator it = regionStart; it != wmmaEnd; ++it) {
            StinkyInstruction& inst = getStinkyInst(it);
            if (!isMatrixInstruction(inst)) continue;
            wmmaIdx++;
            if (srcVGPRsOverlap(inst, loadDestVGPRs)) lastOverlap = wmmaIdx;
        }

        // Step 4: threshold N = wmmaIdx + (targetDSLoadLatency / wmmaIssueConfig.latency) + 1.
        barrierWmmaThresholds_[be.barrier] =
            lastOverlap + (targetDSLoadLatency / wmmaIssueConfig.latency) + 1;
    }
}

// Compute forceBarrierBeforeNthWmma_ for this region from register dependencies.
//
//  Step 1  — for each barrier, find ds_reads whose src PSEUDO token matches.
//  Step 2  — for each such ds_read, find the first WMMA whose src overlaps the
//            ds_read's dest VGPRs; take the maximum such 1-based WMMA index across
//            all matching ds_reads (MaximumWMMAIdx).
//  Step 3  — residualCycles = max(0, targetDSLoadLatency
//                                    - MaximumWMMAIdx * wmmaIssueConfig.latency)
//  Step 4  — forceBarrierBeforeNthWmma = (residualCycles / wmmaIssueConfig.latency) + 1
//            i.e. the barrier is suppressed from firing until WMMA count reaches this value,
//            ensuring it fires before the computed WMMA slot.
std::unordered_map<StinkyInstruction*, int> CDNA5ReadyQueue::computeBarrierBeforeThresholds(
    IRList::iterator regionStart, IRList::iterator regionEnd) {
    std::unordered_map<StinkyInstruction*, int> result;

    auto barriers = collectBarrierTokens(regionStart, regionEnd, /*useSrc=*/false);

    for (const BarrierTokenEntry& be : barriers) {
        // Step 1: scan (barrier, regionEnd] — collect all ds_reads whose src
        //         PSEUDO token matches a dest token of this barrier.
        struct DSReadMatch {
            uint32_t latency;
            std::unordered_set<uint32_t> destVGPRs;
        };
        std::vector<DSReadMatch> matchingDSReads;
        bool isAfterBarrier = false;
        for (IRList::iterator it = regionStart; it != regionEnd; ++it) {
            StinkyInstruction& inst = getStinkyInst(it);
            if (&inst == be.barrier) isAfterBarrier = true;
            if (!isDSRead(inst) || !isAfterBarrier) continue;
            for (const StinkyRegister& src : inst.getSrcRegs()) {
                if (isPseudoReg(src) && be.tokens.count(src.reg.idx)) {
                    matchingDSReads.push_back(
                        {static_cast<uint32_t>(inst.latencyCycles), collectDestVGPRs(inst)});
                    break;
                }
            }
        }
        if (matchingDSReads.empty()) continue;

        // Step 2: for each matching ds_read, find the first WMMA in [regionStart, regionEnd)
        //         whose src VGPRs overlap the ds_read's dest VGPRs; take the maximum
        //         1-based WMMA index across all ds_reads (MaximumWMMAIdx).
        int maximumWMMAIdx = -1;
        int targetDSLoadLatency = 0;
        for (const DSReadMatch& dse : matchingDSReads) {
            int wmmaIdx = 0;
            for (IRList::iterator it = regionStart; it != regionEnd; ++it) {
                StinkyInstruction& inst = getStinkyInst(it);
                if (!isMatrixInstruction(inst)) continue;
                wmmaIdx++;
                if (srcVGPRsOverlap(inst, dse.destVGPRs)) {
                    if (wmmaIdx > maximumWMMAIdx) {
                        maximumWMMAIdx = wmmaIdx;
                        targetDSLoadLatency = (int)dse.latency;
                    }
                    break;  // want the first, not the last
                }
            }
        }
        if (maximumWMMAIdx == -1) continue;

        // Step 3: residualCycles = max(0, targetDSLoadLatency
        //                               - MaximumWMMAIdx * wmmaIssueConfig.latency)
        int residualCycles =
            std::max(0, targetDSLoadLatency - maximumWMMAIdx * (int)wmmaIssueConfig.latency);

        // Step 4: beforeN = (residualCycles / wmmaIssueConfig.latency) + 1
        int beforeN = (residualCycles / (int)wmmaIssueConfig.latency) + 1;
        int maxFinalWmmaIdx = targetDSLoadLatency / (int)wmmaIssueConfig.latency;
        int beforeThreshold =
            std::max(0, wmmaIssueConfig.issuedCount - (beforeN + maxFinalWmmaIdx));
        result[be.barrier] = beforeThreshold;
    }

    return result;
}

// Main scheduling orchestration:
//   Phase A: forced barrier — when wmmaIssuedCountThisRegion_ reaches a per-barrier threshold.
//   Phase B: WMMA if DS latency gate (rule 2) passed, DS distribution hint (rule 4) met,
//            loop head balance (rule 5) ok, and program order (rule 1) allows.
//   Phase C: inside WMMA latency window — fill with non-WMMA work.
//   Phase D: outside WMMA latency — pick smallest-id from any non-WMMA queue.
//   Phase E: forced WMMA — pick most-ready WMMA when all non-WMMA queues are empty.
//   Phase F: barriers — only after all compute queues (WMMA + non-WMMA) are drained.
DAGNode* CDNA5ReadyQueue::pickOne() {
    PASS_DEBUG(
        std::cerr << "[CDNA5 pickOne] prevPick="
                  << (lastPickedNode_ ? std::to_string(lastPickedNode_->id) : std::string("none"))
                  << "\n");
    auto rememberPick = [this](DAGNode* node) {
        lastPickedNode_ = node;
        return node;
    };

    // Phase A — forced barrier: issue the lowest-id barrier whose WMMA threshold is met.
    if (DAGNode* forced = extractForcedBarrier()) {
        PASS_DEBUG(std::cerr << "[CDNA5 pickOne] forced barrier: wmmaIssued="
                             << wmmaIssuedCountThisRegion_
                             << " threshold=" << barrierWmmaThresholds_.at(forced->inst)
                             << " barrierId=" << forced->id << std::flush << "\n");
        updateWMMAStatus(forced);
        return rememberPick(forced);
    }

    // Phase B — try WMMA if all gates pass.
    bool otherQueuesHaveWork = !globalReadQueue.empty() || !localReadQueue.empty() ||
                               !otherQueue.empty() || !valuQueue.empty();

    if (!wmmaQueue.empty()) {
        auto [bestWMMA, bestLatency] = findMostReadyWMMA();

        DAGNode* smallestPickable = nullptr;
        int pickKind = -1;
        findSmallestPickableNonWmma(&smallestPickable, &pickKind);

        const bool blockWmmaForLoopHeadBalance =
            deferHeadBalanceThisRegion_ && deferFirstHeadWmmaActive_ && otherQueuesHaveWork;
        // Hard gate: while the previous WMMA latency window is still active,
        // do not issue another WMMA if any non-WMMA work is pickable.
        const bool blockWmmaForActiveWindow =
            (coIssueCyclePos_ < activeWmmaLatency_) && (smallestPickable != nullptr);

        bool blockWmmaForAtLeastOneNonWmmaInterleaving = false;
        if (lastPickedNode_ != nullptr) {
            blockWmmaForAtLeastOneNonWmmaInterleaving =
                otherQueuesHaveWork && isMatrixInstruction(*lastPickedNode_->inst);
        }
        PASS_DEBUG(std::cerr << "[CDNA5 pickOne] Phase B candidate wmmaId=" << bestWMMA->id
                             << " bestLatency=" << bestLatency
                             << " blockLoopHead=" << blockWmmaForLoopHeadBalance
                             << " blockActiveWindow=" << blockWmmaForActiveWindow
                             << " blockAtLeastOneNonWmmaInterleaving="
                             << blockWmmaForAtLeastOneNonWmmaInterleaving
                             << " localReadQ=" << localReadQueue.size() << " nonWmmaMinId="
                             << (smallestPickable ? std::to_string(smallestPickable->id)
                                                  : std::string("none"))
                             << "\n");
        if (bestLatency <= 0 && !blockWmmaForLoopHeadBalance && !blockWmmaForActiveWindow &&
            !blockWmmaForAtLeastOneNonWmmaInterleaving) {
            DAGNode* node = pickOneFromWMMA(bestWMMA);
            PASS_DEBUG(std::cerr << "[CDNA5 pickOne] Phase B picked WMMA dagId=" << node->id
                                 << "\n");
            return rememberPick(node);
        }
    }

    // Phase C — inside WMMA latency window: fill with non-WMMA work.
    // If nothing non-WMMA is pickable, fast-forward past the window so
    // Phase D/E can pick the next WMMA. Barriers are deferred until all
    // compute queues (including wmmaQueue) are empty.
    if (coIssueCyclePos_ < activeWmmaLatency_) {
        DAGNode* smallestPickable = nullptr;
        int pickKind = -1;
        findSmallestPickableNonWmma(&smallestPickable, &pickKind);
        if (smallestPickable != nullptr) {
            PASS_DEBUG(std::cerr << "[CDNA5 pickOne] Phase C picked non-WMMA dagId="
                                 << smallestPickable->id << " kind=" << pickKind << "\n");
            return rememberPick(popNonWmmaByKind(pickKind));
        }

        advanceTime(activeWmmaLatency_ - coIssueCyclePos_);
    }

    // Phase D — outside WMMA latency: pick smallest-id from any non-WMMA queue.
    {
        DAGNode* smallestPickable = nullptr;
        int pickKind = -1;
        findSmallestPickableNonWmma(&smallestPickable, &pickKind);
        if (smallestPickable != nullptr) {
            return rememberPick(popNonWmmaByKind(pickKind));
        }
    }

    // Phase E — forced WMMA: pick the most-ready WMMA before barriers.
    if (!wmmaQueue.empty()) {
        auto [bestWMMA, bestLatency] = findMostReadyWMMA();
        (void)bestLatency;
        DAGNode* node = pickOneFromWMMA(bestWMMA);
        return rememberPick(node);
    }

    // Phase F — barriers after all compute work is done.
    if (!barrierQueue.empty()) {
        DAGNode* barrier = barrierQueue.top();
        barrierQueue.pop();
        updateWMMAStatus(barrier);
        PASS_DEBUG(std::cerr << "[DAG CDNA5 pickOne] Phase F barrierQueue dagId=" << barrier->id
                             << " (non-barrier buckets empty for this pick)\n";
                   barrier->inst->dump(std::cerr); std::cerr << "\n");
        return rememberPick(barrier);
    }

    // Phase G — final safety net: if any bucket still has ready nodes but all
    // policy gates rejected them, force-pick the oldest DAG node to guarantee
    // progress and avoid scheduler starvation.
    DAGNode* fallback = nullptr;
    int fallbackKind = -1;
    if (findOldestFallbackNonWmma(&fallback, &fallbackKind)) {
        PASS_DEBUG(std::cerr << "[CDNA5 pickOne] Phase G fallback pick dagId=" << fallback->id
                             << " kind=" << fallbackKind << "\n");
        return rememberPick(popFallbackNonWmmaByKind(fallback, fallbackKind));
    }

    assert(false && "CDNA5ReadyQueue::pickOne: all buckets empty");
    return nullptr;
}

// Route ready DAG nodes into priority buckets.
void CDNA5ReadyQueue::push(DAGNode* node) {
    if (isMatrixInstruction(*node->inst)) {
        wmmaQueue.push(node);
        return;
    }

    if (getPassContext().getPassFeatureConfig().dagFeatures.distributeGlobalRead &&
        isTensorLoad(*node->inst)) {
        globalReadQueue.push(node);
        return;
    }

    if (isDSRead(*node->inst)) {
        localReadQueue.push(node);
        return;
    }

    if (isVectorALU(*node->inst) || isTranscendental(*node->inst)) {
        valuQueue.push(node);
        return;
    }

    if (isBarrier(*node->inst)) {
        barrierQueue.push(node);
        return;
    }

    otherQueue.push(node);
}

bool CDNA5ReadyQueue::empty() const {
    return wmmaQueue.empty() && globalReadQueue.empty() && localReadQueue.empty() &&
           valuQueue.empty() && otherQueue.empty() && barrierQueue.empty();
}

// Per-BB init. Rule (5): cross-BB loop tail WMMA detection.
// Resets co-issue timeline. Sets wmmaIssueConfig.latency from first WMMA in block.
void CDNA5ReadyQueue::onInit(IRList::iterator regionStart, IRList::iterator regionEnd) {
    deferFirstHeadWmmaActive_ = false;
    deferHeadBalanceThisRegion_ = false;

    activeCoIssueWindow_ = 0;
    coIssueCyclePos_ = 0;
    activeWmmaLatency_ = 0;

    currentBB_ = (regionStart != regionEnd) ? regionStart->getParent() : nullptr;

    if (getPassContext().getPassFeatureConfig().loopConfig.unrollGemm == false) return;

    const Loop* loop = getLoop();
    if (loop && loop->headerBB && loop->latchBB) {
        if (latchBBTailIsWmma(*loop->latchBB)) deferFirstHeadWmmaActive_ = true;
    }

    wmmaIssueConfig.latency = 0;
    for (IRList::iterator it = regionStart; it != regionEnd; ++it) {
        auto* instPtr = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!instPtr) continue;
        if (isMatrixInstruction(*instPtr)) {
            wmmaIssueConfig.latency = instPtr->latencyCycles;
            break;
        }
    }

    restoreCrossBBStateFromLoop();
}

void CDNA5ReadyQueue::restoreCrossBBStateFromLoop() {
    crossBBDsResiduals_.clear();
    const Loop* loop = getLoop();
    if (!currentBB_ || !loop || !loop->contains(currentBB_) || !getAnalysisCache()) return;

    for (BasicBlock* pred : currentBB_->getPredecessors()) {
        const BBScheduleState* state = getAnalysisCache()->lookup(pred);
        if (!state) continue;
        for (const auto& [regIdx, rem] : state->dsResiduals) {
            if (rem > 0) crossBBDsResiduals_[regIdx] = std::max(crossBBDsResiduals_[regIdx], rem);
        }
    }
}

void CDNA5ReadyQueue::onFinishBB() {
    if (currentBB_ && getAnalysisCache())
        getAnalysisCache()->store(currentBB_, {0, wmmaRegisterLatencyCounters});
}

// Per scheduling region. Rule (4): count ds_loads and WMMAs for adaptive distribution.
// Rule (2): seedWmmaDsLatencyFromPrefix. Rule (5): head balance.
// Barrier thresholds: computeBarrierAfterThresholds / computeBarrierBeforeThresholds.
void CDNA5ReadyQueue::onInitRegion(IRList::iterator regionStart, IRList::iterator regionEnd,
                                   IRList::iterator blockBegin) {
    wmmaIssuedCountThisRegion_ = 0;
    dsInsertedSinceLastWmma_ = 0;
    lastPickedNode_ = nullptr;
    if (getPassContext().getPassFeatureConfig().loopConfig.unrollGemm == false) return;

    const Loop* loop = getLoop();
    deferHeadBalanceThisRegion_ = deferFirstHeadWmmaActive_ && loop &&
                                  loop->headerBB == blockBegin->getParent() &&
                                  regionStart != blockBegin;

    seedWmmaDsLatencyFromPrefix(blockBegin, regionStart, wmmaRegisterLatencyCounters,
                                crossBBDsResiduals_);

    wmmaIssueConfig.issuedCount = 0;
    totalDsRemaining_ = 0;
    totalWmmaRemaining_ = 0;
    consecutiveDsCount_ = 0;
    hasWMMAInRegion_ = false;
    for (IRList::iterator it = regionStart; it != regionEnd; ++it) {
        auto* instPtr = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!instPtr) continue;
        StinkyInstruction& inst = *instPtr;

        if (isDSRead(inst)) totalDsRemaining_++;

        if (isMatrixInstruction(inst)) {
            wmmaIssueConfig.issuedCount++;
            totalWmmaRemaining_++;
            hasWMMAInRegion_ = true;
        }
    }

    minDsPerWmma_ = (totalWmmaRemaining_ > 0)
                        ? (int)std::ceil((float)totalDsRemaining_ / totalWmmaRemaining_)
                        : 0;

    barrierWmmaThresholds_.clear();
    if (hasWMMAInRegion_) {
        computeBarrierAfterThresholds(regionStart, regionEnd);
        auto beforeThresholds = computeBarrierBeforeThresholds(regionStart, regionEnd);
        for (auto& [barrier, beforeVal] : beforeThresholds) {
            auto it = barrierWmmaThresholds_.find(barrier);
            if (it != barrierWmmaThresholds_.end())
                it->second = (it->second + beforeVal) / 2;
            else
                barrierWmmaThresholds_[barrier] = beforeVal;
        }
    }
}
}  // namespace
