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
#include <climits>
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

#include "InFlightQueue.hpp"
#include "ReadyQueue.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"

namespace {
using namespace stinkytofu;

enum NonWmmaKind { kGlobalRead = 0, kLocalRead, kOther, kValu };

// CDNA5 (Gfx1250) scheduling defaults. Used when dagFeatures still hold the
// PassFeatureConfig sentinel values (0 / INT_MAX). Explicit non-sentinel config wins.
constexpr int kCdna5DsReadQueueDepth = 16;
constexpr int kCdna5DsReadDrainLatency = 72;
constexpr int kCdna5DsReadPerWmma = 3;
constexpr int kCdna5GlobalReadPerWmma = 1;

// -------------------------------------------------------------------------
// Prefix / loop analysis (free functions; no CDNA5ReadyQueue state)
// -------------------------------------------------------------------------

// Scheduling rule (2): simulate producer completion over [blockBegin, regionStart) —
// outstanding data-ready latencies decrease by each instruction's issueCycles; each
// producer overwrites its dest VGPRs with that op's latencyCycles. Remaining counts seed
// regDataReadyCounters so the first WMMA/consumer in a region sees preloop / in-BB
// producers the register DAG may not edge to (double-buffer: WMMA on X0 while in-loop ds
// fills X1). Generalized from ds_load-only to all producers, with the same
// latencyCycles > issueCycles self-clear skip used at issue time; short-latency ops
// decay away across the prefix, only long ones (ds_load) persist. Caller: onInitRegion.
//
// crossBBResiduals: data-ready residuals from predecessor BBs (merged in onInit).
// They decay through the prefix alongside within-BB producers so cross-BB loads
// properly gate consumers.
static void seedWmmaDsLatencyFromPrefix(IRList::iterator blockBegin, IRList::iterator regionStart,
                                        std::map<int, int>& regDataReadyCounters,
                                        const std::map<int, int>& crossBBResiduals) {
    regDataReadyCounters.clear();
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

        if (inst.latencyCycles <= inst.issueCycles) continue;
        for (const StinkyRegister& dstReg : inst.getDestRegs()) {
            if (!dstReg.isRegister() || isPseudoReg(dstReg)) continue;
            for (unsigned off = 0; off < dstReg.reg.num; ++off)
                pending[dstReg.reg.idx + off] = inst.latencyCycles;
        }
    }

    for (const auto& [regIdx, rem] : pending) {
        if (rem > 0) regDataReadyCounters[regIdx] = rem;
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
        if (isLabel(*instPtr) || isBranch(*instPtr) || isCall(*instPtr)) continue;
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
    IRList::iterator it;
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
        entry.it = it;
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
//  (4) Per-WMMA-window DS cap — dagFeatures.dsReadPerWmma ds_loads per WMMA window
//      (INT_MAX = unconstrained).
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
    int globalReadPerWMMA = kCdna5GlobalReadPerWmma;

    InFlightQueue globalReadInflight_;
    int crossBBGlobalReadCount_ = 0;
    int crossBBGlobalReadResidual_ = 0;

    InFlightQueue dsReadInflight_;

    int globalReadQueueDepth() const {
        return getPassContext().getPassFeatureConfig().dagFeatures.globalReadQueueDepth;
    }
    int globalReadDrainLatency() const {
        return getPassContext().getPassFeatureConfig().dagFeatures.globalReadDrainLatency;
    }
    bool globalReadQueueFull() const {
        return globalReadInflight_.full();
    }

    int dsReadQueueDepth() const {
        const int cfg = getPassContext().getPassFeatureConfig().dagFeatures.dsReadQueueDepth;
        return cfg > 0 ? cfg : kCdna5DsReadQueueDepth;
    }
    int dsReadDrainLatency() const {
        const int cfg = getPassContext().getPassFeatureConfig().dagFeatures.dsReadDrainLatency;
        return cfg > 0 ? cfg : kCdna5DsReadDrainLatency;
    }
    int dsReadPerWmma() const {
        const int cfg = getPassContext().getPassFeatureConfig().dagFeatures.dsReadPerWmma;
        return cfg < INT_MAX ? cfg : kCdna5DsReadPerWmma;
    }
    bool dsReadQueueFull() const {
        return dsReadInflight_.full();
    }

    // --- VALU co-issue timeline tracker ---
    uint16_t activeCoIssueWindow_ = 0;
    int coIssueCyclePos_ = 0;
    int activeWmmaLatency_ = 0;
    // WMMA that opened the current latency window (valid while coIssueCyclePos_ <
    // activeWmmaLatency_). Used to detect ds_load dest / WMMA src VGPR overlap hazards.
    DAGNode* activeWmmaNode_ = nullptr;

    // --- Per-WMMA-window DS cap (dagFeatures.dsReadPerWmma) ---
    int maxDsPerWmmaWindow_ = 0;
    int dsInsertedSinceLastWmma_ = 0;

    // (A) RAW data-ready gate. Per reg index: remaining modeled latency until a
    // producer's result is safe to consume (e.g. ds_load LDS->VGPR, 56 cyc). Any
    // long-latency producer stamps it; a consumer whose src is still in flight is not
    // "free". Decays in advanceTime. Crosses BBs via BBScheduleState.dsResiduals.
    std::map<int, int> regDataReadyCounters;

    // (B) elapse-time ordering. Timeline (advanceTime clock) at which each reg was last
    // touched by any operand (dst or src) of an issued instruction. Used to order
    // already-free nodes within a bucket: prefer the node whose operands were touched
    // longest ago, so a register-overwrite (e.g. a VALU reusing a just-read ds_load
    // address) is naturally deferred behind other work. Per-region; reset each region.
    std::map<int, int> regLastTouch_;
    int clock_ = 0;

    WMMAIssueConfig wmmaIssueConfig;

    bool hasWMMAInRegion_ = false;

    // --- Loop head balancing ---
    bool deferFirstHeadWmmaActive_ = false;
    bool deferHeadBalanceThisRegion_ = false;

    // Per-barrier forced-issue threshold: maps StinkyInstruction* -> N.
    std::unordered_map<StinkyInstruction*, int> barrierWmmaThresholds_;
    // Per-barrier matching ds_load count collected in computeBarrierBeforeThresholds.
    std::unordered_map<StinkyInstruction*, int> barrierDsLoadCounts_;
    struct BarrierBeforeOutput {
        int beforeThreshold = 0;
        int wmmaWindowsNeeded = 0;
    };
    struct BarrierAfterOutput {
        int afterThreshold = 0;
        int overlapWmmaWindow = 0;
    };

    int wmmaIssuedCountThisRegion_ = 0;

    BasicBlock* currentBB_ = nullptr;

    DAGNode* lastPickedNode_ = nullptr;

    std::map<int, int> crossBBDsResiduals_;

    void advanceTime(int cycles);
    int computeValuAdvanceCycles(int issueCycles) const;
    void updateWMMAStatus(DAGNode* node);
    void stampDataReady(const StinkyInstruction& inst);
    void touchOperands(const StinkyInstruction& inst);
    int getMaxSrcDataWait(DAGNode* node) const;
    bool nonWmmaHasHazard(DAGNode* node) const;
    int nodeElapseKey(DAGNode* node) const;
    DAGNode* pickFreeBest(const ReadySetByDAGid& queue) const;
    std::pair<DAGNode*, int> findMostReadyWMMA();
    DAGNode* pickOneFromWMMA(DAGNode* pick = nullptr);
    bool findSmallestPickableNonWmma(DAGNode* pickedDS, DAGNode** outNode, int* kindOut) const;
    bool destOverlapsActiveWmmaSrc(DAGNode* node) const;
    bool findOldestFallbackNonWmma(DAGNode* pickedDS, DAGNode** outNode, int* kindOut) const;
    DAGNode* extractForcedBarrier();
    std::unordered_map<StinkyInstruction*, BarrierAfterOutput> computeBarrierAfterThresholds(
        IRList::iterator regionStart, IRList::iterator regionEnd);
    std::unordered_map<StinkyInstruction*, BarrierBeforeOutput> computeBarrierBeforeThresholds(
        IRList::iterator regionStart, IRList::iterator regionEnd);
    bool isValuPickable() const;
    DAGNode* popNonWmma(DAGNode* node, int pickKind);

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

// Advance the co-issue timeline and the elapse-time clock, and decay the RAW
// data-ready counters by \p cycles.
void CDNA5ReadyQueue::advanceTime(int cycles) {
    coIssueCyclePos_ += cycles;
    clock_ += cycles;
    globalReadInflight_.advance(cycles);
    dsReadInflight_.advance(cycles);
    for (auto it = regDataReadyCounters.begin(); it != regDataReadyCounters.end();) {
        it->second -= cycles;
        if (it->second <= 0)
            it = regDataReadyCounters.erase(it);
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

// After a picked instruction: advance the co-issue timeline. Barriers use result latency
// (latencyCycles); VALU/transcendentals use co-issue-aware issue progress; others use issueCycles.
void CDNA5ReadyQueue::updateWMMAStatus(DAGNode* node) {
    int elapsedCycles = node->inst->issueCycles;
    if (isBarrier(*node->inst))
        elapsedCycles = node->inst->latencyCycles;
    else if (isVectorALU(*node->inst) || isTranscendental(*node->inst))
        elapsedCycles = computeValuAdvanceCycles(node->inst->issueCycles);
    advanceTime(elapsedCycles);
}

// True if VALU can be picked in the current co-issue timeline position.
bool CDNA5ReadyQueue::isValuPickable() const {
    if (coIssueCyclePos_ >= activeWmmaLatency_) return true;
    return (activeCoIssueWindow_ >> coIssueCyclePos_) & 1;
}

// Remove a specific non-WMMA node from its queue by kind (0=global, 1=local,
// 2=other, 3=valu), update all scheduling counters, and return the node.
DAGNode* CDNA5ReadyQueue::popNonWmma(DAGNode* node, int pickKind) {
    assert(node != nullptr);
    if (pickKind == kGlobalRead) {
        globalReadQueue.erase(node);
        globalReadCounter++;
        if (globalReadQueueDepth() > 0) globalReadInflight_.push(globalReadDrainLatency());
    } else if (pickKind == kLocalRead) {
        localReadQueue.erase(node);
        dsReadInflight_.push(dsReadDrainLatency());
        dsInsertedSinceLastWmma_++;
    } else if (pickKind == kOther) {
        otherQueue.erase(node);
    } else {
        assert(pickKind == kValu);
        valuQueue.erase(node);
    }
    // (A) RAW: stamp this producer's dest data-ready latency (e.g. ds_load).
    // (B) elapse: record the timeline touch for all operands (dst + src).
    stampDataReady(*node->inst);
    touchOperands(*node->inst);
    updateWMMAStatus(node);
    if (deferHeadBalanceThisRegion_) deferFirstHeadWmmaActive_ = false;
    return node;
}

// (A) RAW stamp: record each dest reg's data-ready latency. Skip when
// latencyCycles <= issueCycles: after a pick, updateWMMAStatus -> advanceTime(issueCycles)
// runs before the next pickOne and would immediately decay such a stamp to 0, so it can
// never gate a consumer. Skipping it makes the dominant gfx1250 case (VALU/SALU
// latency == issue == 1) a no-op with no map churn; only ds_load (56) and rare latency-2
// ops persist.
void CDNA5ReadyQueue::stampDataReady(const StinkyInstruction& inst) {
    if (inst.latencyCycles <= inst.issueCycles) return;
    for (const StinkyRegister& dst : inst.getDestRegs()) {
        if (!dst.isRegister() || isPseudoReg(dst)) continue;
        for (unsigned off = 0; off < dst.reg.num; ++off)
            regDataReadyCounters[dst.reg.idx + off] = inst.latencyCycles;
    }
}

// (B) elapse touch: record the current timeline clock for every operand reg (dst + src)
// of an issued instruction, so a later node reusing a just-touched reg can be deferred.
void CDNA5ReadyQueue::touchOperands(const StinkyInstruction& inst) {
    for (const StinkyRegister& dst : inst.getDestRegs()) {
        if (!dst.isRegister() || isPseudoReg(dst)) continue;
        for (unsigned off = 0; off < dst.reg.num; ++off) regLastTouch_[dst.reg.idx + off] = clock_;
    }
    for (const StinkyRegister& src : inst.getSrcRegs()) {
        if (!src.isRegister() || isPseudoReg(src)) continue;
        for (unsigned off = 0; off < src.reg.num; ++off) regLastTouch_[src.reg.idx + off] = clock_;
    }
}

// (A) RAW data-ready gate: max outstanding data-ready latency over a node's src VGPRs.
// Returns 0 if all src data is ready (safe to consume), >0 if hardware would stall.
int CDNA5ReadyQueue::getMaxSrcDataWait(DAGNode* node) const {
    int maxLat = 0;
    for (const StinkyRegister& srcReg : node->inst->getSrcRegs()) {
        if (!srcReg.isRegister()) continue;
        for (unsigned off = 0; off < srcReg.reg.num; ++off) {
            auto it = regDataReadyCounters.find(srcReg.reg.idx + off);
            if (it != regDataReadyCounters.end() && it->second > maxLat) maxLat = it->second;
        }
    }
    return maxLat;
}

// A non-WMMA compute node is not "free" while a src still consumes in-flight producer
// data (RAW). WAR (overwrite of a just-read reg) is no longer a hard gate — it is
// handled by elapse-time ordering in pickFreeBest.
bool CDNA5ReadyQueue::nonWmmaHasHazard(DAGNode* node) const {
    return getMaxSrcDataWait(node) > 0;
}

// (B) elapse key: min over the node's operand regs (dst + src) of (clock_ - lastTouch).
// The most-recently-touched operand binds (smallest elapse), so a node reusing a
// just-touched reg ranks low and is deferred. Regs never touched => INT_MAX (very old).
int CDNA5ReadyQueue::nodeElapseKey(DAGNode* node) const {
    int minElapse = INT_MAX;
    auto consider = [&](const StinkyRegister& r) {
        if (!r.isRegister() || isPseudoReg(r)) return;
        for (unsigned off = 0; off < r.reg.num; ++off) {
            auto it = regLastTouch_.find(r.reg.idx + off);
            const int elapse = (it == regLastTouch_.end()) ? INT_MAX : (clock_ - it->second);
            if (elapse < minElapse) minElapse = elapse;
        }
    };
    for (const StinkyRegister& dst : node->inst->getDestRegs()) consider(dst);
    for (const StinkyRegister& src : node->inst->getSrcRegs()) consider(src);
    return minElapse;
}

// Best hazard-free node in \p queue: largest elapse key (operands touched longest ago),
// tie broken by smallest DAG id. Returns nullptr if every ready node has a RAW hazard.
DAGNode* CDNA5ReadyQueue::pickFreeBest(const ReadySetByDAGid& queue) const {
    DAGNode* best = nullptr;
    int bestElapse = INT_MIN;
    for (DAGNode* n : queue) {  // iterates smallest-id first, so ties keep the oldest id
        if (nonWmmaHasHazard(n)) continue;
        const int elapse = nodeElapseKey(n);
        if (elapse > bestElapse) {
            bestElapse = elapse;
            best = n;
        }
    }
    return best;
}

// Find the WMMA in wmmaQueue with the smallest max data-ready latency (most ready).
// Returns the node and its latency. Ties broken by DAG id (program order).
std::pair<DAGNode*, int> CDNA5ReadyQueue::findMostReadyWMMA() {
    DAGNode* best = nullptr;
    int bestLatency = INT_MAX;
    for (DAGNode* n : wmmaQueue) {
        int lat = getMaxSrcDataWait(n);
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
    activeWmmaNode_ = node;
    // Advance by WMMA issue cycles after opening a new timeline window.
    // This keeps coIssueCyclePos_ aligned with elapsed cycles right after WMMA issue.
    advanceTime(node->inst->issueCycles);
    wmmaIssueConfig.issuedCount--;

    if (deferHeadBalanceThisRegion_) deferFirstHeadWmmaActive_ = false;
    wmmaIssuedCountThisRegion_++;

    dsInsertedSinceLastWmma_ = 0;
    maxDsPerWmmaWindow_ = dsReadPerWmma();

    globalReadCounter = 0;
    // (A) RAW: stamp the WMMA's dest (accumulator) data-ready latency.
    // (B) elapse: record the timeline touch for all its operands.
    stampDataReady(*node->inst);
    touchOperands(*node->inst);
    return node;
}

// True if issuing \p node now would risk a hazard: while the WMMA that opened the current
// latency window is still in flight, \p node's dest VGPRs overlap that WMMA's src VGPRs, so
// the write could clobber a source the WMMA is still reading. Applies to any writer whose
// dest could alias the active WMMA's srcs (e.g. a ds_load dest or a VALU dest).
bool CDNA5ReadyQueue::destOverlapsActiveWmmaSrc(DAGNode* node) const {
    if (node == nullptr || activeWmmaNode_ == nullptr) return false;
    // Only relevant while the WMMA latency window is still active.
    if (coIssueCyclePos_ >= activeWmmaLatency_) return false;

    for (const StinkyRegister& dstReg : node->inst->getDestRegs()) {
        if (!dstReg.isRegister() || isPseudoReg(dstReg)) continue;
        for (const StinkyRegister& srcReg : activeWmmaNode_->inst->getSrcRegs()) {
            if (!srcReg.isRegister() || isPseudoReg(srcReg)) continue;
            // isOverlap already requires the same register class before checking indices.
            if (dstReg.isOverlap(srcReg)) return true;
        }
    }
    return false;
}

// Pick minimum DAG id among ready non-WMMA nodes.
// Queues: globalReadQueue (throttled), localReadQueue, valuQueue (co-issue gated), otherQueue.
// SALU/other and VALU picks go through pickFreeBest: a node whose src still consumes
// in-flight producer data (RAW) is skipped; among the rest, the node whose operands were
// touched longest ago wins (elapse ordering — defers a reg-overwrite behind other work),
// tie broken by DAG id. When none is free those queues contribute nothing and Phase G
// falls back to the oldest.
// kind: 0=global, 1=local, 2=other, 3=valu.
bool CDNA5ReadyQueue::findSmallestPickableNonWmma(DAGNode* pickedDS, DAGNode** outNode,
                                                  int* kindOut) const {
    *outNode = nullptr;
    *kindOut = -1;
    DAGNode* best = nullptr;
    int kind = -1;

    // Per-WMMA-window DS cap (rule 4) only spreads ds_loads across an active WMMA
    // co-issue window; it is meaningless when no WMMA is available to issue, so it
    // is applied only while a WMMA is pending. Otherwise ds_loads drain freely,
    // bounded only by the real hardware limiter dsReadQueueFull().
    const bool dsCapReached = !wmmaQueue.empty() && dsInsertedSinceLastWmma_ >= maxDsPerWmmaWindow_;
    bool dsWindowOk = pickedDS && !dsCapReached && !dsReadQueueFull();

    if (!globalReadQueue.empty() && !globalReadQueueFull() &&
        (globalReadCounter < globalReadPerWMMA || otherQueue.empty())) {
        best = globalReadQueue.top();
        kind = kGlobalRead;
    }
    if (dsWindowOk) {
        if (!best || pickedDS->id < best->id) {
            best = pickedDS;
            kind = kLocalRead;
        }
    }
    if (DAGNode* t = pickFreeBest(otherQueue)) {
        if (!best || t->id < best->id) {
            best = t;
            kind = kOther;
        }
    }
    if (isValuPickable()) {
        if (DAGNode* t = pickFreeBest(valuQueue)) {
            if (!dsWindowOk && (!best || t->id < best->id)) {
                best = t;
                kind = kValu;
            }
        }
    }

    if (!best) return false;
    *outNode = best;
    *kindOut = kind;
    return true;
}

// Final-fallback candidate search across non-WMMA queues.
// kind: 0=global, 1=local, 2=other, 3=valu.
bool CDNA5ReadyQueue::findOldestFallbackNonWmma(DAGNode* pickedDS, DAGNode** outNode,
                                                int* kindOut) const {
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

    if (!globalReadQueue.empty()) consider(globalReadQueue.top(), kGlobalRead);
    consider(pickedDS, kLocalRead);
    if (!otherQueue.empty()) consider(otherQueue.top(), kOther);
    if (!valuQueue.empty()) consider(valuQueue.top(), kValu);

    if (best == nullptr) return false;
    *outNode = best;
    *kindOut = kind;
    return true;
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
//  Step 4 — threshold N = max(lastOverlap, wmmaWindowsNeeded) + latencyWmmaBudget;
//            latencyWmmaBudget = (latency / wmmaIssueConfig.latency) + 1.
//            wmmaWindowsNeeded is derived from matching ds_read count and DS per-WMMA cap.
//            use dsReadDrainLatency when matchingDsLoadCount > dsReadQueueDepth(), else
//            targetDSLoadLatency from the latest matching ds_read.
std::unordered_map<StinkyInstruction*, CDNA5ReadyQueue::BarrierAfterOutput>
CDNA5ReadyQueue::computeBarrierAfterThresholds(IRList::iterator regionStart,
                                               IRList::iterator regionEnd) {
    std::unordered_map<StinkyInstruction*, BarrierAfterOutput> result;
    struct BarrierTokenGroup {
        std::vector<StinkyInstruction*> barriers;
        std::unordered_set<uint32_t> tokens;
        IRList::iterator firstIt;
        IRList::iterator lastIt;
    };
    struct BarrierAfterSummary {
        StinkyInstruction* barrierKey;
        std::vector<StinkyInstruction*> barriers;
        int afterThreshold;
        int lastOverlap;
    };

    // Step 1a: collect all movable barriers with their PSEUDO src token sets.
    auto barriers = collectBarrierTokens(regionStart, regionEnd, /*useSrc=*/true);
    std::vector<BarrierTokenGroup> barrierGroups;
    for (const BarrierTokenEntry& be : barriers) {
        const bool canPairWithLast = !barrierGroups.empty() &&
                                     barrierGroups.back().tokens == be.tokens &&
                                     barrierGroups.back().barriers.size() < 2 &&
                                     std::next(barrierGroups.back().lastIt) == be.it;
        if (canPairWithLast) {
            barrierGroups.back().barriers.push_back(be.barrier);
            barrierGroups.back().lastIt = be.it;
        } else {
            barrierGroups.push_back({{be.barrier}, be.tokens, be.it, be.it});
        }
    }

    std::vector<BarrierAfterSummary> overlapChecks;
    for (const BarrierTokenGroup& group : barrierGroups) {
        // For a signal/wait pair, use the first barrier as the "before barrier" anchor.
        StinkyInstruction* groupBarrier = group.barriers.front();

        // Step 1b: scan [regionStart, groupBarrier) — find the latest ds_read whose
        //          dest PSEUDO token matches a src token of this barrier group.
        StinkyInstruction* targetDSLoad = nullptr;
        IRList::iterator targetDSLoadIt = regionEnd;
        uint32_t targetDSLoadLatency = 0;
        int matchingDsLoadCount = 0;
        for (IRList::iterator it = regionStart; it != regionEnd; ++it) {
            StinkyInstruction& inst = getStinkyInst(it);
            if (&inst == groupBarrier) break;
            if (!isDSRead(inst)) continue;
            for (const StinkyRegister& src : inst.getSrcRegs()) {
                if (isPseudoReg(src) && group.tokens.count(src.reg.idx)) {
                    targetDSLoad = &inst;
                    targetDSLoadIt = it;  // keep updating → ends up as latest
                    targetDSLoadLatency = inst.latencyCycles;
                    matchingDsLoadCount++;
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

        // Step 4: threshold N = lastOverlap + (latency / wmmaIssueConfig.latency) + 1.
        // When matching ds_load count exceeds queue depth, use dsReadDrainLatency; otherwise
        // use the latest matching ds_read latency.
        const int latencyForAfterThreshold = matchingDsLoadCount > dsReadQueueDepth()
                                                 ? dsReadDrainLatency()
                                                 : (int)targetDSLoadLatency;
        const int latencyWmmaBudget = (latencyForAfterThreshold / wmmaIssueConfig.latency) + 1;
        int maxDsPerWmmaWindow = dsReadPerWmma();
        int wmmaWindowsNeeded = (matchingDsLoadCount + maxDsPerWmmaWindow - 1) / maxDsPerWmmaWindow;
        if (matchingDsLoadCount > dsReadQueueDepth()) {
            int stallLatency = dsReadDrainLatency();
            float cyclePerDs = (float)stallLatency / (float)dsReadQueueDepth();
            float cyclesNeeded = cyclePerDs * (matchingDsLoadCount - dsReadQueueDepth());
            int wmmaNeeded = (int)std::ceil(cyclesNeeded / (float)wmmaIssueConfig.latency);
            wmmaWindowsNeeded += wmmaNeeded;
        }
        const int overlapOrWindowBase = std::max(lastOverlap, wmmaWindowsNeeded);
        int afterThreshold = overlapOrWindowBase + latencyWmmaBudget;
        for (StinkyInstruction* barrier : group.barriers) {
            barrierWmmaThresholds_[barrier] = afterThreshold;
            result[barrier] = {afterThreshold, 0};
        }
        overlapChecks.push_back({groupBarrier, group.barriers, afterThreshold, lastOverlap});
        PASS_DEBUG(std::cerr << "[CDNA5 computeBarrierAfterThresholds] barrier=" << groupBarrier
                             << " barrierGroupSize=" << group.barriers.size() << " afterThreshold="
                             << afterThreshold << " matchingDsLoadCount=" << matchingDsLoadCount
                             << " latencyWmmaBudget=" << latencyWmmaBudget << " wmmaWindowsNeeded="
                             << wmmaWindowsNeeded << " overlapOrWindowBase=" << overlapOrWindowBase
                             << " lastOverlap=" << lastOverlap << "\n");
    }

    // Group-level overlap check uses [0, lastOverlap) as each group's interval.
    // When overlap exists, add overlap budget to afterThreshold.
    std::vector<int> overlapPoints;
    overlapPoints.reserve(overlapChecks.size() * 2);
    for (const BarrierAfterSummary& summary : overlapChecks) {
        overlapPoints.push_back(0);
        overlapPoints.push_back(summary.lastOverlap);
    }
    std::sort(overlapPoints.begin(), overlapPoints.end());
    overlapPoints.erase(std::unique(overlapPoints.begin(), overlapPoints.end()),
                        overlapPoints.end());

    std::unordered_map<StinkyInstruction*, int> overlapWmmaWindowCountByGroup;
    std::unordered_map<StinkyInstruction*, int> overlapPeakConcurrencyByGroup;
    int globalOverlapPeakConcurrency = 0;

    for (size_t p = 0; p + 1 < overlapPoints.size(); ++p) {
        int segBegin = overlapPoints[p];
        int segEnd = overlapPoints[p + 1];
        if (segEnd <= segBegin) continue;

        std::vector<const BarrierAfterSummary*> activeGroups;
        for (const BarrierAfterSummary& summary : overlapChecks) {
            int begin = 0;
            int end = summary.lastOverlap;
            if (begin <= segBegin && segEnd <= end) activeGroups.push_back(&summary);
        }

        int concurrent = static_cast<int>(activeGroups.size());
        globalOverlapPeakConcurrency = std::max(globalOverlapPeakConcurrency, concurrent);
        if (concurrent < 2) continue;

        int segLen = segEnd - segBegin;
        for (const BarrierAfterSummary* active : activeGroups) {
            overlapWmmaWindowCountByGroup[active->barrierKey] += segLen;
            overlapPeakConcurrencyByGroup[active->barrierKey] =
                std::max(overlapPeakConcurrencyByGroup[active->barrierKey], concurrent);
        }
    }

    for (const BarrierAfterSummary& summary : overlapChecks) {
        int overlapCount = overlapWmmaWindowCountByGroup[summary.barrierKey];
        const int overlapBudget = std::max(0, overlapCount);
        int adjustedAfterThreshold =
            std::min((int)wmmaIssueConfig.issuedCount, summary.afterThreshold + overlapBudget);
        for (StinkyInstruction* barrier : summary.barriers) {
            barrierWmmaThresholds_[barrier] = adjustedAfterThreshold;
            result[barrier] = {adjustedAfterThreshold, overlapBudget};
        }
        PASS_DEBUG(
            std::cerr << "[CDNA5 computeBarrierAfterThresholds overlap] barrier="
                      << summary.barrierKey << " barrierGroupSize=" << summary.barriers.size()
                      << " baseAfterThreshold=" << summary.afterThreshold
                      << " adjustedAfterThreshold=" << adjustedAfterThreshold
                      << " adjustedOverlapWmmaWindow=" << overlapBudget
                      << " lastOverlap=" << summary.lastOverlap
                      << " overlapWmmaWindowCount=" << overlapCount << " overlapPeakConcurrency="
                      << overlapPeakConcurrencyByGroup[summary.barrierKey]
                      << " globalOverlapPeakConcurrency=" << globalOverlapPeakConcurrency << "\n");
    }
    return result;
}

// Compute "before" forced-barrier thresholds for this region.
//
//  Step 1  — for each barrier, collect all ds_reads after the barrier whose src
//            PSEUDO token matches a dest token produced by that barrier.
//  Step 2  — for each matching ds_read, starting from its post-barrier WMMA index,
//            find the first consumer WMMA whose src overlaps the ds_read dest VGPRs
//            (scan ds_read -> regionEnd, then wrap regionStart -> ds_read). Keep the
//            largest consumer index across ds_reads (MaximumWMMAIdx), and remember
//            the latency of the ds_read that defines that max.
//  Step 3  — residualCycles = max(0, MaximumWMMAIdx * wmmaIssueConfig.latency
//                                    - targetDSLoadLatency)
//  Step 4  — build candidate "before" caps from:
//            - beforeN: residualCycles / wmmaIssueConfig.latency
//            - maxFinalWmmaIdx: targetDSLoadLatency / wmmaIssueConfig.latency
//            - wmmaWindowsNeeded: WMMA windows needed to issue all matching ds_reads.
//            Final threshold = max(0, min(beforeN, issuedCount
//                                             - max(maxFinalWmmaIdx, wmmaWindowsNeeded))).
//  Step 5  — overlap adjustment: detect barriers whose WMMA-window intervals overlap,
//            then for each such barrier pull its forced start earlier by the overlap
//            budget and widen its window span (kept within the region), so overlapping
//            barriers leave enough WMMA-window space to issue their ds_loads.
std::unordered_map<StinkyInstruction*, CDNA5ReadyQueue::BarrierBeforeOutput>
CDNA5ReadyQueue::computeBarrierBeforeThresholds(IRList::iterator regionStart,
                                                IRList::iterator regionEnd) {
    std::unordered_map<StinkyInstruction*, BarrierBeforeOutput> result;
    struct BarrierTokenGroup {
        std::vector<StinkyInstruction*> barriers;
        std::unordered_set<uint32_t> tokens;
        IRList::iterator lastIt;
    };
    struct BarrierBeforeSummary {
        StinkyInstruction* barrierKey;
        std::vector<StinkyInstruction*> barriers;
        int beforeThreshold;
        int wmmaWindowsNeeded;
    };
    std::vector<BarrierBeforeSummary> overlapChecks;

    auto barriers = collectBarrierTokens(regionStart, regionEnd, /*useSrc=*/false);
    std::vector<BarrierTokenGroup> barrierGroups;
    for (const BarrierTokenEntry& be : barriers) {
        const bool canPairWithLast = !barrierGroups.empty() &&
                                     barrierGroups.back().tokens == be.tokens &&
                                     barrierGroups.back().barriers.size() < 2 &&
                                     std::next(barrierGroups.back().lastIt) == be.it;
        if (canPairWithLast) {
            barrierGroups.back().barriers.push_back(be.barrier);
            barrierGroups.back().lastIt = be.it;
        } else {
            barrierGroups.push_back({{be.barrier}, be.tokens, be.it});
        }
    }

    for (const BarrierTokenGroup& group : barrierGroups) {
        StinkyInstruction* groupBarrier = group.barriers.back();
        for (StinkyInstruction* barrier : group.barriers) barrierDsLoadCounts_[barrier] = 0;
        // Step 1: scan (barrier, regionEnd] — collect all ds_reads whose src
        //         PSEUDO token matches a dest token of this barrier.
        struct DSReadMatch {
            uint32_t latency;
            std::unordered_set<uint32_t> destVGPRs;
            IRList::iterator it;
            int dsWmmaIdx;
        };
        int dsWmmaIdx = 0;
        std::vector<DSReadMatch> matchingDSReads;
        bool isAfterBarrier = false;
        for (IRList::iterator it = regionStart; it != regionEnd; ++it) {
            StinkyInstruction& inst = getStinkyInst(it);
            if (&inst == groupBarrier) isAfterBarrier = true;
            if (isMatrixInstruction(inst)) dsWmmaIdx++;
            if (!isDSRead(inst) || !isAfterBarrier) continue;
            for (const StinkyRegister& src : inst.getSrcRegs()) {
                if (isPseudoReg(src) && group.tokens.count(src.reg.idx)) {
                    matchingDSReads.push_back({static_cast<uint32_t>(inst.latencyCycles),
                                               collectDestVGPRs(inst), it, dsWmmaIdx});
                    break;
                }
            }
            // NOTE: currently a no-op detection stub. It scans for another barrier that
            // carries a dest PSEUDO token overlapping this group's tokens, but the result
            // is not recorded or acted upon yet (kept as a placeholder for future
            // cross-barrier token handling).
            if (isBarrier(inst) && &inst != groupBarrier) {
                for (const StinkyRegister& dest : inst.getDestRegs()) {
                    if (isPseudoReg(dest) && group.tokens.count(dest.reg.idx)) {
                        // Found one matching token on this barrier; no need to keep scanning
                        // its remaining dest operands.
                        break;
                    }
                }
            }
        }
        if (matchingDSReads.empty()) continue;

        // Step 2: for each matching ds_read, find the first consumer WMMA (with wrap-around
        //         search order) whose src VGPRs overlap the ds_read dest VGPRs; take the
        //         maximum resulting WMMA index across all ds_reads (MaximumWMMAIdx).
        int maximumWMMAIdx = -1;
        int targetDSLoadLatency = 0;
        for (const DSReadMatch& dse : matchingDSReads) {
            int wmmaIdx = dse.dsWmmaIdx;
            bool found = false;
            // Scan in two segments: first from the ds_read position to regionEnd,
            // then wrap around from regionStart up to (but not including) the ds_read.
            auto scanWMMA = [&](IRList::iterator scanStart, IRList::iterator scanEnd) {
                for (IRList::iterator it = scanStart; it != scanEnd; ++it) {
                    StinkyInstruction& inst = getStinkyInst(it);
                    if (!isMatrixInstruction(inst)) continue;
                    wmmaIdx++;
                    if (srcVGPRsOverlap(inst, dse.destVGPRs)) {
                        if (wmmaIdx > maximumWMMAIdx) {
                            maximumWMMAIdx = wmmaIdx;
                            targetDSLoadLatency = (int)dse.latency;
                        }
                        found = true;
                        return;  // Keep the first consumer WMMA for this ds_read.
                    }
                }
            };
            scanWMMA(dse.it, regionEnd);
            if (!found) scanWMMA(regionStart, dse.it);
        }
        if (maximumWMMAIdx == -1) continue;

        // Step 3: residualCycles = max(0, MaximumWMMAIdx * wmmaIssueConfig.latency
        //                               - targetDSLoadLatency)
        int residualCycles =
            std::max(0, maximumWMMAIdx * (int)wmmaIssueConfig.latency - targetDSLoadLatency);

        // Step 4: base before cap (in WMMA count units) from residual cycles.
        int beforeN = (residualCycles / (int)wmmaIssueConfig.latency);
        int maxFinalWmmaIdx = targetDSLoadLatency / (int)wmmaIssueConfig.latency;
        // Step 4.1: Consider the number of ds_load to be issued in this range.
        const int dsLoadCount = static_cast<int>(matchingDSReads.size());
        for (StinkyInstruction* barrier : group.barriers)
            barrierDsLoadCounts_[barrier] = dsLoadCount;
        int maxDsPerWmmaWindow = dsReadPerWmma();
        int wmmaWindowsNeeded = (dsLoadCount + maxDsPerWmmaWindow - 1) / maxDsPerWmmaWindow;
        if (dsLoadCount > dsReadQueueDepth()) {
            int stallLatency = dsReadDrainLatency();
            float cyclePerDs = (float)stallLatency / (float)dsReadQueueDepth();
            float cyclesNeeded = cyclePerDs * (dsLoadCount - dsReadQueueDepth());
            int wmmaNeeded = (int)std::ceil(cyclesNeeded / (float)wmmaIssueConfig.latency);
            wmmaWindowsNeeded += wmmaNeeded;
        }
        // WMMA issue count that forces the barrier early enough for all dependent ds_reads.
        // Take the latest of three constraints, then subtract from total WMMAs in the region:
        //   beforeN — remaining latency after the last consumer WMMA
        //   maxFinalWmmaIdx — absolute cap after the 1st ds_load (DS load latency / WMMA latency)
        //   wmmaWindowsNeeded — DS issue bandwidth (enough WMMA windows for all ds_loads); when
        //       dsLoadCount > dsReadQueueDepth(), add extra drain windows from dsReadDrainLatency.
        int beforeThreshold =
            std::max(0, std::min(beforeN, wmmaIssueConfig.issuedCount -
                                              std::max(maxFinalWmmaIdx, wmmaWindowsNeeded)));
        for (StinkyInstruction* barrier : group.barriers)
            result[barrier] = {beforeThreshold, wmmaWindowsNeeded};
        overlapChecks.push_back({groupBarrier, group.barriers, beforeThreshold, wmmaWindowsNeeded});
        PASS_DEBUG(std::cerr << "[CDNA5 computeBarrierBeforeThresholds] barrier="
                             << " beforeThreshold=" << beforeThreshold
                             << " barrierGroupSize=" << group.barriers.size()
                             << " beforeN=" << beforeN << " maxFinalWmmaIdx=" << maxFinalWmmaIdx
                             << " wmmaWindowsNeeded=" << wmmaWindowsNeeded
                             << " numDsLoad=" << dsLoadCount << "\n");
    }

    // Overlap check by WMMA-window segments (not pair-only): handles 3+ barriers
    // overlapping at the same time and reports per-barrier aggregate overlap.
    std::vector<int> overlapPoints;
    overlapPoints.reserve(overlapChecks.size() * 2);
    for (const BarrierBeforeSummary& summary : overlapChecks) {
        overlapPoints.push_back(summary.beforeThreshold);
        overlapPoints.push_back(summary.beforeThreshold + summary.wmmaWindowsNeeded);
    }
    std::sort(overlapPoints.begin(), overlapPoints.end());
    overlapPoints.erase(std::unique(overlapPoints.begin(), overlapPoints.end()),
                        overlapPoints.end());

    std::unordered_map<StinkyInstruction*, int> overlapWmmaWindowCountByGroup;
    std::unordered_map<StinkyInstruction*, int> overlapPeakConcurrencyByGroup;
    int globalOverlapPeakConcurrency = 0;

    for (size_t p = 0; p + 1 < overlapPoints.size(); ++p) {
        int segBegin = overlapPoints[p];
        int segEnd = overlapPoints[p + 1];
        if (segEnd <= segBegin) continue;

        std::vector<const BarrierBeforeSummary*> activeBarriers;
        for (const BarrierBeforeSummary& summary : overlapChecks) {
            int begin = summary.beforeThreshold;
            int end = summary.beforeThreshold + summary.wmmaWindowsNeeded;
            if (begin <= segBegin && segEnd <= end) activeBarriers.push_back(&summary);
        }

        int concurrent = static_cast<int>(activeBarriers.size());
        globalOverlapPeakConcurrency = std::max(globalOverlapPeakConcurrency, concurrent);
        if (concurrent < 2) continue;

        int segLen = segEnd - segBegin;
        for (const BarrierBeforeSummary* active : activeBarriers) {
            overlapWmmaWindowCountByGroup[active->barrierKey] += segLen;
            overlapPeakConcurrencyByGroup[active->barrierKey] =
                std::max(overlapPeakConcurrencyByGroup[active->barrierKey], concurrent);
        }
    }

    for (const BarrierBeforeSummary& summary : overlapChecks) {
        int overlapCount = overlapWmmaWindowCountByGroup[summary.barrierKey];
        // Final one-shot expansion: pull the barrier's forced start earlier by the
        // overlap budget and widen its WMMA-window span, so overlapping barriers leave
        // enough WMMA-window space for ds_load issue (interval kept within the region).
        const int overlapBudget = std::max(0, overlapCount);
        int adjustedBeforeThreshold = std::max(0, summary.beforeThreshold - overlapBudget);
        int adjustedWmmaWindowsNeeded =
            std::max(0, std::min((int)wmmaIssueConfig.issuedCount - adjustedBeforeThreshold,
                                 summary.wmmaWindowsNeeded + overlapBudget));
        for (StinkyInstruction* barrier : summary.barriers)
            result[barrier] = {adjustedBeforeThreshold, adjustedWmmaWindowsNeeded};
        PASS_DEBUG(
            std::cerr << "[CDNA5 computeBarrierBeforeThresholds overlap] barrier="
                      << summary.barrierKey << " barrierGroupSize=" << summary.barriers.size()
                      << " baseBeforeThreshold=" << summary.beforeThreshold
                      << " adjustedBeforeThreshold=" << adjustedBeforeThreshold
                      << " dsLoadCount=" << barrierDsLoadCounts_[summary.barrierKey]
                      << " baseWmmaWindowsNeeded=" << summary.wmmaWindowsNeeded
                      << " adjustedWmmaWindowsNeeded=" << adjustedWmmaWindowsNeeded
                      << " overlapWmmaWindowCount=" << overlapCount << " overlapPeakConcurrency="
                      << overlapPeakConcurrencyByGroup[summary.barrierKey]
                      << " globalOverlapPeakConcurrency=" << globalOverlapPeakConcurrency << "\n");
    }

    return result;
}

// Main scheduling orchestration:
//   Phase A: forced barrier — when wmmaIssuedCountThisRegion_ reaches a per-barrier threshold.
//   Phase B: WMMA if DS latency gate (rule 2) passed, DS window cap (rule 4) respected,
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

    // Pre-compute the best DS read by dsReadPriority once for all phases.
    DAGNode* pickedDS = nullptr;
    for (DAGNode* n : localReadQueue) {
        if (!pickedDS || n->dsReadPriority < pickedDS->dsReadPriority) pickedDS = n;
    }

    // Phase B — try WMMA if all gates pass.
    bool otherQueuesHaveWork = !globalReadQueue.empty() || !localReadQueue.empty() ||
                               !otherQueue.empty() || !valuQueue.empty();

    if (!wmmaQueue.empty()) {
        auto [bestWMMA, bestLatency] = findMostReadyWMMA();

        DAGNode* smallestPickable = nullptr;
        int pickKind = -1;
        // Returns false when no non-WMMA can be issued right now (e.g. the only
        // pending ds_load is held back by the co-exec hazard gate). In that case
        // there is nothing to interleave, so the next WMMA should be allowed to go.
        const bool hasPickableNonWmma =
            findSmallestPickableNonWmma(pickedDS, &smallestPickable, &pickKind);

        const bool blockWmmaForLoopHeadBalance =
            deferHeadBalanceThisRegion_ && deferFirstHeadWmmaActive_ && otherQueuesHaveWork;
        const bool blockWmmaForActiveWindow =
            (coIssueCyclePos_ < activeWmmaLatency_) && (smallestPickable != nullptr);

        bool blockWmmaForAtLeastOneNonWmmaInterleaving = false;
        if (lastPickedNode_ != nullptr) {
            blockWmmaForAtLeastOneNonWmmaInterleaving =
                hasPickableNonWmma && isMatrixInstruction(*lastPickedNode_->inst);
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
    if (coIssueCyclePos_ < activeWmmaLatency_) {
        DAGNode* smallestPickable = nullptr;
        int pickKind = -1;
        findSmallestPickableNonWmma(pickedDS, &smallestPickable, &pickKind);
        if (smallestPickable != nullptr) {
            PASS_DEBUG(std::cerr << "[CDNA5 pickOne] Phase C picked non-WMMA dagId="
                                 << smallestPickable->id << " kind=" << pickKind << "\n");
            return rememberPick(popNonWmma(smallestPickable, pickKind));
        }

        advanceTime(activeWmmaLatency_ - coIssueCyclePos_);
    }

    // Phase D — outside WMMA latency: pick smallest-id from any non-WMMA queue.
    {
        DAGNode* smallestPickable = nullptr;
        int pickKind = -1;
        findSmallestPickableNonWmma(pickedDS, &smallestPickable, &pickKind);
        if (smallestPickable != nullptr) {
            return rememberPick(popNonWmma(smallestPickable, pickKind));
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

    // Phase G — final safety net: force-pick the oldest DAG node to guarantee progress.
    DAGNode* fallback = nullptr;
    int fallbackKind = -1;
    if (findOldestFallbackNonWmma(pickedDS, &fallback, &fallbackKind)) {
        // Safety net must guarantee progress, so the global-read gate is not
        // applied here. If a tensor load is the only remaining work but the
        // credit pool is full, idle until the earliest credit drains so the
        // pick below stays within the queue depth.
        if (fallbackKind == kGlobalRead && globalReadQueueFull()) {
            int minDrain = globalReadInflight_.minResidual();
            if (minDrain > 0) advanceTime(minDrain);
        }
        PASS_DEBUG(std::cerr << "[CDNA5 pickOne] Phase G fallback pick dagId=" << fallback->id
                             << " kind=" << fallbackKind << "\n");
        return rememberPick(popNonWmma(fallback, fallbackKind));
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
// Resets co-issue timeline. Sets WMMA issue config from first WMMA in block.
void CDNA5ReadyQueue::onInit(IRList::iterator regionStart, IRList::iterator regionEnd) {
    deferFirstHeadWmmaActive_ = false;
    deferHeadBalanceThisRegion_ = false;

    activeCoIssueWindow_ = 0;
    coIssueCyclePos_ = 0;
    activeWmmaLatency_ = 0;
    activeWmmaNode_ = nullptr;
    globalReadInflight_ = InFlightQueue(globalReadQueueDepth());
    dsReadInflight_ = InFlightQueue(dsReadQueueDepth());

    currentBB_ = (regionStart != regionEnd) ? regionStart->getParent() : nullptr;

    if (getPassContext().getPassFeatureConfig().loopConfig.unrollGemm == false) return;

    const Loop* loop = getLoop();
    if (loop && loop->headerBB && loop->latchBB) {
        if (latchBBTailIsWmma(*loop->latchBB)) deferFirstHeadWmmaActive_ = true;
    }

    wmmaIssueConfig.latency = 0;
    wmmaIssueConfig.issueCycles = 1;
    for (IRList::iterator it = regionStart; it != regionEnd; ++it) {
        auto* instPtr = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!instPtr) continue;
        if (isMatrixInstruction(*instPtr)) {
            wmmaIssueConfig.latency = instPtr->latencyCycles;
            wmmaIssueConfig.issueCycles = instPtr->issueCycles;
            break;
        }
    }

    restoreCrossBBStateFromLoop();

    // Seed the in-flight credit pool from loop-carried state: crossBBGlobalReadCount_
    // credits, each stamped with the worst-case remaining drain (reconstructs the
    // most-constrained predecessor so no incoming path over-issues).
    if (globalReadQueueDepth() > 0 && crossBBGlobalReadCount_ > 0)
        globalReadInflight_.seed(crossBBGlobalReadCount_, crossBBGlobalReadResidual_);
}

void CDNA5ReadyQueue::restoreCrossBBStateFromLoop() {
    crossBBDsResiduals_.clear();
    crossBBGlobalReadCount_ = 0;
    crossBBGlobalReadResidual_ = 0;
    const Loop* loop = getLoop();
    if (!currentBB_ || !loop || !loop->contains(currentBB_) || !getAnalysisCache()) return;

    // Global-read credits: loop predecessors take priority over non-loop ones
    // (the loop body runs many iterations, so its carried state governs steady
    // state). Take the max within the chosen group on BOTH axes — occupancy and
    // residual drain — so no predecessor path is left over-issuing.
    int loopCount = 0, loopRes = 0, nonLoopCount = 0, nonLoopRes = 0;
    bool sawLoopPred = false;

    for (BasicBlock* pred : currentBB_->getPredecessors()) {
        const BBScheduleState* state = getAnalysisCache()->lookup(pred);
        if (!state) continue;
        for (const auto& [regIdx, rem] : state->dsResiduals) {
            if (rem > 0) crossBBDsResiduals_[regIdx] = std::max(crossBBDsResiduals_[regIdx], rem);
        }
        if (loop->contains(pred)) {
            sawLoopPred = true;
            loopCount = std::max(loopCount, state->globalReadInflightCount);
            loopRes = std::max(loopRes, state->globalReadResidual);
        } else {
            nonLoopCount = std::max(nonLoopCount, state->globalReadInflightCount);
            nonLoopRes = std::max(nonLoopRes, state->globalReadResidual);
        }
    }
    crossBBGlobalReadCount_ = sawLoopPred ? loopCount : nonLoopCount;
    crossBBGlobalReadResidual_ = sawLoopPred ? loopRes : nonLoopRes;
}

void CDNA5ReadyQueue::onFinishBB() {
    if (!currentBB_ || !getAnalysisCache()) return;
    getAnalysisCache()->store(currentBB_, {0, regDataReadyCounters, globalReadInflight_.size(),
                                           globalReadInflight_.maxResidual()});
}

// Per scheduling region. Rule (4): per-WMMA-window DS cap (computed in pickOneFromWMMA).
// Rule (2): seedWmmaDsLatencyFromPrefix. Rule (5): head balance.
// Barrier thresholds: computeBarrierAfterThresholds / computeBarrierBeforeThresholds.
void CDNA5ReadyQueue::onInitRegion(IRList::iterator regionStart, IRList::iterator regionEnd,
                                   IRList::iterator blockBegin) {
    wmmaIssuedCountThisRegion_ = 0;
    dsInsertedSinceLastWmma_ = 0;
    lastPickedNode_ = nullptr;
    // (B) elapse ordering state is per-region: reset the touch map and clock so a new
    // region starts with all regs "very old" (no spurious deferrals from a prior region).
    regLastTouch_.clear();
    clock_ = 0;
    if (getPassContext().getPassFeatureConfig().loopConfig.unrollGemm == false) return;

    const Loop* loop = getLoop();
    deferHeadBalanceThisRegion_ = deferFirstHeadWmmaActive_ && loop &&
                                  loop->headerBB == blockBegin->getParent() &&
                                  regionStart != blockBegin;

    seedWmmaDsLatencyFromPrefix(blockBegin, regionStart, regDataReadyCounters, crossBBDsResiduals_);

    wmmaIssueConfig.issuedCount = 0;
    hasWMMAInRegion_ = false;
    for (IRList::iterator it = regionStart; it != regionEnd; ++it) {
        auto* instPtr = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!instPtr) continue;
        StinkyInstruction& inst = *instPtr;

        if (isMatrixInstruction(inst)) {
            wmmaIssueConfig.issuedCount++;
            hasWMMAInRegion_ = true;
        }
    }

    barrierWmmaThresholds_.clear();
    barrierDsLoadCounts_.clear();
    if (hasWMMAInRegion_) {
        auto afterThresholds = computeBarrierAfterThresholds(regionStart, regionEnd);
        for (auto& [barrier, afterOutput] : afterThresholds) {
            barrierWmmaThresholds_[barrier] = afterOutput.afterThreshold;
        }
        auto beforeThresholds = computeBarrierBeforeThresholds(regionStart, regionEnd);
        for (auto& [barrier, beforeOutput] : beforeThresholds) {
            auto it = barrierWmmaThresholds_.find(barrier);
            if (it != barrierWmmaThresholds_.end())
                it->second = (it->second + beforeOutput.beforeThreshold) / 2;
            else
                barrierWmmaThresholds_[barrier] = beforeOutput.beforeThreshold;
        }
        // Cross-check exclusive barrier pairs:
        // one barrier appears only in afterThresholds and another appears only in beforeThresholds.
        // Compare ranges and report overlap.
        for (const auto& [afterBarrier, afterOutput] : afterThresholds) {
            if (beforeThresholds.find(afterBarrier) != beforeThresholds.end()) continue;
            const int afterWindow = std::max(0, afterOutput.overlapWmmaWindow);
            const int afterEnd = afterOutput.afterThreshold;
            const int afterBegin = std::max(0, afterEnd - afterWindow);
            for (const auto& [beforeBarrier, beforeOutput] : beforeThresholds) {
                if (afterBarrier == beforeBarrier) continue;
                if (afterThresholds.find(beforeBarrier) != afterThresholds.end()) continue;
                const int beforeBegin = beforeOutput.beforeThreshold;
                const int beforeWindow = std::max(0, beforeOutput.wmmaWindowsNeeded);
                const int beforeEnd = beforeBegin + beforeWindow;
                const bool overlap = (afterBegin < beforeEnd) && (beforeBegin < afterEnd);
                if (overlap) {
                    auto afterIt = barrierWmmaThresholds_.find(afterBarrier);
                    auto beforeIt = barrierWmmaThresholds_.find(beforeBarrier);
                    if (afterIt != barrierWmmaThresholds_.end() &&
                        beforeIt != barrierWmmaThresholds_.end()) {
                        const int mergedThreshold = (afterIt->second + beforeIt->second) / 2;
                        afterIt->second = mergedThreshold;
                        beforeIt->second = mergedThreshold;
                    }
                }
                PASS_DEBUG(std::cerr
                           << "[CDNA5 onInitRegion after-before exclusive overlap] afterBarrier="
                           << afterBarrier << " beforeBarrier=" << beforeBarrier
                           << " afterThreshold=" << afterEnd << " afterWmmaWindow=" << afterWindow
                           << " beforeThreshold=" << beforeBegin << " beforeWmmaWindow="
                           << beforeWindow << " overlap=" << overlap << "\n");
            }
        }

        // Final pair normalization: keep barrier_signal/barrier_wait pairs on the same threshold.
        auto normalizeBarrierPairs = [&](bool useSrcTokens) {
            auto barrierEntries = collectBarrierTokens(regionStart, regionEnd, useSrcTokens);
            struct BarrierTokenGroup {
                std::vector<StinkyInstruction*> barriers;
                std::unordered_set<uint32_t> tokens;
                IRList::iterator lastIt;
            };
            std::vector<BarrierTokenGroup> barrierGroups;
            for (const BarrierTokenEntry& be : barrierEntries) {
                const bool canPairWithLast = !barrierGroups.empty() &&
                                             barrierGroups.back().tokens == be.tokens &&
                                             barrierGroups.back().barriers.size() < 2 &&
                                             std::next(barrierGroups.back().lastIt) == be.it;
                if (canPairWithLast) {
                    barrierGroups.back().barriers.push_back(be.barrier);
                    barrierGroups.back().lastIt = be.it;
                } else {
                    barrierGroups.push_back({{be.barrier}, be.tokens, be.it});
                }
            }

            for (const auto& group : barrierGroups) {
                if (group.barriers.size() < 2) continue;
                int sum = 0;
                int count = 0;
                for (StinkyInstruction* barrier : group.barriers) {
                    auto it = barrierWmmaThresholds_.find(barrier);
                    if (it == barrierWmmaThresholds_.end()) continue;
                    sum += it->second;
                    count++;
                }
                if (count < 2) continue;
                const int mergedThreshold = sum / count;
                for (StinkyInstruction* barrier : group.barriers) {
                    auto it = barrierWmmaThresholds_.find(barrier);
                    if (it != barrierWmmaThresholds_.end()) it->second = mergedThreshold;
                }
                PASS_DEBUG(std::cerr << "[CDNA5 onInitRegion pair normalize] groupSize="
                                     << group.barriers.size()
                                     << " mergedThreshold=" << mergedThreshold
                                     << " useSrcTokens=" << useSrcTokens << "\n");
            }
        };
        normalizeBarrierPairs(/*useSrcTokens=*/true);
        normalizeBarrierPairs(/*useSrcTokens=*/false);
    }
}
}  // namespace
