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
// then drains ready nodes via this queue. CDNA5 adds WMMA-centric heuristics on top of that
// DAG: tensor/global prioritization, DS latency modeling, program-order ties, loop-aware head
// balancing (stops WMMA-at-loop-tail plus WMMA-right-after-head in the next iteration), and
// cross-region WMMA spacing.
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
        StinkyInstruction& inst = getStinkyInst(it);
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
        StinkyInstruction& inst = getStinkyInst(it);
        if (isLabel(inst) || isBranch(inst)) continue;
        return isMatrixInstruction(inst);
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
//   Step 3 — pickOne Phase A: block WMMA while non-WMMA queues have work.
//   Step 4 — pickOneFromWMMA: clear deferral after first head WMMA is issued.

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
// Scheduling menu (every pick still respects the DAG: in-degree 0 only):
//
//  (1) Program order vs WMMA — prefer WMMA in Phase B, but not before pickable
//      global/tensor/local/other with a smaller DAG id (preload / X0 vs X1 double-buffer).
//  (2) DS / VGPR latency — block WMMA until modeled ds_load latency for WMMA src VGPRs
//      has decayed; seed from the BB prefix before each region.
//  (3) WMMA–WMMA spacing — after each WMMA pick, sibling WMMAs wait (per-node counters)
//      until mandated gap cycles have been “spent” by issued instructions.
//  (4) Density + cross-region gap — avgIssueInterval from region WMMA/DS/issue totals;
//      wmmaPipelineGapCycles survives region splits so WMMAs do not pack only because
//      micro-regions reset per-node spacing.
//  (5) Loop tail vs head — defer first WMMA in the loop header BB until non-WMMA queues
//      drain once. Uses cross-BB Loop detection (LoopDetection.hpp) via setLoopContext().
//      Skipped when loopConfig.unrollGemm is false.
//
// Where rules are implemented:
//
//   pickOne                 — main orchestration: (1)(2)(3)(4)(5); (5) Step 3
//   blockWmmaForLoopHeadBalance pickOneFromWMMA         — (3)(4) inter-Wmma delay +
//   wmmaPipelineGapCycles; (5) Step 4 clear deferral updateWMMAStatus        — (2)(3)(4) time step
//   for counters after any picked insn findMostReadyWMMA       — (2) gate before WMMA
//   findSmallestPickableNonWmma — (1) min-id non-WMMA vs WMMA program order
//   push                    — (1) bucket routing into ready queues
//   onInit                  — (4) gap reset + WMMA latency snap; (5) Step 1 latchBBTailIsWmma
//   onInitRegion            — (2) prefix DS seed; (4) WMMAIssueConfig budget; (5) Step 2 defer flag
//   seedWmmaDsLatencyFromPrefix — (2) prefix-only DS latency → wmmaRegisterLatencyCounters
//   latchBBTailIsWmma       — (5) check if latch BB ends with WMMA before back-edge branch
//
// Temporarily named CDNA5 for MI450-class hardware; rename when marketing settles.
//
class CDNA5ReadyQueue : public ReadyQueue {
    // --- Priority buckets (DAG ids compare smaller = earlier in source within region) ---
    ReadySetByDAGid wmmaQueue;
    ReadySetByDAGid globalReadQueue;  // tensor_load_to_lds when distributeGlobalRead
    ReadySetByDAGid localReadQueue;   // reserved; same priority scheme as CDNA3
    ReadySetByDAGid barrierQueue;
    ReadySetByDAGid otherQueue;  // scalars, ds_load, waits in region, etc.

    // Throttle tensor issues vs other work (mirrors CDNA3 globalReadPerMFMA idea).
    int globalReadCounter = 0;
    int globalReadPerWMMA = 1;

    // --- WMMA interlock state (decayed by updateWMMAStatus on every picked instruction) ---

    // Per other WMMA DAG node: cycles until that WMMA may be considered for priority 1.
    std::map<DAGNode*, int> wmmaNodeCounters;

    // Per VGPR index: remaining modeled latency until ds_load result is safe for WMMA src.
    std::map<int, int> wmmaRegisterLatencyCounters;

    // Region snapshot for pickOneFromWMMA delay math (totalIssuedCycles is also decremented
    // each pick so "rest" shrinks as the region is scheduled).
    WMMAIssueConfig wmmaIssueConfig;

    // Global WMMA gap: unlike wmmaNodeCounters, survives onInitRegion so waitcnt boundaries
    // do not erase spacing between consecutive WMMAs. Forced pick at end of pickOne bypasses
    // the priority-1 check that requires this to be 0.
    int wmmaPipelineGapCycles = 0;

    // After issuing WMMA, skip priority-1 WMMA once if global/local/other still have nodes,
    // so we do not string WMMAs when scalar/tensor work is still ready.
    bool lastPickedWasWMMA = false;

    // This region contains WMMA.
    bool hasWMMAInRegion_ = false;

    // --- Loop head balancing (computed once per BB in onInit via Loop from setLoopContext) ---
    bool deferFirstHeadWmmaActive_ = false;    // true if latch BB tail is WMMA
    bool deferHeadBalanceThisRegion_ = false;  // true if this BB is the loop header

    // Per-barrier forced-issue threshold: maps StinkyInstruction* → N.
    // When wmmaIssuedCountThisRegion_ >= N for a barrier in barrierQueue, that barrier
    // is immediately issued at the top of pickOne() before any other phase.
    // Counter resets to 0 each time a forced barrier fires.
    std::unordered_map<StinkyInstruction*, int> barrierWmmaThresholds_;

    int wmmaIssuedCountThisRegion_ = 0;

    BasicBlock* currentBB_ = nullptr;

    // Cross-BB DS residuals merged from predecessors in onInit. Kept separate
    // from wmmaRegisterLatencyCounters to avoid double-decay when
    // seedWmmaDsLatencyFromPrefix re-simulates the within-BB prefix.
    std::map<int, int> crossBBDsResiduals_;

    void updateWMMAStatus(DAGNode* node);
    int getMaxDsLatency(DAGNode* node);
    std::pair<DAGNode*, int> findMostReadyWMMA();
    DAGNode* pickOneFromWMMA(DAGNode* pick = nullptr);
    bool findSmallestPickableNonWmma(DAGNode** outNode, int* kindOut) const;
    DAGNode* extractForcedBarrier();
    void computeBarrierAfterThresholds(IRList::iterator regionStart, IRList::iterator regionEnd);
    std::unordered_map<StinkyInstruction*, int> computeBarrierBeforeThresholds(
        IRList::iterator regionStart, IRList::iterator regionEnd);
    /// True if any non-barrier bucket still has work this pick could take (min-id policy +
    /// tensor throttle). Barriers must not run while this holds — avoids corner cases where
    /// WMMA heuristics defer compute but the barrier queue would still drain.
    bool nonBarrierNodesStillReady() const;

    // Restore cross-BB state from loop body predecessors only.
    // TODO: extend to all BBs once scheduling heuristics are verified safe.
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

// Scheduling rules (2)(3)(4): after any picked instruction, advance modeled time — decay
// wmmaPipelineGapCycles, wmmaNodeCounters, wmmaRegisterLatencyCounters; subtract cycles from
// wmmaIssueConfig.totalIssuedCycles. Callers: pickOne (all paths), pickOneFromWMMA.
void CDNA5ReadyQueue::updateWMMAStatus(DAGNode* node) {
    const int cycles = node->inst->issueCycles;

    // Every scheduled insn advances "time" for WMMA gap and WMMA/DS countdowns alike.
    wmmaPipelineGapCycles = std::max(0, wmmaPipelineGapCycles - cycles);

    wmmaIssueConfig.totalIssuedCycles -= cycles;

    // Decrement per-WMMA issue interval counters.
    for (auto& [n, counter] : wmmaNodeCounters) {
        if (counter > 0) counter -= cycles;
    }

    // Decrement ds_read latency per VGPR; remove when latency reaches 0 (VGPR ready).
    for (auto it = wmmaRegisterLatencyCounters.begin(); it != wmmaRegisterLatencyCounters.end();) {
        if (it->second > 0) {
            it->second -= cycles;
            if (it->second <= 0)
                it = wmmaRegisterLatencyCounters.erase(it);
            else
                ++it;
        } else
            ++it;
    }
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

// Scheduling rules (3)(4): compute delay from wmmaIssueConfig (rest, avgIssueInterval,
// latencyCycles); raise wmmaNodeCounters on sibling WMMAs; set wmmaPipelineGapCycles.
// Rule (5): if deferHeadBalanceThisRegion_, first pick clears deferFirstHeadWmmaActive_.
// Then updateWMMAStatus; resets globalReadCounter. Callers: pickOne Phase B / Phase D.
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

    // Push out sibling WMMAs: delay is derived from avgIssueInterval and remaining non-WMMA
    // slack (rest) in the region snapshot, similar in spirit to CDNA3 MFMA spacing.
    int delay;
    auto rest = (int)((wmmaIssueConfig.totalIssuedCycles - wmmaIssueConfig.totalWmmaIssuedCycles) /
                      wmmaIssueConfig.issuedCount);
    if (wmmaIssueConfig.issuedCount <= 0 ||
        (wmmaIssueConfig.issuedCount > 0 && rest < node->inst->latencyCycles)) {
        if (node->inst->latencyCycles >= wmmaIssueConfig.avgIssueInterval)
            delay = std::max(rest, 1);
        else
            delay = node->inst->latencyCycles;
    } else
        delay = wmmaIssueConfig.avgIssueInterval;

    for (auto& [n, counter] : wmmaNodeCounters) {
        if (n != node) counter = std::max(counter, delay);
    }
    wmmaNodeCounters.erase(node);

    wmmaPipelineGapCycles = std::max(wmmaPipelineGapCycles, delay);

    wmmaIssueConfig.issuedCount--;
    wmmaIssueConfig.totalWmmaIssuedCycles -= node->inst->issueCycles;
    // One balanced head WMMA is enough to drop loop-head deferral for the rest of the BB.
    if (deferHeadBalanceThisRegion_) deferFirstHeadWmmaActive_ = false;
    wmmaIssuedCountThisRegion_++;
    updateWMMAStatus(node);
    globalReadCounter = 0;
    return node;
}

// Scheduling rule (1): pick minimum DAG id among ready non-WMMA nodes. Queues: globalReadQueue
// (throttled by globalReadCounter vs globalReadPerWMMA), localReadQueue, otherQueue.
bool CDNA5ReadyQueue::findSmallestPickableNonWmma(DAGNode** outNode, int* kindOut) const {
    // Among currently pickable global (if throttle allows), local, and other nodes, return
    // the one with minimum DAG id — stable tie-break that matches source order when the DAG
    // leaves multiple nodes ready.
    *outNode = nullptr;
    *kindOut = -1;
    DAGNode* best = nullptr;
    int kind = -1;

    if (!globalReadQueue.empty() && (globalReadCounter < globalReadPerWMMA || otherQueue.empty())) {
        best = globalReadQueue.top();
        kind = 0;
    }
    if (!localReadQueue.empty()) {
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

    if (!best) return false;
    *outNode = best;
    *kindOut = kind;
    return true;
}

bool CDNA5ReadyQueue::nonBarrierNodesStillReady() const {
    DAGNode* n = nullptr;
    int kind = -1;
    if (findSmallestPickableNonWmma(&n, &kind)) return true;
    return !wmmaQueue.empty();
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
        int beforeThreshold = wmmaIssueConfig.issuedCount - (beforeN + maxFinalWmmaIdx);
        result[be.barrier] = beforeThreshold;
    }

    return result;
}

// Main scheduling orchestration — rules (1)–(5):
//   Phase A: forced barrier — when wmmaIssuedCountThisRegion_ reaches a per-barrier threshold
//            (barrierWmmaThresholds_), immediately issue the barrier before any other phase.
//   Phase B: WMMA if wmmaNodeCounters, wmmaPipelineGapCycles, findMostReadyWMMA, programOrderOk
//            (vs findSmallestPickableNonWmma), lastPickedWasWMMA / otherQueuesHaveWork, and
//            !blockWmmaForLoopHeadBalance (rule 5).
//   Phase C: pop non-WMMA; on ds_load, extend wmmaRegisterLatencyCounters (rule 2).
//   Phase D: forced WMMA (pickOneFromWMMA) when Phase B skipped it but wmmaQueue still ready.
//   Phase E: barriers only after no global/local/other non-WMMA and no schedulable WMMA remains.
DAGNode* CDNA5ReadyQueue::pickOne() {
    // Phase A — forced barrier when per-barrier WMMA threshold is met.
    // Phase B — try WMMA first if all WMMA-specific gates pass.
    // Phase C — smallest-id non-WMMA among global/local/other.
    // Phase D — WMMA via pickOneFromWMMA if queue non-empty (Phase B may have deferred it).
    // Phase E — barriers when nothing else above can run this step.

    // Phase A — forced barrier: issue the lowest-id barrier whose WMMA threshold is met.
    if (DAGNode* forced = extractForcedBarrier()) {
        PASS_DEBUG(std::cerr << "[CDNA5 pickOne] forced barrier: wmmaIssued="
                             << wmmaIssuedCountThisRegion_
                             << " threshold=" << barrierWmmaThresholds_.at(forced->inst)
                             << " barrierId=" << forced->id << std::flush << "\n");
        updateWMMAStatus(forced);
        lastPickedWasWMMA = false;
        return forced;
    }

    // Phase B — try WMMA first if all WMMA-specific gates pass.
    bool otherQueuesHaveWork =
        !globalReadQueue.empty() || !localReadQueue.empty() || !otherQueue.empty();

    DAGNode* smallestPickable = nullptr;
    int pickKind = -1;
    findSmallestPickableNonWmma(&smallestPickable, &pickKind);

    if (!wmmaQueue.empty()) {
        // Pick the WMMA with the smallest outstanding DS latency (most ready).
        auto [bestWMMA, bestLatency] = findMostReadyWMMA();
        int c = wmmaNodeCounters.count(bestWMMA) ? wmmaNodeCounters.at(bestWMMA) : 0;
        bool programOrderOk = hasWMMAInRegion_ ||
                              (smallestPickable == nullptr || bestWMMA->id < smallestPickable->id);
        // Rule (5): avoid WMMA at loop tail then WMMA as first work after the head label.
        const bool blockWmmaForLoopHeadBalance =
            deferHeadBalanceThisRegion_ && deferFirstHeadWmmaActive_ && otherQueuesHaveWork;
        if (c <= 0 && wmmaPipelineGapCycles <= 0 && programOrderOk &&
            (!lastPickedWasWMMA || !otherQueuesHaveWork) && !blockWmmaForLoopHeadBalance) {
            DAGNode* node = pickOneFromWMMA(bestWMMA);
            lastPickedWasWMMA = true;
            return node;
        }
    }

    // Phase C — smallest-id non-WMMA among global/local/other.
    if (smallestPickable != nullptr) {
        DAGNode* node = nullptr;
        if (pickKind == 0) {
            node = globalReadQueue.top();
            globalReadQueue.pop();
            globalReadCounter++;
        } else if (pickKind == 1) {
            // Pick ds_read with smallest pre-computed dsReadPriority.
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
        } else {
            assert(pickKind == 2);
            node = otherQueue.top();
            otherQueue.pop();
        }
        updateWMMAStatus(node);
        lastPickedWasWMMA = false;
        // Rule (5): one non-WMMA issued after loop head is sufficient — lift the deferral so
        // WMMA can compete again on the very next pick instead of waiting for all non-WMMAs.
        if (deferHeadBalanceThisRegion_) deferFirstHeadWmmaActive_ = false;
        return node;
    }

    // Phase D — forced WMMA: pick the most-ready WMMA before barriers.
    if (!wmmaQueue.empty()) {
        auto [bestWMMA, bestLatency] = findMostReadyWMMA();
        (void)bestLatency;
        DAGNode* node = pickOneFromWMMA(bestWMMA);
        lastPickedWasWMMA = true;
        return node;
    }

    assert(!nonBarrierNodesStillReady() &&
           "CDNA5 pickOne: barrier only after pickable global/local/other and WMMA are gone");

    // Phase E — movable barriers only when nothing else in the ready set can be issued this step.
    if (!barrierQueue.empty()) {
        DAGNode* barrier = barrierQueue.top();
        barrierQueue.pop();
        updateWMMAStatus(barrier);
        lastPickedWasWMMA = false;
        PASS_DEBUG(std::cerr << "[DAG CDNA5 pickOne] Phase E barrierQueue dagId=" << barrier->id
                             << " (non-barrier buckets empty for this pick)\n";
                   barrier->inst->dump(std::cerr); std::cerr << "\n");
        return barrier;
    }

    assert(false && "CDNA5ReadyQueue::pickOne: all buckets empty");
    return nullptr;
}

// Scheduling rule (1): route ready DAG nodes into wmmaQueue, globalReadQueue (tensor_load when
// distributeGlobalRead), barrierQueue, or otherQueue. pickOne drains buckets; order here is not
// schedule order.
void CDNA5ReadyQueue::push(DAGNode* node) {
    // Route ready nodes into buckets. Order here does not imply schedule order; pickOne
    // implements the actual priority and min-id policy.
    if (isMatrixInstruction(*node->inst)) {
        wmmaNodeCounters[node] = 0;  // pickOneFromWMMA may raise before this WMMA becomes top
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

    if (isBarrier(*node->inst)) {
        barrierQueue.push(node);
        return;
    }

    otherQueue.push(node);
}

// ReadyQueue API: true when all scheduling buckets are empty (no rule logic).
bool CDNA5ReadyQueue::empty() const {
    return wmmaQueue.empty() && globalReadQueue.empty() && otherQueue.empty() &&
           barrierQueue.empty() && localReadQueue.empty();
}

// Once per BB. Rule (4): reset/restore wmmaPipelineGapCycles; compute loop-wide
// avgIssueInterval. Rule (5): cross-BB loop tail WMMA detection via setLoopContext().
// Sets wmmaIssueConfig.latency from first WMMA/SWMMA in block.
//
// Per-BB init. Restores cross-BB state from predecessors via bbEndStates_.
void CDNA5ReadyQueue::onInit(IRList::iterator regionStart, IRList::iterator regionEnd) {
    deferFirstHeadWmmaActive_ = false;
    deferHeadBalanceThisRegion_ = false;

    currentBB_ = (regionStart != regionEnd) ? getStinkyInst(regionStart).getParent() : nullptr;

    if (getPassContext().getPassFeatureConfig().loopConfig.unrollGemm == false) return;

    wmmaPipelineGapCycles = 4;  // FIXME: should use jump latency instead.

    // Rule (5): cross-BB loop tail WMMA detection via LoopDetection.
    const Loop* loop = getLoop();
    if (loop && loop->headerBB && loop->latchBB) {
        if (latchBBTailIsWmma(*loop->latchBB)) deferFirstHeadWmmaActive_ = true;
    }

    wmmaIssueConfig.latency = 0;
    for (IRList::iterator it = regionStart; it != regionEnd; ++it) {
        StinkyInstruction& inst = getStinkyInst(it);
        if (isMatrixInstruction(inst)) {
            wmmaIssueConfig.latency = inst.latencyCycles;
            break;
        }
    }

    restoreCrossBBStateFromLoop();
}

// Restore cross-BB state from all predecessors via ScheduleAnalysisCache.
// Only triggered for loop body BBs, but reads from all predecessors
// (including pre-loop) — findMostReadyWMMA handles residual latencies gracefully.
void CDNA5ReadyQueue::restoreCrossBBStateFromLoop() {
    crossBBDsResiduals_.clear();
    const Loop* loop = getLoop();
    if (!currentBB_ || !loop || !loop->contains(currentBB_) || !getAnalysisCache()) return;

    for (BasicBlock* pred : currentBB_->getPredecessors()) {
        const BBScheduleState* state = getAnalysisCache()->lookup(pred);
        if (!state) continue;
        wmmaPipelineGapCycles = std::max(wmmaPipelineGapCycles, state->gapCycles);
        for (const auto& [regIdx, rem] : state->dsResiduals) {
            if (rem > 0) crossBBDsResiduals_[regIdx] = std::max(crossBBDsResiduals_[regIdx], rem);
        }
    }
}

// Save end-of-BB state to ScheduleAnalysisCache for all BBs.
void CDNA5ReadyQueue::onFinishBB() {
    if (currentBB_ && getAnalysisCache())
        getAnalysisCache()->store(currentBB_, {wmmaPipelineGapCycles, wmmaRegisterLatencyCounters});
}

// Per scheduling region. avgIssueInterval uses region-local totals only (no loop-wide aggregation).
// Rule (2): seedWmmaDsLatencyFromPrefix. Rule (4): wmmaIssueConfig. Rule (5): head balance.
void CDNA5ReadyQueue::onInitRegion(IRList::iterator regionStart, IRList::iterator regionEnd,
                                   IRList::iterator blockBegin) {
    // Per scheduling region (between non-movable side effects). blockBegin is the BB start
    // so prefix seeding and loop-head detection see preloop / prior regions in file order.
    lastPickedWasWMMA = false;
    wmmaIssuedCountThisRegion_ = 0;
    if (getPassContext().getPassFeatureConfig().loopConfig.unrollGemm == false) return;

    // Rule (5): only defer in the header BB of the loop (where back-edge lands),
    // and only for regions after the first one (matching the old prefix-based check
    // where the first region's empty prefix meant no deferral).
    const Loop* loop = getLoop();
    deferHeadBalanceThisRegion_ = deferFirstHeadWmmaActive_ && loop &&
                                  loop->headerBB == getStinkyInst(blockBegin).getParent() &&
                                  regionStart != blockBegin;

    seedWmmaDsLatencyFromPrefix(blockBegin, regionStart, wmmaRegisterLatencyCounters,
                                crossBBDsResiduals_);

    wmmaIssueConfig.totalIssuedCycles = 0;
    wmmaIssueConfig.totalWmmaIssuedCycles = 0;
    wmmaIssueConfig.issuedCount = 0;
    int totalDSLatency = 0;
    hasWMMAInRegion_ = false;
    for (IRList::iterator it = regionStart; it != regionEnd; ++it) {
        StinkyInstruction& inst = getStinkyInst(it);

        wmmaIssueConfig.totalIssuedCycles += inst.issueCycles;

        if (isDSRead(inst)) totalDSLatency += inst.latencyCycles;

        if (isMatrixInstruction(inst)) {
            wmmaIssueConfig.issuedCount += 1;
            wmmaIssueConfig.totalWmmaIssuedCycles += inst.issueCycles;
        }
    }
    // Budget per WMMA (avgIssueInterval): spread WMMAs when the region is DS- or issue-heavy.
    //
    //   totalIssuedCycles   |========================|  all insn issueCycles in region
    //   totalDSLatency      |==============================|  sum of ds_load latencyCycles
    //                                    |
    //                         numer = max(both)   <-- whichever rail is longer wins
    //                                    |
    //                                    v
    //              totalAvgLatency = ceil( numer / issuedCount )   "room" per WMMA slot
    //
    //   wmmaIssueConfig.latency   |===|  one WMMA op's latencyCycles (floor on spacing)
    //
    //   avgIssueInterval = max(totalAvgLatency, latency)
    //   pickOneFromWMMA uses avgIssueInterval so sibling WMMAs wait longer when DS dominates.
    //
    int totalAvgLatency;
    if (wmmaIssueConfig.issuedCount <= 0)
        totalAvgLatency = wmmaIssueConfig.latency;
    else {
        totalAvgLatency =
            (int)std::ceil((float)std::max(wmmaIssueConfig.totalIssuedCycles, totalDSLatency) /
                           wmmaIssueConfig.issuedCount);
        hasWMMAInRegion_ = true;
    }
    if (totalAvgLatency > wmmaIssueConfig.latency) {
        wmmaIssueConfig.avgIssueInterval = totalAvgLatency;
    } else {
        wmmaIssueConfig.avgIssueInterval = wmmaIssueConfig.latency;
    }

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
