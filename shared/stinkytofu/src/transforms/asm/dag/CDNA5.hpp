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
#include <array>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <map>
#include <tuple>
#include <vector>

#include "InFlightQueue.hpp"
#include "ReadyQueue.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/transforms/asm/dag/HazardRules.hpp"

namespace {
using namespace stinkytofu;

enum NonWmmaKind { kGlobalRead = 0, kLocalRead, kOther, kValu };

// -------------------------------------------------------------------------
// Hardware hazard rules: a fixed cycle gap required between a producer writing a
// register and a specific class of consumer reading it as a source (not either op's
// own issue/latency cycles — a producer->consumer edge gap). Data-driven so a new
// hazard pair is a new table row, not new code. Consumer-side correctness (the gate in
// CDNA5ReadyQueue::hazardGates_) is unconditional regardless of scheduling order;
// producer-side hoisting (CDNA5ReadyQueue::decidePromote(), via DAGNode::hazardDeadline)
// is a throughput heuristic layered on top, never required for correctness.
//
// HazardRule itself, and the family-wide kCdna5HazardRules table, live in HazardRules.hpp
// (included above) so HazardGapAnalysisPass can share them without duplicating this logic.
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// Per-arch CDNA5 scheduling POLICY. CDNA5.hpp is the CDNA5 *family* ready queue: both
// gfx1250 and gfx1250v0 compile against it, so these are selected per arch here rather
// than baked in as single family-wide constants. A new CDNA5-family arch adds one case
// to cdna5ConfigForArch(). User PassFeatureConfig overrides still win over these
// per-arch defaults (see the dsReadPerWmma accessor).
//
// Only the scheduling *ratios* live here. The physical facts this queue also needs -
// LDS queue depth, drain/throttle latency, and the hazard-rule table - are hardware,
// not policy, and live in HWModel (stinkytofu/hardware/HWModel.hpp), reached through
// passCtx.getHWModel(). Keeping them there is what lets other passes see the same
// numbers instead of each redeclaring them.
// -------------------------------------------------------------------------
struct CDNA5Config {
    // Used when dagFeatures still hold the PassFeatureConfig INT_MAX sentinel;
    // explicit non-sentinel user config wins over these.
    int dsReadPerWmma;
    int globalReadPerWmma;
};

constexpr CDNA5Config kGfx1250Config = {
    /*dsReadPerWmma=*/3,
    /*globalReadPerWmma=*/1,
};

// gfx1250v0: starts from the gfx1250 values. TODO(tuning): fill in gfx1250v0's real
// per-WMMA ratios. Kept as its own case so those numbers can be changed here without
// touching gfx1250. Its physical facts are likewise a separate HWModel entry.
constexpr CDNA5Config kGfx1250v0Config = kGfx1250Config;

// Select the CDNA5-family policy for \p arch. Private to the ready queue / scheduler
// (not shared infrastructure): each CDNA5 arch's knobs live next to the family model
// that consumes them. gfx1250 is the default for any unlisted arch (the pipeline only
// runs the CDNA5 ready queue on CDNA5-family archs).
//
// Keyed off the shared kArchKey* constants (HWModel.hpp) so this policy table and the
// HWModel fact table cannot be restepped independently.
inline const CDNA5Config& cdna5ConfigForArch(const std::array<int, 3>& arch) {
    switch (archKey(arch)) {
        case kArchKeyGfx1250v0:
            return kGfx1250v0Config;
        case kArchKeyGfx1250:
        default:
            return kGfx1250Config;
    }
}

// -------------------------------------------------------------------------
// Prefix / loop analysis (free functions; no CDNA5ReadyQueue state)
// -------------------------------------------------------------------------

// Register-file-aware key for the data-ready (RAW) and elapse-touch maps. These maps
// were keyed on reg.idx alone, which conflates register files: e.g. a WMMA writing its
// accumulator v[12:20) would stamp indices 12..19 and falsely gate a later SALU that
// reads s14/s15 (same indices, different file). Fold the register type into the key so
// vector, scalar, and accumulator registers of the same index never collide.
static inline int regDepKey(RegType type, uint32_t idx) {
    return (static_cast<int>(type) << 20) | static_cast<int>(idx & 0xFFFFF);
}

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
                pending[regDepKey(dstReg.reg.type, dstReg.reg.idx + off)] = inst.latencyCycles;
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

static inline int popcount16(uint16_t v) {
    return __builtin_popcount(static_cast<unsigned>(v));
}

// Prev WMMA's D feeds next WMMA's A/B (or SWMMAC index).
static bool wmmaToWmmaCoexecOverlap(const StinkyInstruction& prod, const StinkyInstruction& cons) {
    if (prod.getDestRegs().empty()) return false;
    const StinkyRegister& d = prod.getDestRegs()[0];
    const auto& srcs = cons.getSrcRegs();
    if (srcs.size() > 0 && d.isOverlap(srcs[0])) return true;                   // A
    if (srcs.size() > 1 && d.isOverlap(srcs[1])) return true;                   // B
    if (isSWMMA(cons) && srcs.size() > 2 && d.isOverlap(srcs[2])) return true;  // index
    return false;
}

// WMMA D vs a co-executable VALU consumer: RAW (D->src), WAW (D->dst), WAR (prod A/B or
// SWMMAC index -> cons dst).
static bool wmmaToValuCoexecOverlap(const StinkyInstruction& prod, const StinkyInstruction& cons) {
    if (prod.getDestRegs().empty()) return false;
    const StinkyRegister& d = prod.getDestRegs()[0];
    for (const StinkyRegister& s : cons.getSrcRegs())
        if (d.isOverlap(s)) return true;  // RAW
    for (const StinkyRegister& cd : cons.getDestRegs())
        if (d.isOverlap(cd)) return true;  // WAW
    const auto& psrc = prod.getSrcRegs();
    const size_t nWar = isSWMMA(prod) ? 3 : 2;
    for (size_t i = 0; i < psrc.size() && i < nWar; ++i)
        for (const StinkyRegister& cd : cons.getDestRegs())
            if (psrc[i].isOverlap(cd)) return true;  // WAR
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

// A run of barriers that share the same PSEUDO token set. barrier_signal/barrier_wait
// pairs (adjacent in the IR, same tokens) are merged so both get the same threshold.
struct BarrierTokenGroup {
    std::vector<StinkyInstruction*> barriers;
    std::unordered_set<uint32_t> tokens;
    IRList::iterator firstIt;
    IRList::iterator lastIt;
};

// Merge consecutive same-token barriers into groups of up to two (a signal/wait pair).
// Two barriers only pair when they carry identical tokens and are adjacent in the IR.
static std::vector<BarrierTokenGroup> groupBarrierTokens(
    const std::vector<BarrierTokenEntry>& entries) {
    std::vector<BarrierTokenGroup> groups;
    for (const BarrierTokenEntry& be : entries) {
        const bool canPairWithLast = !groups.empty() && groups.back().tokens == be.tokens &&
                                     groups.back().barriers.size() < 2 &&
                                     std::next(groups.back().lastIt) == be.it;
        if (canPairWithLast) {
            groups.back().barriers.push_back(be.barrier);
            groups.back().lastIt = be.it;
        } else {
            groups.push_back({{be.barrier}, be.tokens, be.it, be.it});
        }
    }
    return groups;
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
    // Which phase, if any, decidePromote() forces this pick (None = normal selection).
    // Values name the gateable phases of pickOne(); isPromote(p) is true when nothing
    // is promoted (every phase runs) or when p is the promoted phase (only it runs).
    enum class PromotePhase { None, Wmma, NonWmmaFill, ForcedWmma, Barrier };

    // --- Priority buckets (DAG ids compare smaller = earlier in source) ---
    ReadySetByDAGid wmmaQueue;
    ReadySetByDAGid globalReadQueue;  // tensor_load_to_lds when distributeGlobalRead
    ReadySetByDAGid localReadQueue;   // ds_load
    ReadySetByDAGid valuQueue;        // VALU and transcendental instructions
    ReadySetByDAGid barrierQueue;
    ReadySetByDAGid otherQueue;  // scalars, waits in region, etc.

    // Per-arch CDNA5 scheduling policy (the per-WMMA ratios), selected by arch in the
    // constructor. Points at a static constexpr CDNA5Config, so this is a stable ref.
    const CDNA5Config& config_;

    // Physical hardware facts for this arch (LDS queue model, hazard-rule table).
    // Owned by the library, one object per arch, so this reference is stable too.
    const HWModel& hw_;

    // Throttle tensor issues vs other work.
    int globalReadCounter = 0;
    int globalReadPerWMMA = config_.globalReadPerWmma;

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
        return cfg > 0 ? cfg : hw_.lds.readQueueDepth;
    }
    // Barrier-timing only (computeBarrierAfterThresholds): the cycles a barrier must wait for
    // its dependent ds_reads to return. 0 means "derive dynamically from the matching
    // ds_read count and target ds_read latency." Queue occupancy / pacing uses
    // dsReadThrottleLatency, not this.
    int dsReadDrainLatency() const {
        const int cfg = getPassContext().getPassFeatureConfig().dagFeatures.dsReadDrainLatency;
        return cfg > 0 ? cfg : hw_.lds.readDrainLatency;
    }
    // Lifetime of one in-flight ds_read credit; also sets the saturated-queue issue interval
    // (dsReadThrottleLatency/dsReadQueueDepth cycles per ds_read).
    int dsReadThrottleLatency() const {
        const int cfg = getPassContext().getPassFeatureConfig().dagFeatures.dsReadThrottleLatency;
        return cfg > 0 ? cfg : hw_.lds.readThrottleLatency;
    }
    int dsReadPerWmma() const {
        const int cfg = getPassContext().getPassFeatureConfig().dagFeatures.dsReadPerWmma;
        return cfg > 0 ? (cfg < INT_MAX ? cfg : config_.dsReadPerWmma) : config_.dsReadPerWmma;
    }
    bool dsReadQueueFull() const {
        return dsReadInflight_.full();
    }
    int dsReadThrottleWait() const {
        return dsReadInflight_.throttleWait();
    }

    // --- VALU co-issue timeline tracker ---
    uint16_t activeCoIssueWindow_ = 0;
    int coIssueCyclePos_ = 0;
    int activeWmmaLatency_ = 0;
    // WMMA that opened the current latency window (valid while coIssueCyclePos_ <
    // activeWmmaLatency_). Used to detect ds_load dest / WMMA src VGPR overlap hazards.
    DAGNode* activeWmmaNode_ = nullptr;

    // Non-WMMA fills since the active WMMA opened its window. A dependent next WMMA is held
    // in Phase B until this reaches popcount(coIssueWindow)+1 (WMMA->WMMA coexec slots + 1).
    int nonWmmaFillsSinceActiveWmma_ = 0;

    // Filler v_nops to emit before the next pickOne() result (drained by scheduleInDAG).
    // Set at a dependent WMMA's issue point when no independent VALU remained to fill the gap.
    int pendingFillerVNops_ = 0;

    // VGPR-MSB bank currently in effect (mirrors InsertVgprMsbPass); updated on issue,
    // reset per region. pickFreeBest prefers a free candidate matching it. -1 = unknown.
    int currentMsb_ = -1;

    // --- Per-WMMA-window DS cap (dagFeatures.dsReadPerWmma) ---
    int maxDsPerWmmaWindow_ = 0;
    int dsInsertedSinceLastWmma_ = 0;
    // Per-window override for maxDsPerWmmaWindow_; empty => use the flat value.
    std::vector<int> dsTargetPerWindow_;

    // (A) RAW data-ready gate. Per reg index: remaining modeled latency until a
    // producer's result is safe to consume (e.g. ds_load LDS->VGPR, 56 cyc). Any
    // long-latency producer stamps it; a consumer whose src is still in flight is not
    // "free". Decays in advanceTime. Crosses BBs via BBScheduleState.dsResiduals.
    std::map<int, int> regDataReadyCounters;

    // Hazard gates, one independent lane per config_.hazardRules entry. Per reg key:
    // remaining cycles until a rule.isConsumer instruction may read it (stamped
    // rule.cycles when a flagged producer issues). Kept SEPARATE per rule (and
    // separate from regDataReadyCounters) because each hazard is consumer-type- and
    // register-file-specific: e.g. a VALU/SALU reading the same sgpr a flagged SALU
    // just wrote is not gated by the SaluSgprToMemAddr lane. Decays in advanceTime.
    // Sized to config_.numHazardRules at construction (runtime, since the rule count is
    // per-arch), indexed by the same ruleIdx the scheduler pre-scan assigns.
    std::vector<std::map<int, int>> hazardGates_;

    // Ready, flagged (non-empty hazardFlags), not-yet-issued hazard producers, tracked
    // so decidePromote() doesn't need to scan every queue each pick to find the ones
    // whose hazardDeadline might have arrived. Populated in push(), erased once the
    // node is picked (popNonWmma).
    struct HazardHoistCandidate {
        DAGNode* node;
        int kind;  // NonWmmaKind: which queue this producer sits in (kOther or kValu).
    };
    std::vector<HazardHoistCandidate> hazardHoistCandidates_;

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

    // Set by decidePromote() each pick: which phase is forced (None = normal selection)
    // and the exact node that phase will issue. Read via isPromote() to gate the phases.
    PromotePhase promotedPhase_ = PromotePhase::None;
    DAGNode* promotedNode_ = nullptr;
    // Valid only when promotedPhase_ == PromotePhase::NonWmmaFill via the hazard-hoist
    // case: which queue (NonWmmaKind: kOther or kValu) promotedNode_ must be popped from.
    int promotedKind_ = -1;

    std::map<int, int> crossBBDsResiduals_;

    void advanceTime(int cycles);
    int computeValuAdvanceCycles(int issueCycles) const;
    void updateWMMAStatus(DAGNode* node);
    void stampDataReady(const StinkyInstruction& inst);
    void touchOperands(const StinkyInstruction& inst);
    int getMaxSrcDataWait(DAGNode* node) const;
    int getHazardWait(DAGNode* node) const;
    bool destOverlapsActiveWmmaSrc(DAGNode* node) const;
    int nodeElapseKey(DAGNode* node) const;
    DAGNode* pickFreeBest(const ReadySetByDAGid& queue, int* outWait = nullptr,
                          bool allowHiddenStall = false) const;
    std::pair<DAGNode*, int> findMostReadyWMMA();
    DAGNode* pickOneFromWMMA(DAGNode* pick = nullptr);
    bool findSmallestPickableNonWmma(DAGNode* pickedDS, DAGNode** outNode, int* kindOut,
                                     int* outWait = nullptr) const;

    bool findOldestFallbackNonWmma(DAGNode* pickedDS, DAGNode** outNode, int* kindOut,
                                   int* outWait = nullptr) const;

    // Promotion = split the old forced-barrier phase into a pure decision + a per-phase
    // gate, run once before the normal phases. decidePromote() records which phase must
    // fire now (promotedPhase_) and the exact node it will issue (promotedNode_),
    // mutating no queue. Each phase guards its body with isPromote(ThisPhase): when
    // nothing is promoted every phase runs normally; when something is promoted only
    // that phase runs, so the promoted node issues through its own existing phase (one
    // entry point), never a second dedicated path. A new forcing rule is a new case in
    // decidePromote(), not a new phase. Two promotions: a barrier whose per-barrier
    // WMMA-issued threshold is met (formerly the forced-barrier phase), and a
    // hazard-hoist producer whose live clock_ has reached its hazardDeadline (forces
    // the producer to issue now, through its own NonWmmaFill phase, so it lands before
    // its hazarded consumer needs the gap instead of after).
    void decidePromote();
    bool isPromote(PromotePhase phase) const {
        return promotedPhase_ == PromotePhase::None || promotedPhase_ == phase;
    }
    DAGNode* extractForcedBarrier();
    std::unordered_map<StinkyInstruction*, BarrierAfterOutput> computeBarrierAfterThresholds(
        IRList::iterator regionStart, IRList::iterator regionEnd);
    std::unordered_map<StinkyInstruction*, BarrierBeforeOutput> computeBarrierBeforeThresholds(
        IRList::iterator regionStart, IRList::iterator regionEnd);
    int computeWmmaWindowsNeeded(int dsLoadCount) const;
    bool isValuPickable() const;
    DAGNode* popNonWmma(DAGNode* node, int pickKind);

    void restoreCrossBBStateFromLoop();

   public:
    explicit CDNA5ReadyQueue(const PassContext& passCtx)
        : ReadyQueue(passCtx),
          config_(cdna5ConfigForArch(passCtx.getGemmTileConfig().arch)),
          hw_(passCtx.getHWModel()),
          hazardGates_(hw_.hazards.numRules) {}

    DAGNode* pickOne() override;
    void push(DAGNode* node) override;
    bool empty() const override;

    // Build the coexec filler v_nops counted during the last pickOne() as detached insts.
    std::vector<StinkyInstruction*> takePendingFillerInsts() override {
        std::vector<StinkyInstruction*> fillers;
        if (pendingFillerVNops_ > 0) {
            const auto& arch = getPassContext().getGemmTileConfig().arch;
            const GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);
            const HwInstDesc* vnopDesc = getMCIDByUOp(GFX::v_nop, archId);
            fillers.reserve(pendingFillerVNops_);
            for (int i = 0; i < pendingFillerVNops_; ++i)
                fillers.push_back(IRBase::createIR<StinkyInstruction>(vnopDesc));
        }
        pendingFillerVNops_ = 0;
        return fillers;
    }

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
    for (auto& gate : hazardGates_) {
        for (auto it = gate.begin(); it != gate.end();) {
            it->second -= cycles;
            if (it->second <= 0)
                it = gate.erase(it);
            else
                ++it;
        }
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
        dsReadInflight_.pushWithThrottle(dsReadThrottleLatency());
        dsInsertedSinceLastWmma_++;
    } else if (pickKind == kOther) {
        otherQueue.erase(node);
    } else {
        assert(pickKind == kValu);
        valuQueue.erase(node);
    }
    // WMMA->VALU: if this VALU depends on the active WMMA's D and fewer than slots fills
    // landed (no independent VALU left to space it), emit the shortfall as v_nops before it.
    if (pickKind == kValu && activeWmmaNode_ != nullptr &&
        wmmaToValuCoexecOverlap(*activeWmmaNode_->inst, *node->inst)) {
        const int slots = popcount16(activeWmmaNode_->inst->coIssueWindow);
        const int shortfall = slots - nonWmmaFillsSinceActiveWmma_;
        if (shortfall > 0) {
            pendingFillerVNops_ = shortfall;
            // Credit fillers as slot fills so a later dependent VALU in this window is not
            // re-padded.
            nonWmmaFillsSinceActiveWmma_ += shortfall;
        }
    }
    // Only VALU-pipe ops fill a coexec slot.
    if (pickKind == kValu) nonWmmaFillsSinceActiveWmma_++;
    // (A) RAW: stamp this producer's dest data-ready latency (e.g. ds_load).
    // (B) elapse: record the timeline touch for all operands (dst + src).
    touchOperands(*node->inst);
    updateWMMAStatus(node);
    stampDataReady(*node->inst);
    // Advance the tracked MSB bank; ops with no opinion (-1) leave it unchanged.
    if (node->requiredMsb != -1) currentMsb_ = node->requiredMsb;
    // Hazard gates: stamp exactly the (rule, register) pairs the pre-scan found a
    // rule.isConsumer instruction reads from this producer, so that consumer's pick
    // waits the fixed hazard out. Per-rule lane (not regDataReadyCounters) so an
    // unrelated instruction reading the same register is not wrongly gated.
    for (const HazardFlag& hf : node->hazardFlags)
        // rule.cycles == -1 ("hoist as far as possible"): the strategy is producer-side
        // hoisting (deadline forced to 0 in the pre-scan), not a consumer-side hold, so
        // clamp the gate to 0 rather than stamping a negative wait.
        hazardGates_[hf.ruleIdx][hf.regKey] = std::max(0, hw_.hazards.rules[hf.ruleIdx].cycles);
    // No longer a live hoist candidate once issued (decidePromote() must not try to
    // force it again).
    if (!node->hazardFlags.empty()) {
        auto it = std::find_if(hazardHoistCandidates_.begin(), hazardHoistCandidates_.end(),
                               [node](const HazardHoistCandidate& hc) { return hc.node == node; });
        if (it != hazardHoistCandidates_.end()) hazardHoistCandidates_.erase(it);
    }
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
            regDataReadyCounters[regDepKey(dst.reg.type, dst.reg.idx + off)] =
                inst.latencyCycles - inst.issueCycles;
    }
}

// (B) elapse touch: record the current timeline clock for every operand reg (dst + src)
// of an issued instruction, so a later node reusing a just-touched reg can be deferred.
void CDNA5ReadyQueue::touchOperands(const StinkyInstruction& inst) {
    for (const StinkyRegister& dst : inst.getDestRegs()) {
        if (!dst.isRegister() || isPseudoReg(dst)) continue;
        for (unsigned off = 0; off < dst.reg.num; ++off)
            regLastTouch_[regDepKey(dst.reg.type, dst.reg.idx + off)] = clock_;
    }
    for (const StinkyRegister& src : inst.getSrcRegs()) {
        if (!src.isRegister() || isPseudoReg(src)) continue;
        for (unsigned off = 0; off < src.reg.num; ++off)
            regLastTouch_[regDepKey(src.reg.type, src.reg.idx + off)] = clock_;
    }
}

// (A) RAW data-ready gate: max outstanding data-ready latency over a node's src VGPRs.
// Returns 0 if all src data is ready (safe to consume), >0 if hardware would stall.
int CDNA5ReadyQueue::getMaxSrcDataWait(DAGNode* node) const {
    int maxLat = 0;
    for (const StinkyRegister& srcReg : node->inst->getSrcRegs()) {
        if (!srcReg.isRegister()) continue;
        for (unsigned off = 0; off < srcReg.reg.num; ++off) {
            auto it = regDataReadyCounters.find(regDepKey(srcReg.reg.type, srcReg.reg.idx + off));
            if (it != regDataReadyCounters.end() && it->second > maxLat) maxLat = it->second;
        }
    }
    return maxLat;
}

// Hazard gate: max remaining hazard cycles over \p node's sources, across every rule
// where \p node's instruction matches rule.isConsumer. Returns 0 when \p node may issue
// now with respect to every hazard rule, >0 when it must still wait for some flagged
// producer's write to clear the hardware gap. Consulted for any candidate that could be
// a hazarded consumer (tensor_load/s_load/global_read/etc. — see call sites in
// pickFreeBest and findSmallestPickableNonWmma).
int CDNA5ReadyQueue::getHazardWait(DAGNode* node) const {
    int maxLat = 0;
    for (int ruleIdx = 0; ruleIdx < hw_.hazards.numRules; ++ruleIdx) {
        const HazardRule& rule = hw_.hazards.rules[ruleIdx];
        const auto& gate = hazardGates_[ruleIdx];
        if (gate.empty() || !rule.isConsumer(*node->inst)) continue;
        for (const StinkyRegister& srcReg : node->inst->getSrcRegs()) {
            if (!srcReg.isRegister() || isPseudoReg(srcReg) || srcReg.reg.type != rule.regType)
                continue;
            for (unsigned off = 0; off < srcReg.reg.num; ++off) {
                auto it = gate.find(regDepKey(srcReg.reg.type, srcReg.reg.idx + off));
                if (it != gate.end() && it->second > maxLat) maxLat = it->second;
            }
        }
    }
    return maxLat;
}

// True if issuing \p node now would risk a co-execution hazard: while the WMMA that
// opened the current latency window is still in flight, \p node's dest VGPRs overlap
// that WMMA's src VGPRs, so the write could clobber a source the WMMA is still reading.
bool CDNA5ReadyQueue::destOverlapsActiveWmmaSrc(DAGNode* node) const {
    if (node == nullptr || activeWmmaNode_ == nullptr) return false;
    if (coIssueCyclePos_ >= activeWmmaLatency_) return false;
    for (const StinkyRegister& dstReg : node->inst->getDestRegs()) {
        if (!dstReg.isRegister() || isPseudoReg(dstReg)) continue;
        for (const StinkyRegister& srcReg : activeWmmaNode_->inst->getSrcRegs()) {
            if (!srcReg.isRegister() || isPseudoReg(srcReg)) continue;
            if (dstReg.isOverlap(srcReg)) return true;
        }
    }
    return false;
}

// (B) elapse key: min over the node's operand regs (dst + src) of (clock_ - lastTouch).
// The most-recently-touched operand binds (smallest elapse), so a node reusing a
// just-touched reg ranks low and is deferred. Regs never touched => INT_MAX (very old).
int CDNA5ReadyQueue::nodeElapseKey(DAGNode* node) const {
    int minElapse = INT_MAX;
    auto consider = [&](const StinkyRegister& r) {
        if (!r.isRegister() || isPseudoReg(r)) return;
        for (unsigned off = 0; off < r.reg.num; ++off) {
            auto it = regLastTouch_.find(regDepKey(r.reg.type, r.reg.idx + off));
            const int elapse = (it == regLastTouch_.end()) ? INT_MAX : (clock_ - it->second);
            if (elapse < minElapse) minElapse = elapse;
        }
    };
    for (const StinkyRegister& dst : node->inst->getDestRegs()) consider(dst);
    for (const StinkyRegister& src : node->inst->getSrcRegs()) consider(src);
    return minElapse;
}

// Updates (best, bestKey) to (cand, key) if key sorts before bestKey (lexicographic
// std::tuple compare) or best is not yet set. Shared by the "min by (metric, id)"
// candidate pickers below; each caller supplies its own key shape.
template <typename Key>
static bool considerBest(DAGNode* cand, Key key, DAGNode*& best, Key& bestKey) {
    if (!cand) return false;
    if (best == nullptr || key < bestKey) {
        best = cand;
        bestKey = key;
        return true;
    }
    return false;
}

// Best issuable node in \p queue: RAW/hazard-free nodes are preferred; when none is
// free and \p allowHiddenStall is set, a node whose remaining wait (RAW data-ready or
// hazard-gate, whichever is larger) still fits under the active WMMA's latency shadow
// is also eligible (the wait is hidden by the in-flight WMMA, so it is free to
// co-issue). Within the same free/hazard tier the operand touched longest ago wins
// (largest elapse), tie broken by DAG id. Producer-side hazard hoisting is handled
// separately by decidePromote(), not here — this function only decides whether a
// candidate is safe to issue *now*, not whether it should be forced early.
// \p outWait (optional) receives the cycles the caller must advanceTime() before
// issuing the returned node (0 for a free pick). Returns nullptr if none is eligible.
DAGNode* CDNA5ReadyQueue::pickFreeBest(const ReadySetByDAGid& queue, int* outWait,
                                       bool allowHiddenStall) const {
    const int coIssueSpace = activeWmmaLatency_ - coIssueCyclePos_;
    DAGNode* best = nullptr;
    int bestElapse = INT_MIN;
    int bestWait = 0;
    int bestAff = 0;
    for (DAGNode* n : queue) {  // iterates smallest-id first, so ties keep the oldest id
        const int wait = std::max(getMaxSrcDataWait(n), getHazardWait(n));
        // Tolerate a wait only if it fits the WMMA latency shadow and the dest does
        // not clobber a live WMMA src (then the stall is free).
        if (wait > 0 && (!allowHiddenStall || coIssueSpace <= 0 || wait > coIssueSpace ||
                         destOverlapsActiveWmmaSrc(n)))
            continue;
        const int elapse = nodeElapseKey(n);
        // MSB bank affinity: a tiebreak BELOW the free/hazard tier and ABOVE elapse —
        // prefer a same-bank candidate to avoid an s_set_vgpr_msb switch. Inert when
        // currentMsb_ == -1 (nothing matches), so unchanged behavior until a real tie.
        const int aff = (n->requiredMsb != -1 && n->requiredMsb == currentMsb_) ? 1 : 0;
        // Free nodes rank above hidden-stall ones; within a tier, prefer same-bank
        // (larger aff), then largest elapse.
        const bool nHazard = wait > 0;
        const bool curHazard = bestWait > 0;
        bool better;
        if (best == nullptr)
            better = true;
        else
            better = (!nHazard && curHazard) ||
                     (nHazard == curHazard &&
                      (aff > bestAff || (aff == bestAff && elapse > bestElapse)));
        if (better) {
            best = n;
            bestElapse = elapse;
            bestWait = wait;
            bestAff = aff;
        }
    }
    if (best && outWait) *outWait = bestWait;
    return best;
}

// Find the WMMA in wmmaQueue with the smallest max data-ready latency (most ready).
// Returns the node and its latency. Ties broken by DAG id (program order).
std::pair<DAGNode*, int> CDNA5ReadyQueue::findMostReadyWMMA() {
    DAGNode* best = nullptr;
    std::tuple<int, int> bestKey{INT_MAX, 0};
    for (DAGNode* n : wmmaQueue) {
        considerBest(n, std::make_tuple(getMaxSrcDataWait(n), (int)n->id), best, bestKey);
    }
    return {best, std::get<0>(bestKey)};
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

    // Dependent b2b WMMA with too few fills: emit the shortfall as v_nops.
    if (activeWmmaNode_ != nullptr &&
        wmmaToWmmaCoexecOverlap(*activeWmmaNode_->inst, *node->inst)) {
        const int slotsPlusOne = popcount16(activeWmmaNode_->inst->coIssueWindow) + 1;
        const int shortfall = slotsPlusOne - nonWmmaFillsSinceActiveWmma_;
        if (shortfall > 0) pendingFillerVNops_ = shortfall;
    }

    // consume the time that is not used by the WMMA
    if (coIssueCyclePos_ < activeWmmaLatency_) advanceTime(activeWmmaLatency_ - coIssueCyclePos_);

    activeCoIssueWindow_ = node->inst->coIssueWindow;
    coIssueCyclePos_ = 0;
    activeWmmaLatency_ = node->inst->latencyCycles;
    activeWmmaNode_ = node;
    nonWmmaFillsSinceActiveWmma_ = 0;  // new window: restart WMMA->WMMA fill count
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
    if (node->requiredMsb != -1) currentMsb_ = node->requiredMsb;
    return node;
}

// Pick among ready non-WMMA nodes, preferring genuinely RAW-free work.
// Queues: globalReadQueue (throttled), localReadQueue, valuQueue (co-issue gated), otherQueue.
// SALU/other and VALU picks go through pickFreeBest (elapse ordering — defers a
// reg-overwrite behind other work). SALU/other may additionally be co-issued with a
// hidden stall when its src RAW wait fits under the active WMMA's latency shadow; that
// wait is returned via \p outWait so the caller advances the timeline before issuing.
// A hidden-stall candidate never outranks a genuinely free one. When nothing qualifies,
// these queues contribute nothing and Phase G falls back to the oldest.
// kind: 0=global, 1=local, 2=other, 3=valu.
bool CDNA5ReadyQueue::findSmallestPickableNonWmma(DAGNode* pickedDS, DAGNode** outNode,
                                                  int* kindOut, int* outWait) const {
    *outNode = nullptr;
    *kindOut = -1;
    if (outWait) *outWait = 0;
    DAGNode* best = nullptr;
    int kind = -1;
    int bestWait = 0;
    std::tuple<bool, int> bestKey{};

    // Ordering, highest key first: (1) free work beats a hidden-stall candidate;
    // (2) smallest id. Producer-side hazard hoisting is handled separately by
    // decidePromote(), not here — a flagged producer competes on equal terms with
    // everything else unless/until decidePromote() forces it.
    auto consider = [&](DAGNode* cand, int candKind, int candWait) {
        if (!cand) return;
        if (considerBest(cand, std::make_tuple(candWait > 0, (int)cand->id), best, bestKey)) {
            kind = candKind;
            bestWait = candWait;
        }
    };

    // Per-WMMA-window DS cap (rule 4) only spreads ds_loads across an active WMMA
    // co-issue window; it is meaningless when no WMMA is available to issue, so it
    // is applied only while a WMMA is pending. When the DS queue reaches depth,
    // ds_load can still issue, but is throttled to dsReadThrottleLatency/dsReadQueueDepth.
    int windowCap = maxDsPerWmmaWindow_;
    if (!dsTargetPerWindow_.empty()) {
        const int w = std::min((int)wmmaIssuedCountThisRegion_, (int)dsTargetPerWindow_.size() - 1);
        windowCap = dsTargetPerWindow_[w];
    }
    const bool dsCapReached = !wmmaQueue.empty() && dsInsertedSinceLastWmma_ >= windowCap;
    const bool dsBaseOk = pickedDS && !dsCapReached && !destOverlapsActiveWmmaSrc(pickedDS);
    int dsThrottleWait = 0;
    if (dsBaseOk) {
        dsThrottleWait = dsReadThrottleWait();
        consider(pickedDS, kLocalRead, dsThrottleWait);
    }
    const bool dsWindowOk = dsBaseOk && dsThrottleWait == 0;

    if (!globalReadQueue.empty() && !globalReadQueueFull() &&
        (globalReadCounter < globalReadPerWMMA || otherQueue.empty())) {
        // A tensor_load whose source is still inside a live hazard-gate window carries
        // that wait, so it ranks as a hidden-stall candidate and defers behind free
        // work (whatever fills the gap). It is still eligible when nothing else can
        // go, paying the remaining wait before issue.
        DAGNode* gr = globalReadQueue.top();
        consider(gr, kGlobalRead, getHazardWait(gr));
    }
    // SALU/other allows hidden stalls (see pickFreeBest); VALU stays RAW-free-only.
    int otherWait = 0;
    if (DAGNode* t = pickFreeBest(otherQueue, &otherWait, /*allowHiddenStall=*/true)) {
        consider(t, kOther, otherWait);
    }
    if (isValuPickable() || best == nullptr) {
        if (DAGNode* t = pickFreeBest(valuQueue)) {
            if (!dsWindowOk && !destOverlapsActiveWmmaSrc(t)) consider(t, kValu, 0);
        }
    }

    if (!best) return false;
    *outNode = best;
    *kindOut = kind;
    if (outWait) *outWait = bestWait;
    return true;
}

// Final-fallback candidate search across non-WMMA queues. Ranks by smallest
// outstanding wait (not DAG id) so real work fills gaps where possible.
// kind: 0=global, 1=local, 2=other, 3=valu.
bool CDNA5ReadyQueue::findOldestFallbackNonWmma(DAGNode* pickedDS, DAGNode** outNode, int* kindOut,
                                                int* outWait) const {
    *outNode = nullptr;
    *kindOut = -1;
    if (outWait) *outWait = 0;
    DAGNode* best = nullptr;
    int kind = -1;
    std::tuple<int, int> bestKey{};

    auto consider = [&](DAGNode* cand, int candKind) {
        if (cand == nullptr) return;
        const int wait = std::max(getMaxSrcDataWait(cand), getHazardWait(cand));
        if (considerBest(cand, std::make_tuple(wait, (int)cand->id), best, bestKey))
            kind = candKind;
    };

    if (!globalReadQueue.empty()) consider(globalReadQueue.top(), kGlobalRead);
    consider(pickedDS, kLocalRead);
    if (!otherQueue.empty()) consider(otherQueue.top(), kOther);
    if (!valuQueue.empty()) consider(valuQueue.top(), kValu);

    if (best == nullptr) return false;
    *outNode = best;
    *kindOut = kind;
    if (outWait) *outWait = std::get<0>(bestKey);
    return true;
}

// Pure promotion decision, run once at the top of each pickOne(). Records which phase
// must fire now and the exact node it will issue, without mutating any queue. Two
// rules: (1) a barrier whose WMMA-issued threshold is met (this is what the dedicated
// forced-barrier phase used to do ahead of WMMA); (2) a hazard-hoist producer whose
// live clock_ has reached its hazardDeadline -- forces the producer to issue now,
// through its own NonWmmaFill phase, so it lands before its hazarded consumer needs
// the gap instead of after. Add further forcing rules here as new PromotePhase cases
// (or new decisions within an existing phase, as below).
void CDNA5ReadyQueue::decidePromote() {
    promotedPhase_ = PromotePhase::None;
    promotedNode_ = nullptr;
    promotedKind_ = -1;

    if (!barrierQueue.empty() && !barrierWmmaThresholds_.empty()) {
        for (DAGNode* node : barrierQueue) {
            auto thIt = barrierWmmaThresholds_.find(node->inst);
            if (thIt != barrierWmmaThresholds_.end() &&
                wmmaIssuedCountThisRegion_ >= thIt->second) {
                promotedPhase_ = PromotePhase::Barrier;
                promotedNode_ = node;
                PASS_DEBUG(std::cerr << "[CDNA5 decidePromote] promote barrier=" << node->inst
                                     << " wmmaIssuedCountThisRegion_=" << wmmaIssuedCountThisRegion_
                                     << " threshold=" << thIt->second << "\n");
                return;
            }
        }
    }

    // Hazard hoist: fires once the live elapse clock reaches this producer's
    // hazardDeadline, or earlier if a pending WMMA would otherwise jump clock_ past
    // it first (a WMMA can otherwise steal the slot right before the deadline).
    // Deliberately clock_-based, not a proxy node's readiness: clock_ only advances
    // via cycles actually issued (advanceTime, called on every real pick), so it can't
    // run ahead of the true schedule the way "some node's inDegree hit 0" can when
    // that node is structurally unblocked long before it is actually picked. Before
    // the deadline, the producer competes as ordinary work -- no special priority --
    // so it never crowds out ds/wmma while it still has slack. Also requires the
    // producer to be genuinely free to issue right now (no outstanding RAW/hazard
    // wait of its own, and no WMMA-src overlap) -- this is a throughput heuristic
    // layered on an unconditionally-correct consumer-side gate, so it must never
    // force an otherwise-unsafe issue; missing the deadline just costs a later
    // explicit stall via that gate.
    for (const HazardHoistCandidate& hc : hazardHoistCandidates_) {
        bool deadlineReached = clock_ >= hc.node->hazardDeadline;
        if (!deadlineReached && !wmmaQueue.empty()) {
            auto [bestWMMA, bestLatency] = findMostReadyWMMA();
            if (bestWMMA && bestLatency <= 0 &&
                clock_ + bestWMMA->inst->latencyCycles > hc.node->hazardDeadline) {
                deadlineReached = true;
            }
        }
        if (!deadlineReached) continue;
        if (getMaxSrcDataWait(hc.node) > 0 || getHazardWait(hc.node) > 0) continue;
        if (destOverlapsActiveWmmaSrc(hc.node)) continue;
        promotedPhase_ = PromotePhase::NonWmmaFill;
        promotedNode_ = hc.node;
        promotedKind_ = hc.kind;
        return;
    }
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

// WMMA windows needed to issue dsLoadCount ds_reads given the per-window DS cap. When the
// count exceeds the DS read queue depth, the queue is saturated and the extra ds_reads are
// paced at dsReadThrottleLatency/dsReadQueueDepth cycles each, so add enough extra windows
// to cover that pacing cost.
int CDNA5ReadyQueue::computeWmmaWindowsNeeded(int dsLoadCount) const {
    const int maxDsPerWmmaWindow = dsReadPerWmma();
    int wmmaWindowsNeeded = (dsLoadCount + maxDsPerWmmaWindow - 1) / maxDsPerWmmaWindow;
    // Depth 0 means the arch has no modeled LDS return queue, so there is no drain
    // to account for. Guarding here matches onInit(), which already skips the same
    // ratio when the depth is zero.
    if (dsReadQueueDepth() > 0 && dsLoadCount > dsReadQueueDepth()) {
        const float cyclePerDs = (float)dsReadThrottleLatency() / (float)dsReadQueueDepth();
        const float cyclesNeeded = cyclePerDs * (dsLoadCount - dsReadQueueDepth());
        const float baseWindows =
            (float)(dsReadQueueDepth() + maxDsPerWmmaWindow - 1) / (float)maxDsPerWmmaWindow;
        const float latencyWindows = cyclesNeeded / (float)wmmaIssueConfig.latency;
        wmmaWindowsNeeded = (int)std::ceil(baseWindows + latencyWindows);
    }
    return wmmaWindowsNeeded;
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
//            latency = dsReadDrainLatency when it is configured (> 0), else
//            computeDynamicDrainLatency(hw, matchingDsLoadCount, targetDSLoadLatency, numWaves).
std::unordered_map<StinkyInstruction*, CDNA5ReadyQueue::BarrierAfterOutput>
CDNA5ReadyQueue::computeBarrierAfterThresholds(IRList::iterator regionStart,
                                               IRList::iterator regionEnd) {
    std::unordered_map<StinkyInstruction*, BarrierAfterOutput> result;
    struct BarrierAfterSummary {
        StinkyInstruction* barrierKey;
        std::vector<StinkyInstruction*> barriers;
        int afterThreshold;
        int lastOverlap;
        int wmmaWindowsNeeded;
    };

    // Step 1a: collect all movable barriers with their PSEUDO src token sets, then merge
    //          signal/wait pairs so both halves share one threshold.
    auto barrierGroups =
        groupBarrierTokens(collectBarrierTokens(regionStart, regionEnd, /*useSrc=*/true));

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
        // A positive dsReadDrainLatency pins the latency. A non-positive value
        // (default 0) means "use dynamic drain latency," derived from the matching
        // ds_load count and the latest matching ds_read latency by the HWModel helper,
        // keyed by this pass context's NumWaves.
        const int configuredDrainLatency = dsReadDrainLatency();
        const int numWaves = static_cast<int>(getPassContext().getGemmTileConfig().NumWaves);
        const int latencyForAfterThreshold =
            configuredDrainLatency > 0
                ? configuredDrainLatency
                : computeDynamicDrainLatency(hw_, matchingDsLoadCount, (int)targetDSLoadLatency,
                                             numWaves);
        const int latencyWmmaBudget = (latencyForAfterThreshold / wmmaIssueConfig.latency) + 1;
        const int wmmaWindowsNeeded = computeWmmaWindowsNeeded(matchingDsLoadCount);
        const int overlapOrWindowBase = std::max(lastOverlap, wmmaWindowsNeeded);
        int afterThreshold = overlapOrWindowBase + latencyWmmaBudget;
        for (StinkyInstruction* barrier : group.barriers) {
            barrierWmmaThresholds_[barrier] = afterThreshold;
            result[barrier] = {afterThreshold, wmmaWindowsNeeded};
        }
        overlapChecks.push_back(
            {groupBarrier, group.barriers, afterThreshold, lastOverlap, wmmaWindowsNeeded});
        PASS_DEBUG(std::cerr << "[CDNA5 computeBarrierAfterThresholds] barrier=" << groupBarrier
                             << " barrierGroupSize=" << group.barriers.size() << " afterThreshold="
                             << afterThreshold << " matchingDsLoadCount=" << matchingDsLoadCount
                             << " latencyWmmaBudget=" << latencyWmmaBudget << " wmmaWindowsNeeded="
                             << wmmaWindowsNeeded << " overlapOrWindowBase=" << overlapOrWindowBase
                             << " latencyForAfterThreshold=" << latencyForAfterThreshold
                             << " lastOverlap=" << lastOverlap << "\n");
    }

    // Step 5: each group's interval is [afterThreshold - wmmaWindowsNeeded, afterThreshold).
    // Process groups in ascending afterThreshold order so that a "front" (earlier) barrier is
    // always fully resolved before the "later" barriers that it pushes back. A stable_sort
    // keeps groups with equal afterThreshold in their original appearance order, which gives
    // the tie-break: the earlier-appearing group is treated as the front (earlier) one.
    // After sorting, for any pair i < j we have afterThreshold[i] <= afterThreshold[j], so j is
    // always the later barrier and i the earlier one:
    //   - the later barrier (larger afterThreshold) is pushed back until its interval start
    //     clears the overlapping earlier barrier's end (== that barrier's afterThreshold):
    //     newAfterThreshold = maxEarlierAfterThreshold + wmmaWindowsNeeded.
    //   - the earlier barrier (smaller afterThreshold) keeps the prior behavior of extending
    //     afterThreshold by the shared overlap length.
    // Results are capped at issuedCount.
    std::stable_sort(overlapChecks.begin(), overlapChecks.end(),
                     [](const BarrierAfterSummary& a, const BarrierAfterSummary& b) {
                         return a.afterThreshold < b.afterThreshold;
                     });
    const size_t n = overlapChecks.size();
    // pushedStart[i] starts at the barrier's own interval start; overlapping earlier barriers
    // push it up to their end so (pushedStart + wmmaWindowsNeeded) clears the overlap.
    std::vector<int> pushedStart(n);
    std::vector<int> frontOverlapBudget(n, 0);  // shared length with overlapping later barriers
    for (size_t i = 0; i < n; ++i)
        pushedStart[i] = overlapChecks[i].afterThreshold - overlapChecks[i].wmmaWindowsNeeded;

    for (size_t i = 0; i < n; ++i) {
        const int endI = overlapChecks[i].afterThreshold;
        for (size_t j = i + 1; j < n; ++j) {
            const int endJ = overlapChecks[j].afterThreshold;
            const int overlapLen = std::min(endI, endJ) - std::max(pushedStart[i], pushedStart[j]);
            if (overlapLen <= 0) continue;

            // Sorted ascending, so j is the later barrier and i the earlier (front) one.
            pushedStart[j] = std::max(pushedStart[j], overlapChecks[i].afterThreshold);
            frontOverlapBudget[i] += overlapLen;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        const BarrierAfterSummary& summary = overlapChecks[i];
        const int adjustedAfterThreshold =
            std::min((int)wmmaIssueConfig.issuedCount,
                     pushedStart[i] + summary.wmmaWindowsNeeded + frontOverlapBudget[i]);

        const int shift = adjustedAfterThreshold - summary.afterThreshold;
        for (StinkyInstruction* barrier : summary.barriers) {
            barrierWmmaThresholds_[barrier] = adjustedAfterThreshold;
            result[barrier] = {adjustedAfterThreshold, shift};
        }
        PASS_DEBUG(
            std::cerr << "[CDNA5 computeBarrierAfterThresholds overlap] barrier="
                      << summary.barrierKey << " barrierGroupSize=" << summary.barriers.size()
                      << " baseAfterThreshold=" << summary.afterThreshold
                      << " adjustedAfterThreshold=" << adjustedAfterThreshold << " shift=" << shift
                      << " intervalStart=" << (summary.afterThreshold - summary.wmmaWindowsNeeded)
                      << " intervalEnd=" << summary.afterThreshold << " wmmaWindowsNeeded="
                      << summary.wmmaWindowsNeeded << " pushedStart=" << pushedStart[i]
                      << " frontOverlapBudget=" << frontOverlapBudget[i]
                      << " lastOverlap=" << summary.lastOverlap << "\n");
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
    struct BarrierBeforeSummary {
        StinkyInstruction* barrierKey;
        std::vector<StinkyInstruction*> barriers;
        int beforeThreshold;
        int wmmaWindowsNeeded;
    };
    std::vector<BarrierBeforeSummary> overlapChecks;

    auto barrierGroups =
        groupBarrierTokens(collectBarrierTokens(regionStart, regionEnd, /*useSrc=*/false));

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
        const int wmmaWindowsNeeded = computeWmmaWindowsNeeded(dsLoadCount);
        // WMMA issue count that forces the barrier early enough for all dependent ds_reads.
        // Take the latest of three constraints, then subtract from total WMMAs in the region:
        //   beforeN — remaining latency after the last consumer WMMA
        //   maxFinalWmmaIdx — absolute cap after the 1st ds_load (DS load latency / WMMA latency)
        //   wmmaWindowsNeeded — DS issue bandwidth (enough WMMA windows for all ds_loads); when
        //       dsLoadCount > dsReadQueueDepth(), add extra drain windows from
        //       dsReadThrottleLatency.
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

    // Step 5: each group's interval is [beforeThreshold, beforeThreshold + wmmaWindowsNeeded).
    // Mirror of computeBarrierAfterThresholds but walking back-to-front: scan from the last
    // barrier group backward, comparing each group only with the groups before it. For an
    // overlapping pair, decide the back barrier by beforeThreshold (smaller = earlier on the
    // WMMA axis); on a tie the later-appearing group (larger index) is the back one:
    //   - the back barrier (smaller beforeThreshold) is pulled earlier by its own window until
    //     its interval end clears the overlapping front barrier's start (== that barrier's
    //     beforeThreshold): newBeforeThreshold = frontBeforeThreshold - wmmaWindowsNeeded.
    //   - the front barrier (larger beforeThreshold) keeps the prior behavior of pulling its
    //     beforeThreshold earlier by the shared overlap length.
    // Results are clamped to [0, issuedCount].
    const size_t n = overlapChecks.size();
    // pulledEnd[i] starts at the barrier's own interval end; overlapping front barriers pull it
    // down to their start so (pulledEnd - wmmaWindowsNeeded) clears the overlap.
    std::vector<int> pulledEnd(n);
    std::vector<int> frontOverlapBudget(n, 0);  // shared length with overlapping back barriers
    for (size_t i = 0; i < n; ++i)
        pulledEnd[i] = overlapChecks[i].beforeThreshold + overlapChecks[i].wmmaWindowsNeeded;

    for (size_t i = n; i-- > 0;) {
        const int startI = overlapChecks[i].beforeThreshold;
        for (size_t j = i; j-- > 0;) {
            const int startJ = overlapChecks[j].beforeThreshold;
            const int overlapLen = std::min(pulledEnd[i], pulledEnd[j]) - std::max(startI, startJ);
            if (overlapLen <= 0) continue;

            // On a tie, the later-appearing group (i) is the back one, so j is the front.
            const size_t back = startJ < startI ? j : i;
            const size_t front = back == j ? i : j;
            pulledEnd[back] = std::min(pulledEnd[back], overlapChecks[front].beforeThreshold);
            frontOverlapBudget[front] += overlapLen;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        const BarrierBeforeSummary& summary = overlapChecks[i];
        const int adjustedBeforeThreshold =
            std::max(0, pulledEnd[i] - summary.wmmaWindowsNeeded - frontOverlapBudget[i]);
        const int adjustedWmmaWindowsNeeded =
            std::max(0, std::min((int)wmmaIssueConfig.issuedCount - adjustedBeforeThreshold,
                                 summary.wmmaWindowsNeeded + frontOverlapBudget[i]));
        for (StinkyInstruction* barrier : summary.barriers)
            result[barrier] = {adjustedBeforeThreshold, adjustedWmmaWindowsNeeded};
        PASS_DEBUG(std::cerr << "[CDNA5 computeBarrierBeforeThresholds overlap] barrier="
                             << summary.barrierKey
                             << " barrierGroupSize=" << summary.barriers.size()
                             << " baseBeforeThreshold=" << summary.beforeThreshold
                             << " adjustedBeforeThreshold=" << adjustedBeforeThreshold
                             << " dsLoadCount=" << barrierDsLoadCounts_[summary.barrierKey]
                             << " baseWmmaWindowsNeeded=" << summary.wmmaWindowsNeeded
                             << " adjustedWmmaWindowsNeeded=" << adjustedWmmaWindowsNeeded
                             << " pulledEnd=" << pulledEnd[i]
                             << " frontOverlapBudget=" << frontOverlapBudget[i] << "\n");
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

    // Promotion decision (formerly the forced-barrier phase). Records which phase must
    // fire now; the non-promoted phases below gate themselves off via isPromote(), and
    // the promoted node issues through its own phase — one entry point, no side effects
    // here.
    decidePromote();

    // Pre-compute the best DS read by dsReadPriority once for all phases.
    DAGNode* pickedDS = nullptr;
    for (DAGNode* n : localReadQueue) {
        if (!pickedDS || n->dsReadPriority < pickedDS->dsReadPriority) pickedDS = n;
    }

    // Phase B — try WMMA if all gates pass.
    bool otherQueuesHaveWork = !globalReadQueue.empty() || !localReadQueue.empty() ||
                               !otherQueue.empty() || !valuQueue.empty();

    if (isPromote(PromotePhase::Wmma) && !wmmaQueue.empty()) {
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
        // Hold a dependent bestWMMA while non-WMMA work remains; else emit shortfall as v_nops.
        bool blockWmmaForCoexecSpacing = false;
        if (activeWmmaNode_ != nullptr && hasPickableNonWmma &&
            wmmaToWmmaCoexecOverlap(*activeWmmaNode_->inst, *bestWMMA->inst)) {
            const int slotsPlusOne = popcount16(activeWmmaNode_->inst->coIssueWindow) + 1;
            blockWmmaForCoexecSpacing = nonWmmaFillsSinceActiveWmma_ < slotsPlusOne;
        }
        PASS_DEBUG(std::cerr << "[CDNA5 pickOne] Phase B candidate wmmaId=" << bestWMMA->id
                             << " bestLatency=" << bestLatency
                             << " blockLoopHead=" << blockWmmaForLoopHeadBalance
                             << " blockActiveWindow=" << blockWmmaForActiveWindow
                             << " blockAtLeastOneNonWmmaInterleaving="
                             << blockWmmaForAtLeastOneNonWmmaInterleaving
                             << " blockCoexecSpacing=" << blockWmmaForCoexecSpacing
                             << " fills=" << nonWmmaFillsSinceActiveWmma_
                             << " localReadQ=" << localReadQueue.size() << " nonWmmaMinId="
                             << (smallestPickable ? std::to_string(smallestPickable->id)
                                                  : std::string("none"))
                             << "\n");
        if (bestLatency <= 0 && !blockWmmaForLoopHeadBalance && !blockWmmaForActiveWindow &&
            !blockWmmaForAtLeastOneNonWmmaInterleaving && !blockWmmaForCoexecSpacing) {
            DAGNode* node = pickOneFromWMMA(bestWMMA);
            PASS_DEBUG(std::cerr << "[CDNA5 pickOne] Phase B picked WMMA dagId=" << node->id
                                 << "\n");
            return rememberPick(node);
        }
    }

    // Phase C+D — NonWmmaFill: fill with non-WMMA work (inside then outside the window).
    if (isPromote(PromotePhase::NonWmmaFill)) {
        // Hazard-hoist promotion: decidePromote() found a flagged producer whose
        // hazardDeadline the live clock_ has reached, and confirmed it's genuinely
        // free to issue now. Issue it directly through this, its own existing phase —
        // no separate entry point — bypassing the normal findSmallestPickableNonWmma
        // selection (which would otherwise rank it as ordinary work and might not pick
        // it this cycle, missing the deadline).
        if (promotedPhase_ == PromotePhase::NonWmmaFill && promotedNode_) {
            PASS_DEBUG(std::cerr << "[CDNA5 pickOne] hazard-hoist promoted dagId="
                                 << promotedNode_->id << " kind=" << promotedKind_ << " deadline="
                                 << promotedNode_->hazardDeadline << " clock=" << clock_ << "\n");
            return rememberPick(popNonWmma(promotedNode_, promotedKind_));
        }

        // Phase C — inside WMMA latency window.
        if (coIssueCyclePos_ < activeWmmaLatency_) {
            DAGNode* smallestPickable = nullptr;
            int pickKind = -1;
            int pickWait = 0;
            if (findSmallestPickableNonWmma(pickedDS, &smallestPickable, &pickKind, &pickWait)) {
                PASS_DEBUG(std::cerr << "[CDNA5 pickOne] Phase C picked non-WMMA dagId="
                                     << smallestPickable->id << " kind=" << pickKind
                                     << " wait=" << pickWait << "\n");
                // Pay any hidden stall (hidden under the WMMA latency) before issuing.
                if (pickWait > 0) advanceTime(pickWait);
                return rememberPick(popNonWmma(smallestPickable, pickKind));
            }

            advanceTime(activeWmmaLatency_ - coIssueCyclePos_);
        }

        // Phase D — outside WMMA latency.
        DAGNode* smallestPickable = nullptr;
        int pickKind = -1;
        int pickWait = 0;
        if (findSmallestPickableNonWmma(pickedDS, &smallestPickable, &pickKind, &pickWait)) {
            // No latency shadow here, so pickWait is 0; advance kept for safety.
            if (pickWait > 0) advanceTime(pickWait);
            return rememberPick(popNonWmma(smallestPickable, pickKind));
        }
    }

    // Phase E — forced WMMA: pick the most-ready WMMA before barriers.
    if (isPromote(PromotePhase::ForcedWmma) && !wmmaQueue.empty()) {
        auto [bestWMMA, bestLatency] = findMostReadyWMMA();
        (void)bestLatency;
        DAGNode* node = pickOneFromWMMA(bestWMMA);
        return rememberPick(node);
    }

    // Barrier phase — the single entry point for issuing a barrier. When a barrier was
    // promoted (threshold met), the phases above gated off and we arrive here to issue
    // that exact node (formerly the forced-barrier phase ahead of WMMA). Otherwise this
    // is the drain case: issue the lowest-id barrier once all compute has been picked.
    if (isPromote(PromotePhase::Barrier) && !barrierQueue.empty()) {
        DAGNode* barrier;
        if (promotedPhase_ == PromotePhase::Barrier) {
            barrier = extractForcedBarrier();  // removes the promoted (threshold-met) node
        } else {
            barrier = barrierQueue.top();
            barrierQueue.pop();
        }
        updateWMMAStatus(barrier);
        PASS_DEBUG(std::cerr << "[DAG CDNA5 pickOne] barrier dagId=" << barrier->id
                             << " promoted=" << (promotedPhase_ == PromotePhase::Barrier) << "\n";
                   barrier->inst->dump(std::cerr); std::cerr << "\n");
        return rememberPick(barrier);
    }

    // Phase G — final safety net: force-pick the least-blocked ready node to
    // guarantee progress.
    DAGNode* fallback = nullptr;
    int fallbackKind = -1;
    int fallbackWait = 0;
    if (findOldestFallbackNonWmma(pickedDS, &fallback, &fallbackKind, &fallbackWait)) {
        // Throttle (queue depth) is skipped for progress, but the hazard gate is
        // unconditional (see config_.hazardRules) and still has to be paid here too.
        int waitCycles = fallbackWait;
        if (fallbackKind == kGlobalRead && globalReadQueueFull())
            waitCycles = std::max(waitCycles, globalReadInflight_.minResidual());
        if (fallbackKind == kLocalRead) waitCycles = std::max(waitCycles, dsReadThrottleWait());
        if (waitCycles > 0) advanceTime(waitCycles);
        PASS_DEBUG(std::cerr << "[CDNA5 pickOne] Phase G fallback pick dagId=" << fallback->id
                             << " kind=" << fallbackKind << " wait=" << waitCycles << "\n");
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
        if (!node->hazardFlags.empty()) hazardHoistCandidates_.push_back({node, kValu});
        return;
    }

    if (isBarrier(*node->inst)) {
        barrierQueue.push(node);
        return;
    }

    otherQueue.push(node);
    if (!node->hazardFlags.empty()) hazardHoistCandidates_.push_back({node, kOther});
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
    nonWmmaFillsSinceActiveWmma_ = 0;
    globalReadInflight_ = InFlightQueue(globalReadQueueDepth());
    dsReadInflight_ = InFlightQueue(dsReadQueueDepth());
    const int dsDepth = dsReadQueueDepth();
    const double dsThrottleInterval =
        dsDepth > 0 ? (double)dsReadThrottleLatency() / (double)dsDepth : 0.0;
    dsReadInflight_.setThrottleInterval(dsThrottleInterval);

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
    // Per-region: MSB state is not carried across a region boundary (side-effect cut).
    currentMsb_ = -1;
    // Clear per-region node ptr; it dangles into the previous region's freed DAGNodeList.
    activeWmmaNode_ = nullptr;
    nonWmmaFillsSinceActiveWmma_ = 0;
    // Hazard state is per-region: hazardHoistCandidates_ holds DAGNode* into the prior
    // region's freed DAGNodeList, and any in-flight gate window is already locked into
    // the prior region's fixed instruction order (region boundaries are side-effect
    // cuts — no reordering crosses them), so it has nothing left to gate here.
    hazardHoistCandidates_.clear();
    for (auto& gate : hazardGates_) gate.clear();
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

    // Flat fill (one entry per window); a later commit computes per-window targets.
    dsTargetPerWindow_.assign(wmmaIssueConfig.issuedCount + 1, dsReadPerWmma());

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
            auto barrierGroups =
                groupBarrierTokens(collectBarrierTokens(regionStart, regionEnd, useSrcTokens));
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
