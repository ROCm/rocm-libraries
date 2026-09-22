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
// StinkyDAGSchedulerPass splits each basic block into regions at non-movable
// side effects (waits, stores, branches, etc.), builds a per-region dependency
// DAG from physical registers, then drains ready nodes via this queue. CDNA5
// models the WMMA–VALU co-issue timeline: WMMA issues in 1 cycle; VALU is only
// gated by the co-issue window. Memory ops and SALU use independent pipelines.
//
#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "InFlightQueue.hpp"
#include "ReadyQueue.hpp"
#include "stinkytofu/analysis/asm/WmmaHideBudgetAnalysis.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"
#include "stinkytofu/transforms/asm/dag/HazardRules.hpp"

namespace {
using namespace stinkytofu;
using namespace stinkytofu::dag;

enum NonWmmaKind { kGlobalRead = 0, kLocalRead, kOther, kValu };

// -------------------------------------------------------------------------
// Hardware hazard rules: a fixed cycle gap required between a producer writing
// a register and a specific class of consumer reading it as a source (not
// either op's own issue/latency cycles — a producer->consumer edge gap).
// Data-driven so a new hazard pair is a new table row, not new code.
// Consumer-side correctness (the gate in CDNA5ReadyQueue::hazardGates_) is
// unconditional regardless of scheduling order; producer-side hoisting
// (CDNA5ReadyQueue::decidePromote(), via DAGNode::hazardDeadline) is a
// throughput heuristic layered on top, never required for correctness.
//
// HazardRule itself, and the family-wide kCdna5HazardRules table, live in
// HazardRules.hpp (included above) so HazardGapAnalysisPass can share them
// without duplicating this logic.
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// Per-arch CDNA5 scheduling POLICY. CDNA5.hpp is the CDNA5 *family* ready
// queue: both gfx1250 and gfx1250v0 compile against it, so these are selected
// per arch here rather than baked in as single family-wide constants. A new
// CDNA5-family arch adds one case to cdna5ConfigForArch(). User
// PassFeatureConfig overrides still win over these per-arch defaults (see the
// dsReadPerWmma accessor).
//
// Only the scheduling *ratios* live here. The physical facts this queue also
// needs - LDS queue depth, drain/throttle latency, and the hazard-rule table -
// are hardware, not policy, and live in HWModel
// (stinkytofu/hardware/HWModel.hpp), reached through passCtx.getHWModel().
// Keeping them there is what lets other passes see the same numbers instead of
// each redeclaring them.
// -------------------------------------------------------------------------
struct CDNA5Config {
    // Used when dagFeatures still hold the PassFeatureConfig INT_MAX sentinel;
    // explicit non-sentinel user config wins over these.
    int dsReadPerWmma;
    int globalReadPerWmma;
    int tensorLoadWmmaSpace;
    // Cycle span the dsReadPerWmma ceiling applies over. Paired with
    // dsReadPerWmma: the two together are the cap, and neither means anything
    // without the other.
    //
    // Arch data, deliberately NOT read off a matrix op in the region. The
    // ceiling protects the LDS return queue, so it has to be defined where no
    // WMMA is -- in a region's tail and in a region that has none at all. A
    // per-region scan gives 0 in exactly those cases.
    int dsIssueCapSpanCycles;
};

constexpr CDNA5Config kGfx1250Config = {
    /*dsReadPerWmma=*/3,
    /*globalReadPerWmma=*/1,
    /*tensorLoadWmmaSpace=*/0,
    // v_wmma_* is .cost = {1, 8} (Gfx1250Formats.def): issue 1, latency 8, so
    // the co-issue window is 7 cycles. 8 here is the same number the old
    // per-WMMA window had, which is what keeps the issue RATE unchanged while
    // the ceiling's SCOPE widens.
    /*dsIssueCapSpanCycles=*/8,
};

// gfx1250v0: starts from the gfx1250 values. TODO(tuning): fill in gfx1250v0's
// real per-WMMA ratios. Kept as its own case so those numbers can be changed
// here without touching gfx1250. Its physical facts are likewise a separate
// HWModel entry.
constexpr CDNA5Config kGfx1250v0Config = kGfx1250Config;

// Select the CDNA5-family policy for \p arch. Private to the ready queue /
// scheduler (not shared infrastructure): each CDNA5 arch's knobs live next to
// the family model that consumes them. gfx1250 is the default for any unlisted
// arch (the pipeline only runs the CDNA5 ready queue on CDNA5-family archs).
//
// Keyed off the shared kArchKey* constants (HWModel.hpp) so this policy table
// and the HWModel fact table cannot be restepped independently.
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

// Register-file-aware key for the data-ready (RAW) and elapse-touch maps. These
// maps were keyed on reg.idx alone, which conflates register files: e.g. a WMMA
// writing its accumulator v[12:20) would stamp indices 12..19 and falsely gate
// a later SALU that reads s14/s15 (same indices, different file). Fold the
// register type into the key so vector, scalar, and accumulator registers of
// the same index never collide.
static inline int regDepKey(RegType type, uint32_t idx) {
    return (static_cast<int>(type) << 20) | static_cast<int>(idx & 0xFFFFF);
}

// Scheduling rule (2): simulate producer completion over [blockBegin,
// regionStart) — outstanding data-ready latencies decrease by each
// instruction's issueCycles; each producer overwrites its dest VGPRs with that
// op's latencyCycles. Remaining counts seed regDataReadyCounters so the first
// WMMA/consumer in a region sees preloop / in-BB producers the register DAG may
// not edge to (double-buffer: WMMA on X0 while in-loop ds fills X1).
// Generalized from ds_load-only to all producers, with the same latencyCycles >
// issueCycles self-clear skip used at issue time; short-latency ops decay away
// across the prefix, only long ones (ds_load) persist. Caller: onInitRegion.
//
// crossBBResiduals: data-ready residuals from predecessor BBs (merged in
// onInit). They decay through the prefix alongside within-BB producers so
// cross-BB loads properly gate consumers.
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

// Scheduling rule (5) helper: walk backward from the end of a BB, skipping
// LABEL and branch ops; true if the first real instruction found is WMMA/SWMMA.
// Used to detect “tail WMMA” in the latch BB of a loop (cross-BB aware).
static bool latchBBTailIsWmma(BasicBlock& latchBB) {
    if (latchBB.empty()) return false;
    // Walk from tail toward head. Do not use std::prev(end()) on
    // IRList::iterator: end() is nullptr and IntrusiveListIterator::operator--
    // does not step to tail_.
    for (auto it = latchBB.rbegin(); it != latchBB.rend(); ++it) {
        auto* instPtr = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!instPtr) continue;
        if (isLabel(*instPtr) || isBranch(*instPtr) || isCall(*instPtr)) continue;
        return isMatrixInstruction(*instPtr);
    }
    return false;
}

// Rule (5) — loop tail WMMA / head WMMA deferral (cross-BB aware via
// LoopDetection).
//
// Uses the Loop* from setLoopContext() to detect whether the loop’s latch BB
// ends with WMMA before its back-edge branch. If so, the header BB’s first WMMA
// should be deferred to avoid back-to-back WMMA across iterations.
//
// When the loop is split across multiple BBs (e.g. unrolled loops), the latch
// BB (containing the back-edge branch) may be different from the header BB.
//
// Steps (unchanged from the old pipeline, but loop detection is now cross-BB):
//   Step 1 — onInit: check latchBB tail for WMMA via latchBBTailIsWmma().
//   Step 2 — onInitRegion: deferHeadBalanceThisRegion_ if this BB is the loop
//   header. Step 3 — pickOne Phase B: block WMMA while non-WMMA queues have
//   work. Step 4 — pickOneFromWMMA / popNonWmmaByKind: clear deferral after
//   first pick.

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

// WMMA D vs a co-executable VALU consumer: RAW (D->src), WAW (D->dst), WAR
// (prod A/B or SWMMAC index -> cons dst).
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

// Collect all movable barriers in [regionStart, regionEnd) with their PSEUDO
// token sets. useSrc: true → collect from getSrcRegs(), false → collect from
// getDestRegs().
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

// A run of barriers that share the same PSEUDO token set.
// barrier_signal/barrier_wait pairs (adjacent in the IR, same tokens) are
// merged so both get the same threshold.
struct BarrierTokenGroup {
    std::vector<StinkyInstruction*> barriers;
    std::unordered_set<uint32_t> tokens;
    IRList::iterator firstIt;
    IRList::iterator lastIt;
};

// Merge consecutive same-token barriers into groups of up to two (a signal/wait
// pair). Two barriers only pair when they carry identical tokens and are
// adjacent in the IR.
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
//  (4) Per-WMMA-window DS cap — dagFeatures.dsReadPerWmma ds_loads per WMMA
//  window
//      (INT_MAX = unconstrained).
//  (5) Loop tail vs head — defer first WMMA in the loop header BB until
//      non-WMMA queues drain once. Cross-BB via LoopDetection.
//
class CDNA5ReadyQueue : public ReadyQueue {
    // Which phase, if any, decidePromote() forces this pick (None = normal
    // selection). Values name the gateable phases of pickOne(); isPromote(p) is
    // true when nothing is promoted (every phase runs) or when p is the promoted
    // phase (only it runs).
    enum class PromotePhase { None, Wmma, NonWmmaFill, ForcedWmma, Barrier };

    // --- Priority buckets (DAG ids compare smaller = earlier in source) ---
    ReadySetByDAGid wmmaQueue;
    ReadySetByDAGid globalReadQueue;  // tensor_load_to_lds when distributeGlobalRead
    ReadySetByDAGid localReadQueue;   // ds_load
    ReadySetByDAGid valuQueue;        // VALU and transcendental instructions
    ReadySetByDAGid barrierQueue;
    ReadySetByDAGid otherQueue;  // scalars, waits in region, etc.

    // Per-arch CDNA5 scheduling policy (the per-WMMA ratios), selected by arch in
    // the constructor. Points at a static constexpr CDNA5Config, so this is a
    // stable ref.
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
    int crossBBDsReadCount_ = 0;
    int crossBBDsReadResidual_ = 0;

    // Rule (4) ds_load issue cap (dagFeatures.dsReadPerWmma), as a sliding
    // window on the real timeline: depth = the ceiling N, entry lifetime = the
    // window span, so full() means "N already issued within the last span".
    //
    // Deliberately NOT anchored to a WMMA issue. The old counter reset when the
    // next WMMA issued, which left the ceiling undefined once none remained (the
    // region tail) and let 2N issue back-to-back across a reset. A sliding
    // window is defined everywhere and has no boundary to forget at.
    //
    // It stays a CAP: N may issue back-to-back while the window has room, and a
    // busy in-flight queue places fewer, so windows stay unevenly filled.
    InFlightQueue dsIssueCap_;

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
    // Barrier-timing only (computeBarrierAfterThresholds): the cycles a barrier
    // must wait for its dependent ds_reads to return. 0 means "derive dynamically
    // from the matching ds_read count and target ds_read latency." Queue
    // occupancy / pacing uses dsReadThrottleLatency, not this.
    int dsReadDrainLatency() const {
        const int cfg = getPassContext().getPassFeatureConfig().dagFeatures.dsReadDrainLatency;
        return cfg > 0 ? cfg : hw_.lds.readDrainLatency;
    }
    // Lifetime of one in-flight ds_read credit; also sets the saturated-queue
    // issue interval (dsReadThrottleLatency/dsReadQueueDepth cycles per ds_read).
    int dsReadThrottleLatency() const {
        const int cfg = getPassContext().getPassFeatureConfig().dagFeatures.dsReadThrottleLatency;
        if (cfg > 0) return cfg;
        if (hw_.lds.readThrottleLatency > 0) return hw_.lds.readThrottleLatency;
        return 4 * dsReadQueueDepth();
    }
    double dsReadThrottleTransitionFactor() const {
        const double cfg =
            getPassContext().getPassFeatureConfig().dagFeatures.dsReadThrottleTransitionFactor;
        return std::clamp(cfg, 0.0, 1.0);
    }
    int dsReadThrottleTransitionEntries() const {
        const int cfg =
            getPassContext().getPassFeatureConfig().dagFeatures.dsReadThrottleTransitionEntries;
        return cfg >= 0 ? cfg : dsReadQueueDepth();
    }
    // Effective ds issue cost for one wave. The ISA number is single-wave; the
    // ds issue pipe is shared, so resident waves round-robin it and one wave's
    // issues are spaced out (see HWModel::Lds::wavesPerDsIssuePipe). Distinct
    // from dsReadPerWmma below, which is a manually tuned ceiling, not a cost.
    int dsIssueCost(const StinkyInstruction& inst) const {
        return dsIssueCyclesForWaves(
            hw_, inst.issueCycles,
            static_cast<int>(getPassContext().getGemmTileConfig().NumWaves));
    }
    int dsReadPerWmma() const {
        const int cfg = getPassContext().getPassFeatureConfig().dagFeatures.dsReadPerWmma;
        // INT_MAX is the "unset" sentinel and resolves to the arch default. A
        // non-positive value is not a sentinel and is not a cap anyone can mean:
        // it used to fall through to the arch default silently, so a caller that
        // asked for 0 got 3. Reject it rather than guess.
        if (cfg <= 0) {
            report_fatal_error(
                "dagFeatures.dsReadPerWmma must be positive (or INT_MAX to take the arch "
                "default); got " +
                std::to_string(cfg) + ".");
        }
        const int resolved = cfg < INT_MAX ? cfg : config_.dsReadPerWmma;
        // The arch default is static data, so a bad one is a build-time mistake
        // in this file rather than a caller error.
        assert(resolved > 0 && "arch config dsReadPerWmma must be positive");
        return resolved;
    }
    int tensorLoadWmmaSpace() const {
        const int cfg = getPassContext().getPassFeatureConfig().dagFeatures.tensorLoadWmmaSpace;
        return cfg > 0 ? cfg : config_.tensorLoadWmmaSpace;
    }
    // Whether to run the per-window hide-budget policy at the top of each region.
    // The gfx1250 production backend enables it; the standalone pass keeps an
    // explicit flag so tests and custom pipelines can opt in.
    bool hideBudgetPrescanEnabled() const {
        return getPassContext().getPassFeatureConfig().dagFeatures.enableWmmaHideBudgetPrescan;
    }
    bool dsReadQueueFull() const {
        return dsReadInflight_.full();
    }
    // Span of the rule (4) cap window, in cycles: at most dsReadPerWmma
    // ds_loads may issue in any dsIssueCapSpan() cycles of the real timeline.
    //
    // Arch data (config_), not a property of any matrix op in the region. The
    // ceiling exists to stop the LDS return queue being overrun, which is true
    // whether or not a WMMA is in flight, so the span has to be defined where
    // none is -- the region tail, and a region containing no matrix op at all.
    //
    // This deliberately does NOT read wmmaIssueConfig.latency, which was the
    // obvious source and is wrong three ways:
    //   - it is the FIRST matrix op's latency in the region (see the scan in
    //     onInit), while latencyCycles is overridden per matrix format pair via
    //     HwInstDesc::matrixFmtCostOverrides -- so a region mixing formats
    //     would size every window from whichever WMMA happened to come first;
    //   - it is 0 in a region with no matrix op, and
    //   - it is 0 for the whole region when loopConfig.unrollGemm is off, since
    //     the scan sits below that early return.
    // The last two are silent: they do not disable the cap, they shrink its
    // window to nothing, which would make the ceiling bind far harder in
    // exactly the regions it was never measured in.
    //
    // The arch value reproduces today's per-WMMA window length, so widening the
    // ceiling's scope does not also change its rate. It is a rate-preserving
    // choice, not a derived one: the queue-derived span is
    // dsReadPerWmma * dsReadThrottleLatency / dsReadQueueDepth, which puts the
    // cap's long-run rate exactly at what the LDS queue sustains. Measured,
    // that is ~2x tighter than today's and starves the scheduler of the
    // ds_loads it uses to fill hazard gaps, so adopting it is a retune that
    // needs dsReadPerWmma re-tuned with it against hardware.
    //
    // dagFeatures.dsIssueCapSpanCycles overrides the arch value.
    int dsIssueCapSpan() const {
        const int cfg = getPassContext().getPassFeatureConfig().dagFeatures.dsIssueCapSpanCycles;
        const int resolved = cfg > 0 ? cfg : config_.dsIssueCapSpanCycles;
        // A non-positive span is not a window anyone can mean, and a 0 would
        // expire every entry immediately and silently disable rule (4).
        assert(resolved > 0 && "arch config dsIssueCapSpanCycles must be positive");
        return resolved;
    }
    int dsReadThrottleWait() const {
        return dsReadInflight_.throttleWait();
    }

    // --- VALU co-issue timeline tracker ---
    uint16_t activeCoIssueWindow_ = 0;
    int coIssueCyclePos_ = 0;
    int activeWmmaLatency_ = 0;
    // HwInstDesc::blockedScaleMask of the active matrix op: cycles of its window
    // that no instruction of any kind can be issued into, anchored at the window
    // END (bit 0 = last cycle). activeCoIssueWindow_ cannot express this -- it
    // only gates VALU, while a scale WMMA's LD_SCALE sub-issue blocks SALU and
    // memory picks just as much.
    uint16_t activeWmmaBlockedScale_ = 0;
    // WMMA that opened the current latency window (valid while coIssueCyclePos_ <
    // activeWmmaLatency_). Used to detect ds_load dest / WMMA src VGPR overlap
    // hazards.
    DAGNode* activeWmmaNode_ = nullptr;

    // Non-WMMA fills since the active WMMA opened its window. A dependent next
    // WMMA is held in Phase B until this reaches popcount(coIssueWindow)+1
    // (WMMA->WMMA coexec slots + 1).
    int nonWmmaFillsSinceActiveWmma_ = 0;
    // Region-wide actual and required cumulative non-WMMA issue counts.
    int nonWmmaIssuedThisRegion_ = 0;
    int cumulativeWmmaHideBudget_ = 0;
    int dsLoadIssuedThisRegion_ = 0;
    int cumulativeWmmaDsLoadBudget_ = 0;
    RegionHideBudget hideBudget_;

    // Filler v_nops to emit before the next pickOne() result (drained by
    // scheduleInDAG). Set at a dependent WMMA's issue point when no independent
    // VALU remained to fill the gap.
    int pendingFillerVNops_ = 0;

    // VGPR-MSB bank currently in effect (mirrors InsertVgprMsbPass); updated on
    // issue, reset per region. pickFreeBest prefers a free candidate matching it.
    // -1 = unknown.
    int currentMsb_ = -1;

    // --- Rule (4) ds_load cap (dagFeatures.dsReadPerWmma) ---
    // Enforced by dsIssueCap_ above; no per-window counter is kept.
    //
    // Diagnostic only (PASS_DEBUG): which constraint is actually binding when a
    // ds_load is available to pick. A cap that binds on nearly every sample is
    // behaving as an assignment -- it, not the queue, is choosing placement --
    // which is the thing to know before rebalancing the cap against the queue.
    // Sampled once per pickOne(), so the four counters partition those samples.
    int dsBindNeither_ = 0;   // free: could issue now
    int dsBindCapOnly_ = 0;   // only the [X,Y) cap says stop
    int dsBindQueueOnly_ = 0; // only the ds queue (full / throttled) says stop
    int dsBindBoth_ = 0;
    // Synthetic throttle cycles charged to DS placement in the current WMMA.
    // Kept separate from coIssueCyclePos_, the real hardware/hazard timeline.
    int dsSchedulingBudgetUsed_ = 0;

    // (A) RAW data-ready gate. Per reg index: remaining modeled latency until a
    // producer's result is safe to consume (e.g. ds_load LDS->VGPR, 56 cyc). Any
    // long-latency producer stamps it; a consumer whose src is still in flight is
    // not "free". Decays in advanceTime. Crosses BBs via
    // BBScheduleState.dsResiduals.
    std::map<int, int> regDataReadyCounters;

    // Hazard gates, one independent lane per config_.hazardRules entry. Per reg
    // key: remaining cycles until a rule.isConsumer instruction may read it
    // (stamped rule.cycles when a flagged producer issues). Kept SEPARATE per
    // rule (and separate from regDataReadyCounters) because each hazard is
    // consumer-type- and register-file-specific: e.g. a VALU/SALU reading the
    // same sgpr a flagged SALU just wrote is not gated by the SaluSgprToMemAddr
    // lane. Decays in advanceTime. Sized to config_.numHazardRules at
    // construction (runtime, since the rule count is per-arch), indexed by the
    // same ruleIdx the scheduler pre-scan assigns.
    std::vector<std::map<int, int>> hazardGates_;

    // Ready, flagged (non-empty hazardFlags), not-yet-issued hazard producers,
    // tracked so decidePromote() doesn't need to scan every queue each pick to
    // find the ones whose hazardDeadline might have arrived. Populated in push(),
    // erased once the node is picked (popNonWmma).
    struct HazardHoistCandidate {
        DAGNode* node;
        int kind;  // NonWmmaKind: which queue this producer sits in (kOther or
                   // kValu).
    };
    std::vector<HazardHoistCandidate> hazardHoistCandidates_;

    // (B) elapse-time ordering. Timeline (advanceTime clock) at which each reg
    // was last touched by any operand (dst or src) of an issued instruction. Used
    // to order already-free nodes within a bucket: prefer the node whose operands
    // were touched longest ago, so a register-overwrite (e.g. a VALU reusing a
    // just-read ds_load address) is naturally deferred behind other work.
    // Per-region; reset each region.
    std::map<int, int> regLastTouch_;
    int clock_ = 0;

    WMMAIssueConfig wmmaIssueConfig;

    bool hasWMMAInRegion_ = false;

    // --- Loop head balancing ---
    bool deferFirstHeadWmmaActive_ = false;
    bool deferHeadBalanceThisRegion_ = false;

    // Per-barrier forced-issue threshold: maps StinkyInstruction* -> N.
    std::unordered_map<StinkyInstruction*, int> barrierWmmaThresholds_;
    // Per-barrier matching ds_load count collected in
    // computeBarrierBeforeThresholds.
    std::unordered_map<StinkyInstruction*, int> barrierDsLoadCounts_;
    struct BarrierBeforeOutput {
        int beforeThreshold = 0;
        int baseBeforeThreshold = 0;
        int wmmaWindowsNeeded = 0;
        int dsLoadCount = 0;
    };
    struct BarrierAfterOutput {
        int afterThreshold = 0;
        int baseAfterThreshold = 0;
        // Drain latency expressed in WMMA-window units from Step 4
        // ((latency / wmmaIssueConfig.latency) + 1). Used by Layer 2 with
        // wmmaWindowsNeeded to form the unclamped after claim window.
        int latencyWmmaBudget = 0;
        int wmmaWindowsNeeded = 0;
        int dsLoadCount = 0;
    };

    int wmmaIssuedCountThisRegion_ = 0;

    BasicBlock* currentBB_ = nullptr;
    std::vector<Layer2BarrierOverlapCandidate> layer2BarrierOverlapCandidates_;

    DAGNode* lastPickedNode_ = nullptr;

    // Set by decidePromote() each pick: which phase is forced (None = normal
    // selection) and the exact node that phase will issue. Read via isPromote()
    // to gate the phases.
    PromotePhase promotedPhase_ = PromotePhase::None;
    DAGNode* promotedNode_ = nullptr;
    // Valid only when promotedPhase_ == PromotePhase::NonWmmaFill via the
    // hazard-hoist case: which queue (NonWmmaKind: kOther or kValu) promotedNode_
    // must be popped from.
    int promotedKind_ = -1;

    // Open SCC chain blocks handshake barriers (see cluster-barrier.md).
    unsigned openSccChain_ = 0;
    unsigned sccReadersLeft_ = 0;

    bool sccChainBlocks(const DAGNode* node) const {
        if (!clusterBarrierEnabled()) return false;
        return openSccChain_ != 0 && node->handshakeBarrier;
    }

    // True when decidePromote() / pickOne() may consider issuing this barrier
    // now. No-op when clusterBarrier is off (see sccChainBlocks()).
    bool isBarrierEligibleNow(const DAGNode* node) const {
        return !sccChainBlocks(node);
    }

    // kRule3CrossLoop false: no-op (earliestClock unset).
    bool heldBackForLead(const DAGNode* node) const {
        if (!clusterBarrierEnabled()) return false;
        return clock_ < node->earliestClock;
    }

    void noteSccChainIssue(DAGNode* node);

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

    // Promotion = split the old forced-barrier phase into a pure decision + a
    // per-phase gate, run once before the normal phases. decidePromote() records
    // which phase must fire now (promotedPhase_) and the exact node it will issue
    // (promotedNode_), mutating no queue. Each phase guards its body with
    // isPromote(ThisPhase): when nothing is promoted every phase runs normally;
    // when something is promoted only that phase runs, so the promoted node
    // issues through its own existing phase (one entry point), never a second
    // dedicated path. A new forcing rule is a new case in decidePromote(), not a
    // new phase. Two promotions: a barrier whose per-barrier WMMA-issued
    // threshold is met (formerly the forced-barrier phase), and a hazard-hoist
    // producer whose live clock_ has reached its hazardDeadline (forces the
    // producer to issue now, through its own NonWmmaFill phase, so it lands
    // before its hazarded consumer needs the gap instead of after).
    void decidePromote();
    bool isPromote(PromotePhase phase) const {
        return promotedPhase_ == PromotePhase::None || promotedPhase_ == phase;
    }
    DAGNode* extractForcedBarrier();
    std::unordered_map<StinkyInstruction*, BarrierAfterOutput> computeBarrierAfterThresholds(
        IRList::iterator regionStart, IRList::iterator regionEnd);
    std::unordered_map<StinkyInstruction*, BarrierBeforeOutput> computeBarrierBeforeThresholds(
        IRList::iterator regionStart, IRList::iterator regionEnd);
    DsLoadBudgetConfig dsLoadBudgetConfig() const;
    int computeWmmaWindowsNeeded(int dsLoadCount) const;
    bool isValuPickable() const;
    bool isBlockedCycle(int pos) const;
    int freeCoIssueSpace() const;
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
    std::vector<Layer2BarrierOverlapCandidate> takeLayer2BarrierOverlapCandidates() override {
        return std::exchange(layer2BarrierOverlapCandidates_, {});
    }
    bool empty() const override;

    // Build the coexec filler v_nops counted during the last pickOne() as
    // detached insts.
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
                      IRList::iterator blockBegin, const RegionDependencies& deps) override;

    void onFinishBB() override;
};

// True when \p pos -- cycles elapsed since the active matrix op issued -- lands
// on a cycle of its window the hardware occupies outright (LD_SCALE, on a scale
// WMMA). No instruction of any pipe can issue there. The mask is end-anchored,
// so bit 0 is the window's last cycle.
bool CDNA5ReadyQueue::isBlockedCycle(int pos) const {
    return isBlockedWindowCycle(pos, activeWmmaLatency_, activeWmmaBlockedScale_);
}

// Cycles of genuinely free latency shadow left in the active op's window: it
// ends at the window close, or at the first blocked cycle if one comes sooner.
int CDNA5ReadyQueue::freeCoIssueSpace() const {
    for (int pos = coIssueCyclePos_; pos < activeWmmaLatency_; ++pos)
        if (isBlockedCycle(pos)) return pos - coIssueCyclePos_;
    return activeWmmaLatency_ - coIssueCyclePos_;
}

// Advance the co-issue timeline and the elapse-time clock, and decay the RAW
// data-ready counters by \p cycles.
void CDNA5ReadyQueue::advanceTime(int cycles) {
    // Never let the timeline come to rest on a blocked cycle -- every pick path
    // reads coIssueCyclePos_ to decide what may issue next, and nothing may issue
    // there. Roll on to the next issuable cycle instead; the skipped cycles still
    // elapse, the hardware is just spending them itself.
    int landing = coIssueCyclePos_ + cycles;
    while (isBlockedCycle(landing)) ++landing;
    cycles = landing - coIssueCyclePos_;
    coIssueCyclePos_ += cycles;
    clock_ += cycles;
    globalReadInflight_.advance(cycles);
    dsReadInflight_.advance(cycles);
    dsIssueCap_.advance(cycles);
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
// During an active WMMA window, only allowed positions contribute to VALU
// progress.
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
            canIssue = (pos < kCoIssueBits) && (((activeCoIssueWindow_ >> pos) & 1u) != 0u) &&
                       !isBlockedCycle(pos);
        }
        if (canIssue) issued++;
        elapsed++;
    }
    return elapsed;
}

// After a picked instruction: advance the co-issue timeline. Barriers use
// result latency (latencyCycles); VALU/transcendentals use co-issue-aware issue
// progress; others use issueCycles.
void CDNA5ReadyQueue::updateWMMAStatus(DAGNode* node) {
    int elapsedCycles = node->inst->issueCycles;
    if (isBarrier(*node->inst))
        elapsedCycles = node->inst->latencyCycles;
    else if (isVectorALU(*node->inst) || isTranscendental(*node->inst))
        elapsedCycles = computeValuAdvanceCycles(node->inst->issueCycles);
    else if (isDSRead(*node->inst))
        elapsedCycles = dsIssueCost(*node->inst);
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
        dsIssueCap_.push(dsIssueCapSpan());
    } else if (pickKind == kOther) {
        otherQueue.erase(node);
    } else {
        assert(pickKind == kValu);
        valuQueue.erase(node);
    }
    // WMMA->VALU: if this VALU depends on the active WMMA's D and fewer than
    // slots fills landed (no independent VALU left to space it), emit the
    // shortfall as v_nops before it.
    if (pickKind == kValu && activeWmmaNode_ != nullptr &&
        wmmaToValuCoexecOverlap(*activeWmmaNode_->inst, *node->inst)) {
        const int slots = popcount16(activeWmmaNode_->inst->coIssueWindow);
        const int shortfall = slots - nonWmmaFillsSinceActiveWmma_;
        if (shortfall > 0) {
            pendingFillerVNops_ = shortfall;
            // Credit fillers as slot fills so a later dependent VALU in this window
            // is not re-padded.
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
    // rule.isConsumer instruction reads from this producer, so that consumer's
    // pick waits the fixed hazard out. Per-rule lane (not regDataReadyCounters)
    // so an unrelated instruction reading the same register is not wrongly gated.
    for (const HazardFlag& hf : node->hazardFlags)
        // rule.cycles == -1 ("hoist as far as possible"): the strategy is
        // producer-side hoisting (deadline forced to 0 in the pre-scan), not a
        // consumer-side hold, so clamp the gate to 0 rather than stamping a
        // negative wait.
        hazardGates_[hf.ruleIdx][hf.regKey] = std::max(0, hw_.hazards.rules[hf.ruleIdx].cycles);
    // No longer a live hoist candidate once issued (decidePromote() must not try
    // to force it again).
    if (!node->hazardFlags.empty()) {
        auto it = std::find_if(hazardHoistCandidates_.begin(), hazardHoistCandidates_.end(),
                               [node](const HazardHoistCandidate& hc) { return hc.node == node; });
        if (it != hazardHoistCandidates_.end()) hazardHoistCandidates_.erase(it);
    }
    if (deferHeadBalanceThisRegion_) deferFirstHeadWmmaActive_ = false;
    return node;
}

// (A) RAW stamp: record each dest reg's data-ready latency. Skip when
// latencyCycles <= issueCycles: after a pick, updateWMMAStatus ->
// advanceTime(issueCycles) runs before the next pickOne and would immediately
// decay such a stamp to 0, so it can never gate a consumer. Skipping it makes
// the dominant gfx1250 case (VALU/SALU latency == issue == 1) a no-op with no
// map churn; only ds_load (56) and rare latency-2 ops persist.
void CDNA5ReadyQueue::stampDataReady(const StinkyInstruction& inst) {
    if (inst.latencyCycles <= inst.issueCycles) return;
    for (const StinkyRegister& dst : inst.getDestRegs()) {
        if (!dst.isRegister() || isPseudoReg(dst)) continue;
        for (unsigned off = 0; off < dst.reg.num; ++off)
            regDataReadyCounters[regDepKey(dst.reg.type, dst.reg.idx + off)] =
                inst.latencyCycles - inst.issueCycles;
    }
}

// (B) elapse touch: record the current timeline clock for every operand reg
// (dst + src) of an issued instruction, so a later node reusing a just-touched
// reg can be deferred.
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

// (A) RAW data-ready gate: max outstanding data-ready latency over a node's src
// VGPRs. Returns 0 if all src data is ready (safe to consume), >0 if hardware
// would stall.
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

// Hazard gate: max remaining hazard cycles over \p node's sources, across every
// rule where \p node's instruction matches rule.isConsumer. Returns 0 when \p
// node may issue now with respect to every hazard rule, >0 when it must still
// wait for some flagged producer's write to clear the hardware gap. Consulted
// for any candidate that could be a hazarded consumer
// (tensor_load/s_load/global_read/etc. — see call sites in pickFreeBest and
// findSmallestPickableNonWmma).
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

// True if issuing \p node now would risk a co-execution hazard: while the WMMA
// that opened the current latency window is still in flight, \p node's dest
// VGPRs overlap that WMMA's src VGPRs, so the write could clobber a source the
// WMMA is still reading.
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

// (B) elapse key: min over the node's operand regs (dst + src) of (clock_ -
// lastTouch). The most-recently-touched operand binds (smallest elapse), so a
// node reusing a just-touched reg ranks low and is deferred. Regs never touched
// => INT_MAX (very old).
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

// Updates (best, bestKey) to (cand, key) if key sorts before bestKey
// (lexicographic std::tuple compare) or best is not yet set. Shared by the "min
// by (metric, id)" candidate pickers below; each caller supplies its own key
// shape.
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

// Best issuable node in \p queue: RAW/hazard-free nodes are preferred; when
// none is free and \p allowHiddenStall is set, a node whose remaining wait (RAW
// data-ready or hazard-gate, whichever is larger) still fits under the active
// WMMA's latency shadow is also eligible (the wait is hidden by the in-flight
// WMMA, so it is free to co-issue). Within the same free/hazard tier the
// operand touched longest ago wins (largest elapse), tie broken by DAG id.
// Producer-side hazard hoisting is handled separately by decidePromote(), not
// here — this function only decides whether a candidate is safe to issue *now*,
// not whether it should be forced early. \p outWait (optional) receives the
// cycles the caller must advanceTime() before issuing the returned node (0 for
// a free pick). Returns nullptr if none is eligible.
DAGNode* CDNA5ReadyQueue::pickFreeBest(const ReadySetByDAGid& queue, int* outWait,
                                       bool allowHiddenStall) const {
    // A stall is only hidden if the instruction can actually issue when it
    // expires, so the shadow stops at the first blocked cycle.
    const int coIssueSpace = freeCoIssueSpace();
    DAGNode* best = nullptr;
    int bestElapse = INT_MIN;
    int bestWait = 0;
    int bestAff = 0;
    for (DAGNode* n : queue) {  // iterates smallest-id first, so ties keep the oldest id
        if (heldBackForLead(n)) continue;
        const int wait = std::max(getMaxSrcDataWait(n), getHazardWait(n));
        // Tolerate a wait only if it fits the WMMA latency shadow and the dest does
        // not clobber a live WMMA src (then the stall is free).
        if (wait > 0 && (!allowHiddenStall || coIssueSpace <= 0 || wait > coIssueSpace ||
                         destOverlapsActiveWmmaSrc(n)))
            continue;
        const int elapse = nodeElapseKey(n);
        // MSB bank affinity: a tiebreak BELOW the free/hazard tier and ABOVE elapse
        // — prefer a same-bank candidate to avoid an s_set_vgpr_msb switch. Inert
        // when currentMsb_ == -1 (nothing matches), so unchanged behavior until a
        // real tie.
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

// Find the WMMA in wmmaQueue with the smallest max data-ready latency (most
// ready). Returns the node and its latency. Ties broken by DAG id (program
// order).
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
    // Set before the advanceTime() below so those cycles are never picked into.
    activeWmmaBlockedScale_ = node->inst->getHwInstDesc()->blockedScaleMask;
    activeWmmaNode_ = node;
    nonWmmaFillsSinceActiveWmma_ = 0;  // new window: restart WMMA->WMMA fill count
    dsSchedulingBudgetUsed_ = 0;
    // Advance by WMMA issue cycles after opening a new timeline window.
    // This keeps coIssueCyclePos_ aligned with elapsed cycles right after WMMA
    // issue.
    advanceTime(node->inst->issueCycles);
    wmmaIssueConfig.issuedCount--;

    if (deferHeadBalanceThisRegion_) deferFirstHeadWmmaActive_ = false;
    wmmaIssuedCountThisRegion_++;
    if (hideBudget_.numWindows() > 0) {
        if (hideBudget_.issueBudgetByWmmaIndex) {
            const int windowIndex = wmmaIssuedCountThisRegion_ - 1;
            cumulativeWmmaHideBudget_ += hideBudget_.issueBudgetFor(windowIndex);
            cumulativeWmmaDsLoadBudget_ += hideBudget_.dsLoadBudgetFor(windowIndex);
        } else {
            cumulativeWmmaHideBudget_ += hideBudget_.issueBudgetFor(node->inst);
        }
    }
    globalReadCounter = 0;
    // (A) RAW: stamp the WMMA's dest (accumulator) data-ready latency.
    // (B) elapse: record the timeline touch for all its operands.
    stampDataReady(*node->inst);
    touchOperands(*node->inst);
    if (node->requiredMsb != -1) currentMsb_ = node->requiredMsb;
    return node;
}

// Pick among ready non-WMMA nodes, preferring genuinely RAW-free work.
// Queues: globalReadQueue (throttled), localReadQueue, valuQueue (co-issue
// gated), otherQueue. SALU/other and VALU picks go through pickFreeBest (elapse
// ordering — defers a reg-overwrite behind other work). SALU/other may
// additionally be co-issued with a hidden stall when its src RAW wait fits
// under the active WMMA's latency shadow; that wait is returned via \p outWait
// so the caller advances the timeline before issuing. A hidden-stall candidate
// never outranks a genuinely free one. When nothing qualifies, these queues
// contribute nothing and Phase G falls back to the oldest. kind: 0=global,
// 1=local, 2=other, 3=valu.
bool CDNA5ReadyQueue::findSmallestPickableNonWmma(DAGNode* pickedDS, DAGNode** outNode,
                                                  int* kindOut, int* outWait) const {
    *outNode = nullptr;
    *kindOut = -1;
    if (outWait) *outWait = 0;
    DAGNode* best = nullptr;
    int kind = -1;
    int bestWait = 0;
    std::tuple<int, int, int> bestKey{};
    const bool dsLoadBudgetEnabled = !hideBudget_.barriers.empty();
    const bool dsLoadBudgetPending = dsLoadIssuedThisRegion_ < cumulativeWmmaDsLoadBudget_;

    // Ordering, lowest key first: (1) genuinely free work; (2) a throttled DS
    // whose pacing debt fits the active WMMA's scheduling budget; (3) work that
    // still needs a real RAW/hazard stall. Within a tier, global_read beats
    // other non-WMMA kinds, then smallest id wins.
    // Producer-side hazard hoisting is handled separately by
    // decidePromote(), not here — a flagged producer competes on equal terms with
    // everything else unless/until decidePromote() forces it.
    auto consider = [&](DAGNode* cand, int candKind, int candWait) {
        if (!cand) return;
        const int availabilityRank = candWait == 0 ? 0 : (candKind == kLocalRead ? 1 : 2);
        const int kindRank = (candKind == kGlobalRead) ? 0 : 1;
        if (considerBest(cand, std::make_tuple(availabilityRank, kindRank, (int)cand->id), best,
                         bestKey)) {
            kind = candKind;
            bestWait = candWait;
        }
    };

    // Rule (4) ds_load cap. dsIssueCap_ holds the ds_loads issued within the
    // last dsIssueCapSpan() cycles of the real timeline; full() means the
    // ceiling is reached. The window slides on that clock and is never reset,
    // so it is defined everywhere -- including the region tail, where the
    // pre-refactor per-WMMA counter had nothing to gate against.
    //
    // Unconditional, and a WAIT rather than a veto. As a veto it dropped the
    // ds_load from the candidate set, which does not make the scheduler wait --
    // it makes it issue whatever else is ready, hoisting a worse candidate (see
    // SgprToTensorLoadHazard_AtLeast8CycleGap). As a wait it competes on price,
    // so with nothing better to run the scheduler idles until the window frees.
    const int dsCapWait = dsIssueCap_.full() ? std::max(1, dsIssueCap_.minResidual()) : 0;
    // A DS budget is an upper gate, not permission to bypass queue pacing:
    // while pending, DS reads still follow the normal throttle/scheduling-space
    // checks below; once satisfied, they leave normal selection until the next
    // WMMA contributes another window's DS budget. Phase G remains the progress
    // fallback when no normally pickable instruction exists.
    const bool dsBudgetAllowsIssue = !dsLoadBudgetEnabled || dsLoadBudgetPending;
    const bool dsBaseOk = pickedDS && dsBudgetAllowsIssue && !destOverlapsActiveWmmaSrc(pickedDS);
    int dsThrottleWait = 0;
    if (dsBaseOk) {
        dsThrottleWait = std::max(dsCapWait, dsReadThrottleWait());
        const int schedulingPos = coIssueCyclePos_ + dsSchedulingBudgetUsed_;
        int schedulingSpace = activeWmmaLatency_ - schedulingPos;
        for (int pos = schedulingPos; pos < activeWmmaLatency_; ++pos) {
            if (isBlockedCycle(pos)) {
                schedulingSpace = pos - schedulingPos;
                break;
            }
        }
        // There is no hide-budget bypass here -- #12458 removed it so a pending
        // DS budget cannot skip throttle pacing. outOfWmmaWindow is the only
        // one left, and it deliberately requires dsReadThrottleWait() == 0: it
        // covers a CAP wait, never a throttle wait. The cap window slides on
        // the real clock, so waiting on it is elapsed time that frees a slot;
        // throttle debt is pacing and drains nothing, so an out-of-window
        // throttled ds_load keeps falling through to Phase G, which charges it
        // to the throttle clock (DsReadThrottle_PhaseGFallbackUsesOnlyThrottleClock).
        //
        // It is needed because activeWmmaLatency_ is 0 out of window, so the
        // comparison below is false for every wait and would drop the ds_load
        // -- the veto this change removed, reached by another route.
        const bool outOfWmmaWindow = activeWmmaLatency_ <= 0 && dsReadThrottleWait() == 0;
        const bool fitsSchedulingBudget =
            dsThrottleWait == 0 || outOfWmmaWindow ||
            (schedulingPos < activeWmmaLatency_ &&
             dsThrottleWait + dsIssueCost(*pickedDS->inst) <= schedulingSpace);
        if (fitsSchedulingBudget) consider(pickedDS, kLocalRead, dsThrottleWait);
    }
    const bool dsWindowOk = dsBaseOk && dsThrottleWait == 0;

    if (!globalReadQueue.empty() && !globalReadQueueFull() &&
        (globalReadCounter < globalReadPerWMMA || otherQueue.empty())) {
        // A tensor_load whose source is still inside a live hazard-gate window
        // carries that wait, so it ranks as a hidden-stall candidate and defers
        // behind free work (whatever fills the gap). It is still eligible when
        // nothing else can go, paying the remaining wait before issue.
        DAGNode* gr = globalReadQueue.top();
        consider(gr, kGlobalRead, getHazardWait(gr));
    }
    // SALU/other allows hidden stalls (see pickFreeBest); VALU stays
    // RAW-free-only.
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

// Pure promotion decision, run once at the top of each pickOne(). Records which
// phase must fire now and the exact node it will issue, without mutating any
// queue. Two rules: (1) a barrier whose WMMA-issued threshold is met (this is
// what the dedicated forced-barrier phase used to do ahead of WMMA); (2) a
// hazard-hoist producer whose live clock_ has reached its hazardDeadline --
// forces the producer to issue now, through its own NonWmmaFill phase, so it
// lands before its hazarded consumer needs the gap instead of after. Add
// further forcing rules here as new PromotePhase cases (or new decisions within
// an existing phase, as below). Open the chain on its def, close it on its last
// reader. Between those two picks sccChainBlocks() holds back every workgroup
// barrier.
void CDNA5ReadyQueue::noteSccChainIssue(DAGNode* node) {
    if (!clusterBarrierEnabled()) return;
    if (node->sccChainId == 0) return;

    if (node->sccChainDef) {
        openSccChain_ = node->sccChainReaders > 0 ? node->sccChainId : 0;
        sccReadersLeft_ = node->sccChainReaders;
        return;
    }
    // A reader of a chain other than the open one means its def was never issued
    // here (a value defined in an earlier region); nothing to close.
    if (node->sccChainId != openSccChain_) return;
    if (--sccReadersLeft_ == 0) openSccChain_ = 0;
}

void CDNA5ReadyQueue::decidePromote() {
    promotedPhase_ = PromotePhase::None;
    promotedNode_ = nullptr;
    promotedKind_ = -1;

    if (!barrierQueue.empty() && !barrierWmmaThresholds_.empty()) {
        for (DAGNode* node : barrierQueue) {
            if (!isBarrierEligibleNow(node)) continue;
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
    // hazardDeadline, or earlier if a pending WMMA would otherwise jump clock_
    // past it first (a WMMA can otherwise steal the slot right before the
    // deadline). Deliberately clock_-based, not a proxy node's readiness: clock_
    // only advances via cycles actually issued (advanceTime, called on every real
    // pick), so it can't run ahead of the true schedule the way "some node's
    // inDegree hit 0" can when that node is structurally unblocked long before it
    // is actually picked. Before the deadline, the producer competes as ordinary
    // work -- no special priority -- so it never crowds out ds/wmma while it
    // still has slack. Also requires the producer to be genuinely free to issue
    // right now (no outstanding RAW/hazard wait of its own, and no WMMA-src
    // overlap) -- this is a throughput heuristic layered on an
    // unconditionally-correct consumer-side gate, so it must never force an
    // otherwise-unsafe issue; missing the deadline just costs a later explicit
    // stall via that gate.
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
        if (!isBarrierEligibleNow(node)) continue;
        auto thIt = barrierWmmaThresholds_.find(node->inst);
        if (thIt != barrierWmmaThresholds_.end() && wmmaIssuedCountThisRegion_ >= thIt->second) {
            forced = node;
            break;
        }
    }
    if (forced) barrierQueue.erase(forced);
    return forced;
}

// WMMA windows needed to issue dsLoadCount ds_reads given the per-window DS
// cap. After the count reaches the DS read queue depth, configured transition
// entries use the scaled throttle interval and later entries use the full one.
DsLoadBudgetConfig CDNA5ReadyQueue::dsLoadBudgetConfig() const {
    DsLoadBudgetConfig config;
    config.dsReadPerWmma = dsReadPerWmma();
    config.dsReadQueueDepth = dsReadQueueDepth();
    config.dsReadThrottleLatency = dsReadThrottleLatency();
    config.dsReadThrottleTransitionFactor = dsReadThrottleTransitionFactor();
    config.dsReadThrottleTransitionEntries = dsReadThrottleTransitionEntries();
    config.wmmaLatency = wmmaIssueConfig.latency;
    return config;
}

int CDNA5ReadyQueue::computeWmmaWindowsNeeded(int dsLoadCount) const {
    return computeDsLoadWmmaWindowsNeeded(dsLoadCount, dsLoadBudgetConfig());
}

// Compute forceBarrierAfterNthWmma_ for this region from register dependencies.
//
//  Step 1a — collect all movable barriers with their PSEUDO src token sets.
//  Step 1b — for each barrier, find the latest ds_read whose dest PSEUDO token
//  matches. Step 2 & 3 — find the last WMMA in [regionStart, that ds_read]
//  whose src VGPRs
//               overlap the ds_read's dest VGPRs; record its 1-based index
//               (wmmaIdx).
//  Step 4 — threshold N = max(lastOverlap, wmmaWindowsNeeded) +
//  latencyWmmaBudget;
//            latencyWmmaBudget = (latency / wmmaIssueConfig.latency) + 1.
//            wmmaWindowsNeeded is derived from matching ds_read count and DS
//            per-WMMA cap. latency = dsReadDrainLatency when it is configured
//            (> 0), else computeDynamicDrainLatencyForLoads(hw, matchingLoads,
//            numWaves) over every matching ds_read (last-load latency,
//            count-weighted average throughput, max maxDrain over the burst).
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
        int latencyWmmaBudget;
        int dsLoadCount;
    };

    // Step 1a: collect all movable barriers with their PSEUDO src token sets,
    // then merge
    //          signal/wait pairs so both halves share one threshold.
    auto barrierGroups =
        groupBarrierTokens(collectBarrierTokens(regionStart, regionEnd, /*useSrc=*/true));

    std::vector<BarrierAfterSummary> overlapChecks;
    for (const BarrierTokenGroup& group : barrierGroups) {
        // For a signal/wait pair, use the first barrier as the "before barrier"
        // anchor.
        StinkyInstruction* groupBarrier = group.barriers.front();

        // Step 1b: scan [regionStart, groupBarrier) — collect every matching
        //          ds_read's drain entry (latency + HwInstDesc throughput /
        //          maxDrain) in order; the latest also anchors the VGPR / WMMA
        //          overlap scan below.
        StinkyInstruction* targetDSLoad = nullptr;
        IRList::iterator targetDSLoadIt = regionEnd;
        std::vector<DsLoadDrainEntry> matchingDsLoads;
        for (IRList::iterator it = regionStart; it != regionEnd; ++it) {
            StinkyInstruction& inst = getStinkyInst(it);
            if (&inst == groupBarrier) break;
            if (!isDSRead(inst)) continue;
            for (const StinkyRegister& src : inst.getSrcRegs()) {
                if (isPseudoReg(src) && group.tokens.count(src.reg.idx)) {
                    const HwInstDesc* desc = inst.getHwInstDesc();
                    matchingDsLoads.push_back(makeDsLoadDrainEntry(
                        hw_, static_cast<int>(inst.latencyCycles), desc ? desc->dsThroughput : 0,
                        desc ? desc->dsMaxDrain : 0));
                    targetDSLoad = &inst;
                    targetDSLoadIt = it;  // keep updating → ends up as latest
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

        // Step 4: threshold N = lastOverlap + (latency / wmmaIssueConfig.latency)
        // + 1. A positive dsReadDrainLatency pins the latency. A non-positive value
        // (default 0) means "use dynamic drain latency," derived from all matching
        // ds_loads via computeDynamicDrainLatencyForLoads (last-load latency,
        // count-weighted average throughput, max maxDrain over the burst), keyed
        // by this pass context's NumWaves.
        const int configuredDrainLatency = dsReadDrainLatency();
        const int numWaves = static_cast<int>(getPassContext().getGemmTileConfig().NumWaves);
        const int matchingDsLoadCount = static_cast<int>(matchingDsLoads.size());
        const int latencyForAfterThreshold =
            configuredDrainLatency > 0
                ? configuredDrainLatency
                : computeDynamicDrainLatencyForLoads(hw_, matchingDsLoads, numWaves);
        const int latencyWmmaBudget = (latencyForAfterThreshold / wmmaIssueConfig.latency) + 1;
        const int wmmaWindowsNeeded = computeWmmaWindowsNeeded(matchingDsLoadCount);
        const int overlapOrWindowBase = std::max(lastOverlap, wmmaWindowsNeeded);
        int afterThreshold = overlapOrWindowBase + latencyWmmaBudget;
        for (StinkyInstruction* barrier : group.barriers) {
            barrierWmmaThresholds_[barrier] = afterThreshold;
            result[barrier] = {afterThreshold, afterThreshold, latencyWmmaBudget, wmmaWindowsNeeded,
                               matchingDsLoadCount};
        }
        overlapChecks.push_back({groupBarrier, group.barriers, afterThreshold, lastOverlap,
                                 wmmaWindowsNeeded, latencyWmmaBudget, matchingDsLoadCount});
        PASS_DEBUG(std::cerr << "[CDNA5 computeBarrierAfterThresholds] barrier=" << groupBarrier
                             << " barrierGroupSize=" << group.barriers.size() << " afterThreshold="
                             << afterThreshold << " matchingDsLoadCount=" << matchingDsLoadCount
                             << " latencyWmmaBudget=" << latencyWmmaBudget << " wmmaWindowsNeeded="
                             << wmmaWindowsNeeded << " overlapOrWindowBase=" << overlapOrWindowBase
                             << " latencyForAfterThreshold=" << latencyForAfterThreshold
                             << " lastOverlap=" << lastOverlap << "\n");
    }

    // Step 5: each group's interval is [afterThreshold - wmmaWindowsNeeded,
    // afterThreshold). Process groups in ascending afterThreshold order so that a
    // "front" (earlier) barrier is always fully resolved before the "later"
    // barriers that it pushes back. A stable_sort keeps groups with equal
    // afterThreshold in their original appearance order, which gives the
    // tie-break: the earlier-appearing group is treated as the front (earlier)
    // one. After sorting, for any pair i < j we have afterThreshold[i] <=
    // afterThreshold[j], so j is always the later barrier and i the earlier one:
    //   - the later barrier (larger afterThreshold) is pushed back until its
    //   interval start
    //     clears the overlapping earlier barrier's end (== that barrier's
    //     afterThreshold): newAfterThreshold = maxEarlierAfterThreshold +
    //     wmmaWindowsNeeded.
    //   - the earlier barrier (smaller afterThreshold) keeps the prior behavior
    //   of extending
    //     afterThreshold by the shared overlap length.
    // Results are capped at issuedCount.
    std::stable_sort(overlapChecks.begin(), overlapChecks.end(),
                     [](const BarrierAfterSummary& a, const BarrierAfterSummary& b) {
                         return a.afterThreshold < b.afterThreshold;
                     });
    const size_t n = overlapChecks.size();
    // pushedStart[i] starts at the barrier's own interval start; overlapping
    // earlier barriers push it up to their end so (pushedStart +
    // wmmaWindowsNeeded) clears the overlap.
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

            // Sorted ascending, so j is the later barrier and i the earlier (front)
            // one.
            pushedStart[j] = std::max(pushedStart[j], overlapChecks[i].afterThreshold);
            frontOverlapBudget[i] += overlapLen;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        const BarrierAfterSummary& summary = overlapChecks[i];
        const int adjustedAfterThreshold =
            std::min((int)wmmaIssueConfig.issuedCount,
                     pushedStart[i] + summary.wmmaWindowsNeeded + frontOverlapBudget[i]);

        for (StinkyInstruction* barrier : summary.barriers) {
            barrierWmmaThresholds_[barrier] = adjustedAfterThreshold;
            result[barrier] = {adjustedAfterThreshold, summary.afterThreshold,
                               summary.latencyWmmaBudget, summary.wmmaWindowsNeeded,
                               summary.dsLoadCount};
        }
        PASS_DEBUG(
            std::cerr << "[CDNA5 computeBarrierAfterThresholds overlap] barrier="
                      << summary.barrierKey << " barrierGroupSize=" << summary.barriers.size()
                      << " baseAfterThreshold=" << summary.afterThreshold
                      << " adjustedAfterThreshold=" << adjustedAfterThreshold
                      << " latencyWmmaBudget=" << summary.latencyWmmaBudget
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
//  Step 2  — for each matching ds_read, starting from its post-barrier WMMA
//  index,
//            find the first consumer WMMA whose src overlaps the ds_read dest
//            VGPRs (scan ds_read -> regionEnd, then wrap regionStart ->
//            ds_read). Keep the largest consumer index across ds_reads
//            (MaximumWMMAIdx), and remember the latency of the ds_read that
//            defines that max.
//  Step 3  — residualCycles = max(0, MaximumWMMAIdx * wmmaIssueConfig.latency
//                                    - targetDSLoadLatency)
//  Step 4  — build candidate "before" caps from:
//            - beforeN: residualCycles / wmmaIssueConfig.latency
//            - maxFinalWmmaIdx: targetDSLoadLatency / wmmaIssueConfig.latency
//            - wmmaWindowsNeeded: WMMA windows needed to issue all matching
//            ds_reads. Final threshold = max(0, min(beforeN, issuedCount
//                                             - max(maxFinalWmmaIdx,
//                                             wmmaWindowsNeeded))).
//  Step 5  — overlap adjustment: detect barriers whose WMMA-window intervals
//  overlap,
//            then for each such barrier pull its forced start earlier by the
//            overlap budget and widen its window span (kept within the region),
//            so overlapping barriers leave enough WMMA-window space to issue
//            their ds_loads.
std::unordered_map<StinkyInstruction*, CDNA5ReadyQueue::BarrierBeforeOutput>
CDNA5ReadyQueue::computeBarrierBeforeThresholds(IRList::iterator regionStart,
                                                IRList::iterator regionEnd) {
    std::unordered_map<StinkyInstruction*, BarrierBeforeOutput> result;
    struct BarrierBeforeSummary {
        StinkyInstruction* barrierKey;
        std::vector<StinkyInstruction*> barriers;
        int beforeThreshold;
        int wmmaWindowsNeeded;
        int dsLoadCount;
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
            // NOTE: currently a no-op detection stub. It scans for another barrier
            // that carries a dest PSEUDO token overlapping this group's tokens, but
            // the result is not recorded or acted upon yet (kept as a placeholder for
            // future cross-barrier token handling).
            if (isBarrier(inst) && &inst != groupBarrier) {
                for (const StinkyRegister& dest : inst.getDestRegs()) {
                    if (isPseudoReg(dest) && group.tokens.count(dest.reg.idx)) {
                        // Found one matching token on this barrier; no need to keep
                        // scanning its remaining dest operands.
                        break;
                    }
                }
            }
        }
        if (matchingDSReads.empty()) continue;

        // Step 2: for each matching ds_read, find the first consumer WMMA (with
        // wrap-around
        //         search order) whose src VGPRs overlap the ds_read dest VGPRs;
        //         take the maximum resulting WMMA index across all ds_reads
        //         (MaximumWMMAIdx).
        int maximumWMMAIdx = -1;
        int targetDSLoadLatency = 0;
        for (const DSReadMatch& dse : matchingDSReads) {
            int wmmaIdx = dse.dsWmmaIdx;
            bool found = false;
            // Scan in two segments: first from the ds_read position to regionEnd,
            // then wrap around from regionStart up to (but not including) the
            // ds_read.
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
        // WMMA issue count that forces the barrier early enough for all dependent
        // ds_reads. Take the latest of three constraints, then subtract from total
        // WMMAs in the region:
        //   beforeN — remaining latency after the last consumer WMMA
        //   maxFinalWmmaIdx — absolute cap after the 1st ds_load (DS load latency /
        //   WMMA latency) wmmaWindowsNeeded — DS issue bandwidth (enough WMMA
        //   windows for all ds_loads); when
        //       dsLoadCount > dsReadQueueDepth(), add extra drain windows from
        //       dsReadThrottleLatency.
        int beforeThreshold =
            std::max(0, std::min(beforeN, wmmaIssueConfig.issuedCount -
                                              std::max(maxFinalWmmaIdx, wmmaWindowsNeeded)));
        for (StinkyInstruction* barrier : group.barriers)
            result[barrier] = {beforeThreshold, beforeThreshold, wmmaWindowsNeeded, dsLoadCount};
        overlapChecks.push_back(
            {groupBarrier, group.barriers, beforeThreshold, wmmaWindowsNeeded, dsLoadCount});
        PASS_DEBUG(std::cerr << "[CDNA5 computeBarrierBeforeThresholds] barrier="
                             << " beforeThreshold=" << beforeThreshold
                             << " barrierGroupSize=" << group.barriers.size()
                             << " beforeN=" << beforeN << " maxFinalWmmaIdx=" << maxFinalWmmaIdx
                             << " wmmaWindowsNeeded=" << wmmaWindowsNeeded
                             << " numDsLoad=" << dsLoadCount << "\n");
    }

    // Step 5: each group's interval is [beforeThreshold, beforeThreshold +
    // wmmaWindowsNeeded). Mirror of computeBarrierAfterThresholds but walking
    // back-to-front: scan from the last barrier group backward, comparing each
    // group only with the groups before it. For an overlapping pair, decide the
    // back barrier by beforeThreshold (smaller = earlier on the WMMA axis); on a
    // tie the later-appearing group (larger index) is the back one:
    //   - the back barrier (smaller beforeThreshold) is pulled earlier by its own
    //   window until
    //     its interval end clears the overlapping front barrier's start (== that
    //     barrier's beforeThreshold): newBeforeThreshold = frontBeforeThreshold -
    //     wmmaWindowsNeeded.
    //   - the front barrier (larger beforeThreshold) keeps the prior behavior of
    //   pulling its
    //     beforeThreshold earlier by the shared overlap length.
    // Results are clamped to [0, issuedCount].
    const size_t n = overlapChecks.size();
    // pulledEnd[i] starts at the barrier's own interval end; overlapping front
    // barriers pull it down to their start so (pulledEnd - wmmaWindowsNeeded)
    // clears the overlap.
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

            // On a tie, the later-appearing group (i) is the back one, so j is the
            // front.
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
            result[barrier] = {adjustedBeforeThreshold, summary.beforeThreshold,
                               adjustedWmmaWindowsNeeded, summary.dsLoadCount};
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
//   Phase A: forced barrier — when wmmaIssuedCountThisRegion_ reaches a
//   per-barrier threshold. Phase B: WMMA if DS latency gate (rule 2) passed, DS
//   window cap (rule 4) respected,
//            loop head balance (rule 5) ok, and program order (rule 1) allows.
//   Phase C: inside WMMA latency window — fill with non-WMMA work.
//   Phase D: outside WMMA latency — pick smallest-id from any non-WMMA queue.
//   Phase E: forced WMMA — pick most-ready WMMA when all non-WMMA queues are
//   empty. Phase F: barriers — only after all compute queues (WMMA + non-WMMA)
//   are drained.
DAGNode* CDNA5ReadyQueue::pickOne() {
    PASS_DEBUG(
        std::cerr << "[CDNA5 pickOne] prevPick="
                  << (lastPickedNode_ ? std::to_string(lastPickedNode_->id) : std::string("none"))
                  << "\n");
    auto rememberPick = [this](DAGNode* node) {
        if (node != nullptr && !isMatrixInstruction(*node->inst)) {
            nonWmmaIssuedThisRegion_++;
            if (isDSRead(*node->inst)) dsLoadIssuedThisRegion_++;
        }
        lastPickedNode_ = node;
        noteSccChainIssue(node);
        return node;
    };

    // Promotion decision (formerly the forced-barrier phase). Records which phase
    // must fire now; the non-promoted phases below gate themselves off via
    // isPromote(), and the promoted node issues through its own phase — one entry
    // point, no side effects here.
    decidePromote();

    // Pre-compute the best DS read by dsReadPriority once for all phases.
    DAGNode* pickedDS = nullptr;
    for (DAGNode* n : localReadQueue) {
        if (!pickedDS || n->dsReadPriority < pickedDS->dsReadPriority) pickedDS = n;
    }

    // Diagnostic: classify what would stop this ds_load right now. Read both
    // constraints unconditionally, so a ds_load held back by either shows up
    // whichever one is binding.
    if (pickedDS != nullptr) {
        const bool capStops = dsIssueCap_.full();
        const bool queueStops = dsReadInflight_.full() || dsReadInflight_.throttleWait() > 0;
        if (capStops && queueStops)
            ++dsBindBoth_;
        else if (capStops)
            ++dsBindCapOnly_;
        else if (queueStops)
            ++dsBindQueueOnly_;
        else
            ++dsBindNeither_;
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
        // Hold a dependent bestWMMA while non-WMMA work remains; else emit
        // shortfall as v_nops.
        bool blockWmmaForCoexecSpacing = false;
        if (activeWmmaNode_ != nullptr && hasPickableNonWmma &&
            wmmaToWmmaCoexecOverlap(*activeWmmaNode_->inst, *bestWMMA->inst)) {
            const int slotsPlusOne = popcount16(activeWmmaNode_->inst->coIssueWindow) + 1;
            blockWmmaForCoexecSpacing = nonWmmaFillsSinceActiveWmma_ < slotsPlusOne;
        }
        const int hideBudget = cumulativeWmmaHideBudget_;
        const int dsLoadBudget = cumulativeWmmaDsLoadBudget_;
        const bool blockWmmaForHideBudget =
            hasPickableNonWmma &&
            (nonWmmaIssuedThisRegion_ < hideBudget || dsLoadIssuedThisRegion_ < dsLoadBudget);
        PASS_DEBUG(std::cerr << "[CDNA5 pickOne] Phase B candidate wmmaId=" << bestWMMA->id
                             << " bestLatency=" << bestLatency
                             << " blockLoopHead=" << blockWmmaForLoopHeadBalance
                             << " blockActiveWindow=" << blockWmmaForActiveWindow
                             << " blockAtLeastOneNonWmmaInterleaving="
                             << blockWmmaForAtLeastOneNonWmmaInterleaving
                             << " blockCoexecSpacing=" << blockWmmaForCoexecSpacing
                             << " blockHideBudget=" << blockWmmaForHideBudget << " hideBudget="
                             << hideBudget << " nonWmmaIssued=" << nonWmmaIssuedThisRegion_
                             << " dsLoadBudget=" << dsLoadBudget << " dsLoadIssued="
                             << dsLoadIssuedThisRegion_ << " fills=" << nonWmmaFillsSinceActiveWmma_
                             << " localReadQ=" << localReadQueue.size() << " nonWmmaMinId="
                             << (smallestPickable ? std::to_string(smallestPickable->id)
                                                  : std::string("none"))
                             << "\n");
        if (bestLatency <= 0 && !blockWmmaForLoopHeadBalance && !blockWmmaForActiveWindow &&
            !blockWmmaForAtLeastOneNonWmmaInterleaving && !blockWmmaForCoexecSpacing &&
            !blockWmmaForHideBudget) {
            DAGNode* node = pickOneFromWMMA(bestWMMA);
            PASS_DEBUG(std::cerr << "[CDNA5 pickOne] Phase B picked WMMA dagId=" << node->id
                                 << "\n");
            return rememberPick(node);
        }
    }

    // Phase C+D — NonWmmaFill: fill with non-WMMA work (inside then outside the
    // window).
    if (isPromote(PromotePhase::NonWmmaFill)) {
        // Hazard-hoist promotion: decidePromote() found a flagged producer whose
        // hazardDeadline the live clock_ has reached, and confirmed it's genuinely
        // free to issue now. Issue it directly through this, its own existing phase
        // — no separate entry point — bypassing the normal
        // findSmallestPickableNonWmma selection (which would otherwise rank it as
        // ordinary work and might not pick it this cycle, missing the deadline).
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
                // DS throttle wait consumes only its independent scheduling
                // budget. RAW/hazard waits remain genuine elapsed stalls.
                if (pickWait > 0) {
                    if (pickKind == kLocalRead) {
                        dsSchedulingBudgetUsed_ += pickWait;
                        dsReadInflight_.advanceThrottle(pickWait);
                    } else {
                        advanceTime(pickWait);
                    }
                }
                return rememberPick(popNonWmma(smallestPickable, pickKind));
            }

            advanceTime(activeWmmaLatency_ - coIssueCyclePos_);
        }

        // Phase D — outside WMMA latency.
        DAGNode* smallestPickable = nullptr;
        int pickKind = -1;
        int pickWait = 0;
        if (findSmallestPickableNonWmma(pickedDS, &smallestPickable, &pickKind, &pickWait)) {
            // Same split as Phase C: DS throttle wait is pacing-only; RAW/hazard
            // waits are genuine elapsed stalls. pickWait can be non-zero here
            // (e.g. throttled DS while its DS budget is pending, or a hazard
            // stall).
            if (pickWait > 0) {
                if (pickKind == kLocalRead) {
                    dsSchedulingBudgetUsed_ += pickWait;
                    dsReadInflight_.advanceThrottle(pickWait);
                } else {
                    advanceTime(pickWait);
                }
            }
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

    // Barrier phase — the single entry point for issuing a barrier. When a
    // barrier was promoted (threshold met), the phases above gated off and we
    // arrive here to issue that exact node (formerly the forced-barrier phase
    // ahead of WMMA). Otherwise this is the drain case: issue the lowest-id
    // barrier once all compute has been picked.
    if (isPromote(PromotePhase::Barrier) && !barrierQueue.empty()) {
        DAGNode* barrier = nullptr;
        if (promotedPhase_ == PromotePhase::Barrier) {
            barrier = extractForcedBarrier();  // removes the promoted (threshold-met) node
        } else if (clusterBarrierEnabled()) {
            // Skip barriers held back by an open SCC chain; a later phase issues them
            // once the chain's last reader has gone out.
            for (DAGNode* cand : barrierQueue) {
                if (!isBarrierEligibleNow(cand)) continue;
                barrier = cand;
                break;
            }
            if (barrier) barrierQueue.erase(barrier);
        } else {
            barrier = barrierQueue.top();
            barrierQueue.pop();
        }
        if (barrier) {
            updateWMMAStatus(barrier);
            PASS_DEBUG(std::cerr << "[DAG CDNA5 pickOne] barrier dagId=" << barrier->id
                                 << " promoted=" << (promotedPhase_ == PromotePhase::Barrier)
                                 << "\n";
                       barrier->inst->dump(std::cerr); std::cerr << "\n");
            return rememberPick(barrier);
        }
    }

    // Phase G — final safety net: force-pick the least-blocked ready node to
    // guarantee progress.
    DAGNode* fallback = nullptr;
    int fallbackKind = -1;
    int fallbackWait = 0;
    if (findOldestFallbackNonWmma(pickedDS, &fallback, &fallbackKind, &fallbackWait)) {
        // RAW/hazard and credit-drain waits are real elapsed time. A DS throttle
        // wait is only pacing debt: real waits satisfy as much of it as they
        // cover, and any remainder advances only the independent throttle clock.
        int realWait = fallbackWait;
        if (fallbackKind == kGlobalRead && globalReadQueueFull())
            realWait = std::max(realWait, globalReadInflight_.minResidual());
        if (realWait > 0) advanceTime(realWait);

        int throttleWait = 0;
        if (fallbackKind == kLocalRead) {
            throttleWait = dsReadThrottleWait();
            if (throttleWait > 0) {
                dsSchedulingBudgetUsed_ += throttleWait;
                dsReadInflight_.advanceThrottle(throttleWait);
            }
        }
        PASS_DEBUG(std::cerr << "[CDNA5 pickOne] Phase G fallback pick dagId=" << fallback->id
                             << " kind=" << fallbackKind << " wait=" << realWait
                             << " throttleWait=" << throttleWait << "\n");
        return rememberPick(popNonWmma(fallback, fallbackKind));
    }

    // Only barriers are left and an SCC chain is still holding them back.
    // applyClusterBarrierSccRule only locks chains whose readers can all issue
    // without the barrier going first, so a locked chain always has a reader to
    // make progress on; reaching here means the SCC/barrier ordering invariant
    // broke.
    if (clusterBarrierEnabled() && openSccChain_ != 0 && !barrierQueue.empty()) {
        STINKY_UNREACHABLE("CDNA5ReadyQueue::pickOne: open SCC chain but only barriers are ready");
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
    activeWmmaBlockedScale_ = 0;
    activeWmmaNode_ = nullptr;
    nonWmmaFillsSinceActiveWmma_ = 0;
    dsSchedulingBudgetUsed_ = 0;
    nonWmmaIssuedThisRegion_ = 0;
    cumulativeWmmaHideBudget_ = 0;
    dsLoadIssuedThisRegion_ = 0;
    cumulativeWmmaDsLoadBudget_ = 0;
    hideBudget_ = {};
    globalReadInflight_ = InFlightQueue(globalReadQueueDepth());
    dsReadInflight_ = InFlightQueue(dsReadQueueDepth());
    // Built here rather than per region: the cap window slides on the real
    // clock, so it must not forget at a region boundary any more than it does
    // at a WMMA. dsReadPerWmma() is guaranteed positive, so the depth is real --
    // InFlightQueue::full() is `depth_ > 0 && size >= depth_`, and a depth of 0
    // would report "not full" forever and silently disable rule (4).
    //
    // Not seeded from a predecessor BB. The cross-BB carry is inert on the
    // scheduler's single RPO pass (a loop header is visited before its latch, so
    // sawLoopPred never goes true -- see restoreCrossBBStateFromLoop), so
    // carrying the cap window would be code with no effect until that is fixed.
    dsIssueCap_ = InFlightQueue(dsReadPerWmma());
    assert(dsIssueCap_.depth() > 0 && "rule (4) cap must have a positive depth");
    const int dsDepth = dsReadQueueDepth();
    const double dsThrottleInterval =
        dsDepth > 0 ? (double)dsReadThrottleLatency() / (double)dsDepth : 0.0;
    dsReadInflight_.setThrottleInterval(dsThrottleInterval, dsReadThrottleTransitionFactor(),
                                        dsReadThrottleTransitionEntries());

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

    // Seed the in-flight credit pool from loop-carried state:
    // crossBBGlobalReadCount_ credits, each stamped with the worst-case remaining
    // drain (reconstructs the most-constrained predecessor so no incoming path
    // over-issues).
    if (globalReadQueueDepth() > 0 && crossBBGlobalReadCount_ > 0)
        globalReadInflight_.seed(crossBBGlobalReadCount_, crossBBGlobalReadResidual_);
    // Same credit-pool seeding for the ds_load (LDS return queue) pacer, so a
    // real loop re-entry doesn't model the queue as empty while hardware still
    // has the prior iteration's tail draining (see crossBBDsReadCount_).
    if (dsReadQueueDepth() > 0 && crossBBDsReadCount_ > 0)
        dsReadInflight_.seed(crossBBDsReadCount_, crossBBDsReadResidual_);
}

// KNOWN LIMITATION: on the scheduler's single RPO pass this currently carries
// nothing across a loop back-edge. StinkyDAGSchedulerPass walks blocks in RPO
// and ScheduleAnalysisCache is written in onFinishBB, so a loop header is
// visited BEFORE its latch: the back-edge predecessor has no stored state yet,
// lookup() returns null, and sawLoopPred stays false -- the header falls back to
// its (cold) preheader. A single-BB loop body hits the same thing via its own
// self-edge. So the seeding below only takes effect if a block is scheduled
// twice, which nothing does today.
//
// Confirm with a PASS_DEBUG on sawLoopPred before relying on it. Making it live
// needs either a second pass over loop bodies, a fixed-point iteration, or an
// analytical steady-state estimate at region init -- not a change here.
void CDNA5ReadyQueue::restoreCrossBBStateFromLoop() {
    crossBBDsResiduals_.clear();
    crossBBGlobalReadCount_ = 0;
    crossBBGlobalReadResidual_ = 0;
    crossBBDsReadCount_ = 0;
    crossBBDsReadResidual_ = 0;
    const Loop* loop = getLoop();
    if (!currentBB_ || !loop || !loop->contains(currentBB_) || !getAnalysisCache()) return;

    // Global-read credits: loop predecessors take priority over non-loop ones
    // (the loop body runs many iterations, so its carried state governs steady
    // state). Take the max within the chosen group on BOTH axes — occupancy and
    // residual drain — so no predecessor path is left over-issuing. ds_load
    // credits follow the identical rule.
    int loopCount = 0, loopRes = 0, nonLoopCount = 0, nonLoopRes = 0;
    int dsLoopCount = 0, dsLoopRes = 0, dsNonLoopCount = 0, dsNonLoopRes = 0;
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
            dsLoopCount = std::max(dsLoopCount, state->dsReadInflightCount);
            dsLoopRes = std::max(dsLoopRes, state->dsReadResidual);
        } else {
            nonLoopCount = std::max(nonLoopCount, state->globalReadInflightCount);
            nonLoopRes = std::max(nonLoopRes, state->globalReadResidual);
            dsNonLoopCount = std::max(dsNonLoopCount, state->dsReadInflightCount);
            dsNonLoopRes = std::max(dsNonLoopRes, state->dsReadResidual);
        }
    }
    crossBBGlobalReadCount_ = sawLoopPred ? loopCount : nonLoopCount;
    crossBBGlobalReadResidual_ = sawLoopPred ? loopRes : nonLoopRes;
    crossBBDsReadCount_ = sawLoopPred ? dsLoopCount : dsNonLoopCount;
    crossBBDsReadResidual_ = sawLoopPred ? dsLoopRes : dsNonLoopRes;
}

void CDNA5ReadyQueue::onFinishBB() {
    PASS_DEBUG({
        const int total = dsBindNeither_ + dsBindCapOnly_ + dsBindQueueOnly_ + dsBindBoth_;
        if (total > 0) {
            auto pct = [total](int n) { return (100 * n + total / 2) / total; };
            std::cerr << "[CDNA5 dsBind] bb="
                      << (currentBB_ ? currentBB_->getLabel() : "?") << " samples=" << total
                      << " free=" << dsBindNeither_ << "(" << pct(dsBindNeither_) << "%)"
                      << " capOnly=" << dsBindCapOnly_ << "(" << pct(dsBindCapOnly_) << "%)"
                      << " queueOnly=" << dsBindQueueOnly_ << "(" << pct(dsBindQueueOnly_) << "%)"
                      << " both=" << dsBindBoth_ << "(" << pct(dsBindBoth_) << "%)"
                      << " dsReadPerWmma=" << dsReadPerWmma()
                      << " queueDepth=" << dsReadQueueDepth() << "\n";
        }
    });
    dsBindNeither_ = dsBindCapOnly_ = dsBindQueueOnly_ = dsBindBoth_ = 0;
    if (!currentBB_ || !getAnalysisCache()) return;
    getAnalysisCache()->store(currentBB_, {0, regDataReadyCounters, globalReadInflight_.size(),
                                           globalReadInflight_.maxResidual(),
                                           dsReadInflight_.size(), dsReadInflight_.maxResidual()});
}

// Per scheduling region. Rule (4): per-WMMA-window DS cap (computed in
// pickOneFromWMMA). Rule (2): seedWmmaDsLatencyFromPrefix. Rule (5): head
// balance. Barrier thresholds: computeBarrierAfterThresholds /
// computeBarrierBeforeThresholds.
void CDNA5ReadyQueue::onInitRegion(IRList::iterator regionStart, IRList::iterator regionEnd,
                                   IRList::iterator blockBegin, const RegionDependencies& deps) {
    wmmaIssuedCountThisRegion_ = 0;
    lastPickedNode_ = nullptr;
    // SCC chain locks are per-region: chain ids index the prior region's
    // DAGNodeList, and region boundaries are side-effect cuts no reordering
    // crosses anyway.
    openSccChain_ = 0;
    sccReadersLeft_ = 0;
    // (B) elapse ordering state is per-region: reset the touch map and clock so a
    // new region starts with all regs "very old" (no spurious deferrals from a
    // prior region).
    regLastTouch_.clear();
    clock_ = 0;
    // Per-region: MSB state is not carried across a region boundary (side-effect
    // cut).
    currentMsb_ = -1;
    // Clear per-region node ptr; it dangles into the previous region's freed
    // DAGNodeList.
    activeWmmaNode_ = nullptr;
    nonWmmaFillsSinceActiveWmma_ = 0;
    nonWmmaIssuedThisRegion_ = 0;
    cumulativeWmmaHideBudget_ = 0;
    dsLoadIssuedThisRegion_ = 0;
    cumulativeWmmaDsLoadBudget_ = 0;
    hideBudget_ = {};
    // Hazard state is per-region: hazardHoistCandidates_ holds DAGNode* into the
    // prior region's freed DAGNodeList, and any in-flight gate window is already
    // locked into the prior region's fixed instruction order (region boundaries
    // are side-effect cuts — no reordering crosses them), so it has nothing left
    // to gate here.
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
    int wmmaHideBudgetBase = 0;
    bool hasWmmaHideBudgetBase = false;
    std::unordered_set<StinkyInstruction*> regionInsts;
    for (IRList::iterator it = regionStart; it != regionEnd; ++it) {
        auto* instPtr = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!instPtr) continue;
        StinkyInstruction& inst = *instPtr;
        regionInsts.insert(instPtr);

        if (isMatrixInstruction(inst)) {
            wmmaIssueConfig.issuedCount++;
            hasWMMAInRegion_ = true;
            // Pre-existing, flagged while reworking the ds cap: this takes the
            // FIRST matrix op in the region and keeps it. latencyCycles and
            // issueCycles are overridden per matrix format pair via
            // HwInstDesc::matrixFmtCostOverrides, so a region mixing formats
            // sizes every window's hide budget from whichever WMMA came first.
            // Not changed here -- it is outside this refactor and would move
            // scheduling behaviour -- but it is the same first-WMMA-only
            // assumption that dsIssueCapSpan() warns against reusing.
            if (!hasWmmaHideBudgetBase) {
                const HwInstDesc* desc = inst.getHwInstDesc();
                const int ldScaleCycles = desc != nullptr && desc->blockedScaleMask != 0 ? 1 : 0;
                wmmaHideBudgetBase =
                    std::max(0, inst.latencyCycles - inst.issueCycles - ldScaleCycles);
                hasWmmaHideBudgetBase = true;
            }
        }
    }

    // Rule (4) ds_load cap: at most dsReadPerWmma ds_loads in any
    // dsIssueCapSpan() cycles of the real timeline. Sliding, so it is defined
    // in the region tail too, where no WMMA remains to delimit a window. The
    // window itself lives across regions -- it is built in onInit(), not here.
    PASS_DEBUG(std::cerr << "[CDNA5 dsCap] dsReadPerWmma=" << dsReadPerWmma()
                         << " span=" << dsIssueCapSpan() << "\n");

    barrierWmmaThresholds_.clear();
    barrierDsLoadCounts_.clear();
    std::vector<WmmaHideBudgetBarrierInfo> hideBudgetBarriers;
    if (hasWMMAInRegion_) {
        // Layer 1/3 (base merge):
        // Build one threshold map from two independent estimators:
        //   - afterThresholds : "barrier should be after at least N WMMA"
        //   - beforeThresholds: "barrier should be before/around WMMA N"
        // If a barrier appears in both maps, average them to get a single
        // neutral placement point in barrierWmmaThresholds_.
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
        // Layer 2/3 (exclusive overlap reconcile):
        // Some signal/wait-like pairs are split: one member only lands in the
        // "after" model while its counterpart only lands in the "before" model.
        // For those cross-map-only pairs, compare their implied WMMA ranges.
        // On overlap, split totalWmma proportionally: pull after earlier (min)
        // and push before later (max) so the two exclusive groups stop fighting
        // for the same windows.
        struct BarrierGroupThresholdSummary {
            StinkyInstruction* anchor = nullptr;
            std::vector<StinkyInstruction*> barriers;
            int threshold = 0;      // current promote threshold
            int baseThreshold = 0;  // unclamped estimator end/begin for overlap
            // Claimed WMMA span for overlap detection:
            //   after:  unclamped wmmaWindowsNeeded + latencyWmmaBudget
            //   before: wmmaWindowsNeeded
            int claimWindow = 0;
            // Proportional-split demand. After intentionally uses issue-only
            // windows (no drain); before uses the same value as claimWindow.
            int splitNeeded = 0;
            // Conservative Layer-2 target across all overlapping pairs
            // (after: min, before: max), applied after the pair loop.
            int pendingThreshold = 0;
            // Descendants used when publishing hard orderings on overlap.
            std::vector<StinkyInstruction*> descendantLoads;
        };
        auto setGroupThreshold = [&](const BarrierGroupThresholdSummary& group, int threshold) {
            for (StinkyInstruction* barrier : group.barriers) {
                auto it = barrierWmmaThresholds_.find(barrier);
                if (it != barrierWmmaThresholds_.end()) it->second = threshold;
            }
        };

        auto buildExclusiveAfterGroups = [&]() {
            std::vector<BarrierGroupThresholdSummary> groups;
            auto grouped =
                groupBarrierTokens(collectBarrierTokens(regionStart, regionEnd, /*useSrc=*/true));
            for (const auto& group : grouped) {
                int thresholdSum = 0;
                int thresholdCount = 0;
                int baseThresholdSum = 0;
                int baseThresholdCount = 0;
                int maxClaimWindow = 0;
                int maxSplitNeeded = 0;
                bool hasPrimary = false;
                bool hasCross = false;
                for (StinkyInstruction* barrier : group.barriers) {
                    auto thIt = barrierWmmaThresholds_.find(barrier);
                    if (thIt == barrierWmmaThresholds_.end()) continue;
                    thresholdSum += thIt->second;
                    thresholdCount++;
                    auto pIt = afterThresholds.find(barrier);
                    if (pIt != afterThresholds.end()) {
                        hasPrimary = true;
                        baseThresholdSum += pIt->second.baseAfterThreshold;
                        baseThresholdCount++;
                        // Full unclamped after demand for overlap detection.
                        maxClaimWindow = std::max(maxClaimWindow,
                                                  std::max(0, pIt->second.wmmaWindowsNeeded +
                                                                  pIt->second.latencyWmmaBudget));
                        maxSplitNeeded =
                            std::max(maxSplitNeeded, std::max(0, pIt->second.wmmaWindowsNeeded));
                    }
                    if (beforeThresholds.find(barrier) != beforeThresholds.end()) hasCross = true;
                }
                if (!hasPrimary || hasCross || thresholdCount == 0 || baseThresholdCount == 0)
                    continue;
                BarrierGroupThresholdSummary summary;
                summary.anchor = group.barriers.front();
                summary.barriers = group.barriers;
                summary.threshold = thresholdSum / thresholdCount;
                summary.baseThreshold = baseThresholdSum / baseThresholdCount;
                summary.claimWindow = maxClaimWindow;
                summary.splitNeeded = maxSplitNeeded;
                summary.pendingThreshold = summary.threshold;
                groups.push_back(std::move(summary));
            }
            return groups;
        };

        auto buildExclusiveBeforeGroups = [&]() {
            std::vector<BarrierGroupThresholdSummary> groups;
            auto grouped =
                groupBarrierTokens(collectBarrierTokens(regionStart, regionEnd, /*useSrc=*/false));
            for (const auto& group : grouped) {
                int thresholdSum = 0;
                int thresholdCount = 0;
                int baseThresholdSum = 0;
                int baseThresholdCount = 0;
                int maxClaimWindow = 0;
                bool hasPrimary = false;
                bool hasCross = false;
                for (StinkyInstruction* barrier : group.barriers) {
                    auto thIt = barrierWmmaThresholds_.find(barrier);
                    if (thIt == barrierWmmaThresholds_.end()) continue;
                    thresholdSum += thIt->second;
                    thresholdCount++;
                    auto pIt = beforeThresholds.find(barrier);
                    if (pIt != beforeThresholds.end()) {
                        hasPrimary = true;
                        baseThresholdSum += pIt->second.baseBeforeThreshold;
                        baseThresholdCount++;
                        maxClaimWindow =
                            std::max(maxClaimWindow, std::max(0, pIt->second.wmmaWindowsNeeded));
                    }
                    if (afterThresholds.find(barrier) != afterThresholds.end()) hasCross = true;
                }
                if (!hasPrimary || hasCross || thresholdCount == 0 || baseThresholdCount == 0)
                    continue;
                BarrierGroupThresholdSummary summary;
                summary.anchor = group.barriers.front();
                summary.barriers = group.barriers;
                summary.threshold = thresholdSum / thresholdCount;
                summary.baseThreshold = baseThresholdSum / baseThresholdCount;
                summary.claimWindow = maxClaimWindow;
                // before: split demand matches claim window (prior window usage).
                summary.splitNeeded = maxClaimWindow;
                summary.pendingThreshold = summary.threshold;
                groups.push_back(std::move(summary));
            }
            return groups;
        };

        auto exclusiveAfterGroups = buildExclusiveAfterGroups();
        auto exclusiveBeforeGroups = buildExclusiveBeforeGroups();
        std::unordered_set<StinkyInstruction*> overlappingHideBudgetBarriers;
        const int totalWmma = std::max(1, wmmaIssueConfig.issuedCount);
        const int targetTensorLoadWmmaSpace = this->tensorLoadWmmaSpace();

        // Collect transitive def-use descendants of \p seeds matching \p isMatch.
        // Traversal stops at instructions outside this scheduling region: their own
        // descendants can never map to a valid hard constraint (the scheduler only
        // knows about in-region instructions), so there is nothing to gain by
        // walking past them.
        auto collectDescendants = [&](const std::vector<StinkyInstruction*>& seeds,
                                      auto&& isMatch) {
            std::vector<StinkyInstruction*> pending(seeds.begin(), seeds.end());
            std::unordered_set<StinkyInstruction*> visited;
            std::vector<StinkyInstruction*> result;
            while (!pending.empty()) {
                StinkyInstruction* descendant = pending.back();
                pending.pop_back();
                if (!visited.insert(descendant).second) continue;
                if (isMatch(*descendant)) result.push_back(descendant);
                if (!regionInsts.contains(descendant)) continue;
                for (StinkyInstruction* user : descendant->getUsers()) pending.push_back(user);
            }
            return result;
        };

        // Computed once per group (not once per after x before pair).
        for (auto& afterGroup : exclusiveAfterGroups) {
            afterGroup.descendantLoads = collectDescendants(
                afterGroup.barriers, [](StinkyInstruction& inst) { return isTensorLoad(inst); });
        }
        for (auto& beforeGroup : exclusiveBeforeGroups) {
            beforeGroup.descendantLoads = collectDescendants(
                beforeGroup.barriers, [](StinkyInstruction& inst) { return isDSRead(inst); });
        }

        // Compute every after-before overlap from the same unmodified group
        // thresholds. Split the available WMMA windows in proportion to each
        // side's demand, then apply the most conservative target after all pairs
        // have been evaluated so the result is independent of iteration order.
        for (auto& afterGroup : exclusiveAfterGroups) {
            for (auto& beforeGroup : exclusiveBeforeGroups) {
                const int baseAfterEnd = afterGroup.baseThreshold;
                const int baseBeforeBegin = beforeGroup.baseThreshold;
                const int baseAfterBegin = std::max(0, baseAfterEnd - afterGroup.claimWindow);
                const int baseBeforeEnd = baseBeforeBegin + beforeGroup.claimWindow;
                const bool overlap =
                    (baseAfterBegin < baseBeforeEnd) && (baseBeforeBegin <= baseAfterEnd);
                int proportionalSplit = -1;
                if (overlap) {
                    overlappingHideBudgetBarriers.insert(afterGroup.barriers.begin(),
                                                         afterGroup.barriers.end());
                    overlappingHideBudgetBarriers.insert(beforeGroup.barriers.begin(),
                                                         beforeGroup.barriers.end());
                    // afterNeeded is issue-only (no drain); beforeNeeded is the
                    // before claim window.
                    const int afterNeeded = std::max(0, afterGroup.splitNeeded);
                    const int beforeNeeded = std::max(0, beforeGroup.splitNeeded);
                    const int totalNeeded = afterNeeded + beforeNeeded;
                    proportionalSplit = totalNeeded > 0
                                            ? static_cast<int>(static_cast<long long>(totalWmma) *
                                                               afterNeeded / totalNeeded)
                                            : totalWmma / 2;
                    const int afterTarget = std::clamp(proportionalSplit - 1, 0, totalWmma);
                    const int beforeTarget = std::clamp(proportionalSplit + 1, 0, totalWmma);

                    // Do not publish overlap yet: some policy constraints can be rejected
                    // later when merged into the register DAG. The scheduler validates
                    // every requested ordering against final instruction order first.
                    Layer2BarrierOverlapCandidate overlapCandidate{
                        afterGroup.barriers, beforeGroup.barriers, {}};
                    auto requireOrdering = [&](StinkyInstruction* predecessor,
                                               StinkyInstruction* successor) {
                        deps.requestedConstraints.emplace_back(predecessor, successor);
                        overlapCandidate.requiredConstraints.emplace_back(predecessor, successor);
                    };

                    // Every overlapping pair needs structural ordering, independent of
                    // which threshold-adjustment branch above was taken.
                    for (StinkyInstruction* barrierAfter : afterGroup.barriers) {
                        for (StinkyInstruction* barrierBefore : beforeGroup.barriers) {
                            requireOrdering(barrierAfter, barrierBefore);
                        }
                    }

                    // Also keep tensor_load descendants of barrierAfter ahead of the
                    // barrierBefore group, and prevent them from interleaving with
                    // ds_load descendants of barrierBefore: all matching tensor_loads
                    // must issue first. Both descendant sets were precomputed once per
                    // group above.
                    for (StinkyInstruction* tensorLoad : afterGroup.descendantLoads) {
                        for (StinkyInstruction* barrierBefore : beforeGroup.barriers) {
                            requireOrdering(tensorLoad, barrierBefore);
                        }
                        for (StinkyInstruction* dsLoad : beforeGroup.descendantLoads) {
                            requireOrdering(tensorLoad, dsLoad);
                        }
                    }
                    layer2BarrierOverlapCandidates_.push_back(std::move(overlapCandidate));

                    afterGroup.pendingThreshold =
                        std::min(afterGroup.pendingThreshold, afterTarget);
                    beforeGroup.pendingThreshold =
                        std::max(beforeGroup.pendingThreshold, beforeTarget);
                }

                PASS_DEBUG(std::cerr
                           << "[CDNA5 onInitRegion after-before exclusive overlap] "
                              "baseAfterThreshold="
                           << afterGroup.baseThreshold
                           << " baseBeforeThreshold=" << beforeGroup.baseThreshold
                           << " currentAfterThreshold=" << afterGroup.threshold
                           << " currentBeforeThreshold=" << beforeGroup.threshold << " afterWindow="
                           << afterGroup.claimWindow << " beforeWindow=" << beforeGroup.claimWindow
                           << " totalWmma=" << totalWmma << " afterGroupAnchor="
                           << afterGroup.anchor << " afterGroupSize=" << afterGroup.barriers.size()
                           << " beforeGroupAnchor=" << beforeGroup.anchor
                           << " beforeGroupSize=" << beforeGroup.barriers.size()
                           << " afterWmmaWindow=" << afterGroup.claimWindow
                           << " beforeWmmaWindow=" << beforeGroup.claimWindow
                           << " overlap=" << overlap << " baseAfterEnd=" << baseAfterEnd
                           << " baseBeforeBegin=" << baseBeforeBegin
                           << " proportionalSplit=" << proportionalSplit
                           << " pendingAfterThreshold=" << afterGroup.pendingThreshold
                           << " pendingBeforeThreshold=" << beforeGroup.pendingThreshold << "\n");
            }
        }

        for (auto& group : exclusiveAfterGroups) group.threshold = group.pendingThreshold;
        for (auto& group : exclusiveBeforeGroups) group.threshold = group.pendingThreshold;

        // Apply tensor-load WMMA spacing once per exclusive group (not once per
        // after×before pair), so thresholds do not compound with group count.
        // Thresholds are written back once below after this optional adjust.
        if (targetTensorLoadWmmaSpace > 0) {
            const int deltaAfter = targetTensorLoadWmmaSpace / 2;
            const int deltaBefore = (targetTensorLoadWmmaSpace + 1) / 2;
            for (auto& afterGroup : exclusiveAfterGroups) {
                afterGroup.threshold = std::clamp(afterGroup.threshold - deltaAfter, 0, totalWmma);
                PASS_DEBUG(std::cerr << "[CDNA5 onInitRegion tensorLoadWmmaSpace] afterGroupAnchor="
                                     << afterGroup.anchor << " threshold=" << afterGroup.threshold
                                     << " deltaAfter=" << deltaAfter
                                     << " targetTensorLoadWmmaSpace=" << targetTensorLoadWmmaSpace
                                     << "\n");
            }
            for (auto& beforeGroup : exclusiveBeforeGroups) {
                beforeGroup.threshold =
                    std::clamp(beforeGroup.threshold + deltaBefore, 0, totalWmma);
                PASS_DEBUG(std::cerr
                           << "[CDNA5 onInitRegion tensorLoadWmmaSpace] beforeGroupAnchor="
                           << beforeGroup.anchor << " threshold=" << beforeGroup.threshold
                           << " deltaBefore=" << deltaBefore
                           << " targetTensorLoadWmmaSpace=" << targetTensorLoadWmmaSpace << "\n");
            }
        }

        for (const auto& group : exclusiveAfterGroups) setGroupThreshold(group, group.threshold);
        for (const auto& group : exclusiveBeforeGroups) setGroupThreshold(group, group.threshold);

        // Final pair normalization: keep barrier_signal/barrier_wait pairs on the
        // same threshold.
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
                                     << group.barriers.size() << " mergedThreshold="
                                     << mergedThreshold << " useSrcTokens=" << useSrcTokens
                                     << " sum=" << sum << " count=" << count << "\n");
            }
        };
        // Layer 3/3 (final pair normalize):
        // Run once with source-token grouping and once with destination-token
        // grouping, because different barrier forms expose their pseudo token on
        // different operand sides. This final pass guarantees each grouped
        // barrier_signal/barrier_wait pair shares one threshold.
        normalizeBarrierPairs(/*useSrcTokens=*/true);
        normalizeBarrierPairs(/*useSrcTokens=*/false);

        // Publish the final, normalized threshold together with each barrier
        // estimator's DS-load demand. Publish once per split-barrier group:
        // computeBarrier* stores the same group result on both signal and wait
        // so the scheduler can look either half up, but counting both here would
        // double the group's DS demand. A group present in both maps still
        // intentionally contributes one Before and one After record.
        auto finalThresholdFor = [&](StinkyInstruction* barrier, int fallback) {
            auto it = barrierWmmaThresholds_.find(barrier);
            return it == barrierWmmaThresholds_.end() ? fallback : it->second;
        };
        auto groupOverlaps = [&](const BarrierTokenGroup& group) {
            return std::any_of(group.barriers.begin(), group.barriers.end(),
                               [&](StinkyInstruction* barrier) {
                                   return overlappingHideBudgetBarriers.contains(barrier);
                               });
        };
        for (const BarrierTokenGroup& group :
             groupBarrierTokens(collectBarrierTokens(regionStart, regionEnd, /*useSrc=*/true))) {
            auto outputIt = afterThresholds.end();
            for (StinkyInstruction* barrier : group.barriers) {
                outputIt = afterThresholds.find(barrier);
                if (outputIt != afterThresholds.end()) break;
            }
            if (outputIt == afterThresholds.end()) continue;
            StinkyInstruction* anchor = group.barriers.front();
            const BarrierAfterOutput& output = outputIt->second;
            hideBudgetBarriers.push_back({anchor, WmmaHideBudgetBarrierPosition::After,
                                          finalThresholdFor(anchor, output.afterThreshold),
                                          output.dsLoadCount, output.wmmaWindowsNeeded,
                                          groupOverlaps(group)});
        }
        for (const BarrierTokenGroup& group :
             groupBarrierTokens(collectBarrierTokens(regionStart, regionEnd, /*useSrc=*/false))) {
            auto outputIt = beforeThresholds.end();
            for (StinkyInstruction* barrier : group.barriers) {
                outputIt = beforeThresholds.find(barrier);
                if (outputIt != beforeThresholds.end()) break;
            }
            if (outputIt == beforeThresholds.end()) continue;
            StinkyInstruction* anchor = group.barriers.back();
            const BarrierBeforeOutput& output = outputIt->second;
            hideBudgetBarriers.push_back({anchor, WmmaHideBudgetBarrierPosition::Before,
                                          finalThresholdFor(anchor, output.beforeThreshold),
                                          output.dsLoadCount, output.wmmaWindowsNeeded,
                                          groupOverlaps(group)});
        }
    }

    // Run after every barrier placement and normalization step so the analysis
    // sees the same final thresholds that the scheduler will enforce.
    if (hideBudgetPrescanEnabled()) {
        hideBudget_ = analyzeWmmaHideBudget(deps.dag, hideBudgetBarriers, wmmaHideBudgetBase,
                                            dsLoadBudgetConfig());
        PASS_DEBUG(std::cerr << "[CDNA5 hideBudget] windows=" << hideBudget_.numWindows()
                             << " wmmaInstructions=" << hideBudget_.wmmaInstructionCount
                             << " nonWmmaInstructions=" << hideBudget_.nonWmmaInstructionCount
                             << " dsLoadInstructions=" << hideBudget_.dsLoadInstructionCount
                             << " nonDsLoadInstructions=" << hideBudget_.nonDsLoadInstructionCount
                             << " wmmaHideBudgetBase=" << hideBudget_.wmmaHideBudgetBase
                             << " barriers=" << hideBudget_.barriers.size() << "\n");
    }
}
}  // namespace
