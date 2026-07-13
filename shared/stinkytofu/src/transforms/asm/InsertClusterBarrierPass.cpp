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
#include "stinkytofu/transforms/asm/InsertClusterBarrierPass.hpp"

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {
namespace {

constexpr int kClusterBarrierId = -3;
constexpr int kWorkgroupBarrierId = -1;
/// Prefix for the WaveIdx-gated skip label (only wave 0 issues the barrier).
constexpr const char* kSkipLabelPrefix = "label_skipCBPreSignal_";
/// Prefix for the outer LoopCounterL-gated skip label guarding a cluster
/// SIGNAL. Distinct from the wait prefix below so signal and wait drain gates
/// read differently in the emitted assembly.
constexpr const char* kSkipSignalLabelPrefixLCL = "label_skipCBSignal_LCL_";
/// Prefix for Rule 4's drain-gated cluster-WAIT skip label (drain-iter gate).
/// Used by `insertLoopCounterLGatedClusterBarrierWaitBefore` when the LCL
/// drain gate is enabled (`kClusterBarrierDrainGateEnabled == true`).
constexpr const char* kSkipWaitLabelPrefixLCL = "label_skipCBWait_LCL_";
constexpr const char* kWaveIdxSymbol = "sgprWaveIdx";
constexpr const char* kLoopCounterLSymbol = "sgprLoopCounterL";
/// Exact label name emitted by Tensile after the GSU==1 early-return
/// guard. Rule 1 anchors on the `LABEL` instruction with this name and
/// emits the signal-only handshake immediately after it.
constexpr const char* kGSU1LabelName = "label_GSU_1";
/// Exact label name emitted by Tensile right before the unrolled
/// summation loop opens. Rule 3 anchors after an existing publication
/// `s_barrier_wait -1` when the backward scan finds one, or at this label
/// (synthesizing a workgroup sync pair) when no publication wait exists
/// before the prefetch boundary.
/// Internal labels inside the prefetch prologue (e.g. `label_skipPGR2_1`)
/// do not match by exact-name comparison and are walked through.
constexpr const char* kOpenLoopLLabelName = "label_openLoopL";
/// Exact label name Tensile emits at the end of the unrolled summation loop.
/// Rule 6: a single bare `s_barrier_wait -3` is planted immediately after
/// this label so the per-iteration cluster wait moves out of the loop body.
constexpr const char* kLoopEndLLabelName = "label_LoopEndL";
/// Substring used to identify the Tensile comment that opens the tail-loop
/// section. Matches the TEXTBLOCK `/* Tail Loop                       */`.
/// Rule 5 (5a + 5b) uses this as its section anchor.
constexpr const char* kTailLoopMarker = "Tail Loop";

/// Master switch selecting the cluster-barrier handshake scheme across ALL
/// rules (1, 3, 4, and 6) -- not just Rule 4. The two schemes are mutually
/// exclusive and must move together to keep `signal -3` / `wait -3` balanced.
/// DEFAULT: true (the GATED scheme).
///
///   - When true (GATED scheme, DEFAULT): every per-iteration / priming cluster signal
///     is wrapped in a LoopCounterL drain gate so it is suppressed on exactly
///     the drain iterations whose paired counterpart is also skipped:
///       * Rule 4: ASYMMETRIC gates on the wait+signal handshake -- the first
///         publication point since the preceding `s_sub LCL` skips its WAIT at
///         `LCL <= pgrValue` and every later pub in the same LCL window skips
///         both halves at `LCL <= pgrValue+1`; the SIGNAL half is always one
///         stage earlier at `LCL <= pgrValue+1`. Both thresholds are lowered by
///         the per-anchor hoisted `lclPreDecrement` (e.g. PGR=1 with three TDM
///         pubs: 1,2,2,2,2,2; PGR=2 first pub 2,3 and later pubs 3,3).
///         When `pgrValue == 0`
///         the wait-side threshold would never skip while the signal-side gate
///         still would (hang); Rules 2/3 are suppressed and Rule 4 emits a bare
///         wait plus an LCL-gated signal only (see `insertClusterBarrierHandshakeBefore`).
///       * Rule 1: priming signal gated on `LCL != 0` (with a leading
///         workgroup sync).
///       * Rule 3: publication-point signal gated on `LCL <= pgrValue` (suppressed
///         when `pgrValue == 0` -- no prefetch publication point to prime).
///       * Rule 6: DISABLED -- the asymmetric Rule 4 gate already leaves the
///         loop with one extra in-loop WAIT that consumes the last signal, so
///         a trailing loop-exit wait would be unpaired.
///   - When false (UNGATED scheme): Rules 1/3/4 emit a plain WaveIdx-gated
///     signal-only (Rule 4: bare `s_barrier_wait -3` then the signal), and
///     Rule 6 plants a single loop-exit `s_barrier_wait -3` to consume the
///     loop's last otherwise-orphaned signal. (This describes the ungated
///     wait-before-signal ordering; the `kRule4SignalBeforeWaitEnabled == true`
///     legacy ordering self-pairs each iteration and therefore disables BOTH
///     Rule 3 and Rule 6 -- see that switch below.)
///
/// In BOTH cases `findLiveSccCmpUpstream` is consulted: if SIA hoisted a live
/// loop-exit `s_cmp_* LCL, imm` whose SCC a downstream `s_cbranch_scc{0,1}`
/// consumes, a clone of it is re-emitted AFTER the entire inserted block
/// (past every skip label, so it runs on the gated AND skip paths) to rebuild
/// that SCC. Placing the restore last is what lets the drain gate coexist with
/// a live SCC -- the gate's own `s_cmp_le` clobbers SCC, but the trailing
/// clone re-establishes it for all outgoing paths.
constexpr bool kClusterBarrierDrainGateEnabled = true;

/// Rule 4 (ungated scheme only) handshake ordering. Ignored by default because
/// the drain gate is enabled by default (the gated scheme owns ordering); it
/// only takes effect when `kClusterBarrierDrainGateEnabled` is manually turned
/// off. When false the ungated per-iteration handshake emits the bare
/// `s_barrier_wait -3` FIRST and then the
/// WaveIdx-gated `s_barrier_signal -3` (the offset ping-pong: each wait consumes
/// the PREVIOUS signal, and the loop's last signal is drained by Rule 6's
/// loop-exit wait). When true it reverts to the legacy pre-rewrite layout: the
/// WaveIdx-gated `s_barrier_signal -3` FIRST, then the bare `s_barrier_wait -3`
/// LAST, so the wait stays the load's immediate predecessor and Rule 2's
/// `isImmediatelyPrecededByClusterBarrierWait` idempotency guard keeps working.
/// Any live-SCC restore clone is then placed immediately BEFORE that trailing
/// wait (not after the whole block).
///
/// Because this layout self-pairs each iteration's signal with its own trailing
/// wait, the loop is already balanced without the offset ping-pong's boundary
/// pieces: when true (and the drain gate is off) BOTH the Rule 3 priming signal
/// and the Rule 6 loop-exit drain wait are disabled (they would otherwise be
/// unpaired). Ignored when `kClusterBarrierDrainGateEnabled == true` -- the
/// gated scheme owns ordering and keeps its own Rule 3 gate.
constexpr bool kRule4SignalBeforeWaitEnabled = false;

/// Build the skip-label name for a given prefix and cluster-barrier
/// *generation* number. The pass numbers skip labels directly at insertion
/// time (see `InsertClusterBarrierPassImpl::run`) so the suffix reflects which
/// generation of the offset ping-pong the guarded signal/wait belongs to,
/// counting from 0 at the top of each kernel.
std::string makeSkipLabel(const char* prefix, uint64_t gen) {
    return std::string(prefix) + std::to_string(gen);
}

/// Build a symbolic SGPR reference (single dword) for emission as `s[<name>]`.
StinkyRegister makeSymbolicSgpr(const std::string& symbolicName) {
    StinkyRegister reg(RegType::S, /*regIdx=*/0u, /*regNum=*/1u);
    reg.setSymbolicName(symbolicName);
    return reg;
}

/// Shared primitive for the three "is barrier with literal id" checks below.
/// Matches either an `s_barrier_wait` (`wantSignal=false`) or an
/// `s_barrier_signal` (`wantSignal=true`) whose first source operand is a
/// LiteralInt equal to `id`. When `rejectMemToken` is true, also requires
/// the instruction to have no MemTokenData modifier (used to keep the
/// cluster-scope wait idempotency check independent of Tensile-emitted
/// `wait_kmcnt`-style modifiers).
bool isBarrierWithLiteralId(const StinkyInstruction& inst, bool wantSignal, int id,
                            bool rejectMemToken) {
    if (wantSignal ? !isBarrierSignal(inst) : !isBarrierWait(inst)) return false;
    if (rejectMemToken && inst.getModifier<MemTokenData>() != nullptr) return false;
    const auto& srcs = inst.getSrcRegs();
    return !srcs.empty() && srcs[0].dataType == StinkyRegister::Type::LiteralInt &&
           srcs[0].getLiteralInt() == id;
}

/// True if `inst` is a workgroup-scope barrier completion: `s_barrier_wait -1`.
/// The cluster-scope (`-3`) wait we synthesize is intentionally excluded so the
/// pass remains idempotent if re-run.
bool isWorkgroupBarrierWait(const StinkyInstruction& inst) {
    return isBarrierWithLiteralId(inst, /*wantSignal=*/false, kWorkgroupBarrierId,
                                  /*rejectMemToken=*/false);
}

/// True if `inst` is a workgroup-scope signal: `s_barrier_signal -1`. Paired
/// with the workgroup wait above; Rule 4 uses it to place the cluster
/// `s_barrier_wait -3` immediately BEFORE this signal (see
/// `findPrecedingWorkgroupBarrierSignalInBB`).
bool isWorkgroupBarrierSignal(const StinkyInstruction& inst) {
    return isBarrierWithLiteralId(inst, /*wantSignal=*/true, kWorkgroupBarrierId,
                                  /*rejectMemToken=*/false);
}

/// True if `inst` is a bare cluster-scope wait (`s_barrier_wait -3` with no
/// MemTokenData). Used as an idempotency signature: the standalone wait we
/// synthesize for Rule 2, the leading wait of any Rule-4 handshake, and any
/// equivalent wait already present in the IR all match.
bool isClusterBarrierWait(const StinkyInstruction& inst) {
    return isBarrierWithLiteralId(inst, /*wantSignal=*/false, kClusterBarrierId,
                                  /*rejectMemToken=*/true);
}

/// True if `inst` is `s_barrier_signal -3`. Mirror of `isClusterBarrierWait`
/// used by Rule 3's synthesized-publication path for section-level idempotency.
bool isClusterBarrierSignal(const StinkyInstruction& inst) {
    return isBarrierWithLiteralId(inst, /*wantSignal=*/true, kClusterBarrierId,
                                  /*rejectMemToken=*/false);
}

/// Walk backward from \p anchor (exclusive) toward the containing basic block's
/// entry to find the nearest preceding `s_barrier_wait -1`. Stops at the BB
/// boundary so the trigger never crosses a CFG edge. Used by Rule 4's
/// per-`tensor_load_to_lds` scan to resolve each load's anchor wait.
StinkyInstruction* findPrecedingWorkgroupBarrierWaitInBB(StinkyInstruction* anchor) {
    BasicBlock* parent = anchor->getParent();
    if (parent == nullptr) return nullptr;
    auto it = BasicBlock::iterator(anchor);
    while (it != parent->begin()) {
        --it;
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (isWorkgroupBarrierWait(*inst)) return inst;
    }
    return nullptr;
}

/// Walk backward from \p anchor (exclusive, the load's anchoring
/// `s_barrier_wait -1`) toward the containing basic block's entry to find the
/// nearest preceding `s_barrier_signal -1` -- the workgroup signal that pairs
/// with the anchor wait. Rule 4's wait-before-signal schemes place the cluster
/// `s_barrier_wait -3` immediately BEFORE this signal; nullptr means no
/// workgroup signal was found in the BB, in which case the caller falls back to
/// emitting the cluster wait at the post-`wait -1` anchor.
StinkyInstruction* findPrecedingWorkgroupBarrierSignalInBB(StinkyInstruction* anchor) {
    BasicBlock* parent = anchor->getParent();
    if (parent == nullptr) return nullptr;
    auto it = BasicBlock::iterator(anchor);
    while (it != parent->begin()) {
        --it;
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (isWorkgroupBarrierSignal(*inst)) return inst;
    }
    return nullptr;
}

/// True if `uOp` is a scalar value compare (`s_cmp_*`). These are the only
/// SCC producers that can be safely re-emitted to restore SCC, because they
/// write SCC and nothing else (no GPR destination). `s_bitcmp1_b32` is
/// intentionally excluded: it is a bit-test, not an `s_cmp_*` value compare.
bool isCompareClass(uint16_t uOp) {
    switch (uOp) {
        case GFX::s_cmp_eq_i32:
        case GFX::s_cmp_eq_u32:
        case GFX::s_cmp_eq_u64:
        case GFX::s_cmp_ge_i32:
        case GFX::s_cmp_ge_u32:
        case GFX::s_cmp_gt_i32:
        case GFX::s_cmp_gt_u32:
        case GFX::s_cmp_le_i32:
        case GFX::s_cmp_le_u32:
        case GFX::s_cmp_lg_i32:
        case GFX::s_cmp_lg_u32:
        case GFX::s_cmp_lg_u64:
        case GFX::s_cmp_lt_i32:
        case GFX::s_cmp_lt_u32:
            return true;
        default:
            return false;
    }
}

/// Walk backward from \p anchor within the same basic block, looking for a
/// "live" scalar compare (`s_cmp_*`) whose SCC has NOT been consumed or
/// overwritten before \p anchor. Returns nullptr on BB start, label, branch,
/// an SCC reader encountered first, or when the first SCC writer encountered
/// is not a value compare.
///
/// Used by Rule 4 to detect whether SIA scheduling has hoisted a loop-exit
/// compare above the anchor (typical for `ScheduleIterAlg=4`) whose SCC a
/// downstream `s_cbranch_scc{0,1}` still consumes. The WaveIdx `s_cmp_eq_u32`
/// that the cluster-signal block emits clobbers SCC, so when such a live
/// compare exists a clone of it is re-emitted after the inserted block to
/// rebuild SCC (see `insertClusterBarrierHandshakeBefore`).
///
/// Detection rule (a minimal SCC liveness walk): the FIRST SCC event above
/// the anchor decides the result.
///   - SCC reader first (cbranch / IF_ImplicitReadSCC)  -> nullptr (consumed).
///   - SCC writer first:
///       * value compare (`s_cmp_*`) -> live, return it (safe to re-emit:
///         writes SCC only, no GPR destination).
///       * any other writer (s_add_u32, s_and_b32, ...) -> nullptr. It is the
///         real live-SCC producer (it overwrote any compare above it) but
///         cannot be cloned without corrupting its GPR destination, so we give
///         up rather than restore a wrong/stale SCC.
///   - label / unconditional branch -> nullptr (control-flow boundary).
///
/// The compare form and operands are unrestricted: restoration only re-runs
/// the identical compare to rebuild the same SCC, so `le`/`lt`/`ge`/... and
/// any source register are all acceptable (unlike the legacy gate use, which
/// required `s_cmp_eq LCL`).
StinkyInstruction* findLiveSccCmpUpstream(StinkyInstruction* anchor) {
    BasicBlock* parent = anchor->getParent();
    if (parent == nullptr) return nullptr;
    auto it = BasicBlock::iterator(anchor);
    while (it != parent->begin()) {
        --it;
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (isPseudoInst(inst)) continue;
        if (isLabel(*inst)) return nullptr;
        if (isUnconditionalBranch(*inst)) return nullptr;
        if (isConditionalBranch(*inst) || inst->is(InstFlag::IF_ImplicitReadSCC)) {
            return nullptr;
        }
        if (inst->is(InstFlag::IF_ImplicitWriteSCC)) {
            // First SCC producer above the anchor. Restore only if it is a
            // value compare; any other SCC writer is the true live producer
            // yet un-restorable (it has a GPR destination), so give up.
            return isCompareClass(inst->getUnifiedOpcode()) ? inst : nullptr;
        }
    }
    return nullptr;
}

/// True if `inst` is an unconditional self-decrement of the loop counter:
/// `s_sub_{u32,i32} s[sgprLoopCounterL], s[sgprLoopCounterL], <imm>`. The
/// destination and first source must both be `s[sgprLoopCounterL]` and the
/// second source must be a literal immediate. On a match, `*outImm` receives
/// the decrement amount. `s_subrev_*` is intentionally rejected: its operand
/// order (`dst = src1 - src0`) does not express `LCL -= imm`.
bool isLoopCounterLSelfDecrement(const StinkyInstruction& inst, int* outImm) {
    const auto uOp = inst.getUnifiedOpcode();
    if (uOp != GFX::s_sub_u32 && uOp != GFX::s_sub_i32) return false;
    const auto& dsts = inst.getDestRegs();
    const auto& srcs = inst.getSrcRegs();
    if (dsts.empty() || srcs.size() < 2) return false;
    if (dsts[0].getSymbolicName() != kLoopCounterLSymbol) return false;
    if (srcs[0].getSymbolicName() != kLoopCounterLSymbol) return false;
    if (srcs[1].dataType != StinkyRegister::Type::LiteralInt) return false;
    if (outImm != nullptr) *outImm = static_cast<int>(srcs[1].getLiteralInt());
    return true;
}

/// Sum the immediate decrements that `s_sub_{u32,i32} s[sgprLoopCounterL],
/// s[sgprLoopCounterL], <imm>` instructions apply to the loop counter between
/// the containing basic block's entry and \p anchor (exclusive), scanning
/// backward and stopping at the BB boundary.
///
/// Rule 4's asymmetric drain-gate thresholds (`pgrValue` for the first pub's
/// WAIT, `pgrValue+1` for later pubs' WAIT and every SIGNAL) are calibrated
/// against the loop counter value at basic-block entry. Different
/// `ScheduleIterAlg` settings may hoist the per-iteration `s_sub LCL, LCL, 1`
/// ABOVE the workgroup-wait anchor, so the gate then reads an already-decremented
/// LCL. To keep the gate firing on the identical absolute iteration regardless
/// of where the decrement landed, the drain gate subtracts this sum from both
/// thresholds (WAIT: nominal - lclPreDecrement; SIGNAL: pgr+1 - lclPreDecrement).
/// Decrements
/// that remain BELOW the anchor (the default schedule) are not seen by the
/// backward scan, so the sum is 0 and the thresholds are left untouched. (Not
/// consulted when the drain gate is disabled, i.e. `kClusterBarrierDrainGateEnabled
/// == false`.)
int sumLoopCounterLDecrementsBeforeInBB(StinkyInstruction* anchor) {
    BasicBlock* parent = anchor->getParent();
    if (parent == nullptr) return 0;
    int total = 0;
    auto it = BasicBlock::iterator(anchor);
    while (it != parent->begin()) {
        --it;
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (isPseudoInst(inst)) continue;
        int imm = 0;
        if (isLoopCounterLSelfDecrement(*inst, &imm)) total += imm;
    }
    return total;
}

/// True when any `s_sub LCL` sits strictly between \p fromExclusive and
/// \p toInclusive in the same basic block (forward scan).
bool hasLoopCounterLDecrementBetweenInBB(StinkyInstruction* fromExclusive,
                                         StinkyInstruction* toInclusive) {
    BasicBlock* parent = toInclusive->getParent();
    if (parent == nullptr || fromExclusive == nullptr || fromExclusive->getParent() != parent) {
        return false;
    }
    auto it = std::next(BasicBlock::iterator(fromExclusive));
    const auto endIt = BasicBlock::iterator(toInclusive);
    for (; it != endIt; ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (isPseudoInst(inst)) continue;
        if (isLoopCounterLSelfDecrement(*inst, /*outImm=*/nullptr)) return true;
    }
    return false;
}

/// Sum `s_sub LCL` immediates strictly between \p fromExclusive and
/// \p toInclusive in the same basic block (forward scan).
int sumLoopCounterLDecrementsBetweenInBB(StinkyInstruction* fromExclusive,
                                         StinkyInstruction* toInclusive) {
    BasicBlock* parent = toInclusive->getParent();
    if (parent == nullptr || fromExclusive == nullptr || fromExclusive->getParent() != parent) {
        return 0;
    }
    int total = 0;
    auto it = std::next(BasicBlock::iterator(fromExclusive));
    const auto endIt = BasicBlock::iterator(toInclusive);
    for (; it != endIt; ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (isPseudoInst(inst)) continue;
        int imm = 0;
        if (isLoopCounterLSelfDecrement(*inst, &imm)) total += imm;
    }
    return total;
}

struct Rule3PublicationScan {
    StinkyInstruction* existingWait = nullptr;
    bool sectionAlreadyHasClusterBarrier = false;
    IRBase* anchor = nullptr;
    bool needsWorkgroupSync = false;
    bool enabled = false;
};

/// Walk backward from \p openLoopLLabel through its basic block and that block's
/// CFG predecessors (one level toward the prefetch prologue). Stops at the first
/// `tensor_load_to_lds`, which marks the prefetch boundary. Returns the nearest
/// preceding workgroup `s_barrier_wait -1` when present; otherwise requests
/// synthesizing the publication point immediately before \p openLoopLLabel.
Rule3PublicationScan scanRule3PublicationPoint(StinkyInstruction* openLoopLLabel) {
    Rule3PublicationScan result;
    BasicBlock* labelBB = openLoopLLabel->getParent();
    if (labelBB == nullptr) return result;

    std::vector<std::pair<BasicBlock*, BasicBlock::iterator>> segments;
    segments.emplace_back(labelBB, BasicBlock::iterator(openLoopLLabel));
    for (BasicBlock* pred : labelBB->getPredecessors()) {
        segments.emplace_back(pred, pred->end());
    }

    bool hitPrefetchBoundary = false;
    auto scanBackward = [&](BasicBlock* block, BasicBlock::iterator start) {
        if (start == block->end()) {
            for (auto it = block->rbegin(); it != block->rend(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst == nullptr) continue;
                if (isTensorLoad(*inst)) {
                    hitPrefetchBoundary = true;
                    return;
                }
                if (isClusterBarrierWait(*inst) || isClusterBarrierSignal(*inst)) {
                    result.sectionAlreadyHasClusterBarrier = true;
                }
                if (isWorkgroupBarrierWait(*inst)) {
                    result.existingWait = inst;
                    return;
                }
            }
            return;
        }
        auto it = start;
        while (it != block->begin()) {
            --it;
            auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (inst == nullptr) continue;
            if (isTensorLoad(*inst)) {
                hitPrefetchBoundary = true;
                return;
            }
            if (isClusterBarrierWait(*inst) || isClusterBarrierSignal(*inst)) {
                result.sectionAlreadyHasClusterBarrier = true;
            }
            if (isWorkgroupBarrierWait(*inst)) {
                result.existingWait = inst;
                return;
            }
        }
    };

    for (auto& [block, start] : segments) {
        scanBackward(block, start);
        if (hitPrefetchBoundary || result.existingWait != nullptr) break;
    }

    if (result.sectionAlreadyHasClusterBarrier) return result;

    if (result.existingWait != nullptr) {
        BasicBlock* waitBB = result.existingWait->getParent();
        auto nextIt = std::next(BasicBlock::iterator(result.existingWait));
        result.anchor =
            (nextIt != waitBB->end()) ? nextIt.getNodePtr() : static_cast<IRBase*>(openLoopLLabel);
        result.enabled = true;
    } else {
        result.anchor = openLoopLLabel;
        result.needsWorkgroupSync = true;
        result.enabled = true;
    }
    return result;
}

/// Marker-bounded forward scan: walk from \p start (inclusive) up to
/// \p endExclusive and return the first `tensor_load_to_lds` encountered, or
/// nullptr. Does NOT stop at labels or branches -- Rule 5b's tail-loop search
/// must cross the tail's own control-flow labels (e.g.
/// `label_TailLoopBegin_L:`) to reach the tail's publication-point load.
StinkyInstruction* findFirstTensorLoadBetween(BasicBlock::iterator start,
                                              BasicBlock::iterator endExclusive) {
    for (auto it = start; it != endExclusive; ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (isTensorLoad(*inst)) return inst;
    }
    return nullptr;
}

/// Walk forward from `anchor` (exclusive) skipping over non-`StinkyInstruction`
/// IR (e.g. `AsmDirective`) and pseudo instructions (LABEL / PHI / FENCE).
/// Returns the first real `StinkyInstruction*` encountered, or nullptr if the
/// rest of the basic block contains no real instruction. Used as the primitive
/// for forward-direction idempotency checks.
StinkyInstruction* firstRealInstAfter(StinkyInstruction* anchor) {
    BasicBlock* parent = anchor->getParent();
    if (parent == nullptr) return nullptr;
    for (auto it = std::next(BasicBlock::iterator(anchor)); it != parent->end(); ++it) {
        auto* next = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (next == nullptr || isPseudoInst(next)) continue;
        return next;
    }
    return nullptr;
}

/// Emit the wave-gated signal block (cmp + branch + signal + skip-label) so
/// that only wave 0 issues the cluster-scope `s_barrier_signal -3`. All new
/// instructions are inserted before `anchor` (or appended when nullptr).
///
/// Shared by:
///   - Rule 4: emits a leading `s_barrier_wait -3` and then calls this helper.
///   - Rule 5a: calls this helper alone (no preceding wait, no LCL gate).
///   - Rules 1 / 3: wrap this helper in an outer LoopCounterL gate (see
///     `insertLoopCounterLGatedClusterBarrierSignalBefore`).
void insertClusterBarrierSignalOnlyBefore(IRBase* anchor, AsmIRBuilder& irBuilder, GfxArchID archId,
                                          uint64_t signalGen) {
    const std::string labelName = makeSkipLabel(kSkipLabelPrefix, signalGen);

    const HwInstDesc* cmpDesc = getMCIDByUOp(GFX::s_cmp_eq_u32, archId);
    const HwInstDesc* brDesc = getMCIDByUOp(GFX::s_cbranch_scc0, archId);
    const HwInstDesc* signalDesc = getMCIDByUOp(GFX::s_barrier_signal, archId);
    assert(cmpDesc && brDesc && signalDesc &&
           "Cluster-barrier opcodes are not supported on this architecture");

    StinkyInstruction* cmpInst = irBuilder.create(cmpDesc, anchor);
    cmpInst->addSrcReg(makeSymbolicSgpr(kWaveIdxSymbol));
    cmpInst->addSrcReg(StinkyRegister(0));
    cmpInst->addModifier<CommentData>(CommentData{"Check for waveID 0"});

    StinkyInstruction* brInst = irBuilder.create(brDesc, anchor);
    brInst->addSrcReg(StinkyRegister(labelName));
    brInst->addModifier<LabelData>(LabelData{labelName});
    brInst->addModifier<CommentData>(CommentData{"Execute cluster barrier signal for waveID 0"});

    StinkyInstruction* signalInst = irBuilder.create(signalDesc, anchor);
    signalInst->addSrcReg(StinkyRegister(kClusterBarrierId));
    signalInst->addModifier<CommentData>(CommentData{"cluster_barrier signal"});

    static const HwInstDesc labelMCID{
        GFX::LABEL, GFX::LABEL, 0, 0, 0, "LABEL", makeFlagSet({InstFlag::IF_HasSideEffect})};
    StinkyInstruction* lblInst = irBuilder.create(&labelMCID, anchor);
    lblInst->addModifier<LabelData>(LabelData{labelName, /*alignment=*/1});
}

/// Emit a workgroup-scope sync pair (`s_barrier_signal -1` followed by
/// `s_barrier_wait -1`) immediately before `anchor`. Always invoked
/// indirectly via `insertLoopCounterLGatedClusterBarrierSignalBefore`
/// when its `workgroupSyncWaitComment` parameter is non-null, so the
/// emitted pair always lands between the outer LCL skip-branch and the
/// inner WaveIdx gate. Used by:
///   - Rule 1: workgroup pair gates the post-GSU==1 cluster signal so
///     all waves reach the join before any wave publishes (comment:
///     `"sync workgroup before cluster signal"`).
///   - Rule 3 synthesized-publication path: when the backward scan from
///     `label_openLoopL:` finds no `s_barrier_wait -1` before crossing the
///     prefetch boundary, synthesize the LDS publication point here (comment:
///     `"workgroup sync"`) so the cluster signal that immediately follows still
///     sits at a valid LDS-coherence point.
void insertWorkgroupBarrierSyncBefore(IRBase* anchor, AsmIRBuilder& irBuilder, GfxArchID archId,
                                      const char* waitComment) {
    const HwInstDesc* signalDesc = getMCIDByUOp(GFX::s_barrier_signal, archId);
    const HwInstDesc* waitDesc = getMCIDByUOp(GFX::s_barrier_wait, archId);
    assert(signalDesc && waitDesc &&
           "Workgroup-barrier opcodes are not supported on this architecture");

    StinkyInstruction* signalInst = irBuilder.create(signalDesc, anchor);
    signalInst->addSrcReg(StinkyRegister(kWorkgroupBarrierId));

    StinkyInstruction* waitInst = irBuilder.create(waitDesc, anchor);
    waitInst->addSrcReg(StinkyRegister(kWorkgroupBarrierId));
    waitInst->addModifier<CommentData>(CommentData{waitComment});
}

/// Wrap a signal-only handshake in an outer `s[sgprLoopCounterL] <cmp> <imm>`
/// gate (the cbranch is always `s_cbranch_scc1`, so it skips the inner
/// block when the compare sets SCC1). Final shape (all inserted before
/// `anchor`):
///
///     <cmpUOp> s[sgprLoopCounterL], <skipWhenScc1Imm>     // outer LCL gate
///     s_cbranch_scc1 label_skipCBSignal_LCL_<N2>       // skip when SCC1
///     s_cmp_eq_u32 s[sgprWaveIdx], 0                       // inner wave gate
///     s_cbranch_scc0 label_skipCBPreSignal_<N1>
///     s_barrier_signal -3
///   label_skipCBPreSignal_<N1>:                            // inner skip label
///   label_skipCBSignal_LCL_<N2>:                        // outer skip label
///
/// Both skip labels share a target IR position (whatever sits at
/// `anchor`), so the outer cbranch effectively bypasses the entire inner
/// block.
///
/// When \p workgroupSyncWaitComment is non-null, an
/// `s_barrier_signal -1` / `s_barrier_wait -1` workgroup-scope pair is
/// emitted between the outer LCL skip-branch and the inner WaveIdx
/// gate, with the wait carrying the supplied comment text. This
/// guarantees that on the non-skipped path every wave in the workgroup
/// has reached this point before the (first) wave issues the cluster
/// signal -- preventing a fast wave from publishing the cluster
/// handshake while siblings are still doing per-wave teardown above
/// the anchor. The pair sits INSIDE the LCL skip region (so it is
/// bypassed on the skip path together with the cluster signal -- which
/// is desirable: on the skip path Tensile's own loop-entry guard also
/// bypasses the loop body, so neither the LDS publication nor the
/// cluster handshake is needed there), giving:
///     <cmpUOp> s[sgprLoopCounterL], <skipWhenScc1Imm>
///     s_cbranch_scc1 label_skipCBSignal_LCL_<N2>
///     s_barrier_signal -1                                  // workgroup signal
///     s_barrier_wait -1                                    // <workgroupSyncWaitComment>
///     s_cmp_eq_u32 s[sgprWaveIdx], 0
///     s_cbranch_scc0 label_skipCBPreSignal_<N1>
///     s_barrier_signal -3
///   label_skipCBPreSignal_<N1>:
///   label_skipCBSignal_LCL_<N2>:
///
/// Instantiations used today (where `pgr = PrefetchGlobalRead` from module options):
///   - Rule 1: `s_cmp_eq_u32` / imm=0       (skip when LCL == 0;
///             workgroupSyncWaitComment = "sync workgroup before
///             cluster signal" -- the post-GSU==1 join needs the
///             workgroup to be in lockstep before the cluster signal
///             fires)
///   - Rule 3: `s_cmp_le_u32` / imm=pgr (skip when LCL <= pgr). When
///             `needsWorkgroupSync` is true (synthesized-publication path),
///             passes workgroupSyncWaitComment = "workgroup sync" to emit the
///             wg sync pair inside the LCL skip region; when an existing
///             publication wait was found, passes nullptr because the wait
///             already exists in the IR.
///   - Rule 4 signal half: `s_cmp_le_i32` / imm=pgr+1 (minus any LCL
///             pre-decrement) when the drain gate is enabled. The paired WAIT
///             is gated at imm=pgr for the first pub since `s_sub LCL`, or
///             imm=pgr+1 for later pubs in the same LCL window, via
///             `insertLoopCounterLGatedClusterBarrierWaitBefore`. When the
///             drain gate is disabled, Rule 4 emits the ungated shape via
///             `insertClusterBarrierSignalOnlyBefore` /
///             `insertClusterBarrierWaitBefore`.
void insertLoopCounterLGatedClusterBarrierSignalBefore(
    IRBase* anchor, AsmIRBuilder& irBuilder, GfxArchID archId, GFX cmpUOp, int skipWhenScc1Imm,
    const std::string& cmpComment, const std::string& branchComment, uint64_t signalGen,
    const char* workgroupSyncWaitComment = nullptr) {
    const std::string lclLabelName = makeSkipLabel(kSkipSignalLabelPrefixLCL, signalGen);

    const HwInstDesc* cmpDesc = getMCIDByUOp(cmpUOp, archId);
    const HwInstDesc* brDesc = getMCIDByUOp(GFX::s_cbranch_scc1, archId);
    assert(cmpDesc && brDesc && "LoopCounterL gate opcodes are not supported on this architecture");

    StinkyInstruction* cmpInst = irBuilder.create(cmpDesc, anchor);
    cmpInst->addSrcReg(makeSymbolicSgpr(kLoopCounterLSymbol));
    cmpInst->addSrcReg(StinkyRegister(skipWhenScc1Imm));
    cmpInst->addModifier<CommentData>(CommentData{cmpComment});

    StinkyInstruction* brInst = irBuilder.create(brDesc, anchor);
    brInst->addSrcReg(StinkyRegister(lclLabelName));
    brInst->addModifier<LabelData>(LabelData{lclLabelName});
    brInst->addModifier<CommentData>(CommentData{branchComment});

    if (workgroupSyncWaitComment != nullptr) {
        insertWorkgroupBarrierSyncBefore(anchor, irBuilder, archId, workgroupSyncWaitComment);
    }

    insertClusterBarrierSignalOnlyBefore(anchor, irBuilder, archId, signalGen);

    static const HwInstDesc labelMCID{
        GFX::LABEL, GFX::LABEL, 0, 0, 0, "LABEL", makeFlagSet({InstFlag::IF_HasSideEffect})};
    StinkyInstruction* lclLblInst = irBuilder.create(&labelMCID, anchor);
    lclLblInst->addModifier<LabelData>(LabelData{lclLabelName, /*alignment=*/1});
}

/// Wrap a cluster-barrier WAIT in an outer `s[sgprLoopCounterL] <cmp> <imm>`
/// gate so the wait is skipped (branched over) when the compare sets SCC1.
/// Unlike the signal helper there is NO inner WaveIdx gate -- every wave
/// executes (or skips) the wait in lockstep. Final shape (all before
/// `anchor`):
///
///     <cmpUOp> s[sgprLoopCounterL], <skipWhenScc1Imm>
///     s_cbranch_scc1 label_skipCBWait_LCL_<N>
///     s_barrier_wait -3
///   label_skipCBWait_LCL_<N>:
///
/// Used by Rule 4's LCL drain gate (`kClusterBarrierDrainGateEnabled == true`)
/// to skip the cluster wait on the drain iterations.
void insertLoopCounterLGatedClusterBarrierWaitBefore(
    IRBase* anchor, AsmIRBuilder& irBuilder, GfxArchID archId, GFX cmpUOp, int skipWhenScc1Imm,
    const std::string& cmpComment, const std::string& branchComment, uint64_t waitGen) {
    const std::string lclLabelName = makeSkipLabel(kSkipWaitLabelPrefixLCL, waitGen);

    const HwInstDesc* cmpDesc = getMCIDByUOp(cmpUOp, archId);
    const HwInstDesc* brDesc = getMCIDByUOp(GFX::s_cbranch_scc1, archId);
    const HwInstDesc* waitDesc = getMCIDByUOp(GFX::s_barrier_wait, archId);
    assert(cmpDesc && brDesc && waitDesc &&
           "Cluster-barrier wait gate opcodes are not supported on this architecture");

    StinkyInstruction* cmpInst = irBuilder.create(cmpDesc, anchor);
    cmpInst->addSrcReg(makeSymbolicSgpr(kLoopCounterLSymbol));
    cmpInst->addSrcReg(StinkyRegister(skipWhenScc1Imm));
    cmpInst->addModifier<CommentData>(CommentData{cmpComment});

    StinkyInstruction* brInst = irBuilder.create(brDesc, anchor);
    brInst->addSrcReg(StinkyRegister(lclLabelName));
    brInst->addModifier<LabelData>(LabelData{lclLabelName});
    brInst->addModifier<CommentData>(CommentData{branchComment});

    StinkyInstruction* waitInst = irBuilder.create(waitDesc, anchor);
    waitInst->addSrcReg(StinkyRegister(kClusterBarrierId));
    waitInst->addModifier<CommentData>(CommentData{"cluster barrier wait"});

    static const HwInstDesc labelMCID{
        GFX::LABEL, GFX::LABEL, 0, 0, 0, "LABEL", makeFlagSet({InstFlag::IF_HasSideEffect})};
    StinkyInstruction* lclLblInst = irBuilder.create(&labelMCID, anchor);
    lclLblInst->addModifier<LabelData>(LabelData{lclLabelName, /*alignment=*/1});
}

/// Forward declaration: defined further down (Rule 2 / Rule 5b helper).
/// Rule 4's ungated path (`kClusterBarrierDrainGateEnabled == false`) also
/// emits a bare `s_barrier_wait -3` via this helper, so it must be visible
/// before that call site.
void insertClusterBarrierWaitBefore(IRBase* anchor, const char* comment, AsmIRBuilder& irBuilder,
                                    GfxArchID archId);

/// Build the CommentData text for a Rule 4 LCL drain-gate compare. When
/// `lclPreDecrement > 0` the emitted immediate is `pgrNominal -
/// lclPreDecrement` (e.g. nominal pgr 1 with one hoisted decrement -> threshold
/// 0); the comment states both the effective threshold and the compensation.
std::string makeRule4DrainGateCmpComment(const char* barrierHalf, int pgrNominal,
                                         int lclPreDecrement) {
    const int effectiveImm = pgrNominal - lclPreDecrement;
    std::string comment = std::string("gate: skip cluster ") + barrierHalf +
                          " when LCL <= " + std::to_string(effectiveImm);
    if (lclPreDecrement > 0) {
        comment += " (pgr " + std::to_string(pgrNominal) + " - lclPreDecrement " +
                   std::to_string(lclPreDecrement) + ")";
    }
    return comment;
}

/// Emit Rule 4's per-iteration cluster-barrier handshake. Layout is
/// selected by `kClusterBarrierDrainGateEnabled` and (when the drain gate is
/// off) `kRule4SignalBeforeWaitEnabled`.
///
/// Two insertion points are used in the wait-before-signal schemes:
///   - \p signalAnchor is the position right after the load's anchoring
///     `s_barrier_wait -1` (workgroup wait). The WaveIdx-gated cluster signal
///     (and its drain-gate skip, if any) plus the trailing SCC restore land
///     here.
///   - \p waitAnchor is the position of the anchor wait's paired
///     `s_barrier_signal -1` (workgroup signal). The cluster `s_barrier_wait -3`
///     (and its drain-gate skip, if any) is planted immediately BEFORE it, so
///     the cluster wait sits adjacent-above the workgroup signal rather than
///     after the workgroup wait. This puts the workgroup `signal -1`/`wait -1`
///     BETWEEN the cluster wait and the cluster signal, fencing wave 0 so it
///     cannot emit the next-generation `signal -3` before every wave has
///     drained the current-generation `wait -3` (random-hang fix; see the Rule 4
///     comment in `run`). When no workgroup signal was found the caller passes
///     `waitAnchor == signalAnchor` and the cluster wait falls back to the
///     post-`wait -1` position.
///
/// Emission:
///   1. The wait+signal handshake (wait first, then the WaveIdx-gated signal):
///        - drain-gated (`kClusterBarrierDrainGateEnabled == true`), `pgrValue >= 1`:
///          each half is wrapped in its own asymmetric LoopCounterL skip -- the WAIT
///          first pub's WAIT (before \p waitAnchor) is skipped at
///          `LCL <= pgrValue - lclPreDecrement`; later pubs in the same LCL
///          window skip their WAIT at `LCL <= pgrValue+1 - lclPreDecrement`.
///          The SIGNAL (before \p signalAnchor) is skipped one stage earlier at
///          `LCL <= pgrValue+1 - lclPreDecrement` for every pub point.
///          The paired `tensor_load_to_lds`
///          is disabled (TDM enable dword = 0) on those last PGR iterations, so
///          the handshake is unnecessary there; the one-stage offset keeps
///          `signal -3` / `wait -3` balanced despite the ping-pong pairing (each
///          wait consumes the PREVIOUS signal).
///        - drain-gated, `pgrValue == 0`: bare cluster wait (before \p waitAnchor)
///          with no LCL gate, then an LCL-gated signal at
///          `LCL <= pgrValue+1 - lclPreDecrement` only. Rules 2/3 are suppressed
///          elsewhere so Rule 1's priming signal is consumed by the first bare wait.
///        - ungated (`== false`): a bare `s_barrier_wait -3` (before
///          \p waitAnchor) then a WaveIdx-gated `s_barrier_signal -3` (before
///          \p signalAnchor) -- or, when `kRule4SignalBeforeWaitEnabled == true`,
///          the legacy signal-first layout (signal, then the SCC restore, then
///          the trailing bare wait, all at \p signalAnchor; \p waitAnchor is
///          unused).
///   2. SCC restore (both gate states): if `liveSccCmp != nullptr` -- SIA
///      hoisted a live loop-exit `s_cmp_* LCL, imm` above the anchor whose SCC
///      a downstream `s_cbranch_scc{0,1}` consumes -- a clone of it is
///      re-emitted AFTER the whole block (before \p signalAnchor). Because every
///      drain skip label and the WaveIdx cmp sit BEFORE this clone on every
///      outgoing path (gated-skip and fall-through alike), the gate's own
///      `s_cmp_le` clobbering SCC is harmless.
///
/// \p pgrValue and \p lclPreDecrement are consulted only by the drain-gate
/// thresholds. The SCC restore clones the live compare verbatim and needs no
/// `lclPreDecrement` compensation: `findLiveSccCmpUpstream` accepts a compare
/// only when it is the first SCC writer above the anchor, and `s_sub LCL` also
/// writes SCC, so a decrement between the compare and the anchor would have
/// aborted detection. A non-null `liveSccCmp` thus guarantees LCL is unchanged
/// from the compare to the restore point (the counted decrements sit ABOVE the
/// compare, already folded into its operands).
void insertClusterBarrierHandshakeBefore(IRBase* signalAnchor, IRBase* waitAnchor,
                                         AsmIRBuilder& irBuilder, GfxArchID archId, int pgrValue,
                                         bool isFirstPubSinceDec, StinkyInstruction* liveSccCmp,
                                         int lclPreDecrement, uint64_t waitGen,
                                         uint64_t signalGen) {
    if (kClusterBarrierDrainGateEnabled) {
        if (pgrValue == 0) {
            // PGR=0: no prefetch drain iterations -- Rules 2/3 are suppressed and
            // the per-iteration wait does not need an LCL drain gate (there is no
            // `LCL <= pgr` skip counterpart to mirror). Emit a bare cluster wait
            // before the workgroup signal; keep the signal half LCL-gated so load
            // disable on the last iteration still suppresses the next signal.
            insertClusterBarrierWaitBefore(waitAnchor, "cluster barrier wait", irBuilder, archId);
            const int signalThreshold = pgrValue + 1 - lclPreDecrement;
            insertLoopCounterLGatedClusterBarrierSignalBefore(
                signalAnchor, irBuilder, archId, GFX::s_cmp_le_i32, signalThreshold,
                makeRule4DrainGateCmpComment("signal", pgrValue + 1, lclPreDecrement),
                "skip cluster signal (drain)", signalGen);
        } else {
            // WAIT: first pub since the preceding `s_sub LCL` skips at
            // `LCL <= pgrValue`; every later pub in the same LCL window skips
            // both halves at `LCL <= pgrValue+1`. When the schedule hoisted
            // `s_sub LCL` above the anchor, `lclPreDecrement` lowers the
            // compare immediate. Planted before `waitAnchor` (immediately above
            // the workgroup signal).
            const int waitNominal = isFirstPubSinceDec ? pgrValue : (pgrValue + 1);
            const int waitThreshold = waitNominal - lclPreDecrement;
            const int signalThreshold = pgrValue + 1 - lclPreDecrement;
            insertLoopCounterLGatedClusterBarrierWaitBefore(
                waitAnchor, irBuilder, archId, GFX::s_cmp_le_i32, waitThreshold,
                makeRule4DrainGateCmpComment("wait", waitNominal, lclPreDecrement),
                "skip cluster wait (drain)", waitGen);
            // SIGNAL: nominal threshold one stage earlier at `LCL <= pgrValue+1`,
            // with the same `lclPreDecrement` compensation on the immediate.
            insertLoopCounterLGatedClusterBarrierSignalBefore(
                signalAnchor, irBuilder, archId, GFX::s_cmp_le_i32, signalThreshold,
                makeRule4DrainGateCmpComment("signal", pgrValue + 1, lclPreDecrement),
                "skip cluster signal (drain)", signalGen);
        }
    } else if (kRule4SignalBeforeWaitEnabled) {
        // Legacy signal-before-wait layout: emit the
        // WaveIdx-gated signal first, then the live-SCC restore clone (if any),
        // then the bare wait LAST -- keeping the wait as the load's immediate
        // predecessor so Rule 2 stays idempotent. All at `signalAnchor`
        // (`waitAnchor` is unused here). Returns early because the SCC restore
        // is emitted here (before the wait), not by the shared trailing block
        // below.
        insertClusterBarrierSignalOnlyBefore(signalAnchor, irBuilder, archId, signalGen);
        if (liveSccCmp != nullptr) {
            const HwInstDesc* restoreDesc = liveSccCmp->getHwInstDesc();
            StinkyInstruction* restoreInst = irBuilder.create(restoreDesc, signalAnchor);
            for (const auto& src : liveSccCmp->getSrcRegs()) restoreInst->addSrcReg(src);
            restoreInst->addModifier<CommentData>(
                CommentData{"restore SCC for downstream cbranch (Rule 4 signal-before-wait)"});
        }
        insertClusterBarrierWaitBefore(signalAnchor, "cluster barrier wait", irBuilder, archId);
        return;
    } else {
        // Ungated wait-before-signal: the bare cluster wait is planted
        // before `waitAnchor` (immediately above the workgroup signal); the
        // WaveIdx-gated signal before `signalAnchor` (after the workgroup wait).
        insertClusterBarrierWaitBefore(waitAnchor, "cluster barrier wait", irBuilder, archId);
        insertClusterBarrierSignalOnlyBefore(signalAnchor, irBuilder, archId, signalGen);
    }
    if (liveSccCmp != nullptr) {
        // Clone verbatim. No immediate compensation for `lclPreDecrement` is
        // needed here: `findLiveSccCmpUpstream` returns a compare only when it
        // is the FIRST SCC writer above the anchor, and any `s_sub LCL` also
        // writes SCC -- so a decrement between the live compare and the anchor
        // would have aborted detection (returning nullptr). A non-null
        // `liveSccCmp` therefore guarantees LCL is unchanged between the
        // compare and this restore point, so re-running the identical compare
        // reproduces the original SCC. (Decrements counted in
        // `lclPreDecrement` sit ABOVE the compare and are already baked into
        // its operands; they matter only for the drain-gate thresholds.)
        const HwInstDesc* restoreDesc = liveSccCmp->getHwInstDesc();
        StinkyInstruction* restoreInst = irBuilder.create(restoreDesc, signalAnchor);
        for (const auto& src : liveSccCmp->getSrcRegs()) restoreInst->addSrcReg(src);
        restoreInst->addModifier<CommentData>(
            CommentData{"restore SCC for downstream cbranch (Rule 4)"});
    }
}

/// True if `inst` is a `LABEL` pseudo whose `LabelData.label` matches `name`
/// exactly. Anchors:
///   - Rule 1: `kGSU1LabelName` (`label_GSU_1:`)
///   - Rule 3 synthesized-publication path: `kOpenLoopLLabelName` (`label_openLoopL:`)
/// Internal control-flow labels (e.g. `label_skipPGR2_1`) do not match by
/// exact-name comparison and are scanned through.
bool isLabelNamed(const StinkyInstruction& inst, const char* name) {
    if (!isLabel(inst)) return false;
    const auto* labelData = inst.getModifier<LabelData>();
    return labelData != nullptr && labelData->label == name;
}

/// True if `ir` is a TEXTBLOCK directive whose value contains `marker` as a
/// substring. Anchor:
///   - Rule 5 (5a / 5b): `kTailLoopMarker` (`/* Tail Loop ... */`)
/// TEXTBLOCK directives are erased at region extraction, so rules using
/// this predicate self-disable at region scope.
bool isTextblockContaining(IRBase* ir, const char* marker) {
    auto* directive = dyn_cast<AsmDirective>(ir);
    return directive != nullptr && directive->kind == AsmDirectiveKind::TEXTBLOCK &&
           directive->value.find(marker) != std::string::npos;
}

struct Rule5TailScan {
    BasicBlock* markerBB = nullptr;
    IRBase* markerNext = nullptr;
    StinkyInstruction* tailTL = nullptr;
    StinkyInstruction* tailWait = nullptr;
    IRBase* signalAnchor = nullptr;
    bool hasRealWait = false;
};

/// Locate the tail-loop publication point across CFG basic blocks. Finds the
/// `/* Tail Loop */` TEXTBLOCK, then the first `tensor_load_to_lds` reachable
/// forward from that marker, and the nearest preceding workgroup
/// `s_barrier_wait -1` on any path back toward the marker.
Rule5TailScan scanRule5TailPoint(Function& func) {
    Rule5TailScan result;
    BasicBlock::iterator markerIt;
    bool foundMarker = false;
    for (BasicBlock& bb : func) {
        for (auto it = bb.begin(); it != bb.end(); ++it) {
            if (isTextblockContaining(it.getNodePtr(), kTailLoopMarker)) {
                result.markerBB = &bb;
                markerIt = it;
                foundMarker = true;
                if (std::next(it) != bb.end()) result.markerNext = std::next(it).getNodePtr();
                break;
            }
        }
        if (foundMarker) break;
    }
    if (!foundMarker) return result;

    std::deque<std::pair<BasicBlock*, BasicBlock::iterator>> forward;
    std::unordered_set<BasicBlock*> forwardSeen;
    forward.emplace_back(result.markerBB, std::next(markerIt));
    forwardSeen.insert(result.markerBB);

    while (!forward.empty() && result.tailTL == nullptr) {
        auto [block, start] = forward.front();
        forward.pop_front();
        for (auto scanIt = start; scanIt != block->end(); ++scanIt) {
            auto* inst = dyn_cast<StinkyInstruction>(scanIt.getNodePtr());
            if (inst != nullptr && isTensorLoad(*inst)) {
                result.tailTL = inst;
                break;
            }
        }
        if (result.tailTL != nullptr) break;
        for (BasicBlock* succ : block->getSuccessors()) {
            if (forwardSeen.insert(succ).second) forward.emplace_back(succ, succ->begin());
        }
    }
    if (result.tailTL == nullptr) return result;

    auto scanBackwardSegment = [&](BasicBlock* block, BasicBlock::iterator start) {
        if (start == block->end()) {
            for (auto rit = block->rbegin(); rit != block->rend(); ++rit) {
                IRBase* node = rit.getNodePtr();
                if (isTextblockContaining(node, kTailLoopMarker)) return;
                auto* inst = dyn_cast<StinkyInstruction>(node);
                if (inst == nullptr) continue;
                if (isWorkgroupBarrierWait(*inst)) {
                    result.tailWait = inst;
                    return;
                }
            }
            return;
        }
        auto it = start;
        while (it != block->begin()) {
            --it;
            if (isTextblockContaining(it.getNodePtr(), kTailLoopMarker)) return;
            auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (inst == nullptr) continue;
            if (isWorkgroupBarrierWait(*inst)) {
                result.tailWait = inst;
                return;
            }
        }
    };

    BasicBlock* tlBB = result.tailTL->getParent();
    scanBackwardSegment(tlBB, BasicBlock::iterator(result.tailTL));
    if (result.tailWait == nullptr) {
        std::deque<BasicBlock*> backward;
        std::unordered_set<BasicBlock*> backwardSeen;
        for (BasicBlock* pred : tlBB->getPredecessors()) {
            if (backwardSeen.insert(pred).second) backward.push_back(pred);
        }
        while (!backward.empty() && result.tailWait == nullptr) {
            BasicBlock* block = backward.front();
            backward.pop_front();
            scanBackwardSegment(block, block->end());
            if (result.tailWait != nullptr) break;
            if (block == result.markerBB) continue;
            for (BasicBlock* pred : block->getPredecessors()) {
                if (backwardSeen.insert(pred).second) backward.push_back(pred);
            }
        }
    }

    if (result.tailWait != nullptr) {
        result.hasRealWait = true;
        BasicBlock* waitBB = result.tailWait->getParent();
        auto nextIt = std::next(BasicBlock::iterator(result.tailWait));
        result.signalAnchor = (nextIt != waitBB->end()) ? nextIt.getNodePtr() : nullptr;
    }
    return result;
}

/// Idempotency check used by Rules 1, 3, 4, and 5a. Examines the first real
/// successor of `anchor` (via `firstRealInstAfter`) and accepts any of:
///   - `s_barrier_wait -3` (Rule 2 / Rule 4's ungated leading wait /
///     Tensile-emitted full handshake)
///   - `s_cmp_eq_u32 s[sgprWaveIdx], 0` (the WaveIdx gate that opens the
///     ungated Rules 1/3 and Rule 5a signal-only emissions)
///   - `s_cmp_eq_u32 s[sgprLoopCounterL], <imm>` (Rule 1's gated `LCL == 0` gate)
///   - `s_cmp_le_u32 s[sgprLoopCounterL], <imm>` (Rule 3's gated `LCL <= pgr` gate)
///   - `s_cmp_le_i32 s[sgprLoopCounterL], <imm>` (Rule 4 LCL drain gate)
///   - `s_cmp_eq_i32 s[sgprLoopCounterL], <imm>` (SCC-restore clone of a live
///     upstream compare)
/// The imm operand is not checked, only the symbolic name, so the predicate
/// is independent of the configured PrefetchGlobalRead value. In any of these cases the
/// anchor is already followed by a cluster-barrier emission and we must
/// not duplicate it.
bool isFollowedByClusterBarrierHandshakeOrSignal(StinkyInstruction* anchor) {
    StinkyInstruction* next = firstRealInstAfter(anchor);
    if (next == nullptr) return false;
    if (isClusterBarrierWait(*next)) return true;
    const auto uOp = next->getUnifiedOpcode();
    if (uOp == GFX::s_cmp_eq_u32 || uOp == GFX::s_cmp_le_u32 || uOp == GFX::s_cmp_eq_i32 ||
        uOp == GFX::s_cmp_le_i32) {
        const auto& srcs = next->getSrcRegs();
        if (srcs.empty()) return false;
        const std::string& sym = srcs[0].getSymbolicName();
        if (sym == kWaveIdxSymbol || sym == kLoopCounterLSymbol) return true;
    }
    return false;
}

/// Function-wide scan: returns the first `tensor_load_to_lds` encountered
/// when walking every BB in order, or nullptr if the function has none.
StinkyInstruction* findFirstTensorLoadInFunc(Function& func) {
    for (BasicBlock& bb : func) {
        for (auto it = bb.begin(); it != bb.end(); ++it) {
            auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (inst == nullptr) continue;
            if (isTensorLoad(*inst)) return inst;
        }
    }
    return nullptr;
}

/// Idempotency guard for Rule 2: walk backward from `anchor` past pseudo
/// instructions (LABEL / PHI / FENCE) and non-`StinkyInstruction` IR. If
/// the first real predecessor is a cluster-scope wait, the load is already
/// gated and we must not emit a duplicate.
bool isImmediatelyPrecededByClusterBarrierWait(StinkyInstruction* anchor) {
    BasicBlock* parent = anchor->getParent();
    if (parent == nullptr) return false;
    auto it = BasicBlock::iterator(anchor);
    while (it != parent->begin()) {
        --it;
        auto* prev = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (prev == nullptr) continue;
        if (isPseudoInst(prev)) continue;
        return isClusterBarrierWait(*prev);
    }
    return false;
}

/// Insert a single `s_barrier_wait -3` (with the given `comment`)
/// immediately before `anchor`. Pass `anchor=nullptr` to append. The
/// comment text varies by rule:
///   - Rule 2 uses `"cluster_barrier wait"` (matches the spelling that
///     already appears at the kernel's first tensor_load in reference asm)
///   - Rule 5b uses `"cluster barrier wait"`.
/// (Used by Rule 2 and Rule 5b.)
void insertClusterBarrierWaitBefore(IRBase* anchor, const char* comment, AsmIRBuilder& irBuilder,
                                    GfxArchID archId) {
    const HwInstDesc* waitDesc = getMCIDByUOp(GFX::s_barrier_wait, archId);
    assert(waitDesc && "Cluster-barrier wait opcode is not supported on this architecture");
    StinkyInstruction* waitInst = irBuilder.create(waitDesc, anchor);
    waitInst->addSrcReg(StinkyRegister(kClusterBarrierId));
    waitInst->addModifier<CommentData>(CommentData{comment});
}

/// True if `inst` is the loop-counter "numIterL == 0" guard compare:
/// `s_cmp_eq_{u32,i32} s[sgprLoopCounterL], 0`. This is the test Tensile emits
/// just before the kernel's first prefetch `tensor_load_to_lds` to decide
/// whether the unrolled summation loop is entered at all.
bool isLoopCounterLZeroCompare(const StinkyInstruction& inst) {
    const auto uOp = inst.getUnifiedOpcode();
    if (uOp != GFX::s_cmp_eq_u32 && uOp != GFX::s_cmp_eq_i32) return false;
    const auto& srcs = inst.getSrcRegs();
    if (srcs.size() < 2) return false;
    if (srcs[0].getSymbolicName() != kLoopCounterLSymbol) return false;
    return srcs[1].dataType == StinkyRegister::Type::LiteralInt && srcs[1].getLiteralInt() == 0;
}

/// Extract the target label of the rocisa long-branch anchor
/// `s_add_i32 sX, <label>, +/-4` (the label is a LiteralString src operand and
/// the +/-4 compensates for `s_getpc_b64`). Returns the label name, or empty
/// string when `inst` is not that anchor. Mirrors LongBranchLoweringPass's
/// `matchAddI32Anchor`, kept local because this pass runs BEFORE long-branch
/// lowering (so the `s_setpc_b64` does not yet carry LabelData).
std::string matchLongBranchAddLabel(const StinkyInstruction& inst) {
    if (inst.getUnifiedOpcode() != GFX::s_add_i32) return "";
    const auto& srcs = inst.getSrcRegs();
    if (srcs.size() != 2) return "";
    if (srcs[0].dataType != StinkyRegister::Type::LiteralString) return "";
    if (srcs[1].dataType != StinkyRegister::Type::LiteralInt) return "";
    const int off = srcs[1].getLiteralInt();
    if (off != 4 && off != -4) return "";
    return srcs[0].getLiteralString();
}

/// Resolve which label control flow lands on when `sgprLoopCounterL == 0`, by
/// scanning backward from the kernel's first `tensor_load_to_lds` (\p firstTL)
/// for the `LCL == 0` guard and decoding the branch that consumes its SCC.
/// Two Tensile encodings are handled:
///   1. short branch:  `s_cmp_eq_u32 LCL, 0` + `s_cbranch_scc1 <label>`
///      -> branch is taken when SCC1 (LCL==0); target is `<label>`
///      (typically `label_LoopEndL`).
///   2. long branch:   `s_cmp_eq_u32 LCL, 0` + `s_cbranch_scc0 <skip>` +
///      `s_getpc_b64` / `s_add_i32 sX, <label>, 4` / ... / `s_setpc_b64`
///      -> the scc0 branch only fires when LCL!=0 (skipping the long branch);
///      when LCL==0 we fall through into the long branch whose static target
///      is `<label>` (typically `label_PrefetchGlobalLastIterEnd`).
/// Returns the resolved label name, or empty string when no guard is found.
std::string resolveLoopCounterLZeroTargetLabel(StinkyInstruction* firstTL) {
    if (firstTL == nullptr) return "";
    BasicBlock* parent = firstTL->getParent();
    if (parent == nullptr) return "";

    // 1) Backward scan: nearest preceding `LCL == 0` guard compare.
    StinkyInstruction* guardCmp = nullptr;
    for (auto it = BasicBlock::iterator(firstTL); it != parent->begin();) {
        --it;
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr || isPseudoInst(inst)) continue;
        if (isLoopCounterLZeroCompare(*inst)) {
            guardCmp = inst;
            break;
        }
    }
    if (guardCmp == nullptr) return "";

    // 2) Forward scan: the first real instruction after the guard is the
    //    branch that consumes its SCC.
    for (auto it = std::next(BasicBlock::iterator(guardCmp)); it != parent->end(); ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr || isPseudoInst(inst)) continue;
        const auto uOp = inst->getUnifiedOpcode();
        if (uOp == GFX::s_cbranch_scc1) {
            // Short form: taken on SCC1 (LCL==0). Target is the branch operand.
            const auto& srcs = inst->getSrcRegs();
            if (!srcs.empty() && srcs[0].dataType == StinkyRegister::Type::LiteralString) {
                return srcs[0].getLiteralString();
            }
            const auto* labelData = inst->getModifier<LabelData>();
            return labelData != nullptr ? labelData->label : "";
        }
        if (uOp == GFX::s_cbranch_scc0) {
            // Long form: scc0 skips the long branch when LCL!=0; the LCL==0
            // path falls through into getpc/add(label)/setpc. Decode the label
            // from the `s_add_i32 sX, <label>, +/-4` anchor.
            for (auto fit = std::next(it); fit != parent->end(); ++fit) {
                auto* fi = dyn_cast<StinkyInstruction>(fit.getNodePtr());
                if (fi == nullptr || isPseudoInst(fi)) continue;
                std::string lbl = matchLongBranchAddLabel(*fi);
                if (!lbl.empty()) return lbl;
                if (fi->getUnifiedOpcode() == GFX::s_setpc_b64) break;
            }
            return "";
        }
        // First real instruction after the guard is not its consuming branch.
        return "";
    }
    return "";
}

class InsertClusterBarrierPassImpl : public Pass {
   public:
    static char ID;

    InsertClusterBarrierPassImpl(int pgrValue) : pgrValue_(pgrValue) {}

    const char* getName() const override {
        return "Insert Cluster Barrier";
    }

    Pass::ID getPassID() const override {
        return &InsertClusterBarrierPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        const auto& arch = passCtx.getGemmTileConfig().arch;
        const GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);

        // Rule 4: for each `tensor_load_to_lds`, plant a cluster-barrier
        // handshake around the load's LDS publication point (see
        // `insertClusterBarrierHandshakeBefore` and the compile-time switches). The nearest
        // preceding `s_barrier_wait -1` is the anchor, together with its paired `s_barrier_signal
        // -1` (workgroup signal) just above it. In the wait-before-signal schemes (the default
        // drain-gated + ungated) the handshake is SPLIT across the workgroup barrier: the cluster
        // `s_barrier_wait -3` (plus its LCL drain gate, if any) goes immediately BEFORE the
        // workgroup `s_barrier_signal -1`, while the WaveIdx-gated `s_barrier_signal -3` (plus its
        // drain gate) and the trailing SCC restore stay AFTER the workgroup `s_barrier_wait -1`.
        //
        // Why the split (random-hang fix): the per-iteration cluster handshake is
        // an offset ping-pong -- each `wait -3` consumes the PREVIOUS iteration's
        // `signal -3`, and only wave 0 issues the (next-generation) `signal -3`.
        // If the workgroup barrier sat BEFORE the cluster wait (the old layout:
        // `signal -1`/`wait -1`/`wait -3`/`signal -3`), nothing fenced wave 0
        // between completing this iteration's `wait -3` and issuing the next
        // iteration's `signal -3`: a fast wave 0 could race ahead and emit the
        // next-generation `signal -3` while sibling waves in the same workgroup
        // were still draining the current-generation `wait -3`. That extra signal
        // lands in the wrong generation, desynchronizing the cluster barrier's
        // signal/wait count and occasionally deadlocking (the random hang).
        // Placing the workgroup `signal -1`/`wait -1` BETWEEN the cluster
        // `wait -3` and `signal -3` fences the workgroup: every wave must finish
        // the current-generation `wait -3` and converge at the workgroup barrier
        // before wave 0 can issue the next-generation `signal -3`, so no signal
        // is ever emitted into the next generation early. (The legacy
        // signal-before-wait layout self-pairs each iteration's own signal/wait
        // and is left untouched.)
        //
        // Triggers are deduplicated by identity, so multiple loads sharing the
        // same anchor wait yield exactly one handshake; the backward scan stays
        // within the load's basic block to avoid crossing a CFG edge. This pass
        // runs after CFGBuilder, so branch/fall-through and label entry points
        // are already materialized as BB boundaries -- Rule 4 keys off those
        // BBs directly rather than simulating segments on flat IR.
        // Cluster-barrier *generation* numbering, threaded through emission in
        // kernel execution order (Rule 1 -> 3 -> 4 -> 5 -> 6; Rule 2's bare wait
        // carries no label). Each cluster signal opens a new generation; each
        // drain-gated wait consumes (drains) the most recent signal's
        // generation -- the offset ping-pong. Function-scoped so every kernel
        // numbers from 0. See the skip-label prefix comments near the top.
        uint64_t clusterGen = 0;
        int64_t lastSignalGen = -1;
        auto nextSignalGen = [&]() {
            const uint64_t gen = clusterGen++;
            lastSignalGen = static_cast<int64_t>(gen);
            return gen;
        };
        auto drainWaitGen = [&]() {
            return lastSignalGen >= 0 ? static_cast<uint64_t>(lastSignalGen) : uint64_t{0};
        };

        Rule5TailScan rule5Scan = scanRule5TailPoint(func);
        if (rule5Scan.tailTL != nullptr &&
            isImmediatelyPrecededByClusterBarrierWait(rule5Scan.tailTL)) {
            rule5Scan.tailTL = nullptr;
        }
        if (rule5Scan.tailWait != nullptr &&
            isFollowedByClusterBarrierHandshakeOrSignal(rule5Scan.tailWait)) {
            rule5Scan.tailWait = nullptr;
            rule5Scan.hasRealWait = false;
            rule5Scan.signalAnchor = nullptr;
        }
        const bool rule5Want5b = rule5Scan.tailTL != nullptr;
        const bool rule5Want5aWait = rule5Scan.tailWait != nullptr;
        const bool rule5Want5aFallback = rule5Scan.tailTL != nullptr &&
                                         rule5Scan.tailWait == nullptr &&
                                         rule5Scan.markerNext != nullptr;
        bool rule5aEmitted = false;
        bool rule5bEmitted = false;

        for (BasicBlock& bb : func) {
            // Tuple: (trigger workgroup wait, anchor iterator next to it,
            //         live upstream LCL cmp at trigger or nullptr, cumulative
            //         LCL pre-decrement before the trigger, paired workgroup
            //         `s_barrier_signal -1` above the trigger or nullptr,
            //         first pub in the current LCL window). The 3rd/4th/5th/6th
            //         elements are captured at scan time -- i.e. against the
            //         pre-mutation IR -- so a later `pending` entry's emission
            //         cannot influence an earlier one's SCC analysis, decrement
            //         count, wait-anchor resolution, or window index.
            std::vector<std::tuple<StinkyInstruction*, BasicBlock::iterator, StinkyInstruction*,
                                   int, StinkyInstruction*, bool>>
                pending;
            std::unordered_set<StinkyInstruction*> seenTriggers;
            StinkyInstruction* lastPendingTrigger = nullptr;
            int pubsInLclWindow = 0;

            // Rule 4 owns only the main-loop region. Any tensor load at or
            // after the `/* Tail Loop */` marker belongs exclusively to Rule 5
            // (5a/5b), so bound the forward scan at the marker. This keeps
            // Rule 4 and Rule 5 from ever sharing an anchor wait -- when no
            // marker exists (e.g. region scope, where it is erased) `rule4End`
            // stays `bb.end()` and Rule 4 sweeps the whole block.
            auto rule4End = bb.end();
            for (auto it = bb.begin(); it != bb.end(); ++it) {
                if (isTextblockContaining(it.getNodePtr(), kTailLoopMarker)) {
                    rule4End = it;
                    break;
                }
            }

            for (auto it = bb.begin(); it != rule4End; ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst == nullptr) continue;
                if (!isTensorLoad(*inst)) continue;

                StinkyInstruction* trigger = findPrecedingWorkgroupBarrierWaitInBB(inst);
                if (trigger == nullptr) continue;
                // Dedup: multiple loads can share the same anchor wait;
                // only the first one queues an emission.
                if (!seenTriggers.insert(trigger).second) continue;
                if (isFollowedByClusterBarrierHandshakeOrSignal(trigger)) continue;

                // Defer the actual mutation until after the scan finishes
                // so we don't invalidate `it`. Capture the live upstream
                // LCL cmp NOW (against the original IR) so each Rule-4 site
                // is analyzed independently from any sibling sites that
                // will be mutated later in the same BB sweep.
                StinkyInstruction* liveSccCmp = findLiveSccCmpUpstream(trigger);
                // Count only `s_sub LCL` between the previous Rule-4 trigger and
                // this one so multi-section loops do not inherit earlier sections'
                // decrements into later sections' drain gates.
                const int lclPreDecrement =
                    (lastPendingTrigger != nullptr)
                        ? sumLoopCounterLDecrementsBetweenInBB(lastPendingTrigger, trigger)
                        : sumLoopCounterLDecrementsBeforeInBB(trigger);
                StinkyInstruction* wgSignal = findPrecedingWorkgroupBarrierSignalInBB(trigger);
                if (lastPendingTrigger != nullptr &&
                    hasLoopCounterLDecrementBetweenInBB(lastPendingTrigger, trigger)) {
                    pubsInLclWindow = 0;
                }
                const bool isFirstPubSinceDec = (pubsInLclWindow == 0);
                ++pubsInLclWindow;
                lastPendingTrigger = trigger;
                pending.emplace_back(trigger, std::next(BasicBlock::iterator(trigger)), liveSccCmp,
                                     lclPreDecrement, wgSignal, isFirstPubSinceDec);
            }

            // Rule 1: signal-only handshake immediately AFTER each
            // `label_GSU_1:` label (emitted by Tensile after the GSU==1
            // early-return guard). The label is a `StinkyInstruction`
            // pseudo, so unlike a TEXTBLOCK it survives region extraction
            // - idempotency (`isFollowedByClusterBarrierHandshakeOrSignal`)
            // handles re-entry across scopes.
            //
            // The emitted sequence is a plain `WaveIdx == 0`-gated cluster
            // signal (same shape as Rules 3/5a). There is NO `LoopCounterL`
            // gate: the cluster barrier's only guard is the WaveIdx check that
            // selects the single signalling wave, so the signal fires on every
            // control-flow path. (The signal is just the priming credit for the
            // paired wait; no LCL-based suppression is needed.)
            //
            // Anchor for emission is the iterator AFTER the label, so the
            // new sequence lands between the label and its successor.
            std::vector<IRBase*> gsu1Anchors;
            for (auto it = bb.begin(); it != bb.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst == nullptr) continue;
                if (!isLabelNamed(*inst, kGSU1LabelName)) continue;
                if (isFollowedByClusterBarrierHandshakeOrSignal(inst)) {
                    continue;
                }
                auto nextIt = std::next(it);
                IRBase* anchor = (nextIt != bb.end()) ? nextIt.getNodePtr() : nullptr;
                gsu1Anchors.push_back(anchor);
            }

            // Rule 6: plant a single bare `s_barrier_wait -3` at the loop-exit
            // convergence label to consume the loop's last, otherwise-orphaned
            // cluster signal.
            //
            // ONLY enabled in the ungated wait-before-signal scheme
            // (`kClusterBarrierDrainGateEnabled == false` AND
            // `kRule4SignalBeforeWaitEnabled == false`). Rationale (the cluster
            // handshake is an offset ping-pong -- each per-iteration WAIT
            // consumes the PREVIOUS SIGNAL):
            //   - Ungated wait-before-signal: the loop emits equal
            //     #WAIT and #SIGNAL, so the final in-loop SIGNAL has no in-loop
            //     WAIT to consume it. Rule 6 supplies that trailing WAIT at loop
            //     exit.
            //   - Drain gate ON: the asymmetric gate skips the WAIT at
            //     `LCL <= pgr` and the SIGNAL one stage earlier at
            //     `LCL <= pgr+1`, leaving the loop with exactly one MORE WAIT
            //     than SIGNAL. That extra WAIT already consumes the last
            //     SIGNAL inside the loop, so a Rule 6 WAIT would be unpaired
            //     (one surplus `wait -3`) and would hang. Hence skip Rule 6.
            //   - Ungated signal-before-wait (legacy,
            //     `kRule4SignalBeforeWaitEnabled == true`): each iteration's
            //     SIGNAL is consumed by its OWN trailing WAIT (same-iteration
            //     self-pairing), so the loop is already balanced -- no priming
            //     signal (Rule 3) and no loop-exit drain (Rule 6) are needed. A
            //     Rule 6 WAIT would be unpaired here too.
            //
            // Anchor selection: instead of unconditionally targeting
            // `label_LoopEndL`, scan backward from the kernel's first
            // `tensor_load_to_lds` for the `sgprLoopCounterL == 0` guard and
            // resolve which label the LCL==0 path actually lands on (see
            // `resolveLoopCounterLZeroTargetLabel`). This is the convergence
            // point of every "loop bypassed" path:
            //   - short-branch encoding -> `label_LoopEndL` (unchanged
            //     behavior; both the normal loop exit and the LCL==0 skip land
            //     here);
            //   - long-branch encoding  -> `label_PrefetchGlobalLastIterEnd`
            //     (the LCL==0 long branch AND the LCL==1 `toPGR1` drain both
            //     fall through here, whereas they bypass `label_LoopEndL`).
            // Planting the wait at this resolved label keeps the hoisted
            // signal paired on the loop-drain paths too. Fall back to
            // `label_LoopEndL` when no guard is found. Idempotency: skip when
            // the label is already followed by a cluster-scope wait.
            std::vector<IRBase*> rule6LoopExitAnchors;
            if (!kClusterBarrierDrainGateEnabled && !kRule4SignalBeforeWaitEnabled) {
                std::string loopExitWaitLabel = resolveLoopCounterLZeroTargetLabel(
                    findFirstTensorLoadBetween(bb.begin(), bb.end()));
                if (loopExitWaitLabel.empty()) loopExitWaitLabel = kLoopEndLLabelName;
                for (auto it = bb.begin(); it != bb.end(); ++it) {
                    auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                    if (inst == nullptr) continue;
                    if (!isLabelNamed(*inst, loopExitWaitLabel.c_str())) continue;
                    StinkyInstruction* next = firstRealInstAfter(inst);
                    if (next != nullptr && isClusterBarrierWait(*next)) continue;
                    auto nextIt = std::next(it);
                    IRBase* anchor = (nextIt != bb.end()) ? nextIt.getNodePtr() : nullptr;
                    rule6LoopExitAnchors.push_back(anchor);
                }
            }

            // Rule 3: signal-only handshake at the LDS publication point
            // that precedes `label_openLoopL:`. Enabled in every scheme EXCEPT
            // the ungated signal-before-wait legacy mode (see the disable guard
            // below `kRule4SignalBeforeWaitEnabled`). The emission is a plain
            // `WaveIdx == 0`-gated cluster signal (no LoopCounterL gate in the
            // ungated scheme) that primes the loop's offset ping-pong: it is
            // consumed by the first per-iteration cluster WAIT (Rule 4) inside
            // the loop body. In the legacy signal-before-wait layout each loop
            // iteration self-pairs its own signal/wait, so no priming signal is
            // needed and Rule 3 is disabled (together with Rule 6).
            //
            // (Historical: this rule used to wrap the signal in an outer
            // `LCL <= pgrValue_` skip that mirrored Tensile's own
            // `s_cmp_le_u32 LCL, pgrValue / s_cbranch_scc1 LoopEndL` entry
            // guard, so the cluster signal is suppressed on the exact
            // same control-flow paths where the corresponding
            // `s_barrier_wait -3` inside the unrolled loop body is
            // skipped -- keeping `signal -3` / `wait -3` paired
            // everywhere).
            //
            // Anchor selection (backward scan from `label_openLoopL:` through
            // its basic block and CFG predecessors):
            //   existing publication wait: nearest preceding `s_barrier_wait -1`
            //       before the prefetch boundary (typical schedules). Anchor at
            //       the successor of that wait; no new workgroup sync is
            //       synthesized. The scan stops if a `tensor_load_to_lds` is
            //       reached first.
            //   synthesized publication sync: no `s_barrier_wait -1` before the
            //       boundary -- anchor at `label_openLoopL:` and synthesize
            //       `s_barrier_signal -1` / `s_barrier_wait -1` (via
            //       `insertWorkgroupBarrierSyncBefore`) before the cluster signal.
            //
            // Internal control-flow labels inside the prefetch prologue
            // (e.g. `label_skipPGR2_*`) do not match `label_openLoopL`
            // by exact-name comparison and are walked through.
            //
            // The label/instruction-based anchor does not depend on
            // TEXTBLOCK directives (and so survives
            // `ScopeAdaptor::moveIRToBlock`, which erases them), so this
            // rule keeps working under the single kernel-scope run that
            // `Gfx1250Backend::buildGfx1250Pipeline` performs when
            // `moduleOptions.ClusterBarrier == true`.
            // Idempotency:
            //   - Section-level: the backward scan also flags whether a
            //     cluster-scope signal/wait already sits in the
            //     section. If so (e.g. a prior pass run already emitted
            //     Rule 3), Rule 3 self-disables.
            //   - Anchor-level (existing-wait path): skip if the existing workgroup
            //     wait is already followed by a cluster handshake, or
            //     if Rule 4 has already queued the same wait as a
            //     trigger (Rule 4 emits the full wait+signal handshake
            //     and supersedes Rule 3 at the same anchor).
            Rule3PublicationScan rule3Scan;
            for (auto it = bb.begin(); it != bb.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst != nullptr && isLabelNamed(*inst, kOpenLoopLLabelName)) {
                    rule3Scan = scanRule3PublicationPoint(inst);
                    break;
                }
            }
            if (rule3Scan.existingWait != nullptr) {
                bool conflictsWithRule4 = false;
                for (const auto& [trigger, _next, _live, _dec, _sig, _first] : pending) {
                    if (trigger == rule3Scan.existingWait) {
                        conflictsWithRule4 = true;
                        break;
                    }
                }
                if (conflictsWithRule4 ||
                    isFollowedByClusterBarrierHandshakeOrSignal(rule3Scan.existingWait)) {
                    rule3Scan.enabled = false;
                    rule3Scan.needsWorkgroupSync = false;
                    rule3Scan.anchor = nullptr;
                }
            }
            // Rule 3 primes the offset ping-pong (a signal before the loop that
            // the first per-iteration WAIT consumes). The ungated legacy
            // signal-before-wait layout self-pairs each iteration's signal/wait,
            // so no priming is needed -- disable Rule 3 there (Rule 6 is likewise
            // disabled). The drain-gated scheme keeps Rule 3 (LCL-gated).
            if (!kClusterBarrierDrainGateEnabled && kRule4SignalBeforeWaitEnabled) {
                rule3Scan.enabled = false;
                rule3Scan.needsWorkgroupSync = false;
                rule3Scan.anchor = nullptr;
            }
            // PGR=0: no prefetch prologue publication point to prime -- Rule 3 is
            // suppressed together with Rule 2 (see the function-wide Rule 2 block).
            if (pgrValue_ == 0) {
                rule3Scan.enabled = false;
                rule3Scan.needsWorkgroupSync = false;
                rule3Scan.anchor = nullptr;
            }

            // Rule 5 anchors are resolved function-wide in `scanRule5TailPoint`
            // (cross-BB forward/backward scan from the `/* Tail Loop */` marker).
            // Emission is split across the BBs that own the 5a/5b anchors.
            const bool rule5WorkInBB =
                (rule5Want5aWait && !rule5aEmitted && rule5Scan.tailWait != nullptr &&
                 rule5Scan.tailWait->getParent() == &bb) ||
                (rule5Want5aFallback && !rule5aEmitted && rule5Scan.markerBB == &bb) ||
                (rule5Want5b && !rule5bEmitted && rule5Scan.tailTL != nullptr &&
                 rule5Scan.tailTL->getParent() == &bb);

            const bool setupNewTileEnabled = rule3Scan.enabled;
            if (pending.empty() && gsu1Anchors.empty() && !setupNewTileEnabled && !rule5WorkInBB &&
                rule6LoopExitAnchors.empty())
                continue;
            AsmIRBuilder irBuilder(bb, archId);
            // Emission runs in kernel execution order so the generation counter
            // (see `nextSignalGen` / `drainWaitGen`) numbers labels top-to-bottom:
            // Rule 1 -> Rule 3 -> Rule 4 -> Rule 6 -> Rule 5. (Rule 2's bare wait,
            // emitted after this loop, carries no label and does not consume a
            // generation number.)

            // Rule 1 -- priming signal at `label_GSU_1:` (top of the kernel).
            for (IRBase* anchor : gsu1Anchors) {
                const uint64_t signalGen = nextSignalGen();
                if (kClusterBarrierDrainGateEnabled) {
                    // Gated scheme (matches Rule 4's drain gate): wrap the
                    // priming signal in an `LCL != 0` gate (skip when LCL == 0)
                    // and front it with a workgroup sync so the post-GSU==1
                    // join is in lockstep before any wave signals. This keeps
                    // the priming signal suppressed on exactly the paths where
                    // the (drain-gated) loop body -- and its paired waits --
                    // are bypassed.
                    insertLoopCounterLGatedClusterBarrierSignalBefore(
                        anchor, irBuilder, archId,
                        /*cmpUOp=*/GFX::s_cmp_eq_u32,
                        /*skipWhenScc1Imm=*/0,
                        /*cmpComment=*/"gate: only signal when LoopCounterL != 0",
                        /*branchComment=*/"skip cluster barrier when LoopCounterL == 0", signalGen,
                        /*workgroupSyncWaitComment=*/"sync workgroup before cluster signal");
                } else {
                    // Ungated scheme: WaveIdx check is the only guard.
                    insertClusterBarrierSignalOnlyBefore(anchor, irBuilder, archId, signalGen);
                }
            }
            // Rule 3 -- priming signal at the loop's LDS publication point.
            if (setupNewTileEnabled) {
                IRBase* anchor = rule3Scan.anchor;
                assert(anchor != nullptr && "Rule 3 enabled without an anchor");
                BasicBlock* rule3BB = anchor->getParent();
                assert(rule3BB != nullptr && "Rule 3 anchor must belong to a basic block");
                AsmIRBuilder rule3Builder(*rule3BB, archId);
                const uint64_t signalGen = nextSignalGen();
                if (kClusterBarrierDrainGateEnabled) {
                    // Gated scheme (matches Rule 4's drain gate): wrap the
                    // publication-point signal in an `LCL <= pgr` gate so it is
                    // suppressed on the drain iterations whose paired loop-body
                    // signals/waits are also skipped.
                    const std::string immStr = std::to_string(pgrValue_);
                    insertLoopCounterLGatedClusterBarrierSignalBefore(
                        anchor, rule3Builder, archId,
                        /*cmpUOp=*/GFX::s_cmp_le_u32,
                        /*skipWhenScc1Imm=*/pgrValue_,
                        /*cmpComment=*/"LoopCounter <= " + immStr + "?",
                        /*branchComment=*/"skip cluster barrier when LoopCounterL <= " + immStr,
                        signalGen,
                        /*workgroupSyncWaitComment=*/
                        rule3Scan.needsWorkgroupSync ? "workgroup sync" : nullptr);
                } else {
                    // Ungated scheme: emit only the WaveIdx-guarded cluster
                    // signal at the LDS publication point (plus the workgroup
                    // sync pair on the synthesized-publication path when
                    // publication point itself).
                    if (rule3Scan.needsWorkgroupSync) {
                        insertWorkgroupBarrierSyncBefore(anchor, rule3Builder, archId,
                                                         "workgroup sync");
                    }
                    insertClusterBarrierSignalOnlyBefore(anchor, rule3Builder, archId, signalGen);
                }
            }
            // Rule 4 -- per-iteration cluster handshake around each workgroup
            // wait. The wait drains the previous generation; the signal opens
            // the next one (offset ping-pong).
            for (const auto& [trigger, nextIt, liveSccCmp, lclPreDecrement, wgSignal,
                              isFirstPubSinceDec] : pending) {
                IRBase* signalAnchor = (nextIt != bb.end()) ? nextIt.getNodePtr() : nullptr;
                IRBase* waitAnchor =
                    (wgSignal != nullptr) ? static_cast<IRBase*>(wgSignal) : signalAnchor;
                const uint64_t waitGen = drainWaitGen();
                const uint64_t signalGen = nextSignalGen();
                insertClusterBarrierHandshakeBefore(signalAnchor, waitAnchor, irBuilder, archId,
                                                    pgrValue_, isFirstPubSinceDec, liveSccCmp,
                                                    lclPreDecrement, waitGen, signalGen);
                (void)trigger;  // queued for ordering only; insertion uses the anchors
            }
            // Rule 5a -- signal-only after the tail loop's preceding workgroup wait.
            if (rule5Want5aWait && !rule5aEmitted && rule5Scan.tailWait != nullptr &&
                rule5Scan.tailWait->getParent() == &bb) {
                bool conflictsWithRule4 = false;
                for (const auto& [trigger, _next, _live, _dec, _sig, _first] : pending) {
                    if (trigger == rule5Scan.tailWait) {
                        conflictsWithRule4 = true;
                        break;
                    }
                }
                if (!conflictsWithRule4) {
                    BasicBlock* waitBB = rule5Scan.tailWait->getParent();
                    assert(waitBB != nullptr);
                    AsmIRBuilder rule5aBuilder(*waitBB, archId);
                    insertClusterBarrierSignalOnlyBefore(rule5Scan.signalAnchor, rule5aBuilder,
                                                         archId, nextSignalGen());
                    rule5aEmitted = true;
                }
            }
            if (rule5Want5aFallback && !rule5aEmitted && rule5Scan.markerBB == &bb) {
                // Rule 5a fallback: synthesize publication sync immediately after marker.
                AsmIRBuilder rule5aBuilder(*rule5Scan.markerBB, archId);
                insertWorkgroupBarrierSyncBefore(rule5Scan.markerNext, rule5aBuilder, archId,
                                                 "tail workgroup sync");
                insertClusterBarrierSignalOnlyBefore(rule5Scan.markerNext, rule5aBuilder, archId,
                                                     nextSignalGen());
                rule5aEmitted = true;
            }
            // Rule 5b -- bare cluster wait immediately before the tail load.
            if (rule5Want5b && !rule5bEmitted && rule5Scan.tailTL != nullptr &&
                rule5Scan.tailTL->getParent() == &bb) {
                BasicBlock* tlBB = rule5Scan.tailTL->getParent();
                assert(tlBB != nullptr);
                AsmIRBuilder rule5bBuilder(*tlBB, archId);
                insertClusterBarrierWaitBefore(rule5Scan.tailTL, "cluster barrier wait",
                                               rule5bBuilder, archId);
                rule5bEmitted = true;
            }
            // Rule 6 -- bare cluster wait immediately after the label
            // the `LCL == 0` guard lands on (see `loopExitWaitLabel`: either
            // `label_LoopEndL` for the short-branch encoding or
            // `label_PrefetchGlobalLastIterEnd` for the long-branch encoding).
            // This is the trailing wait for the loop's last orphaned signal,
            // placed at the convergence point of every loop-drain path.
            // `rule6LoopExitAnchors` is empty (so this is a no-op) unless the ungated
            // wait-before-signal scheme is active (drain gate off AND legacy
            // signal-before-wait off) -- see the Rule 6 collection block above.
            // Bare wait, so it carries no skip label / generation number.
            for (IRBase* anchor : rule6LoopExitAnchors) {
                insertClusterBarrierWaitBefore(anchor, "cluster barrier wait (loop end)", irBuilder,
                                               archId);
            }
        }

        // Rule 2: a single `s_barrier_wait -3` planted immediately before the
        // first `tensor_load_to_lds` of the whole function. This pass must run
        // on the whole kernel IR (see Gfx1250Backend placement before
        // RegionClonePass), not on a region slice from ScopeAdaptor: on a
        // region the "first tensor_load" would be the region's local first,
        // not the kernel's. Idempotency: skip when the load is already gated
        // by a cluster-scope wait (whether ours, Rule 4's, or one already
        // present in the source IR).
        // Suppressed when `pgrValue_ == 0`: with no prefetch iterations there is
        // no orphaned prologue signal for this leading wait to consume; Rule 1's
        // priming signal is paired by the first Rule 4 bare wait instead.
        StinkyInstruction* firstTL = findFirstTensorLoadInFunc(func);
        if (pgrValue_ != 0 && firstTL != nullptr &&
            !isImmediatelyPrecededByClusterBarrierWait(firstTL)) {
            BasicBlock* parent = firstTL->getParent();
            AsmIRBuilder irBuilder(*parent, archId);
            insertClusterBarrierWaitBefore(firstTL, "cluster_barrier wait", irBuilder, archId);
        }

        return PreservedAnalyses::none();
    }

   private:
    const int pgrValue_;
};

char InsertClusterBarrierPassImpl::ID = 0;

}  // namespace

std::unique_ptr<Pass> createInsertClusterBarrierPass(int pgrValue) {
    return std::make_unique<InsertClusterBarrierPassImpl>(pgrValue);
}

}  // namespace stinkytofu
