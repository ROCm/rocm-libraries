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

#include <random>
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
constexpr const char* kSkipLabelPrefix = "label_skipCBPreSignal_";
/// Prefix for the outer LoopCounterL-gated skip label. Distinct from
/// `kSkipLabelPrefix` so it doubles as a signature of the outer gate.
constexpr const char* kSkipLabelPrefixLCL = "label_skipCBPreSignal_LCL_";
/// Prefix for Rule 4's drain-gated cluster-WAIT skip label (drain-iter gate).
/// Used by `insertLoopCounterLGatedClusterBarrierWaitBefore` when Rule 4's
/// mode (d) drain gate is enabled (`kClusterBarrierDrainGateEnabled == true`).
constexpr const char* kSkipWaitLabelPrefixLCL = "label_skipCBWait_LCL_";
constexpr const char* kWaveIdxSymbol = "sgprWaveIdx";
constexpr const char* kLoopCounterLSymbol = "sgprLoopCounterL";
constexpr size_t kHashLen = 16;
/// Exact label name emitted by Tensile after the GSU==1 early-return
/// guard. Rule 1 anchors on the `LABEL` instruction with this name and
/// emits the signal-only handshake immediately after it.
constexpr const char* kGSU1LabelName = "label_GSU_1";
/// Exact label name emitted by Tensile right before the unrolled
/// summation loop opens. Rule 3 anchors here (both for the
/// PrefetchLocalRead>0 backward scan that finds the workgroup sync
/// `s_barrier_wait -1` and for the
/// PrefetchLocalRead=0 fallback that inserts directly before the label).
/// Internal labels inside the prefetch prologue (e.g. `label_skipPGR2_1`)
/// do not match by exact-name comparison and are walked through.
constexpr const char* kOpenLoopLLabelName = "label_openLoopL";
/// Exact label name Tensile emits at the end of the unrolled summation loop.
/// Rule 5: a single bare `s_barrier_wait -3` is planted immediately after
/// this label so the per-iteration cluster wait moves out of the loop body.
constexpr const char* kLoopEndLLabelName = "label_LoopEndL";
/// Substring used to identify the Tensile comment that opens the tail-loop
/// section. Matches the TEXTBLOCK `/* Tail Loop                       */`.
/// Rule 6 (6a + 6b) uses this as its section anchor.
constexpr const char* kTailLoopMarker = "Tail Loop";

/// Master switch selecting the cluster-barrier handshake scheme across ALL
/// rules (1, 3, 4, and 5) -- not just Rule 4. The two schemes are mutually
/// exclusive and must move together to keep `signal -3` / `wait -3` balanced.
///
///   - When true (GATED scheme): every per-iteration / priming cluster signal
///     is wrapped in a LoopCounterL drain gate so it is suppressed on exactly
///     the drain iterations whose paired counterpart is also skipped:
///       * Rule 4: ASYMMETRIC gates on the wait+signal handshake -- the WAIT
///         is skipped at `LCL <= pgrValue` and the SIGNAL one stage earlier at
///         `LCL <= pgrValue+1`, both lowered by the hoisted `lclPreDecrement`
///         so the gate keys off the same absolute iteration.
///       * Rule 1: priming signal gated on `LCL != 0` (with a leading
///         workgroup sync).
///       * Rule 3: publication-point signal gated on `LCL <= pgrValue`.
///       * Rule 5: DISABLED -- the asymmetric Rule 4 gate already leaves the
///         loop with one extra in-loop WAIT that consumes the last signal, so
///         a trailing loop-exit wait would be unpaired.
///   - When false (UNGATED scheme): Rules 1/3/4 emit a plain WaveIdx-gated
///     signal-only (Rule 4: bare `s_barrier_wait -3` then the signal), and
///     Rule 5 plants a single loop-exit `s_barrier_wait -3` to consume the
///     loop's last otherwise-orphaned signal.
///
/// In BOTH cases `findLiveSccCmpUpstream` is consulted: if SIA hoisted a live
/// loop-exit `s_cmp_* LCL, imm` whose SCC a downstream `s_cbranch_scc{0,1}`
/// consumes, a clone of it is re-emitted AFTER the entire inserted block
/// (past every skip label, so it runs on the gated AND skip paths) to rebuild
/// that SCC. Placing the restore last is what lets the drain gate coexist with
/// a live SCC -- the gate's own `s_cmp_le` clobbers SCC, but the trailing
/// clone re-establishes it for all outgoing paths. (The legacy inherited-SCC
/// mode (a) is therefore unnecessary here and stays unused.)
constexpr bool kClusterBarrierDrainGateEnabled = false;

/// Rule 4 (ungated scheme only) handshake ordering. When false (default) the
/// per-iteration handshake emits the bare `s_barrier_wait -3` FIRST and then the
/// WaveIdx-gated `s_barrier_signal -3` (the offset ping-pong: each wait consumes
/// the PREVIOUS signal, and the loop's last signal is drained by Rule 5's
/// loop-exit wait). When true it reverts to the legacy pre-rewrite layout: the
/// WaveIdx-gated `s_barrier_signal -3` FIRST, then the bare `s_barrier_wait -3`
/// LAST, so the wait stays the load's immediate predecessor and Rule 2's
/// `isImmediatelyPrecededByClusterBarrierWait` idempotency guard keeps working.
/// Any live-SCC restore clone is then placed immediately BEFORE that trailing
/// wait (not after the whole block). Ignored when
/// `kClusterBarrierDrainGateEnabled == true` -- the gated scheme owns ordering.
constexpr bool kRule4SignalBeforeWaitEnabled = false;

/// Returns a fresh 16-character alphanumeric identifier. The first call seeds
/// from std::random_device; subsequent calls reuse the engine for low overhead
/// while still producing collision-resistant IDs across all insertions.
std::string makeRandomHash() {
    static thread_local std::mt19937_64 engine{std::random_device{}()};
    static constexpr char kAlphabet[] =
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    static constexpr size_t kAlphaSize = sizeof(kAlphabet) - 1;
    std::uniform_int_distribution<size_t> dist(0, kAlphaSize - 1);

    std::string out;
    out.reserve(kHashLen);
    for (size_t i = 0; i < kHashLen; ++i) out.push_back(kAlphabet[dist(engine)]);
    return out;
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

/// True if `inst` is a bare cluster-scope wait (`s_barrier_wait -3` with no
/// MemTokenData). Used as an idempotency signature: the standalone wait we
/// synthesize for Rule 2, the leading wait of any Rule-4 handshake, and any
/// equivalent wait already present in the IR all match.
bool isClusterBarrierWait(const StinkyInstruction& inst) {
    return isBarrierWithLiteralId(inst, /*wantSignal=*/false, kClusterBarrierId,
                                  /*rejectMemToken=*/true);
}

/// True if `inst` is `s_barrier_signal -3`. Mirror of `isClusterBarrierWait`
/// used by Rule 3's anchor-mode (b) path for section-level idempotency.
bool isClusterBarrierSignal(const StinkyInstruction& inst) {
    return isBarrierWithLiteralId(inst, /*wantSignal=*/true, kClusterBarrierId,
                                  /*rejectMemToken=*/false);
}

/// A segment boundary is either a label (control-flow entry point) or a
/// branch instruction (control-flow exit point). Treating both as boundaries
/// gives us per-CFG-basic-block segmentation even on Tensile's flat IR,
/// which is important for unrolled loops where iter 1/2 and iter 2/2 sit in
/// the same label segment but are separated by a conditional `s_cbranch`.
bool isSegmentBoundary(const StinkyInstruction& inst) {
    return isLabel(inst) || isBranch(inst);
}

/// Walk backward from \p anchor (exclusive) toward \p segmentBegin (inclusive)
/// to find the nearest preceding `s_barrier_wait -1`. Stops as soon as a
/// segment boundary is crossed so the trigger always lives in the same
/// segment as \p anchor. Used by Rule 4's per-`tensor_load_to_lds` scan to
/// resolve each load's anchor wait.
StinkyInstruction* findPrecedingWorkgroupBarrierWaitInSegment(BasicBlock::iterator segmentBegin,
                                                              StinkyInstruction* anchor) {
    auto it = BasicBlock::iterator(anchor);
    while (it != segmentBegin) {
        --it;
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (isSegmentBoundary(*inst)) return nullptr;
        if (isWorkgroupBarrierWait(*inst)) return inst;
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
/// rebuild SCC (see `insertClusterBarrierHandshakeBefore` mode (d); the legacy
/// `insertRule4InheritedSccSignalBlockBefore` inherited-SCC mode (a) is now
/// unused).
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
/// \p segmentBegin (inclusive) and \p anchor (exclusive), scanning backward
/// and stopping at the first segment boundary (label / branch).
///
/// Rule 4's drain-gated mode (b) thresholds (`pgrValue` for the WAIT,
/// `pgrValue+1` for the SIGNAL) are calibrated against the loop counter value
/// at segment entry. Different `ScheduleIterAlg` settings may hoist the
/// per-iteration `s_sub LCL, LCL, 1` ABOVE the workgroup-wait anchor, so the
/// gate then reads an already-decremented LCL. To keep the gate firing on the
/// identical absolute iteration regardless of where the decrement landed,
/// mode (d)'s drain gate subtracts this sum from both thresholds. Decrements
/// that remain BELOW the anchor (the default schedule) are not seen by the
/// backward scan, so the sum is 0 and the thresholds are left untouched. (Not
/// consulted when the drain gate is disabled, i.e. `kClusterBarrierDrainGateEnabled
/// == false`.)
int sumLoopCounterLDecrementsBeforeInSegment(BasicBlock::iterator segmentBegin,
                                             StinkyInstruction* anchor) {
    int total = 0;
    auto it = BasicBlock::iterator(anchor);
    while (it != segmentBegin) {
        --it;
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (isPseudoInst(inst)) continue;
        if (isSegmentBoundary(*inst)) break;
        int imm = 0;
        if (isLoopCounterLSelfDecrement(*inst, &imm)) total += imm;
    }
    return total;
}

/// Marker-bounded forward scan: walk from \p start (inclusive) up to
/// \p endExclusive and return the first `tensor_load_to_lds` encountered, or
/// nullptr. Does NOT stop at labels or branches -- Rule 6b's tail-loop search
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

/// Marker-bounded backward scan: walk from \p anchor (exclusive) back toward
/// \p boundary (exclusive) and return the nearest preceding
/// `s_barrier_wait -1`, or nullptr. Same "no segment-boundary stopping"
/// semantics as `findFirstTensorLoadBetween`. Used by Rule 6a.
StinkyInstruction* findPrecedingWorkgroupBarrierWaitBetween(BasicBlock::iterator boundary,
                                                            StinkyInstruction* anchor) {
    auto it = BasicBlock::iterator(anchor);
    while (it != boundary) {
        --it;
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (isWorkgroupBarrierWait(*inst)) return inst;
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
///   - Rule 6a: calls this helper alone (no preceding wait, no LCL gate).
///   - Rules 1 / 3: wrap this helper in an outer LoopCounterL gate (see
///     `insertLoopCounterLGatedClusterBarrierSignalBefore`).
void insertClusterBarrierSignalOnlyBefore(IRBase* anchor, AsmIRBuilder& irBuilder,
                                          GfxArchID archId) {
    const std::string labelName = kSkipLabelPrefix + makeRandomHash();

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
///   - Rule 3's PrefetchLocalRead=0 anchor-mode (b): when the backward scan from
///     `label_openLoopL:` finds no `s_barrier_wait -1` before crossing
///     the prefetch boundary, we synthesize the LDS publication point
///     here (comment: `"workgroup sync"`) so the cluster signal
///     that immediately follows still sits at a valid LDS-coherence
///     point.
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
///     s_cbranch_scc1 label_skipCBPreSignal_LCL_<H2>       // skip when SCC1
///     s_cmp_eq_u32 s[sgprWaveIdx], 0                       // inner wave gate
///     s_cbranch_scc0 label_skipCBPreSignal_<H1>
///     s_barrier_signal -3
///   label_skipCBPreSignal_<H1>:                            // inner skip label
///   label_skipCBPreSignal_LCL_<H2>:                        // outer skip label
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
///     s_cbranch_scc1 label_skipCBPreSignal_LCL_<H2>
///     s_barrier_signal -1                                  // workgroup signal
///     s_barrier_wait -1                                    // <workgroupSyncWaitComment>
///     s_cmp_eq_u32 s[sgprWaveIdx], 0
///     s_cbranch_scc0 label_skipCBPreSignal_<H1>
///     s_barrier_signal -3
///   label_skipCBPreSignal_<H1>:
///   label_skipCBPreSignal_LCL_<H2>:
///
/// Instantiations used today (where `pgr = PrefetchGlobalRead` from module options):
///   - Rule 1: `s_cmp_eq_u32` / imm=0       (skip when LCL == 0;
///             workgroupSyncWaitComment = "sync workgroup before
///             cluster signal" -- the post-GSU==1 join needs the
///             workgroup to be in lockstep before the cluster signal
///             fires)
///   - Rule 3 (legacy; Rule 3 now emits signal-only and no longer calls
///             this helper): `s_cmp_le_u32` / imm=pgr  (skip when
///             LCL <= pgr; the
///             same gate is used by both Rule 3 anchor modes -- existing
///             workgroup wait (mode a) or synthesized (mode b) -- so
///             cluster signal/wait pairing stays balanced when the
///             unrolled loop body is bypassed. Mode (b) passes
///             workgroupSyncWaitComment = "workgroup sync" to
///             synthesize the LDS publication point inside the LCL
///             skip region; mode (a) passes nullptr because the wait
///             already exists in the IR.)
///   - Rule 4: used by mode (d) when its drain gate is enabled
///             (`kClusterBarrierDrainGateEnabled == true`): `s_cmp_le_i32`
///             / imm=pgr+1 (minus any LCL pre-decrement), so the SIGNAL is
///             skipped one drain stage earlier than the paired WAIT (gated
///             at imm=pgr via
///             `insertLoopCounterLGatedClusterBarrierWaitBefore`).
///             `workgroupSyncWaitComment` is left nullptr here -- the SCC the
///             gate clobbers is rebuilt by the trailing clone in
///             `insertClusterBarrierHandshakeBefore`, not by this helper.
///             When the drain gate is disabled, Rule 4 emits the ungated
///             shape via `insertClusterBarrierSignalOnlyBefore` instead. The
///             legacy inherited-SCC mode (a) helper
///             `insertRule4InheritedSccSignalBlockBefore` stays unused.
void insertLoopCounterLGatedClusterBarrierSignalBefore(
    IRBase* anchor, AsmIRBuilder& irBuilder, GfxArchID archId, GFX cmpUOp, int skipWhenScc1Imm,
    const std::string& cmpComment, const std::string& branchComment,
    const char* workgroupSyncWaitComment = nullptr) {
    const std::string lclLabelName = std::string(kSkipLabelPrefixLCL) + makeRandomHash();

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

    insertClusterBarrierSignalOnlyBefore(anchor, irBuilder, archId);

    static const HwInstDesc labelMCID{
        GFX::LABEL, GFX::LABEL, 0, 0, 0, "LABEL", makeFlagSet({InstFlag::IF_HasSideEffect})};
    StinkyInstruction* lclLblInst = irBuilder.create(&labelMCID, anchor);
    lclLblInst->addModifier<LabelData>(LabelData{lclLabelName, /*alignment=*/1});
}

/// Rule 4's "inherited SCC" emission: the outer `s_cbranch_scc1`
/// consumes SCC from an upstream live `s_cmp_eq LCL, imm` (`liveSccCmp`)
/// that SIA hoisted above this anchor -- no fresh gate cmp is emitted
/// (it would clobber the SCC the downstream cbranch still needs). A
/// clone of `liveSccCmp` is then re-emitted between the inner and outer
/// skip labels to rebuild SCC for that downstream cbranch. Emitted
/// shape:
///
///     s_cbranch_scc1 label_skipCBPreSignal_LCL_<H2> // SCC inherited
///     s_cmp_eq_u32 s[sgprWaveIdx], 0                // inner wave gate
///     s_cbranch_scc0 label_skipCBPreSignal_<H1>
///     s_barrier_signal -3
///   label_skipCBPreSignal_<H1>:
///     <clone of liveSccCmp>                          // restore SCC
///   label_skipCBPreSignal_LCL_<H2>:
///
/// The restore sits between the inner and outer labels so the LCL-skip
/// path bypasses it (its inherited SCC=1 is already what the downstream
/// cbranch expects), while both wave paths (fall-through and wave-skip)
/// land at or past the inner label and re-evaluate the cmp.
[[maybe_unused]] void insertRule4InheritedSccSignalBlockBefore(IRBase* anchor,
                                                              AsmIRBuilder& irBuilder,
                                                              GfxArchID archId,
                                                              StinkyInstruction* liveSccCmp) {
    const std::string innerLabel = std::string(kSkipLabelPrefix) + makeRandomHash();
    const std::string outerLabel = std::string(kSkipLabelPrefixLCL) + makeRandomHash();

    const HwInstDesc* brSccDesc1 = getMCIDByUOp(GFX::s_cbranch_scc1, archId);
    const HwInstDesc* cmpEqU32Desc = getMCIDByUOp(GFX::s_cmp_eq_u32, archId);
    const HwInstDesc* brSccDesc0 = getMCIDByUOp(GFX::s_cbranch_scc0, archId);
    const HwInstDesc* signalDesc = getMCIDByUOp(GFX::s_barrier_signal, archId);
    assert(brSccDesc1 && cmpEqU32Desc && brSccDesc0 && signalDesc &&
           "Rule 4 cluster-barrier opcodes are not supported on this architecture");

    StinkyInstruction* outerBr = irBuilder.create(brSccDesc1, anchor);
    outerBr->addSrcReg(StinkyRegister(outerLabel));
    outerBr->addModifier<LabelData>(LabelData{outerLabel});
    outerBr->addModifier<CommentData>(
        CommentData{"skip cluster barrier (SCC inherited from upstream LCL cmp)"});

    StinkyInstruction* innerCmp = irBuilder.create(cmpEqU32Desc, anchor);
    innerCmp->addSrcReg(makeSymbolicSgpr(kWaveIdxSymbol));
    innerCmp->addSrcReg(StinkyRegister(0));
    innerCmp->addModifier<CommentData>(CommentData{"Check for waveID 0"});

    StinkyInstruction* innerBr = irBuilder.create(brSccDesc0, anchor);
    innerBr->addSrcReg(StinkyRegister(innerLabel));
    innerBr->addModifier<LabelData>(LabelData{innerLabel});
    innerBr->addModifier<CommentData>(CommentData{"Execute cluster barrier signal for waveID 0"});

    StinkyInstruction* signalInst = irBuilder.create(signalDesc, anchor);
    signalInst->addSrcReg(StinkyRegister(kClusterBarrierId));
    signalInst->addModifier<CommentData>(CommentData{"cluster_barrier signal"});

    static const HwInstDesc labelMCID{
        GFX::LABEL, GFX::LABEL, 0, 0, 0, "LABEL", makeFlagSet({InstFlag::IF_HasSideEffect})};

    StinkyInstruction* innerLbl = irBuilder.create(&labelMCID, anchor);
    innerLbl->addModifier<LabelData>(LabelData{innerLabel, /*alignment=*/1});

    const HwInstDesc* restoreDesc = liveSccCmp->getHwInstDesc();
    StinkyInstruction* restoreInst = irBuilder.create(restoreDesc, anchor);
    for (const auto& src : liveSccCmp->getSrcRegs()) restoreInst->addSrcReg(src);
    restoreInst->addModifier<CommentData>(
        CommentData{"restore SCC for downstream cbranch (Rule 4 inherit)"});

    StinkyInstruction* outerLbl = irBuilder.create(&labelMCID, anchor);
    outerLbl->addModifier<LabelData>(LabelData{outerLabel, /*alignment=*/1});
}

/// Wrap a cluster-barrier WAIT in an outer `s[sgprLoopCounterL] <cmp> <imm>`
/// gate so the wait is skipped (branched over) when the compare sets SCC1.
/// Unlike the signal helper there is NO inner WaveIdx gate -- every wave
/// executes (or skips) the wait in lockstep. Final shape (all before
/// `anchor`):
///
///     <cmpUOp> s[sgprLoopCounterL], <skipWhenScc1Imm>
///     s_cbranch_scc1 label_skipCBWait_LCL_<H>
///     s_barrier_wait -3
///   label_skipCBWait_LCL_<H>:
///
/// Used by Rule 4's mode (d) drain gate (`kClusterBarrierDrainGateEnabled == true`)
/// to skip the cluster wait on the drain iterations.
void insertLoopCounterLGatedClusterBarrierWaitBefore(
    IRBase* anchor, AsmIRBuilder& irBuilder, GfxArchID archId, GFX cmpUOp, int skipWhenScc1Imm,
    const std::string& cmpComment, const std::string& branchComment) {
    const std::string lclLabelName = std::string(kSkipWaitLabelPrefixLCL) + makeRandomHash();

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

/// Forward declaration: defined further down (Rule 2 / Rule 6b helper).
/// Rule 4's mode (d) ungated path (`kClusterBarrierDrainGateEnabled == false`) also
/// emits a bare `s_barrier_wait -3` via this helper, so it must be visible
/// before that call site.
void insertClusterBarrierWaitBefore(IRBase* anchor, const char* comment, AsmIRBuilder& irBuilder,
                                    GfxArchID archId);

/// Emit Rule 4's cluster-barrier handshake before `anchor` (the iterator
/// position right after the load's anchoring `s_barrier_wait -1`) using
/// mode (d) -- a unification of the legacy modes (b) and (c) selected by the
/// `kClusterBarrierDrainGateEnabled` master switch.
///
/// Emission:
///   1. The wait+signal handshake (wait first, then the WaveIdx-gated signal):
///        - drain-gated (`kClusterBarrierDrainGateEnabled == true`): each half is
///          wrapped in its own asymmetric LoopCounterL skip -- the WAIT is
///          skipped at `LCL <= pgrValue - lclPreDecrement` and the SIGNAL one
///          stage earlier at `LCL <= pgrValue+1 - lclPreDecrement`. The paired
///          `tensor_load_to_lds` is disabled (TDM enable dword = 0) on those
///          last PGR iterations, so the handshake is unnecessary there; the
///          one-stage offset keeps `signal -3` / `wait -3` balanced despite
///          the ping-pong pairing (each wait consumes the PREVIOUS signal).
///        - ungated (`== false`): a bare `s_barrier_wait -3` then a
///          WaveIdx-gated `s_barrier_signal -3` -- or, when
///          `kRule4SignalBeforeWaitEnabled == true`, the legacy signal-first
///          layout (signal, then the SCC restore, then the trailing bare wait).
///   2. SCC restore (both gate states): if `liveSccCmp != nullptr` -- SIA
///      hoisted a live loop-exit `s_cmp_* LCL, imm` above the anchor whose SCC
///      a downstream `s_cbranch_scc{0,1}` consumes -- a clone of it is
///      re-emitted AFTER the whole block. Because every drain skip label and
///      the WaveIdx cmp sit BEFORE this clone, the restore runs on all
///      outgoing paths (gated-skip and fall-through alike), so the gate's own
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
void insertClusterBarrierHandshakeBefore(IRBase* anchor, AsmIRBuilder& irBuilder, GfxArchID archId,
                                         int pgrValue, StinkyInstruction* liveSccCmp,
                                         int lclPreDecrement) {
    if (kClusterBarrierDrainGateEnabled) {
        // WAIT: skip on the drain iterations `LCL <= pgrValue` (no inner
        // WaveIdx gate -- every wave skips/executes the wait in lockstep).
        insertLoopCounterLGatedClusterBarrierWaitBefore(
            anchor, irBuilder, archId, GFX::s_cmp_le_i32, pgrValue - lclPreDecrement,
            "drain gate: skip cluster wait when LCL <= pgr", "skip cluster wait (drain)");
        // SIGNAL: skip one stage earlier at `LCL <= pgrValue+1` so the last
        // real signal is not orphaned by the offset ping-pong pairing.
        insertLoopCounterLGatedClusterBarrierSignalBefore(
            anchor, irBuilder, archId, GFX::s_cmp_le_i32, pgrValue + 1 - lclPreDecrement,
            "drain gate: skip cluster signal when LCL <= pgr+1", "skip cluster signal (drain)");
    } else if (kRule4SignalBeforeWaitEnabled) {
        // Legacy "signal before wait" layout (pre-rewrite mode): emit the
        // WaveIdx-gated signal first, then the live-SCC restore clone (if any),
        // then the bare wait LAST -- keeping the wait as the load's immediate
        // predecessor so Rule 2 stays idempotent. Returns early because the SCC
        // restore is emitted here (before the wait), not by the shared trailing
        // block below.
        insertClusterBarrierSignalOnlyBefore(anchor, irBuilder, archId);
        if (liveSccCmp != nullptr) {
            const HwInstDesc* restoreDesc = liveSccCmp->getHwInstDesc();
            StinkyInstruction* restoreInst = irBuilder.create(restoreDesc, anchor);
            for (const auto& src : liveSccCmp->getSrcRegs()) restoreInst->addSrcReg(src);
            restoreInst->addModifier<CommentData>(
                CommentData{"restore SCC for downstream cbranch (Rule 4 signal-before-wait)"});
        }
        insertClusterBarrierWaitBefore(anchor, "cluster barrier wait", irBuilder, archId);
        return;
    } else {
        insertClusterBarrierWaitBefore(anchor, "cluster barrier wait", irBuilder, archId);
        insertClusterBarrierSignalOnlyBefore(anchor, irBuilder, archId);
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
        StinkyInstruction* restoreInst = irBuilder.create(restoreDesc, anchor);
        for (const auto& src : liveSccCmp->getSrcRegs()) restoreInst->addSrcReg(src);
        restoreInst->addModifier<CommentData>(
            CommentData{"restore SCC for downstream cbranch (Rule 4 mode d)"});
    }
}

/// True if `inst` is a `LABEL` pseudo whose `LabelData.label` matches `name`
/// exactly. Anchors:
///   - Rule 1: `kGSU1LabelName` (`label_GSU_1:`)
///   - Rule 3 anchor mode (b): `kOpenLoopLLabelName` (`label_openLoopL:`)
/// Internal control-flow labels (e.g. `label_skipPGR2_1`) do not match by
/// exact-name comparison and are scanned through.
bool isLabelNamed(const StinkyInstruction& inst, const char* name) {
    if (!isLabel(inst)) return false;
    const auto* labelData = inst.getModifier<LabelData>();
    return labelData != nullptr && labelData->label == name;
}

/// True if `ir` is a TEXTBLOCK directive whose value contains `marker` as a
/// substring. Anchor:
///   - Rule 6 (6a / 6b): `kTailLoopMarker` (`/* Tail Loop ... */`)
/// TEXTBLOCK directives are erased at region extraction, so rules using
/// this predicate self-disable at region scope.
bool isTextblockContaining(IRBase* ir, const char* marker) {
    auto* directive = dyn_cast<AsmDirective>(ir);
    return directive != nullptr && directive->kind == AsmDirectiveKind::TEXTBLOCK &&
           directive->value.find(marker) != std::string::npos;
}

/// Idempotency check used by Rules 1, 3, 4, and 6a. Examines the first real
/// successor of `anchor` (via `firstRealInstAfter`) and accepts any of:
///   - `s_barrier_wait -3` (Rule 2 / Rule 4's ungated leading wait /
///     Tensile-emitted full handshake)
///   - `s_cmp_eq_u32 s[sgprWaveIdx], 0` (the WaveIdx gate that opens the
///     ungated Rules 1/3 and Rule 6a signal-only emissions)
///   - `s_cmp_eq_u32 s[sgprLoopCounterL], <imm>` (Rule 1's gated `LCL == 0` gate)
///   - `s_cmp_le_u32 s[sgprLoopCounterL], <imm>` (Rule 3's gated `LCL <= pgr` gate)
///   - `s_cmp_le_i32 s[sgprLoopCounterL], <imm>` (Rule 4 drain-gated mode (d) gate)
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
///   - Rule 6b uses `"cluster barrier wait"`.
/// (Used by Rule 2 and Rule 6b.)
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

    InsertClusterBarrierPassImpl(bool isKernelScope, int pgrValue, int plrValue)
        : isKernelScope_(isKernelScope), pgrValue_(pgrValue), plrValue_(plrValue) {}

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
        // handshake immediately after the nearest preceding
        // `s_barrier_wait -1` (the LDS publication point) via mode (d) -- a
        // bare/`LCL`-drain-gated `s_barrier_wait -3` plus a WaveIdx-gated
        // `s_barrier_signal -3`, with a trailing SCC restore when a live
        // upstream compare exists (see `insertClusterBarrierHandshakeBefore`
        // and `kClusterBarrierDrainGateEnabled`). Triggers are deduplicated by
        // identity, so multiple loads sharing the same anchor wait yield
        // exactly one handshake; the backward scan stays within the load's
        // segment to avoid crossing a CFG edge.
        //
        // Tensile lowers everything into one entry basic block with
        // inline label pseudos and branches instead of a real CFG.
        // Labels (entry boundaries) and branches (exit boundaries) both
        // act as segment delimiters so the backward scan from a load
        // never crosses them. This gives the desired per-iteration
        // coverage for ExpandPointerSwap unrolled loops -- iter 1/2 and
        // iter 2/2 sit under a single `label_LoopBeginL` but are split
        // by the odd-exit `s_cbranch`, so their loads anchor on
        // distinct waits and each receive their own handshake.
        for (BasicBlock& bb : func) {
            // Tuple: (trigger workgroup wait, anchor iterator next to it,
            //         live upstream LCL cmp at trigger or nullptr, cumulative
            //         LCL pre-decrement before the trigger). The third and
            //         fourth elements are captured at scan time -- i.e.
            //         against the pre-mutation IR -- so a later `pending`
            //         entry's emission cannot influence an earlier one's SCC
            //         analysis or decrement count.
            std::vector<
                std::tuple<StinkyInstruction*, BasicBlock::iterator, StinkyInstruction*, int>>
                pending;
            std::unordered_set<StinkyInstruction*> seenTriggers;

            auto segBegin = bb.begin();
            for (auto it = bb.begin(); it != bb.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst == nullptr) continue;
                if (isSegmentBoundary(*inst)) {
                    // Labels and branches both end the current segment;
                    // the boundary instruction itself belongs to neither
                    // side, and the new segment begins right after it.
                    segBegin = std::next(it);
                    continue;
                }
                if (!isTensorLoad(*inst)) continue;

                StinkyInstruction* trigger =
                    findPrecedingWorkgroupBarrierWaitInSegment(segBegin, inst);
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
                // Count any `s_sub LCL, LCL, imm` the schedule hoisted above
                // the anchor so mode (d)'s drain-gate thresholds can be
                // compensated. (Unused when the drain gate is disabled.)
                const int lclPreDecrement =
                    sumLoopCounterLDecrementsBeforeInSegment(segBegin, trigger);
                pending.emplace_back(trigger, std::next(BasicBlock::iterator(trigger)), liveSccCmp,
                                     lclPreDecrement);
            }

            // Rule 1: signal-only handshake immediately AFTER each
            // `label_GSU_1:` label (emitted by Tensile after the GSU==1
            // early-return guard). The label is a `StinkyInstruction`
            // pseudo, so unlike a TEXTBLOCK it survives region extraction
            // - idempotency (`isFollowedByClusterBarrierHandshakeOrSignal`)
            // handles re-entry across scopes.
            //
            // The emitted sequence is a plain `WaveIdx == 0`-gated cluster
            // signal (same shape as Rules 3/6a). There is NO `LoopCounterL`
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

            // Rule 5: plant a single bare `s_barrier_wait -3` at the loop-exit
            // convergence label to consume the loop's last, otherwise-orphaned
            // cluster signal.
            //
            // ONLY enabled when Rule 4's drain gate is OFF
            // (`kClusterBarrierDrainGateEnabled == false`). Rationale (the cluster
            // handshake is an offset ping-pong -- each per-iteration WAIT
            // consumes the PREVIOUS SIGNAL):
            //   - Drain gate OFF: the loop emits equal #WAIT and #SIGNAL, so
            //     the final in-loop SIGNAL has no in-loop WAIT to consume it.
            //     Rule 5 supplies that trailing WAIT at loop exit.
            //   - Drain gate ON: the asymmetric gate skips the WAIT at
            //     `LCL <= pgr` and the SIGNAL one stage earlier at
            //     `LCL <= pgr+1`, leaving the loop with exactly one MORE WAIT
            //     than SIGNAL. That extra WAIT already consumes the last
            //     SIGNAL inside the loop, so a Rule 5 WAIT would be unpaired
            //     (one surplus `wait -3`) and would hang. Hence skip Rule 5.
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
            std::vector<IRBase*> loopEndLAnchors;
            if (!kClusterBarrierDrainGateEnabled) {
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
                    loopEndLAnchors.push_back(anchor);
                }
            }

            // Rule 3: signal-only handshake at the LDS publication point
            // that precedes `label_openLoopL:`. Always enabled. The emission
            // is a plain `WaveIdx == 0`-gated cluster signal (no LoopCounterL
            // gate) that primes the loop's ping-pong: it is consumed by the
            // first per-iteration cluster WAIT (Rule 4) inside the loop body.
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
            // Anchor selection (backward scan from `label_openLoopL:`):
            //   (a) `s_barrier_wait -1` -- publication point already
            //       exists (typical for PrefetchLocalRead > 0 schedules). Anchor at
            //       the successor of that wait. No new workgroup sync
            //       is synthesized. The scan stops if a
            //       `tensor_load_to_lds` is reached first: that
            //       instruction marks the prefetch section, before any
            //       workgroup sync could sit, so an earlier workgroup
            //       wait (if one existed) would be unrelated and must
            //       not be re-used as the publication point.
            //   (b) No `s_barrier_wait -1` between the prefetch tail
            //       and `label_openLoopL:` (typical for PrefetchLocalRead == 0
            //       schedules where the prologue has no local-read
            //       preamble barrier). Only act when `plrValue_ == 0`;
            //       anchor at the label and synthesize an
            //       `s_barrier_signal -1` / `s_barrier_wait -1` pair
            //       (via `insertWorkgroupBarrierSyncBefore`) right before
            //       the WaveIdx-gated cluster signal, so the workgroup has
            //       published its LDS writes before any wave issues the
            //       cluster signal.
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
            //   - Anchor-level (mode a): skip if the existing workgroup
            //     wait is already followed by a cluster handshake, or
            //     if Rule 4 has already queued the same wait as a
            //     trigger (Rule 4 emits the full wait+signal handshake
            //     and supersedes Rule 3 at the same anchor).
            BasicBlock::iterator setupNewTileAnchorIt = bb.end();
            bool setupNewTileNeedsWorkgroupSync = false;
            StinkyInstruction* setupNewTileExistingWait = nullptr;
            StinkyInstruction* openLoopLLabel = nullptr;
            for (auto it = bb.begin(); it != bb.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst != nullptr && isLabelNamed(*inst, kOpenLoopLLabelName)) {
                    openLoopLLabel = inst;
                    break;
                }
            }
            if (openLoopLLabel != nullptr) {
                bool sectionAlreadyHasClusterBarrier = false;
                StinkyInstruction* foundWait = nullptr;
                auto it = BasicBlock::iterator(openLoopLLabel);
                while (it != bb.begin()) {
                    --it;
                    auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                    if (inst == nullptr) continue;
                    if (isTensorLoad(*inst)) break;
                    if (isClusterBarrierWait(*inst) || isClusterBarrierSignal(*inst)) {
                        sectionAlreadyHasClusterBarrier = true;
                    }
                    if (isWorkgroupBarrierWait(*inst)) {
                        foundWait = inst;
                        break;
                    }
                }
                if (!sectionAlreadyHasClusterBarrier) {
                    if (foundWait != nullptr) {
                        setupNewTileExistingWait = foundWait;
                        setupNewTileAnchorIt = std::next(BasicBlock::iterator(foundWait));
                    } else if (plrValue_ == 0) {
                        setupNewTileAnchorIt = BasicBlock::iterator(openLoopLLabel);
                        setupNewTileNeedsWorkgroupSync = true;
                    }
                }
            }
            if (setupNewTileExistingWait != nullptr) {
                bool conflictsWithRule4 = false;
                for (const auto& [trigger, _next, _live, _dec] : pending) {
                    if (trigger == setupNewTileExistingWait) {
                        conflictsWithRule4 = true;
                        break;
                    }
                }
                if (conflictsWithRule4 ||
                    isFollowedByClusterBarrierHandshakeOrSignal(setupNewTileExistingWait)) {
                    setupNewTileAnchorIt = bb.end();
                }
            }

            // Rule 6 -- tail-loop cluster handshake (paired, kernel scope
            // effectively). Two emission sites because the workgroup wait
            // and the tail load are in different label/branch-delimited
            // segments; collapsing them would force the cluster to
            // serialize across the tail TDM-reset code that sits between.
            //   - 6a inserts a signal-only handshake (no LoopCounterL
            //     gate) immediately AFTER the nearest preceding
            //     `s_barrier_wait -1` of the tail load (searching
            //     backward from the load, stopping at the marker).
            //   - 6b inserts a bare `s_barrier_wait -3` immediately
            //     BEFORE the first `tensor_load_to_lds` that follows the
            //     `/* Tail Loop */` TEXTBLOCK marker.
            // Region-scope invocations never observe the marker
            // (`ScopeAdaptor::moveIRToBlock` erases TEXTBLOCK directives),
            // so the rule self-disables there.
            StinkyInstruction* tailTL = nullptr;
            StinkyInstruction* tailWait = nullptr;
            BasicBlock::iterator tailWaitNextIt = bb.end();
            {
                BasicBlock::iterator markerIt = bb.end();
                for (auto it = bb.begin(); it != bb.end(); ++it) {
                    if (isTextblockContaining(it.getNodePtr(), kTailLoopMarker)) {
                        markerIt = it;
                        break;
                    }
                }
                if (markerIt != bb.end()) {
                    tailTL = findFirstTensorLoadBetween(std::next(markerIt), bb.end());
                    if (tailTL != nullptr) {
                        tailWait = findPrecedingWorkgroupBarrierWaitBetween(markerIt, tailTL);
                        if (tailWait != nullptr) {
                            tailWaitNextIt = std::next(BasicBlock::iterator(tailWait));
                        }
                    }
                }
            }
            // Rule 6b idempotency (skip if already preceded by a cluster wait).
            if (tailTL != nullptr && isImmediatelyPrecededByClusterBarrierWait(tailTL)) {
                tailTL = nullptr;
            }
            // Rule 6a idempotency (skip if already followed by a cluster handshake)
            // and Rule-4 collision guard (defer to Rule 4 if it already targets
            // the same wait).
            if (tailWait != nullptr) {
                bool conflictsWithRule4 = false;
                for (const auto& [trigger, _next, _live, _dec] : pending) {
                    if (trigger == tailWait) {
                        conflictsWithRule4 = true;
                        break;
                    }
                }
                if (conflictsWithRule4 || isFollowedByClusterBarrierHandshakeOrSignal(tailWait)) {
                    tailWait = nullptr;
                }
            }

            const bool setupNewTileEnabled = (setupNewTileAnchorIt != bb.end());
            if (pending.empty() && gsu1Anchors.empty() && !setupNewTileEnabled &&
                tailTL == nullptr && tailWait == nullptr && loopEndLAnchors.empty())
                continue;
            AsmIRBuilder irBuilder(bb, archId);
            for (const auto& [trigger, nextIt, liveSccCmp, lclPreDecrement] : pending) {
                IRBase* anchor = (nextIt != bb.end()) ? nextIt.getNodePtr() : nullptr;
                insertClusterBarrierHandshakeBefore(anchor, irBuilder, archId, pgrValue_,
                                                    liveSccCmp, lclPreDecrement);
                (void)trigger;  // queued for ordering only; insertion uses `anchor`
            }
            for (IRBase* anchor : gsu1Anchors) {
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
                        /*branchComment=*/"skip cluster barrier when LoopCounterL == 0",
                        /*workgroupSyncWaitComment=*/"sync workgroup before cluster signal");
                } else {
                    // Ungated scheme: WaveIdx check is the only guard.
                    insertClusterBarrierSignalOnlyBefore(anchor, irBuilder, archId);
                }
            }
            if (setupNewTileEnabled) {
                IRBase* anchor = setupNewTileAnchorIt.getNodePtr();
                if (kClusterBarrierDrainGateEnabled) {
                    // Gated scheme (matches Rule 4's drain gate): wrap the
                    // publication-point signal in an `LCL <= pgr` gate so it is
                    // suppressed on the drain iterations whose paired loop-body
                    // signals/waits are also skipped.
                    const std::string immStr = std::to_string(pgrValue_);
                    insertLoopCounterLGatedClusterBarrierSignalBefore(
                        anchor, irBuilder, archId,
                        /*cmpUOp=*/GFX::s_cmp_le_u32,
                        /*skipWhenScc1Imm=*/pgrValue_,
                        /*cmpComment=*/"LoopCounter <= " + immStr + "?",
                        /*branchComment=*/"skip cluster barrier when LoopCounterL <= " + immStr,
                        /*workgroupSyncWaitComment=*/
                        setupNewTileNeedsWorkgroupSync ? "workgroup sync" : nullptr);
                } else {
                    // Ungated scheme: emit only the WaveIdx-guarded cluster
                    // signal at the LDS publication point (plus the workgroup
                    // sync pair when anchor mode (b) has to synthesize the
                    // publication point itself).
                    if (setupNewTileNeedsWorkgroupSync) {
                        insertWorkgroupBarrierSyncBefore(anchor, irBuilder, archId, "workgroup sync");
                    }
                    insertClusterBarrierSignalOnlyBefore(anchor, irBuilder, archId);
                }
            }
            // Rule 6a -- signal-only after the tail loop's preceding workgroup wait.
            if (tailWait != nullptr) {
                IRBase* anchor =
                    (tailWaitNextIt != bb.end()) ? tailWaitNextIt.getNodePtr() : nullptr;
                insertClusterBarrierSignalOnlyBefore(anchor, irBuilder, archId);
            }
            // Rule 6b -- bare cluster wait immediately before the tail load.
            if (tailTL != nullptr) {
                insertClusterBarrierWaitBefore(tailTL, "cluster barrier wait", irBuilder, archId);
            }
            // Rule 5 -- bare cluster wait immediately after the label
            // the `LCL == 0` guard lands on (see `loopExitWaitLabel`: either
            // `label_LoopEndL` for the short-branch encoding or
            // `label_PrefetchGlobalLastIterEnd` for the long-branch encoding).
            // This is the trailing wait for the loop's last orphaned signal,
            // placed at the convergence point of every loop-drain path.
            // `loopEndLAnchors` is empty (so this is a no-op) when the drain
            // gate is enabled -- see the Rule 5 collection block above.
            for (IRBase* anchor : loopEndLAnchors) {
                insertClusterBarrierWaitBefore(anchor, "cluster barrier wait (loop end)", irBuilder,
                                               archId);
            }
        }

        // Rule 2 (kernel scope only): a single `s_barrier_wait -3` planted
        // immediately before the first `tensor_load_to_lds` of the whole
        // function. Region-scope invocations skip this rule because their
        // notion of "first tensor_load" is the region's local first, not
        // the kernel's. Idempotency: skip when the load is already gated
        // by a cluster-scope wait (whether ours, Rule 4's, or one already
        // present in the source IR).
        if (isKernelScope_) {
            StinkyInstruction* firstTL = findFirstTensorLoadInFunc(func);
            if (firstTL != nullptr && !isImmediatelyPrecededByClusterBarrierWait(firstTL)) {
                BasicBlock* parent = firstTL->getParent();
                AsmIRBuilder irBuilder(*parent, archId);
                insertClusterBarrierWaitBefore(firstTL, "cluster_barrier wait", irBuilder, archId);
            }
        }

        return PreservedAnalyses::none();
    }

   private:
    const bool isKernelScope_;
    const int pgrValue_;
    const int plrValue_;
};

char InsertClusterBarrierPassImpl::ID = 0;

}  // namespace

std::unique_ptr<Pass> createInsertClusterBarrierPass(bool isKernelScope, int pgrValue,
                                                     int plrValue) {
    return std::make_unique<InsertClusterBarrierPassImpl>(isKernelScope, pgrValue, plrValue);
}

}  // namespace stinkytofu
