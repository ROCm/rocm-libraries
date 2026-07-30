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
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/EstimateAsmCyclesPass.hpp"

namespace stinkytofu {
namespace {

constexpr int kClusterBarrierId = -3;
constexpr int kWorkgroupBarrierId = -1;
constexpr const char* kSkipLabelPrefix = "label_skipCBPreSignal_";
/// Prefix for the outer LoopCounterL-gated skip label. Distinct from
/// `kSkipLabelPrefix` so it doubles as a signature of the outer gate.
constexpr const char* kSkipLabelPrefixLCL = "label_skipCBPreSignal_LCL_";
/// Prefix for Rule 4's drain-gated cluster-WAIT skip label (drain-iter gate).
/// Used by `insertLoopCounterLGatedClusterBarrierWaitBefore` when Rule 4 runs
/// in drain-gated mode (b) (i.e. when `kRule4ForceUngatedSignalMode` is off).
constexpr const char* kSkipWaitLabelPrefixLCL = "label_skipCBWait_LCL_";
constexpr const char* kWaveIdxSymbol = "sgprWaveIdx";
constexpr const char* kLoopCounterLSymbol = "sgprLoopCounterL";
constexpr size_t kHashLen = 16;
/// Exact label name emitted by Tensile after the GSU==1 early-return
/// guard. Rule 1 anchors on the `LABEL` instruction with this name and
/// emits the signal-only handshake immediately after it.
constexpr const char* kGSU1LabelName = "label_GSU_1";
/// Substring used to identify the Tensile comment that opens the tail-loop
/// section. Matches the TEXTBLOCK `/* Tail Loop                       */`.
/// Rule 5 (5a + 5b) uses this as its section anchor.
constexpr const char* kTailLoopMarker = "Tail Loop";

/// Master switch for Rule 4's "mode (c)" -- the always-ungated signal
/// handshake. When true, `insertClusterBarrierHandshakeBefore` emits a
/// WaveIdx-gated `s_barrier_signal -3` followed by a bare `s_barrier_wait -3`
/// for EVERY trigger; the cluster signal is never wrapped in an LCL skip
/// branch (unlike inherited-SCC mode (a) or drain-gated mode (b)). When
/// false, the original mode-selection logic stays in effect: inherited-SCC
/// mode (a) for `liveLclCmp != nullptr`, drain-gated mode (b) otherwise.
///
/// Mode (c) still detects `liveLclCmp`: if SIA hoisted a loop-exit
/// `s_cmp_eq LCL, imm` whose SCC a downstream `s_cbranch_scc0 LoopBeginL`
/// consumes, the inner `s_cmp_eq_u32 s[sgprWaveIdx], 0` would clobber that
/// SCC, so a clone of the cmp is re-emitted AFTER the bare wait to rebuild
/// it for the loop-back branch.
constexpr bool kRule4ForceUngatedSignalMode = true;

/// How many estimated cycles EARLIER than its paired WAIT (Rule 4(b)) the
/// cluster SIGNAL (Rule 4(a)) should be planted. The signal anchor is walked
/// backward from the trigger `s_barrier_wait -1` until the estimated cumulative
/// cycle drops at least this far below the trigger's cycle, but never past the
/// current segment's first instruction (so the signal stays on the same
/// control-flow path as the wait and the `signal -3` / `wait -3` pairing stays
/// balanced). Set to 0 to keep the signal at its original position (right after
/// the trigger, i.e. co-located with the wait). Requires per-instruction cycle
/// estimates (Gfx1250 loop region only); when unavailable the signal falls back
/// to its original position.
constexpr int kRule4SignalLeadCycles = 500;

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

/// True if `inst` is a workgroup-scope barrier arrival: `s_barrier_signal -1`.
/// Rule 4(b) anchors its cluster `s_barrier_wait -3` immediately BEFORE this
/// signal (the arrival half of the workgroup barrier that pairs with the
/// trigger `s_barrier_wait -1`).
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

/// A segment boundary is either a label (control-flow entry point) or a
/// branch instruction (control-flow exit point). Treating both as boundaries
/// gives us per-CFG-basic-block segmentation even on Tensile's flat IR,
/// which is important for unrolled loops where iter 1/2 and iter 2/2 sit in
/// the same label segment but are separated by a conditional `s_cbranch`.
bool isSegmentBoundary(const StinkyInstruction& inst) {
    return isLabel(inst) || isBranch(inst) || isCall(inst);
}

/// Walk backward from \p anchor (exclusive) toward \p segmentBegin to find the
/// nearest preceding `s_barrier_signal -1` (the arrival half of the workgroup
/// barrier). Stops at a segment boundary. Used by Rule 4 both to resolve each
/// load's trigger signal (the cluster `s_barrier_wait -3` is anchored
/// immediately BEFORE it, so it precedes the whole workgroup barrier pair, and
/// hence the tensor load) and by the Rule 5 conflict guards to bridge a
/// workgroup wait to its paired signal for identity comparison.
StinkyInstruction* findPrecedingWorkgroupBarrierSignalInSegment(BasicBlock::iterator segmentBegin,
                                                                StinkyInstruction* anchor) {
    auto it = BasicBlock::iterator(anchor);
    while (it != segmentBegin) {
        --it;
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (isSegmentBoundary(*inst)) return nullptr;
        if (isWorkgroupBarrierSignal(*inst)) return inst;
    }
    return nullptr;
}

/// Walk backward from \p anchor within the same basic block, looking for a
/// "live" SCC producer whose SCC has NOT been consumed before \p anchor and that
/// can be SAFELY cloned to rebuild SCC at \p anchor. A candidate qualifies when:
///   1. it writes SCC (`IF_ImplicitWriteSCC`),
///   2. it is a pure comparator: none of its destinations is an allocatable
///      (GPR) register. Note an SOPC compare (`s_cmp_*`) models its SCC result
///      as an explicit SCC-typed destination, so it is NOT dest-less; the test
///      is "no GPR dest", not "no dest". An arithmetic/logic SCC writer such as
///      `s_sub_u32` additionally writes a real GPR and is unsafe to clone, and
///   3. none of its source registers are overwritten between it and \p anchor
///      (checked via physical-register `isOverlap`), so a clone at \p anchor
///      recomputes the identical SCC.
/// Returns nullptr on BB start, label, branch, any SCC reader, or an SCC writer
/// failing (2)/(3). The nearest upstream SCC writer is always the candidate, so
/// no other SCC writer can sit between it and \p anchor.
///
/// Used by Rule 4(a) (mode (c)): when found, a clone of the comparator is
/// re-emitted right after the signal block to restore the SCC its WaveIdx
/// `s_cmp_eq_u32` clobbered, before any downstream `s_cbranch_scc*` consumes it.
/// (The legacy modes (a)/(b) also consult the result; they were tuned for the
/// loop-exit `s_cmp_eq LCL, imm` case, which remains the dominant match.)
StinkyInstruction* findLiveRestorableSccCmpUpstream(StinkyInstruction* anchor) {
    BasicBlock* parent = anchor->getParent();
    if (parent == nullptr) return nullptr;
    // Destination registers written by instructions between the candidate and
    // `anchor` (accumulated as we walk backward), for the operand-stability check.
    std::vector<StinkyRegister> clobbered;
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
            // Only pure comparators are safe to clone. In this IR an SOPC compare
            // (`s_cmp_*`) models its SCC result as an explicit SCC-typed
            // destination register, so it is NOT dest-less: a candidate qualifies
            // only when none of its destinations is an allocatable (GPR)
            // register. An arithmetic/logic SCC writer (`s_add_u32`, `s_sub_u32`,
            // ...) additionally writes a real GPR, which makes re-emitting it
            // unsafe -- reject it.
            for (const auto& dst : inst->getDestRegs()) {
                if (dst.isRegister() && isAllocatableReg(dst.reg.type)) return nullptr;
            }
            // Operand stability: reject if any source was clobbered in between.
            for (const auto& src : inst->getSrcRegs()) {
                if (!src.isRegister()) continue;
                for (const auto& w : clobbered) {
                    if (src.isOverlap(w)) return nullptr;
                }
            }
            return inst;
        }
        // Non-SCC-writing instruction on the path: record its dest registers so a
        // later-found candidate can verify its sources stayed stable.
        for (const auto& dst : inst->getDestRegs()) {
            if (dst.isRegister()) clobbered.push_back(dst);
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
/// identical absolute iteration regardless of where the decrement landed, mode
/// (b) subtracts this sum from both thresholds. Decrements that remain BELOW
/// the anchor (the default schedule) are not seen by the backward scan, so the
/// sum is 0 and the thresholds are left untouched. (Not consulted when
/// `kRule4ForceUngatedSignalMode` is on, i.e. mode (c).)
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

/// Marker-bounded backward scan: walk from \p anchor (exclusive) back toward
/// \p boundary (exclusive) and return the nearest preceding
/// `s_barrier_wait -1`, or nullptr. Same "no segment-boundary stopping"
/// semantics as `findFirstTensorLoadBetween`. Used by Rule 5a.
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
///   - Rule 5a: calls this helper alone (no preceding wait, no LCL gate).
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
///   - Rule 4: uses this helper only in its drain-gated fallback mode (b)
///             (when `kRule4ForceUngatedSignalMode` is off): `s_cmp_le_i32`
///             / imm=pgr+1 (minus any LCL pre-decrement), so the SIGNAL is
///             skipped one drain stage earlier than the paired WAIT (gated
///             at imm=pgr via
///             `insertLoopCounterLGatedClusterBarrierWaitBefore`). The
///             active mode (c) and the inherited-SCC mode (a) do NOT use
///             this helper: mode (c) emits via
///             `insertClusterBarrierSignalOnlyBefore` (WaveIdx-gated, no
///             LoopCounterL gate) and mode (a) via
///             `insertRule4InheritedSccSignalBlockBefore`. See
///             `insertClusterBarrierHandshakeBefore`.
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
/// consumes SCC from an upstream live `s_cmp_eq LCL, imm` (`liveLclCmp`)
/// that SIA hoisted above this anchor -- no fresh gate cmp is emitted
/// (it would clobber the SCC the downstream cbranch still needs). A
/// clone of `liveLclCmp` is then re-emitted between the inner and outer
/// skip labels to rebuild SCC for that downstream cbranch. Emitted
/// shape:
///
///     s_cbranch_scc1 label_skipCBPreSignal_LCL_<H2> // SCC inherited
///     s_cmp_eq_u32 s[sgprWaveIdx], 0                // inner wave gate
///     s_cbranch_scc0 label_skipCBPreSignal_<H1>
///     s_barrier_signal -3
///   label_skipCBPreSignal_<H1>:
///     <clone of liveLclCmp>                          // restore SCC
///   label_skipCBPreSignal_LCL_<H2>:
///
/// The restore sits between the inner and outer labels so the LCL-skip
/// path bypasses it (its inherited SCC=1 is already what the downstream
/// cbranch expects), while both wave paths (fall-through and wave-skip)
/// land at or past the inner label and re-evaluate the cmp.
void insertRule4InheritedSccSignalBlockBefore(IRBase* anchor, AsmIRBuilder& irBuilder,
                                              GfxArchID archId, StinkyInstruction* liveLclCmp) {
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

    const HwInstDesc* restoreDesc = liveLclCmp->getHwInstDesc();
    StinkyInstruction* restoreInst = irBuilder.create(restoreDesc, anchor);
    for (const auto& src : liveLclCmp->getSrcRegs()) restoreInst->addSrcReg(src);
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
/// Used by Rule 4's drain-gated mode (b) (when `kRule4ForceUngatedSignalMode`
/// is off) to skip the cluster wait on the drain iterations.
void insertLoopCounterLGatedClusterBarrierWaitBefore(IRBase* anchor, AsmIRBuilder& irBuilder,
                                                     GfxArchID archId, GFX cmpUOp,
                                                     int skipWhenScc1Imm,
                                                     const std::string& cmpComment,
                                                     const std::string& branchComment) {
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

/// Forward declaration: defined further down (Rule 2 / Rule 5b helper).
/// Rule 4's handshake now also emits a bare trailing `s_barrier_wait -3`
/// via this helper, so it must be visible before that call site.
void insertClusterBarrierWaitBefore(IRBase* anchor, const char* comment, AsmIRBuilder& irBuilder,
                                    GfxArchID archId);

/// Re-emit a clone of `liveLclCmp` immediately before `anchor` to rebuild the
/// SCC that Rule 4(a)'s WaveIdx `s_cmp_eq_u32` clobbered (needed when SIA
/// hoisted a live loop-exit `s_cmp_eq LCL, imm` whose SCC a downstream
/// `s_cbranch_scc0 LoopBeginL` consumes). Shared by Rule 4(a) (separated
/// anchors: restore right after the signal block) and Rule 4(b) (co-located
/// anchors: restore after the bare wait).
void insertRule4SccRestore(IRBase* anchor, AsmIRBuilder& irBuilder, StinkyInstruction* liveLclCmp) {
    const HwInstDesc* restoreDesc = liveLclCmp->getHwInstDesc();
    StinkyInstruction* restoreInst = irBuilder.create(restoreDesc, anchor);
    for (const auto& src : liveLclCmp->getSrcRegs()) restoreInst->addSrcReg(src);
    restoreInst->addModifier<CommentData>(
        CommentData{"restore SCC for downstream cbranch (Rule 4 mode c)"});
}

/// Rule 4(a) -- the SIGNAL half of Rule 4's mode (c) handshake. Emits the
/// WaveIdx-gated `s_barrier_signal -3` (only wave 0 signals) before
/// `signalAnchor`. When `sccRestoreCmp` is non-null the SCC restore is emitted
/// right AFTER the signal block (used when the signal is separated from its
/// paired wait, so the clobbered SCC is rebuilt locally before any downstream
/// consumer between the signal and the far-away wait). The signal is never
/// wrapped in an LCL skip branch.
void insertRule4ClusterBarrierSignal(IRBase* signalAnchor, AsmIRBuilder& irBuilder,
                                     GfxArchID archId, StinkyInstruction* sccRestoreCmp) {
    insertClusterBarrierSignalOnlyBefore(signalAnchor, irBuilder, archId);
    if (sccRestoreCmp != nullptr) {
        insertRule4SccRestore(signalAnchor, irBuilder, sccRestoreCmp);
    }
}

/// Rule 4(b) -- the WAIT half of Rule 4's mode (c) handshake. Emits a bare
/// `s_barrier_wait -3` before `waitAnchor`. When `sccRestoreCmp` is non-null a
/// clone of that cmp is re-emitted AFTER the wait to rebuild the SCC that Rule
/// 4(a)'s WaveIdx `s_cmp_eq_u32` clobbered. The wait itself has no SCC side
/// effect, so the restore is safe to place here. `sccRestoreCmp` is passed only
/// when the two halves share an anchor (co-located); when they are separated
/// the restore rides with Rule 4(a) instead (see `insertRule4ClusterBarrierSignal`).
void insertRule4ClusterBarrierWait(IRBase* waitAnchor, AsmIRBuilder& irBuilder, GfxArchID archId,
                                   StinkyInstruction* sccRestoreCmp) {
    insertClusterBarrierWaitBefore(waitAnchor, "cluster barrier wait", irBuilder, archId);
    if (sccRestoreCmp != nullptr) {
        insertRule4SccRestore(waitAnchor, irBuilder, sccRestoreCmp);
    }
}

/// Emit Rule 4's cluster-barrier handshake, split into two independently
/// anchored halves: the SIGNAL (Rule 4(a)) before `signalAnchor` and the WAIT
/// (Rule 4(b)) before `waitAnchor`. Both anchors currently resolve to the same
/// position (right after the load's anchoring `s_barrier_wait -1`), but they
/// are passed separately so each half can be repositioned on its own.
///
/// Mode (c) -- always-ungated signal (active when `kRule4ForceUngatedSignalMode`
/// is true): split into two independently anchored halves --
///   - Rule 4(a): a WaveIdx-gated `s_barrier_signal -3` before `signalAnchor`
///     (via `insertRule4ClusterBarrierSignal`).
///   - Rule 4(b): a bare `s_barrier_wait -3` before `waitAnchor` (via
///     `insertRule4ClusterBarrierWait`).
/// The signal is never wrapped in an LCL skip branch. `liveLclCmp` (if present)
/// drives an SCC restore that ALWAYS rides with Rule 4(a), right after the
/// signal block, so the SCC the downstream `s_cbranch_scc0 LoopBeginL` consumes
/// is rebuilt locally before any instruction between the (possibly far-apart)
/// signal and wait; Rule 4(b) emits a bare, SCC-neutral wait. The caller decides
/// `liveLclCmp` at scan time (null when an intervening SCC writer masks the
/// clobber). It returns before the mode (a)/(b) selection below, so those paths
/// are left untouched.
///
/// When the mode (c) switch is false, two emission modes are selected by the
/// caller via `findLiveRestorableSccCmpUpstream`:
///
///   (a) `liveLclCmp != nullptr` -- SIA=4 inherited-SCC path. Tensile hoisted
///       the loop-exit `s_cmp_eq_i32 LCL, imm` above this anchor and a
///       downstream `s_cbranch_scc0 LoopBeginL` consumes its SCC. Emit an
///       ungated leading `s_barrier_wait -3` followed by the inherited-SCC
///       signal block (the signal is single-iter skipped via the inherited
///       SCC; a clone of the upstream cmp is re-emitted between the inner and
///       outer skip labels to restore SCC). Emitting fresh relational gates
///       here would clobber the live SCC, so this mode is left as-is.
///
///   (b) `liveLclCmp == nullptr` -- drain-gated path. The paired
///       `tensor_load_to_lds` is disabled (TDM enable dword = 0) on the last
///       PGR iterations (`LCL <= pgrValue`), so the handshake is unnecessary
///       there. Because the ping-pong pairing is offset (each wait consumes the
///       PREVIOUS signal), dropping a drain wait would orphan the last real
///       signal. We therefore use ASYMMETRIC gates: skip the WAIT at
///       `LCL <= pgrValue` and the SIGNAL one stage earlier at
///       `LCL <= pgrValue+1`. Both thresholds drop by \p lclPreDecrement so the
///       gate keys off the same absolute iteration even when the schedule
///       decremented LCL before the anchor.
///
/// \p pgrValue and \p lclPreDecrement are consulted by mode (b) only. In modes
/// (a)/(b) `signalAnchor` and `waitAnchor` currently resolve to the same
/// position, so their historical single-anchor behavior is preserved: mode (a)
/// keeps its intricate inherited-SCC block on `waitAnchor`, while mode (b)
/// naturally maps its WAIT gate to `waitAnchor` and its SIGNAL gate to
/// `signalAnchor`.
void insertClusterBarrierHandshakeBefore(IRBase* signalAnchor, IRBase* waitAnchor,
                                         AsmIRBuilder& irBuilder, GfxArchID archId, int pgrValue,
                                         StinkyInstruction* liveLclCmp, int lclPreDecrement) {
    if (kRule4ForceUngatedSignalMode) {
        // Mode (c): always-ungated signal, split into two independently
        // anchored halves. The cluster signal is NEVER wrapped in an LCL skip
        // branch (unlike mode (a)).
        //
        // SCC restore always rides with Rule 4(a): the WaveIdx `s_cmp_eq_u32` in
        // the signal block clobbers SCC at the SIGNAL anchor, so the restore is
        // rebuilt locally -- right after the signal block -- ahead of any
        // downstream consumer (even when the signal leads the wait by many
        // cycles). `liveLclCmp` already carries the caller's restore decision
        // (null when an existing SCC writer between the anchors masks the
        // clobber). Rule 4(b) emits a bare, SCC-neutral wait.
        insertRule4ClusterBarrierSignal(signalAnchor, irBuilder, archId,
                                        liveLclCmp);  // Rule 4(a)
        insertRule4ClusterBarrierWait(waitAnchor, irBuilder, archId,
                                      /*sccRestoreCmp=*/nullptr);  // Rule 4(b)
        return;
    }
    if (liveLclCmp != nullptr) {
        // Mode (a) inherited-SCC: ungated leading wait + a single-iter
        // inherited signal skip. Kept on a single anchor (`waitAnchor`) because
        // the inherited-SCC block interleaves the wait, signal, and SCC restore.
        const HwInstDesc* waitDesc = getMCIDByUOp(GFX::s_barrier_wait, archId);
        assert(waitDesc && "Cluster-barrier wait opcode is not supported on this architecture");
        StinkyInstruction* waitInst = irBuilder.create(waitDesc, waitAnchor);
        waitInst->addSrcReg(StinkyRegister(kClusterBarrierId));
        waitInst->addModifier<CommentData>(CommentData{"cluster barrier wait"});
        insertRule4InheritedSccSignalBlockBefore(waitAnchor, irBuilder, archId, liveLclCmp);
        return;
    }

    // Mode (b) drain-gated: gate the WAIT (Rule 4(b)) at `LCL <= pgr` (drain
    // iters, load disabled) and the SIGNAL (Rule 4(a)) at `LCL <= pgr+1` (one
    // stage earlier so the trailing leftover signal is dropped too). Both
    // thresholds drop by `lclPreDecrement` so the gate keys off the same
    // absolute iteration even when the schedule decremented LCL before the
    // anchor.
    const int waitImm = pgrValue - lclPreDecrement;
    const std::string waitImmStr = std::to_string(waitImm);
    insertLoopCounterLGatedClusterBarrierWaitBefore(
        waitAnchor, irBuilder, archId,
        /*cmpUOp=*/GFX::s_cmp_le_i32,
        /*skipWhenScc1Imm=*/waitImm,
        /*cmpComment=*/"drain iter? LoopCounter <= " + waitImmStr,
        /*branchComment=*/"skip cluster wait when LoopCounterL <= " + waitImmStr);

    const int sigImm = pgrValue + 1 - lclPreDecrement;
    const std::string sigImmStr = std::to_string(sigImm);
    insertLoopCounterLGatedClusterBarrierSignalBefore(
        signalAnchor, irBuilder, archId,
        /*cmpUOp=*/GFX::s_cmp_le_i32,
        /*skipWhenScc1Imm=*/sigImm,
        /*cmpComment=*/"LoopCounter <= " + sigImmStr + "?",
        /*branchComment=*/"skip cluster barrier when LoopCounterL <= " + sigImmStr);
}

/// True if `inst` is a `LABEL` pseudo whose `LabelData.label` matches `name`
/// exactly. Anchors:
///   - Rule 1: `kGSU1LabelName` (`label_GSU_1:`)
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

/// Idempotency check used by Rules 1, 3, 4, and 5a. Examines the first real
/// successor of `anchor` (via `firstRealInstAfter`) and accepts any of:
///   - `s_barrier_wait -3` (Rule 2 / Tensile-emitted full handshake)
///   - `s_cmp_eq_u32 s[sgprWaveIdx], 0` (Rule 5a signal-only AND Rule 4's
///     current emission, whose first instruction is this WaveIdx gate)
///   - `s_cmp_eq_u32 s[sgprLoopCounterL], <imm>` (Rule 1's `LCL == 0` gate)
///   - `s_cmp_le_i32 s[sgprLoopCounterL], <imm>` (Rule 4 drain-gated mode (b) gate)
///   - `s_cmp_eq_i32 s[sgprLoopCounterL], <imm>` (Rule 4 inherited-SCC clone)
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

/// Resolve Rule 4(a)'s signal anchor by walking backward from `referenceAnchor`
/// (Rule 4(b)'s WAIT anchor -- the workgroup `s_barrier_signal -1`) until the
/// estimated cumulative cycle drops at least `leadCycles` below the reference's
/// cycle, then anchoring the signal before that instruction. Measuring the lead
/// from the WAIT anchor (rather than the trigger) guarantees the resolved signal
/// anchor is at or before the wait, so the SIGNAL always leads the WAIT even for
/// small `leadCycles`. The walk never crosses out of the segment: it stops at
/// `segBegin` (the first instruction after the previous label/branch/call
/// boundary), clamping the signal to the segment start so the SIGNAL stays on
/// the same control-flow path as its paired WAIT.
///
/// The walk also stops if it reaches a preceding handshake's WAIT anchor --
/// another workgroup `s_barrier_signal -1` (every Rule-4 trigger is one):
/// hoisting this signal into or above that handshake would let the two cluster
/// `signal -3`/`wait -3` pairs overlap and deadlock. In that case the signal is
/// clamped to just AFTER that handshake's workgroup `s_barrier_wait -1` (the
/// first real instruction following it). The prior handshake is detected via
/// its workgroup signal rather than its cluster `wait -3`, because at
/// anchor-resolution time the cluster waits have not been emitted yet.
///
/// Falls back to `defaultAnchor` (the WAIT anchor, i.e. co-located) when:
///   - `leadCycles <= 0` (feature disabled), or
///   - the reference has no cycle estimate (non-Gfx1250, or outside the modeled
///     loop region, or issue-cycle data absent) -- signaling that cycle-based
///     placement is unavailable here.
/// Instructions without a cycle estimate encountered mid-walk are skipped (they
/// do not advance the search), so a sparse map degrades gracefully.
IRBase* findRule4SignalAnchorByCycleLead(
    StinkyInstruction* referenceAnchor, BasicBlock::iterator segBegin, IRBase* defaultAnchor,
    const std::unordered_map<const StinkyInstruction*, uint32_t>& cycleMap, int leadCycles) {
    if (leadCycles <= 0) return defaultAnchor;
    auto refIt = cycleMap.find(referenceAnchor);
    if (refIt == cycleMap.end()) return defaultAnchor;

    const int64_t target = static_cast<int64_t>(refIt->second) - leadCycles;
    auto it = BasicBlock::iterator(referenceAnchor);
    while (it != segBegin) {
        --it;
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        // Boundary: a PRECEDING handshake's WAIT anchor -- another workgroup
        // `s_barrier_signal -1` (each Rule-4 trigger is one). Never hoist this
        // signal into or above that prior handshake, or the two cluster
        // `signal -3`/`wait -3` pairs overlap and can deadlock. Clamp instead to
        // just AFTER that handshake's workgroup `s_barrier_wait -1` (the tail of
        // its barrier block). NOTE: this is detected via the workgroup signal,
        // not the cluster `wait -3`, because at anchor-resolution time the
        // cluster waits have not been emitted yet -- only the original workgroup
        // barriers exist.
        if (isWorkgroupBarrierSignal(*inst)) {
            for (StinkyInstruction* fwd = firstRealInstAfter(inst);
                 fwd != nullptr && fwd != referenceAnchor; fwd = firstRealInstAfter(fwd)) {
                if (isWorkgroupBarrierWait(*fwd)) {
                    StinkyInstruction* after = firstRealInstAfter(fwd);
                    return (after != nullptr) ? static_cast<IRBase*>(after) : defaultAnchor;
                }
            }
            // No paired workgroup wait below it (unexpected): co-locate the
            // signal with its wait rather than crossing the prior handshake.
            return defaultAnchor;
        }
        auto cycleIt = cycleMap.find(inst);
        if (cycleIt == cycleMap.end()) continue;
        if (static_cast<int64_t>(cycleIt->second) <= target) return inst;
    }
    // Reached the segment start without dropping `leadCycles` cycles: clamp the
    // signal to the first instruction of the segment.
    return segBegin.getNodePtr();
}

/// True if any instruction in the half-open program-order range
/// [`fromInclusive`, `toExclusive`) implicitly writes SCC. Rule 4(a) uses this
/// to decide whether its signal block's WaveIdx `s_cmp_eq_u32` SCC clobber
/// matters: if an existing instruction between the SIGNAL anchor and the WAIT
/// anchor already redefines SCC, the clobber is masked before any downstream
/// consumer sees it, so no SCC restore is needed. Scans only the pre-mutation
/// IR (called at scan time, before any handshake is inserted).
bool anySccWriterInRange(StinkyInstruction* fromInclusive, const IRBase* toExclusive) {
    BasicBlock* parent = fromInclusive->getParent();
    if (parent == nullptr) return false;
    for (auto it = BasicBlock::iterator(fromInclusive); it != parent->end(); ++it) {
        if (it.getNodePtr() == toExclusive) break;
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (inst->is(InstFlag::IF_ImplicitWriteSCC)) return true;
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

class InsertClusterBarrierPassImpl : public Pass {
   public:
    static char ID;

    explicit InsertClusterBarrierPassImpl(int pgrValue) : pgrValue_(pgrValue) {}

    const char* getName() const override {
        return "Insert Cluster Barrier";
    }

    Pass::ID getPassID() const override {
        return &InsertClusterBarrierPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        const auto& arch = passCtx.getGemmTileConfig().arch;
        const GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);

        // Per-instruction estimated cumulative cycle positions, used to place
        // Rule 4(a)'s signal `kRule4SignalLeadCycles` cycles ahead of its
        // paired wait. Computed once against the pre-mutation IR (the query
        // does not annotate the IR). Empty for non-Gfx1250 or when the estimator
        // finds no modeled loop region, in which case Rule 4(a) falls back to
        // co-locating the signal with the wait.
        const std::unordered_map<const StinkyInstruction*, uint32_t> cycleMap =
            (kRule4SignalLeadCycles > 0)
                ? computeEstimatedCyclesPerInstruction(func, passCtx)
                : std::unordered_map<const StinkyInstruction*, uint32_t>{};

        // Rule 4: for each `tensor_load_to_lds`, plant a cluster-barrier
        // handshake immediately after the nearest preceding
        // `s_barrier_wait -1` (the LDS publication point). The handshake is a
        // WaveIdx-gated `s_barrier_signal -3` followed by a bare
        // `s_barrier_wait -3` (no LoopCounterL gate). Triggers are
        // deduplicated by identity, so multiple loads sharing the same
        // anchor wait yield exactly one handshake; the backward scan stays
        // within the load's segment to avoid crossing a CFG edge.
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
            // Tuple: (trigger workgroup signal, resolved Rule 4(a) signal anchor,
            //         resolved Rule 4(b) wait anchor, live upstream LCL cmp at
            //         the SIGNAL anchor or nullptr, cumulative LCL pre-decrement
            //         before the trigger). Everything is resolved at scan time
            //         -- i.e. against the pre-mutation IR -- so a later `pending`
            //         entry's emission cannot influence an earlier one's anchor
            //         resolution, SCC analysis, or decrement count. Crucially the
            //         live-cmp is scanned backward from the SIGNAL anchor (the
            //         actual `s_cmp_eq_u32 sgprWaveIdx` SCC-clobber point), not
            //         from the trigger, so the SCC restore decision reflects
            //         where Rule 4(a) really lands.
            std::vector<
                std::tuple<StinkyInstruction*, IRBase*, IRBase*, StinkyInstruction*, int>>
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

                // Rule 4's trigger IS the workgroup `s_barrier_signal -1` paired
                // with the load's barrier, found directly by walking back from the
                // load (the nearest preceding signal in the segment). Rule 5 keeps
                // using the workgroup-WAIT finder; its conflict guards bridge to
                // this signal via each wait's paired signal.
                StinkyInstruction* trigger =
                    findPrecedingWorkgroupBarrierSignalInSegment(segBegin, inst);
                // A workgroup signal is required: without one there is no barrier
                // pair to anchor the cluster wait before, so Rule 4 does not apply.
                if (trigger == nullptr) continue;
                // Dedup: multiple loads can share the same barrier; only the first
                // one queues an emission.
                if (!seenTriggers.insert(trigger).second) continue;

                // Rule 4(b) WAIT anchor: immediately before the workgroup signal
                // (the trigger), so the cluster `s_barrier_wait -3` precedes the
                // whole workgroup barrier pair -- and hence the tensor load.
                IRBase* waitAnchor = trigger;
                StinkyInstruction* waitAnchorInst = trigger;
                // Idempotency: skip if the WAIT anchor is already immediately
                // preceded by a cluster wait (a prior run already emitted here).
                if (isImmediatelyPrecededByClusterBarrierWait(waitAnchorInst)) continue;

                // Resolve the SIGNAL anchor NOW (against the pre-mutation IR) so a
                // later `pending` entry's emission cannot perturb this resolution.
                //
                // Rule 4(a) SIGNAL anchor: walk backward FROM the WAIT anchor by
                // ~kRule4SignalLeadCycles estimated cycles (bounded by the
                // segment). Measuring the lead from the wait guarantees the signal
                // lands at or before it. Falls back to the wait anchor (co-located)
                // when cycle estimates are unavailable or the feature is disabled.
                IRBase* signalAnchor = findRule4SignalAnchorByCycleLead(
                    waitAnchorInst, segBegin, /*defaultAnchor=*/waitAnchor, cycleMap,
                    kRule4SignalLeadCycles);
                StinkyInstruction* signalAnchorInst =
                    (signalAnchor != nullptr) ? dyn_cast<StinkyInstruction>(signalAnchor) : nullptr;

                // SCC restore decision (all against the pre-mutation IR):
                //   * If any instruction in [signalAnchor, waitAnchor) already
                //     writes SCC, that write masks the signal block's WaveIdx
                //     `s_cmp_eq_u32` clobber before any downstream consumer sees
                //     it -- no restore needed.
                //   * Otherwise scan backward from the SIGNAL anchor for a live
                //     upstream LCL cmp whose SCC a downstream cbranch consumes; if
                //     found, Rule 4(a) re-emits it after the signal block.
                StinkyInstruction* sccRestoreCmp = nullptr;
                if (signalAnchorInst != nullptr &&
                    !anySccWriterInRange(signalAnchorInst, waitAnchor)) {
                    sccRestoreCmp = findLiveRestorableSccCmpUpstream(signalAnchorInst);
                }
                // Count any `s_sub LCL, LCL, imm` the schedule hoisted above
                // the trigger so the drain-gated mode (b) thresholds can be
                // compensated. (Unused when mode (c) is active.)
                const int lclPreDecrement =
                    sumLoopCounterLDecrementsBeforeInSegment(segBegin, trigger);
                pending.emplace_back(trigger, signalAnchor, waitAnchor, sccRestoreCmp,
                                     lclPreDecrement);
            }

            // Rule 1: signal-only handshake immediately AFTER each
            // `label_GSU_1:` label (emitted by Tensile after the GSU==1
            // early-return guard). The label is a `StinkyInstruction`
            // pseudo, so unlike a TEXTBLOCK it survives region extraction
            // - idempotency (`isFollowedByClusterBarrierHandshakeOrSignal`)
            // handles re-entry across scopes.
            //
            // The emitted sequence wraps the inner WaveIdx-gated cluster
            // signal in an outer `LoopCounterL == 0` skip-branch AND
            // plants a workgroup-scope sync (`s_barrier_signal -1` /
            // `s_barrier_wait -1`) between the two gates. The workgroup
            // pair guarantees every wave in the workgroup has reached
            // the post-GSU==1 join before the first wave publishes the
            // cluster signal, so a fast wave cannot race ahead while
            // its siblings are still doing per-wave teardown above the
            // label.
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

            // Rule 5 -- tail-loop cluster handshake (paired, kernel scope
            // effectively). Two emission sites because the workgroup wait
            // and the tail load are in different label/branch-delimited
            // segments; collapsing them would force the cluster to
            // serialize across the tail TDM-reset code that sits between.
            //   - 5a inserts a signal-only handshake (no LoopCounterL
            //     gate) immediately AFTER the nearest preceding
            //     `s_barrier_wait -1` of the tail load (searching
            //     backward from the load, stopping at the marker).
            //   - 5b inserts a bare `s_barrier_wait -3` immediately
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
            // Rule 5b idempotency (skip if already preceded by a cluster wait).
            if (tailTL != nullptr && isImmediatelyPrecededByClusterBarrierWait(tailTL)) {
                tailTL = nullptr;
            }
            // Rule 5a idempotency (skip if already followed by a cluster handshake)
            // and Rule-4 collision guard (defer to Rule 4 if it already targets
            // the same wait).
            if (tailWait != nullptr) {
                // Rule 4's triggers are workgroup SIGNALs; bridge this wait to its
                // paired signal so the identity comparison still detects overlap.
                StinkyInstruction* tailPairedSignal =
                    findPrecedingWorkgroupBarrierSignalInSegment(bb.begin(), tailWait);
                bool conflictsWithRule4 = false;
                for (const auto& [trigger, _sig, _wait, _live, _dec] : pending) {
                    if (tailPairedSignal != nullptr && trigger == tailPairedSignal) {
                        conflictsWithRule4 = true;
                        break;
                    }
                }
                if (conflictsWithRule4 || isFollowedByClusterBarrierHandshakeOrSignal(tailWait)) {
                    tailWait = nullptr;
                }
            }

            if (pending.empty() && gsu1Anchors.empty() && tailTL == nullptr &&
                tailWait == nullptr)
                continue;
            AsmIRBuilder irBuilder(bb, archId);
            for (const auto& [trigger, signalAnchor, waitAnchor, liveLclCmp, lclPreDecrement] :
                 pending) {
                // Rule 4 is split into two independently-anchored halves, both
                // resolved at scan time (see the `pending` tuple comment):
                //   - Rule 4(b) wait at `waitAnchor` (before the trigger workgroup
                //     signal).
                //   - Rule 4(a) signal at `signalAnchor` (cycle-lead backward from
                //     the wait anchor). It owns the SCC restore `liveLclCmp`,
                //     which is null unless an SCC-writer-free window between the
                //     two anchors left a live upstream LCL cmp exposed.
                insertClusterBarrierHandshakeBefore(signalAnchor, waitAnchor, irBuilder, archId,
                                                    pgrValue_, liveLclCmp, lclPreDecrement);
                (void)trigger;  // the workgroup signal; used by the conflict guards above
            }
            for (IRBase* anchor : gsu1Anchors) {
                insertLoopCounterLGatedClusterBarrierSignalBefore(
                    anchor, irBuilder, archId,
                    /*cmpUOp=*/GFX::s_cmp_eq_u32,
                    /*skipWhenScc1Imm=*/0,
                    /*cmpComment=*/"gate: only signal when LoopCounterL != 0",
                    /*branchComment=*/"skip cluster barrier when LoopCounterL == 0",
                    /*workgroupSyncWaitComment=*/"sync workgroup before cluster signal");
            }
            // Rule 5a -- signal-only after the tail loop's preceding workgroup wait.
            if (tailWait != nullptr) {
                IRBase* anchor =
                    (tailWaitNextIt != bb.end()) ? tailWaitNextIt.getNodePtr() : nullptr;
                insertClusterBarrierSignalOnlyBefore(anchor, irBuilder, archId);
            }
            // Rule 5b -- bare cluster wait immediately before the tail load.
            if (tailTL != nullptr) {
                insertClusterBarrierWaitBefore(tailTL, "cluster barrier wait", irBuilder, archId);
            }
        }

        // Rule 2: a single `s_barrier_wait -3` planted immediately before the
        // first `tensor_load_to_lds` of the whole function. Idempotency: skip
        // when the load is already gated by a cluster-scope wait (whether ours,
        // Rule 4's, or one already present in the source IR).
        {
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
    const int pgrValue_;
};

char InsertClusterBarrierPassImpl::ID = 0;

}  // namespace

std::unique_ptr<Pass> createInsertClusterBarrierPass(int pgrValue) {
    return std::make_unique<InsertClusterBarrierPassImpl>(pgrValue);
}

}  // namespace stinkytofu
