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
constexpr const char* kSkipLabelPrefixLCL = "label_skipCBPreSignal_LCL_";
constexpr const char* kDrainBypassLabelSuffix = "_skipCBWait";
constexpr const char* kWaveIdxSymbol = "sgprWaveIdx";
constexpr const char* kLoopCounterLSymbol = "sgprLoopCounterL";
constexpr size_t kHashLen = 16;
constexpr const char* kGSU1LabelName = "label_GSU_1";
constexpr const char* kOpenLoopLabelName = "label_openLoopL";
constexpr const char* kTailLoopMarker = "Tail Loop";

/// Estimated cycles the Rule 3 signal is planted ahead of its paired wait.
/// Set to 0 to co-locate the signal with the wait.
constexpr int kRule3SignalLeadCycles = 500;

/// Minimum Part-A cycles (wait→loop head) before the opening segment may wrap to the latch
/// for the remaining lead. Avoids hoisting when the head climb is too short to justify a
/// loop-carried signal.
constexpr int kMinHeadAccumForLoopWrap = kRule3SignalLeadCycles / 4;

/// Ceiling on how far ahead of its wait the signal may end up after climbing out of a live
/// SCC range. The climb has to clear the whole range, so its cost is the length of that
/// range, not a constant; past this ceiling it buys correctness at more overlap than it is
/// worth. The anchor then drops below the range instead, which lands it closer than
/// kRule3SignalLeadCycles rather than further away.
constexpr int kRule3SignalMaxLeadCycles = 900;

/// Segment edges one handshake may climb across when its own segment is too short to
/// hold kRule3SignalLeadCycles. One hop reaches the previous segment of a pipelined loop
/// body, which leaves exactly one signal in flight and so costs one compensating pair
/// around the loop; more hops would need per-edge accounting for no extra overlap.
constexpr int kMaxSegmentHops = 1;

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

StinkyRegister makeSymbolicSgpr(const std::string& symbolicName) {
    StinkyRegister reg(RegType::S, /*regIdx=*/0u, /*regNum=*/1u);
    reg.setSymbolicName(symbolicName);
    return reg;
}

bool isBarrierWithLiteralId(const StinkyInstruction& inst, bool wantSignal, int id,
                            bool rejectMemToken) {
    if (wantSignal ? !isBarrierSignal(inst) : !isBarrierWait(inst)) return false;
    if (rejectMemToken && inst.getModifier<MemTokenData>() != nullptr) return false;
    const auto& srcs = inst.getSrcRegs();
    return !srcs.empty() && srcs[0].dataType == StinkyRegister::Type::LiteralInt &&
           srcs[0].getLiteralInt() == id;
}

bool isWorkgroupBarrierWait(const StinkyInstruction& inst) {
    return isBarrierWithLiteralId(inst, /*wantSignal=*/false, kWorkgroupBarrierId,
                                  /*rejectMemToken=*/false);
}

bool isWorkgroupBarrierSignal(const StinkyInstruction& inst) {
    return isBarrierWithLiteralId(inst, /*wantSignal=*/true, kWorkgroupBarrierId,
                                  /*rejectMemToken=*/false);
}

bool isClusterBarrierWait(const StinkyInstruction& inst) {
    return isBarrierWithLiteralId(inst, /*wantSignal=*/false, kClusterBarrierId,
                                  /*rejectMemToken=*/true);
}

bool isClusterBarrierSignal(const StinkyInstruction& inst) {
    return isBarrierWithLiteralId(inst, /*wantSignal=*/true, kClusterBarrierId,
                                  /*rejectMemToken=*/false);
}

bool isSegmentBoundary(const StinkyInstruction& inst) {
    return isLabel(inst) || isBranch(inst) || isCall(inst);
}

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

// SCC is written and read both through the descriptor flags and, once the DAG scheduler
// has made the dependency explicit, as an ordinary operand. Check for both.
bool writesScc(const StinkyInstruction& inst) {
    if (inst.is(InstFlag::IF_ImplicitWriteSCC)) return true;
    for (const auto& dst : inst.getDestRegs()) {
        if (dst.isRegister() && dst.reg.type == RegType::SCC) return true;
    }
    return false;
}

bool readsScc(const StinkyInstruction& inst) {
    if (inst.is(InstFlag::IF_ImplicitReadSCC)) return true;
    for (const auto& src : inst.getSrcRegs()) {
        if (src.isRegister() && src.reg.type == RegType::SCC) return true;
    }
    return false;
}

/// Is an SCC value live at the program point immediately in front of \p from, i.e. does
/// something from there on read SCC before anything rewrites it? The backward anchor scan
/// starts at that point, so it has to know what it is already standing in.
///
/// A value read only by a successor block is not detected; in the loop bodies this pass
/// runs on the consuming ``s_cbranch_scc*`` closes the block, so the forward scan sees it.
bool isSccLiveBefore(StinkyInstruction* from) {
    BasicBlock* parent = from->getParent();
    if (parent == nullptr) return false;
    for (auto it = BasicBlock::iterator(from); it != parent->end(); ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (readsScc(*inst)) return true;
        if (writesScc(*inst)) return false;
    }
    return false;
}

/// Walk down from \p from for the first instruction with nothing live in SCC in front of
/// it, i.e. the first spot below a live range that the handshake may clobber. Stops at
/// \p limit (the wait anchor) and returns null when the range stays live that far, since
/// there is then nowhere below it to go.
///
/// A range is held open by its last reader, and in a loop body that reader is the exit
/// branch that closes the segment -- so the spot being looked for is often the first
/// instruction on the branch's fall-through side. That one branch is therefore stepped over
/// rather than treated as a wall; every other boundary still ends the walk, and so does this
/// one when SCC is not what it reads. Stepping over it is safe because the walk only ever
/// moves towards \p limit, which is the wait: taking the fall-through keeps the signal on the
/// path its wait is on.
StinkyInstruction* findSccDeadPointBelow(StinkyInstruction* from, const IRBase* limit) {
    BasicBlock* parent = from->getParent();
    if (parent == nullptr) return nullptr;
    for (auto it = BasicBlock::iterator(from); it != parent->end(); ++it) {
        if (it.getNodePtr() == limit) return nullptr;
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (isSegmentBoundary(*inst)) {
            if (readsScc(*inst) && isConditionalBranch(*inst)) continue;
            return nullptr;
        }
        if (isClusterBarrierWait(*inst)) return nullptr;
        if (isWorkgroupBarrierSignal(*inst) || isWorkgroupBarrierWait(*inst)) continue;
        if (!isSccLiveBefore(inst)) return inst;
    }
    return nullptr;
}

StinkyInstruction* findFirstTensorLoadBetween(BasicBlock::iterator start,
                                              BasicBlock::iterator endExclusive) {
    for (auto it = start; it != endExclusive; ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (isTensorLoad(*inst)) return inst;
    }
    return nullptr;
}

/// Whether a cluster wait already stands anywhere above \p anchor in its block.
bool hasClusterBarrierWaitAbove(StinkyInstruction* anchor) {
    BasicBlock* parent = anchor->getParent();
    if (parent == nullptr) return false;
    auto it = BasicBlock::iterator(anchor);
    while (it != parent->begin()) {
        --it;
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (isClusterBarrierWait(*inst)) return true;
    }
    return false;
}

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

void insertWorkgroupBarrierSyncBefore(IRBase* anchor, AsmIRBuilder& irBuilder, GfxArchID archId) {
    const HwInstDesc* signalDesc = getMCIDByUOp(GFX::s_barrier_signal, archId);
    const HwInstDesc* waitDesc = getMCIDByUOp(GFX::s_barrier_wait, archId);
    assert(signalDesc && waitDesc &&
           "Workgroup-barrier opcodes are not supported on this architecture");

    StinkyInstruction* signalInst = irBuilder.create(signalDesc, anchor);
    signalInst->addSrcReg(StinkyRegister(kWorkgroupBarrierId));

    StinkyInstruction* waitInst = irBuilder.create(waitDesc, anchor);
    waitInst->addSrcReg(StinkyRegister(kWorkgroupBarrierId));
    waitInst->addModifier<CommentData>(CommentData{"sync workgroup before cluster signal"});
}

void insertRule1ClusterBarrierSignalBefore(IRBase* anchor, AsmIRBuilder& irBuilder,
                                           GfxArchID archId) {
    const std::string lclLabelName = std::string(kSkipLabelPrefixLCL) + makeRandomHash();

    const HwInstDesc* cmpDesc = getMCIDByUOp(GFX::s_cmp_eq_u32, archId);
    const HwInstDesc* brDesc = getMCIDByUOp(GFX::s_cbranch_scc1, archId);
    assert(cmpDesc && brDesc && "LoopCounterL gate opcodes are not supported on this architecture");

    StinkyInstruction* cmpInst = irBuilder.create(cmpDesc, anchor);
    cmpInst->addSrcReg(makeSymbolicSgpr(kLoopCounterLSymbol));
    cmpInst->addSrcReg(StinkyRegister(0));
    cmpInst->addModifier<CommentData>(CommentData{"gate: only signal when LoopCounterL != 0"});

    StinkyInstruction* brInst = irBuilder.create(brDesc, anchor);
    brInst->addSrcReg(StinkyRegister(lclLabelName));
    brInst->addModifier<LabelData>(LabelData{lclLabelName});
    brInst->addModifier<CommentData>(CommentData{"skip cluster barrier when LoopCounterL == 0"});

    insertWorkgroupBarrierSyncBefore(anchor, irBuilder, archId);
    insertClusterBarrierSignalOnlyBefore(anchor, irBuilder, archId);

    static const HwInstDesc labelMCID{
        GFX::LABEL, GFX::LABEL, 0, 0, 0, "LABEL", makeFlagSet({InstFlag::IF_HasSideEffect})};
    StinkyInstruction* lclLblInst = irBuilder.create(&labelMCID, anchor);
    lclLblInst->addModifier<LabelData>(LabelData{lclLabelName, /*alignment=*/1});
}

void insertClusterBarrierWaitBefore(IRBase* anchor, const char* comment, AsmIRBuilder& irBuilder,
                                    GfxArchID archId) {
    const HwInstDesc* waitDesc = getMCIDByUOp(GFX::s_barrier_wait, archId);
    assert(waitDesc && "Cluster-barrier wait opcode is not supported on this architecture");
    StinkyInstruction* waitInst = irBuilder.create(waitDesc, anchor);
    waitInst->addSrcReg(StinkyRegister(kClusterBarrierId));
    waitInst->addModifier<CommentData>(CommentData{comment});
}

/// WaveIdx-gated cluster signal at \p signalAnchor, then wait at \p waitAnchor (the trigger).
void insertRule3HandshakeBefore(IRBase* signalAnchor, IRBase* waitAnchor, AsmIRBuilder& irBuilder,
                                GfxArchID archId) {
    insertClusterBarrierSignalOnlyBefore(signalAnchor, irBuilder, archId);
    insertClusterBarrierWaitBefore(waitAnchor, "cluster barrier wait", irBuilder, archId);
}

/// Emit `s_wait_tensorcnt 0` immediately before \p anchor (the instruction
/// right after a cooperative `tensor_load_to_lds` group). Under PGR>=2 the
/// cooperative load is async and produced by a PEER wave, so the consumer's own
/// tensor counter cannot order it. Draining right after the load issues makes
/// the broadcast coherent before the back edge, so the publishing workgroup
/// barrier at the next loop head orders it for the consuming waves. Matches
/// PGR1's per-iteration drain.
void insertProducerTensorDrainBefore(IRBase* anchor, AsmIRBuilder& irBuilder, GfxArchID archId) {
    const HwInstDesc* waitDesc = getMCIDByUOp(GFX::s_wait_tensorcnt, archId);
    assert(waitDesc && "s_wait_tensorcnt opcode is not supported on this architecture");
    StinkyInstruction* w = irBuilder.create(waitDesc, anchor);
    w->addSrcReg(StinkyRegister(0));
    SWaitTensorCntData d;
    d.tlcnt = 0;
    w->addModifier<SWaitTensorCntData>(d);
    w->addModifier<CommentData>(
        CommentData{"retire cooperative tensor_load_to_lds before back-edge (PGR>=2 coherence)"});
}

bool isLabelNamed(const StinkyInstruction& inst, const char* name) {
    if (!isLabel(inst)) return false;
    const auto* labelData = inst.getModifier<LabelData>();
    return labelData != nullptr && labelData->label == name;
}

bool isKernelLabelForPreLoopClimb(const StinkyInstruction& inst) {
    if (!isLabel(inst)) return false;
    const auto* labelData = inst.getModifier<LabelData>();
    if (labelData == nullptr) return true;
    return labelData->label.find("skipCB") == std::string::npos;
}

bool isTextblockContaining(IRBase* ir, const char* marker) {
    auto* directive = dyn_cast<AsmDirective>(ir);
    return directive != nullptr && directive->kind == AsmDirectiveKind::TEXTBLOCK &&
           directive->value.find(marker) != std::string::npos;
}

bool isFollowedByClusterBarrierHandshakeOrSignal(StinkyInstruction* anchor) {
    StinkyInstruction* next = firstRealInstAfter(anchor);
    if (next == nullptr) return false;
    if (isClusterBarrierWait(*next)) return true;
    if (next->getUnifiedOpcode() != GFX::s_cmp_eq_u32) return false;
    const auto& srcs = next->getSrcRegs();
    if (srcs.empty()) return false;
    const std::string& sym = srcs[0].getSymbolicName();
    return sym == kWaveIdxSymbol || sym == kLoopCounterLSymbol;
}

/// Forward scan from ``wgSignal`` for its paired ``s_barrier_wait -1`` and return
/// the first real instruction after that wait.
IRBase* anchorAfterWorkgroupBarrierPair(StinkyInstruction* wgSignal, IRBase* defaultAnchor) {
    BasicBlock* parent = wgSignal->getParent();
    if (parent == nullptr) return defaultAnchor;
    for (auto it = std::next(BasicBlock::iterator(wgSignal)); it != parent->end(); ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr || isPseudoInst(inst)) continue;
        if (isWorkgroupBarrierSignal(*inst)) break;
        if (!isWorkgroupBarrierWait(*inst)) continue;
        for (auto after = std::next(it); after != parent->end(); ++after) {
            auto* next = dyn_cast<StinkyInstruction>(after.getNodePtr());
            if (next == nullptr || isPseudoInst(next)) continue;
            return next;
        }
        break;
    }
    return defaultAnchor;
}

/// Forward scan from ``afterWait`` for the next workgroup barrier handshake
/// (signal then wait).  Returns the first real instruction after that wait, or
/// ``defaultAnchor`` when the pair is missing or incomplete.
IRBase* anchorAfterWorkgroupBarrierFollowing(StinkyInstruction* afterWait, IRBase* defaultAnchor) {
    BasicBlock* parent = afterWait->getParent();
    if (parent == nullptr) return defaultAnchor;
    for (auto it = std::next(BasicBlock::iterator(afterWait)); it != parent->end(); ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr || isPseudoInst(inst)) continue;
        if (!isWorkgroupBarrierSignal(*inst)) continue;
        return anchorAfterWorkgroupBarrierPair(inst, defaultAnchor);
    }
    return defaultAnchor;
}

/// Start of the segment holding \p pos, i.e. just past the closest boundary above it.
BasicBlock::iterator segmentBeginBefore(BasicBlock::iterator pos, BasicBlock::iterator bbBegin) {
    auto it = pos;
    while (it != bbBegin) {
        --it;
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst != nullptr && isSegmentBoundary(*inst)) return std::next(it);
    }
    return bbBegin;
}

/// First segment boundary at or below \p pos, which closes the segment holding it.
/// Used as the forward limit for SCC queries about an anchor that climbed into an
/// earlier segment, where the wait anchor sits above rather than below.
const IRBase* segmentEndAfter(BasicBlock::iterator pos, BasicBlock::iterator bbEnd) {
    for (auto it = pos; it != bbEnd; ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst != nullptr && isSegmentBoundary(*inst)) return it.getNodePtr();
    }
    return nullptr;
}

/// The branch that closes a loop on \p labelInst: a branch back up to it from below.
/// A backward walk leaving a loop head has to follow this edge rather than the textual
/// one, because the textual predecessor is the preheader and runs only on the first trip.
StinkyInstruction* findLatchBranchFor(StinkyInstruction* labelInst) {
    BasicBlock* parent = labelInst->getParent();
    const auto* labelData = labelInst->getModifier<LabelData>();
    if (parent == nullptr || labelData == nullptr) return nullptr;
    for (auto it = std::next(BasicBlock::iterator(labelInst)); it != parent->end(); ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr || !isBranch(*inst)) continue;
        if (getBranchTarget(*inst) == labelData->label) return inst;
    }
    return nullptr;
}

/// Innermost loop head enclosing \p inst: a label whose latch branch sits below \p inst.
/// Hoisting is confined to such a region, because the compensating pair this pass emits
/// only balances a signal that a loop carries from one trip into the next.
StinkyInstruction* findEnclosingLoopHead(StinkyInstruction* inst) {
    BasicBlock* parent = inst->getParent();
    if (parent == nullptr) return nullptr;
    auto it = BasicBlock::iterator(inst);
    while (it != parent->begin()) {
        --it;
        auto* cand = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (cand == nullptr || !isLabel(*cand)) continue;
        StinkyInstruction* latch = findLatchBranchFor(cand);
        if (latch == nullptr) continue;
        for (auto fwd = BasicBlock::iterator(inst); fwd != parent->end(); ++fwd) {
            if (fwd.getNodePtr() == latch) return cand;
        }
    }
    return nullptr;
}

/// Whether \p inst is the compare that opens a loop's skip shortcut, i.e.
/// ``s_cmp_eq_u32 sgprLoopCounterL, <imm>``.
bool isLoopCounterEqCompare(const StinkyInstruction& inst) {
    if (inst.getUnifiedOpcode() != GFX::s_cmp_eq_u32) return false;
    const auto& srcs = inst.getSrcRegs();
    return !srcs.empty() && srcs[0].getSymbolicName() == kLoopCounterLSymbol;
}

/// Where the paths rejoin below a loop that is left by falling off its latch.
///
/// Such a loop has no escape branch to read a label off, and the label sitting just under the
/// latch is not the answer either: the run-up carries a shortcut for the trip counts that
/// never enter the loop at all, and it jumps clean over that label. What every path does
/// reach is the shortcut's own target.
///
/// The shortcut is the first ``s_cmp_eq_u32 sgprLoopCounterL`` below ``label_openLoopL``
/// together with the branch that reads it. Its target speaks for the whole loop only if two
/// things hold, and both are checked here: it lies below the latch, so it names a spot
/// outside the loop rather than one within it; and nothing between the latch and it hands
/// control elsewhere, so whatever falls off the latch arrives there as well.
std::string findLoopSkipShortcutLabel(StinkyInstruction* loopHead, StinkyInstruction* latch) {
    BasicBlock* parent = loopHead->getParent();
    if (parent == nullptr || latch == nullptr) return {};

    StinkyInstruction* openLabel = nullptr;
    auto up = BasicBlock::iterator(loopHead);
    while (up != parent->begin()) {
        --up;
        auto* inst = dyn_cast<StinkyInstruction>(up.getNodePtr());
        if (inst != nullptr && isLabelNamed(*inst, kOpenLoopLabelName)) {
            openLabel = inst;
            break;
        }
    }
    if (openLabel == nullptr) return {};

    std::string target;
    bool sawCompare = false;
    for (auto it = std::next(BasicBlock::iterator(openLabel));
         it != parent->end() && it.getNodePtr() != loopHead; ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (!sawCompare) {
            sawCompare = isLoopCounterEqCompare(*inst);
            continue;
        }
        // Whatever drinks what the compare wrote, and nothing past it: a shortcut is that
        // compare and its own branch, so anything else reading SCC first means there is none.
        if (!readsScc(*inst)) continue;
        if (isBranch(*inst)) target = getBranchTarget(*inst);
        break;
    }
    if (target.empty()) return {};

    for (auto it = std::next(BasicBlock::iterator(latch)); it != parent->end(); ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (isLabelNamed(*inst, target.c_str())) return target;
        if (isUnconditionalBranch(*inst) || isCall(*inst)) break;
    }
    return {};
}

/// Label the loop's escape branches jump to, i.e. where control lands when the body is
/// abandoned. Read before this pass inserts anything, so the only branches between the head
/// and the latch are the loop's own exits.
///
/// A body that holds no such branch is not a loop without a way out; it is one whose way out
/// is spelled by nothing at all, and the skip shortcut above the head is what names it.
std::string findLoopExitLabelName(StinkyInstruction* loopHead) {
    BasicBlock* parent = loopHead->getParent();
    StinkyInstruction* latch = findLatchBranchFor(loopHead);
    const auto* headData = loopHead->getModifier<LabelData>();
    if (parent == nullptr || latch == nullptr || headData == nullptr) return {};
    for (auto it = BasicBlock::iterator(loopHead); it != parent->end(); ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (isBranch(*inst) && inst != latch) {
            const std::string target = getBranchTarget(*inst);
            if (!target.empty() && target != headData->label) return target;
        }
        if (inst == latch) break;
    }
    return findLoopSkipShortcutLabel(loopHead, latch);
}

struct Rule3SignalAnchor {
    IRBase* anchor = nullptr;
    /// Segment edges crossed to reach `anchor`. Non-zero means the signal no longer shares a
    /// segment with its wait, so some edge out of the loop now carries a token.
    int hops = 0;
    /// True when one of those edges was the back edge. The signal then feeds the *next*
    /// trip's wait, which leaves the first trip with nobody to feed it.
    bool crossedLoopHead = false;
};

/// True when \p segBegin is the opening segment of \p loopHead.
bool isFirstLoopSegment(BasicBlock::iterator segBegin, StinkyInstruction* loopHead) {
    if (loopHead == nullptr) return false;
    auto headIt = BasicBlock::iterator(loopHead);
    return segBegin == headIt || segBegin == std::next(headIt);
}

/// Walk backward from the wait for cycle lead. \p maxHops 0 = in-segment only
/// (kRule3CrossLoop false); 1 = one segment hop allowed (kRule3CrossLoop true).
Rule3SignalAnchor findRule3SignalAnchorByCycleLead(
    StinkyInstruction* referenceAnchor, BasicBlock::iterator segBegin, IRBase* defaultAnchor,
    const std::unordered_map<const StinkyInstruction*, uint32_t>& cycleMap, int leadCycles,
    int maxLeadCycles, const std::unordered_set<StinkyInstruction*>& priorWaitAnchors, int maxHops,
    StinkyInstruction* loopHead) {
    if (leadCycles <= 0) return {defaultAnchor, 0};
    auto refIt = cycleMap.find(referenceAnchor);
    if (refIt == cycleMap.end()) return {defaultAnchor, 0};
    BasicBlock* parent = referenceAnchor->getParent();
    if (parent == nullptr) return {defaultAnchor, 0};
    const auto bbBegin = parent->begin();

    // The lead is accumulated from per-instruction costs rather than compared against an
    // absolute cycle position, because a hop across a back edge lands in a segment that is
    // textually below the wait and so carries larger absolute cycles.
    int64_t accum = 0;
    int64_t prevCycle = static_cast<int64_t>(refIt->second);
    int hops = 0;
    bool crossedLoopHead = false;
    auto curSegBegin = segBegin;

    // The handshake opens with `s_cmp_eq_u32 sgprWaveIdx, 0` and is planted in front of
    // whatever this scan returns, so the anchor may only land where SCC holds nothing
    // live. `sccLive` is maintained backwards to describe the point in front of the
    // instruction being looked at, and once the cycle lead is met the scan keeps climbing
    // until that point is clear -- which, for a lead that falls inside a def..reader
    // range, means coming to rest in front of the def. The boundary returns below still
    // win: a cluster wait or a prior handshake's barrier cannot be crossed just to find a
    // better spot, and neither can a segment edge once the hop budget is spent.
    bool sccLive = isSccLiveBefore(referenceAnchor);
    bool targetMet = false;
    StinkyInstruction* leadPoint = nullptr;

    // SCC queries about the anchor scan forward towards the wait, so the wait bounds them --
    // an anchor may not be corrected past the very spot it is leading. Only the back edge
    // puts the anchor textually below its wait; the segment's closing boundary takes over as
    // the limit there, since the wait is no longer ahead of the anchor to be found.
    auto sccLimit = [&](StinkyInstruction* anchorInst) -> const IRBase* {
        if (!crossedLoopHead) return referenceAnchor;
        return segmentEndAfter(BasicBlock::iterator(anchorInst), parent->end());
    };

    // A climb that ends up further than maxLeadCycles from the wait has cleared a range too
    // long to be worth it. Fall the other way instead: down from the lead point to the first
    // spot below the range, which is nearer the wait than the lead asked for.
    // A boundary-forced anchor is a lower bound -- the scan may not go above it -- so when
    // it lands inside a live range the only legal correction is to drop below the range.
    // Failing that the whole segment from the def down to the wait is live and there is no
    // safe spot at all; the caller's default (co-locating with the wait) is then no worse
    // than anything else this pass could pick.
    auto clearScc = [&](IRBase* anchor) -> IRBase* {
        if (anchor == defaultAnchor) return anchor;
        auto* anchorInst = dyn_cast<StinkyInstruction>(anchor);
        if (anchorInst == nullptr || !isSccLiveBefore(anchorInst)) return anchor;
        StinkyInstruction* below = findSccDeadPointBelow(anchorInst, sccLimit(anchorInst));
        return (below != nullptr) ? static_cast<IRBase*>(below) : defaultAnchor;
    };

    auto settle = [&](IRBase* climbed, int64_t totalAccum) -> IRBase* {
        if (leadPoint == nullptr || climbed == leadPoint) return climbed;
        if (totalAccum <= maxLeadCycles) return climbed;
        StinkyInstruction* below = findSccDeadPointBelow(leadPoint, sccLimit(leadPoint));
        return (below != nullptr) ? static_cast<IRBase*>(below) : defaultAnchor;
    };

    // Whether the anchor came to rest in the wait's own segment after all. Both corrections
    // above walk back down, and the boundary they step over on the way is one the climb had
    // already counted.
    auto inWaitSegment = [&](IRBase* anchor) -> bool {
        for (auto it = segBegin; it != BasicBlock::iterator(referenceAnchor); ++it) {
            if (it.getNodePtr() == anchor) return true;
        }
        return false;
    };

    // The hop count is what buys the loop its compensation, so it has to describe the anchor
    // that comes back rather than the climb that looked for it. A scan that climbed over an
    // edge and then dropped back below it crossed nothing in the end and must not be billed
    // for it -- neither when it gave up at the caller's default, which is the wait's own
    // spot, nor when it settled anywhere else the wait can reach without a branch.
    auto report = [&](IRBase* anchor) -> Rule3SignalAnchor {
        if (anchor == defaultAnchor || inWaitSegment(anchor)) return {anchor, 0, false};
        return {anchor, hops, crossedLoopHead};
    };

    // kRule3CrossLoop true only: wait→head; part B latch→anchor for the remainder.
    auto scanTailForRemainder = [&](StinkyInstruction* latch, int64_t headAccum,
                                    int64_t remainder) -> Rule3SignalAnchor {
        crossedLoopHead = true;
        hops = 1;

        int64_t tailAccum = 0;
        auto latchCycleIt = cycleMap.find(latch);
        int64_t tailPrevCycle = (latchCycleIt != cycleMap.end())
                                    ? static_cast<int64_t>(latchCycleIt->second)
                                    : prevCycle;
        bool tailSccLive = readsScc(*latch) || (sccLive && !writesScc(*latch));
        bool tailTargetMet = false;
        StinkyInstruction* tailLeadPoint = nullptr;

        auto tailIt = BasicBlock::iterator(latch);
        while (tailIt != bbBegin) {
            --tailIt;
            auto* tailInst = dyn_cast<StinkyInstruction>(tailIt.getNodePtr());
            if (tailInst == nullptr) continue;

            tailSccLive = readsScc(*tailInst) || (tailSccLive && !writesScc(*tailInst));

            if (isClusterBarrierWait(*tailInst)) {
                return report(
                    clearScc(anchorAfterWorkgroupBarrierFollowing(tailInst, defaultAnchor)));
            }
            if (isSegmentBoundary(*tailInst)) {
                if (isCall(*tailInst) || isUnconditionalBranch(*tailInst)) {
                    return report(clearScc(defaultAnchor));
                }
                if (loopHead != nullptr && tailInst == loopHead) {
                    return report(clearScc(loopHead));
                }
                continue;
            }
            if (isWorkgroupBarrierSignal(*tailInst) && priorWaitAnchors.count(tailInst) != 0) {
                return report(clearScc(anchorAfterWorkgroupBarrierPair(tailInst, defaultAnchor)));
            }
            if (isWorkgroupBarrierSignal(*tailInst) || isWorkgroupBarrierWait(*tailInst)) continue;

            auto tailCycleIt = cycleMap.find(tailInst);
            if (tailCycleIt != cycleMap.end()) {
                const int64_t cyc = static_cast<int64_t>(tailCycleIt->second);
                if (cyc <= tailPrevCycle) tailAccum += tailPrevCycle - cyc;
                tailPrevCycle = cyc;
                if (tailAccum >= remainder) {
                    if (!tailTargetMet) tailLeadPoint = tailInst;
                    tailTargetMet = true;
                }
            }
            leadPoint = tailLeadPoint;
            if (tailTargetMet && !tailSccLive) {
                return report(clearScc(settle(tailInst, headAccum + tailAccum)));
            }
        }
        leadPoint = tailLeadPoint;
        IRBase* fallback = (loopHead != nullptr) ? static_cast<IRBase*>(loopHead) : defaultAnchor;
        return report(clearScc(settle(fallback, headAccum + tailAccum)));
    };

    auto it = BasicBlock::iterator(referenceAnchor);
    while (it != bbBegin) {
        --it;
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;

        // live-before(inst) = reads(inst) | (live-after(inst) & !writes(inst)), for every
        // instruction the climb steps over -- boundaries included. What holds a loop-exit
        // predicate open is the branch that closes the segment, so a climb that took the
        // boundaries as read would walk straight into the range it has to stay out of.
        sccLive = readsScc(*inst) || (sccLive && !writesScc(*inst));

        if (isClusterBarrierWait(*inst)) {
            return report(clearScc(anchorAfterWorkgroupBarrierFollowing(inst, defaultAnchor)));
        }
        if (isSegmentBoundary(*inst)) {
            if (loopHead != nullptr && inst == loopHead) {
                if (targetMet && !sccLive) return report(clearScc(settle(inst, accum)));
                // kRule3CrossLoop true only: latch wrap when head climb falls short.
                if (accum >= kMinHeadAccumForLoopWrap && accum < leadCycles && maxHops > 0 &&
                    hops == 0 && isFirstLoopSegment(segBegin, loopHead)) {
                    StinkyInstruction* latch = findLatchBranchFor(inst);
                    if (latch == nullptr) return report(clearScc(curSegBegin.getNodePtr()));
                    return scanTailForRemainder(latch, accum, leadCycles - accum);
                }
                if (accum >= leadCycles) continue;
            }
            // Stops at segment boundary once maxHops is exhausted (0 when kRule3CrossLoop false).
            if (hops >= maxHops || isCall(*inst) || isUnconditionalBranch(*inst)) {
                return report(clearScc(curSegBegin.getNodePtr()));
            }
            if (isLabel(*inst)) {
                // Leaving a loop head textually would land in the preheader, which runs
                // once; follow the latch so the signal lands on the path that repeats.
                StinkyInstruction* latch = findLatchBranchFor(inst);
                if (latch == nullptr) return report(clearScc(curSegBegin.getNodePtr()));
                it = BasicBlock::iterator(latch);
                // The latch is landed on rather than stepped over, so its own read of the
                // loop condition has to be folded in by hand.
                sccLive = readsScc(*latch) || (sccLive && !writesScc(*latch));
                auto latchCycle = cycleMap.find(latch);
                if (latchCycle != cycleMap.end())
                    prevCycle = static_cast<int64_t>(latchCycle->second);
                crossedLoopHead = true;
            }
            ++hops;
            curSegBegin = segmentBeginBefore(it, bbBegin);
            continue;
        }
        if (isWorkgroupBarrierSignal(*inst) && priorWaitAnchors.count(inst) != 0) {
            return report(clearScc(anchorAfterWorkgroupBarrierPair(inst, defaultAnchor)));
        }

        if (isWorkgroupBarrierSignal(*inst) || isWorkgroupBarrierWait(*inst)) continue;

        auto cycleIt = cycleMap.find(inst);
        if (cycleIt != cycleMap.end()) {
            const int64_t cyc = static_cast<int64_t>(cycleIt->second);
            if (cyc <= prevCycle) accum += prevCycle - cyc;
            prevCycle = cyc;
            if (accum >= leadCycles) {
                if (!targetMet) leadPoint = inst;
                targetMet = true;
            }
        }
        // clearScc has the last word even here. `sccLive` is carried along the one path the
        // climb took, while clearScc reads the range off the code below the anchor, so it
        // also covers a reader the climb never walked past.
        if (targetMet && !sccLive) return report(clearScc(settle(inst, accum)));
    }
    // Running out of block can leave the anchor inside a range that starts above it.
    return report(clearScc(settle(curSegBegin.getNodePtr(), accum)));
}

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

struct PreLoopSignalAnchor {
    IRBase* anchor = nullptr;
    /// True when the climb found no workgroup barrier to sit behind. Only wave 0 issues the
    /// signal, so one has to be planted to hold the rest of the group until it does.
    bool needsWorkgroupBarrier = true;
};

/// First real instruction below \p behind, or \p limit when nothing but pseudo instructions
/// stands between the two. Places the signal as close behind \p behind as it can get without
/// dropping into the loop.
IRBase* anchorJustBelow(StinkyInstruction* behind, StinkyInstruction* limit) {
    BasicBlock* parent = behind->getParent();
    if (parent == nullptr) return limit;
    for (auto it = std::next(BasicBlock::iterator(behind)); it != parent->end(); ++it) {
        // Node by node rather than instruction by instruction: \p limit is a label, and a
        // scan that skipped over labels would never see it.
        if (it.getNodePtr() == limit) break;
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr || isPseudoInst(inst)) continue;
        return inst;
    }
    return limit;
}

/// Where the compensating signal goes in the preheader (`kRule3CrossLoop` only).
///
/// Climb from the loop head upward and take the nearest of ``s_barrier_wait -1``,
/// ``label_*``, or ``tensor_load_to_lds``. Signal-only after an existing workgroup wait;
/// plant a workgroup barrier pair before the signal after a label or load.
PreLoopSignalAnchor findPreLoopSignalAnchor(StinkyInstruction* loopHead) {
    BasicBlock* parent = (loopHead != nullptr) ? loopHead->getParent() : nullptr;
    if (parent == nullptr) return {};

    auto it = BasicBlock::iterator(loopHead);
    while (it != parent->begin()) {
        --it;
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;

        StinkyInstruction* behind = nullptr;
        bool needsWorkgroupBarrier = true;
        if (!isPseudoInst(inst) && isWorkgroupBarrierWait(*inst)) {
            behind = inst;
            needsWorkgroupBarrier = false;
        } else if (isKernelLabelForPreLoopClimb(*inst)) {
            behind = inst;
        } else if (!isPseudoInst(inst) && isTensorLoad(*inst)) {
            behind = inst;
        } else {
            continue;
        }

        IRBase* anchor = anchorJustBelow(behind, loopHead);
        if (anchor == nullptr) anchor = loopHead;

        auto* anchorInst = dyn_cast<StinkyInstruction>(anchor);
        if (anchorInst == nullptr || !isSccLiveBefore(anchorInst)) {
            return {anchor, needsWorkgroupBarrier};
        }
        StinkyInstruction* dead = findSccDeadPointBelow(anchorInst, loopHead);
        if (dead == nullptr) return {};
        return {dead, needsWorkgroupBarrier};
    }
    return {};
}

/// Plant `s_branch <label>` in front of \p anchor.
void insertBranchBefore(IRBase* anchor, const std::string& label, AsmIRBuilder& irBuilder,
                        GfxArchID archId) {
    const HwInstDesc* brDesc = getMCIDByUOp(GFX::s_branch, archId);
    assert(brDesc && "Unconditional branch opcode is not supported on this architecture");
    StinkyInstruction* brInst = irBuilder.create(brDesc, anchor);
    brInst->addSrcReg(StinkyRegister(label));
    brInst->addModifier<LabelData>(LabelData{label});
    brInst->addModifier<CommentData>(CommentData{"nothing in flight: skip the drain wait"});
}

/// Point \p branch at \p newLabel. The target is spelled twice -- as the modifier the
/// printer reads and as the literal operand -- so both have to move.
void retargetBranch(StinkyInstruction& branch, const std::string& newLabel) {
    if (auto* labelData = branch.getModifier<LabelData>()) labelData->label = newLabel;
    const auto& srcs = branch.getSrcRegs();
    for (std::size_t i = 0; i < srcs.size(); ++i) {
        if (srcs[i].dataType != StinkyRegister::Type::LiteralString) continue;
        branch.setSrcReg(i, StinkyRegister(newLabel));
        break;
    }
}

/// kRule3CrossLoop true only. Balance tokens left outstanding by a hoisted loop (see md).
///
/// A handshake that climbed out of its segment leaves its signal above some edge, and every
/// path that leaves the loop below that signal carries a token out with it. Two things are
/// then missing. When the climb crossed the back edge the signal feeds the *next* trip, so
/// the first trip needs one of its own: that is \p pre, a spot found by climbing the
/// preheader, and it is null when nothing crossed the back edge. And whichever paths do
/// carry a token out need a wait to swallow it, which goes just below the loop's exit label
/// where they all land.
///
/// Which paths those are is not the same question as which paths exist. A loop can be
/// hoisted and still let some edges out empty-handed -- the zero-trip guard never entered the
/// body at all, and an exit that sits just below a wait leaves with nothing outstanding.
/// Sending those through the drain would block them on a token nobody posted, so they are
/// routed past it instead.
bool emitLoopCarriedCompensation(StinkyInstruction* loopHead, const std::string& exitLabelName,
                                 const PreLoopSignalAnchor& pre, BasicBlock& preLoopBlock,
                                 Function& func, GfxArchID archId) {
    if (exitLabelName.empty() || loopHead == nullptr) return false;

    // The exit label may be an instruction inside a block or the header of the block that
    // opens at it, depending on where the CFG was last cut; both spell the same point.
    BasicBlock* exitBlock = nullptr;
    IRBase* waitAnchor = nullptr;
    StinkyInstruction* exitLabelInst = nullptr;
    for (BasicBlock& block : func) {
        for (auto it = block.begin(); it != block.end(); ++it) {
            auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (inst == nullptr || !isLabel(*inst)) continue;
            const auto* labelData = inst->getModifier<LabelData>();
            if (labelData == nullptr || labelData->label != exitLabelName) continue;
            if (StinkyInstruction* after = firstRealInstAfter(inst)) {
                exitBlock = &block;
                waitAnchor = after;
                exitLabelInst = inst;
            }
            break;
        }
        if (waitAnchor != nullptr) break;
    }
    if (waitAnchor == nullptr) {
        for (BasicBlock& block : func) {
            if (block.getLabel() != exitLabelName) continue;
            for (auto it = block.begin(); it != block.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst == nullptr || isPseudoInst(inst)) continue;
                exitBlock = &block;
                waitAnchor = inst;
                break;
            }
            break;
        }
    }
    if (waitAnchor == nullptr) return false;

    // A branch to the exit label has to skip the drain wait exactly when no cluster token is
    // in flight where it stands -- when the last cluster instruction on the way there was a
    // wait rather than a signal. Signals and waits strictly alternate, so that state is one
    // bit and a single top-to-bottom sweep settles every branch. The back edge needs no
    // special treatment: the latch is reached with a token in flight, which is also how the
    // preheader arrives at the loop head, so the sweep describes every trip.
    //
    // The sweep opens where the loop's own accounting does: at the preheader signal, which
    // is the first token this loop is answerable for, or at the loop head when there is no
    // such signal because nothing crossed the back edge. Everything above that point leaves
    // for the exit empty-handed. That is not just an assumption about how far the counting
    // has got -- an edge above the preheader signal never entered the body, so there is
    // nothing of this loop's for it to be carrying. The trip-count gate around the Rule 1
    // signal is why reading the state off the text alone would get this wrong: the guard that
    // jumps to the exit fires on the same condition that skips that signal, so the token the
    // text shows in flight was never posted on the path that leaves.
    //
    // One edge into the exit is spelled by no instruction at all: the body simply runs off
    // its end into the label. It is counted here with the rest, because whether it carries a
    // token is exactly as answerable as for a branch and the answer is just as binding.
    std::vector<StinkyInstruction*> bypassBranches;
    bool anyEdgeCarriesToken = false;
    bool fallsThroughToExit = false;
    bool fallThroughCarriesToken = false;
    {
        const IRBase* sweepBegin = (pre.anchor != nullptr) ? pre.anchor : loopHead;
        bool inFlight = false;
        bool countingStarted = false;
        StinkyInstruction* prevReal = nullptr;
        for (BasicBlock& block : func) {
            for (auto it = block.begin(); it != block.end(); ++it) {
                if (it.getNodePtr() == sweepBegin) {
                    countingStarted = true;
                    // The preheader signal is planted in front of its anchor and so is
                    // already outstanding here; arriving at the loop head instead means
                    // reaching it with nothing.
                    inFlight = (pre.anchor != nullptr);
                }
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst == nullptr) continue;
                if (countingStarted) {
                    if (isClusterBarrierSignal(*inst))
                        inFlight = true;
                    else if (isClusterBarrierWait(*inst))
                        inFlight = false;
                }
                if (inst == exitLabelInst && prevReal != nullptr &&
                    !isUnconditionalBranch(*prevReal) && !isCall(*prevReal)) {
                    fallsThroughToExit = true;
                    fallThroughCarriesToken = inFlight;
                    if (inFlight) anyEdgeCarriesToken = true;
                }
                // A label is a place, not a step: what falls into the exit is the last thing
                // that actually ran before it.
                if (!isPseudoInst(inst)) prevReal = inst;
                if (!isBranch(*inst) || getBranchTarget(*inst) != exitLabelName) continue;
                if (inFlight)
                    anyEdgeCarriesToken = true;
                else
                    bypassBranches.push_back(inst);
            }
        }
    }

    // Hoisting inside the body does not by itself strand a token at the exit: a signal that
    // climbed over a plain boundary with no edge out below it leaves every way out of the
    // loop exactly as it found it. Nothing to drain then, and so nothing to route around a
    // drain either.
    if (!anyEdgeCarriesToken) return false;

    AsmIRBuilder exitBuilder(*exitBlock, archId);
    insertClusterBarrierWaitBefore(waitAnchor, "drain loop-carried cluster signal", exitBuilder,
                                   archId);
    const bool fallThroughNeedsBypass = fallsThroughToExit && !fallThroughCarriesToken;
    if (!bypassBranches.empty() || fallThroughNeedsBypass) {
        // Named after the exit it sits just below, so the two read as a pair. That name
        // carries the label prefix already.
        const std::string bypassLabel = exitLabelName + kDrainBypassLabelSuffix;
        static const HwInstDesc labelMCID{
            GFX::LABEL, GFX::LABEL, 0, 0, 0, "LABEL", makeFlagSet({InstFlag::IF_HasSideEffect})};
        StinkyInstruction* bypassLbl = exitBuilder.create(&labelMCID, waitAnchor);
        bypassLbl->addModifier<LabelData>(LabelData{bypassLabel, /*alignment=*/1});
        for (StinkyInstruction* branch : bypassBranches) retargetBranch(*branch, bypassLabel);
        // A branch can be sent past the drain by rewriting where it already goes. The
        // fall-through goes nowhere -- it is the absence of a jump -- so the only way to route
        // it is to give it one.
        if (fallThroughNeedsBypass) {
            insertBranchBefore(exitLabelInst, bypassLabel, exitBuilder, archId);
        }
    }

    if (pre.anchor != nullptr) {
        AsmIRBuilder entryBuilder(preLoopBlock, archId);
        if (pre.needsWorkgroupBarrier) {
            insertWorkgroupBarrierSyncBefore(pre.anchor, entryBuilder, archId);
        }
        insertClusterBarrierSignalOnlyBefore(pre.anchor, entryBuilder, archId);
    }
    return true;
}

class InsertClusterBarrierPassImpl : public Pass {
   public:
    static char ID;

    InsertClusterBarrierPassImpl(bool streamKMulticast, int pgrValue)
        : streamKMulticast_(streamKMulticast), pgrValue_(pgrValue) {}

    const char* getName() const override {
        return "Insert Cluster Barrier";
    }

    Pass::ID getPassID() const override {
        return &InsertClusterBarrierPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        const auto& arch = passCtx.getGemmTileConfig().arch;
        const GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);

        const std::unordered_map<const StinkyInstruction*, uint32_t> cycleMap =
            (kRule3SignalLeadCycles > 0) ? computeEstimatedCyclesPerInstruction(func, passCtx)
                                         : std::unordered_map<const StinkyInstruction*, uint32_t>{};

        // The rules go in the order they are numbered. Each one decides where to put things by
        // reading what is already in the way -- which signal is above, which wait stands
        // between a barrier and its load -- so anything planted out of turn is invisible to
        // the rule that should have seen it, and gets hoisted over or counted twice.
        for (BasicBlock& bb : func) {
            std::vector<IRBase*> gsu1Anchors;
            for (auto it = bb.begin(); it != bb.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst == nullptr) continue;
                if (!isLabelNamed(*inst, kGSU1LabelName)) continue;
                if (isFollowedByClusterBarrierHandshakeOrSignal(inst)) continue;
                auto nextIt = std::next(it);
                gsu1Anchors.push_back((nextIt != bb.end()) ? nextIt.getNodePtr() : nullptr);
            }
            if (gsu1Anchors.empty()) continue;
            AsmIRBuilder irBuilder(bb, archId);
            for (IRBase* anchor : gsu1Anchors) {
                insertRule1ClusterBarrierSignalBefore(anchor, irBuilder, archId);
            }
        }

        // The run-up gets exactly one cluster wait, and this is it. Asking whether one is
        // already there has to look at the whole stretch above the load rather than the
        // instruction immediately in front of it: a pre-loop signal placed by an earlier run
        // sits between the two, and reading only the neighbour would hand the run-up a second
        // wait it has no second token for.
        StinkyInstruction* firstTL = findFirstTensorLoadInFunc(func);
        if (firstTL != nullptr && !hasClusterBarrierWaitAbove(firstTL)) {
            BasicBlock* parent = firstTL->getParent();
            AsmIRBuilder irBuilder(*parent, archId);
            insertClusterBarrierWaitBefore(firstTL, "cluster_barrier wait", irBuilder, archId);
        }

        for (BasicBlock& bb : func) {
            // Collect every trigger in the block before any search runs. A scan that climbs
            // across a back edge walks into handshakes that come later in program order, and
            // it may only stop at their barriers if it already knows they are there.
            struct TriggerSite {
                StinkyInstruction* trigger = nullptr;
                BasicBlock::iterator segBegin;
            };
            std::vector<TriggerSite> triggers;
            std::unordered_set<StinkyInstruction*> seenTriggers;
            // Anchors (instruction right after each cooperative tensor_load
            // group) for the producer-side drain; see
            // insertProducerTensorDrainBefore.
            std::vector<IRBase*> producerDrainAnchors;
            {
                auto segBegin = bb.begin();
                for (auto it = bb.begin(); it != bb.end(); ++it) {
                    auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                    if (inst == nullptr) continue;
                    if (isSegmentBoundary(*inst)) {
                        segBegin = std::next(it);
                        continue;
                    }
                    if (!isTensorLoad(*inst)) continue;

                    StinkyInstruction* trigger =
                        findPrecedingWorkgroupBarrierSignalInSegment(segBegin, inst);
                    if (trigger == nullptr) continue;
                    if (!seenTriggers.insert(trigger).second) continue;
                    if (isImmediatelyPrecededByClusterBarrierWait(trigger)) continue;
                    // Rule 3 speaks for the loop body and nowhere else. Outside a loop there
                    // is no next trip to hand a token to and no exit to compensate at, and the
                    // run-up's own load is Rule 2's business.
                    if (findEnclosingLoopHead(trigger) == nullptr) continue;
                    triggers.push_back({trigger, segBegin});

                    // Record the instruction right after this cooperative
                    // tensor_load group so a producer-side tensor drain can be
                    // planted there (retire the load before the back edge / next
                    // publishing barrier). Advance past any immediately-following
                    // tensor_load(s) so the drain covers the whole group (e.g. the
                    // A/B operand load plus its MX-scale load) rather than landing
                    // between them.
                    if (streamKMulticast_ && pgrValue_ >= 2) {
                        auto postIt = std::next(it);
                        while (postIt != bb.end()) {
                            auto* pinst = dyn_cast<StinkyInstruction>(postIt.getNodePtr());
                            if (pinst != nullptr && isTensorLoad(*pinst)) {
                                ++postIt;
                                continue;
                            }
                            break;
                        }
                        producerDrainAnchors.push_back((postIt != bb.end()) ? postIt.getNodePtr()
                                                                            : nullptr);
                    }
                }
            }

            std::unordered_set<StinkyInstruction*> priorWaitAnchors;
            for (const TriggerSite& site : triggers) {
                priorWaitAnchors.insert(site.trigger);
            }

            struct LoopCompensation {
                StinkyInstruction* head = nullptr;
                std::string exitLabel;
                PreLoopSignalAnchor preLoopSignal;
            };
            std::vector<std::tuple<StinkyInstruction*, IRBase*, IRBase*>> pending;
            std::vector<LoopCompensation> hoistedLoops;
            std::unordered_map<StinkyInstruction*, size_t> hoistedHeads;
            for (const TriggerSite& site : triggers) {
                StinkyInstruction* trigger = site.trigger;
                const BasicBlock::iterator tSegBegin = site.segBegin;
                // What a signal that leaves its segment costs is a wait on whichever edges out
                // of the loop end up carrying it, and that is settled edge by edge below. So
                // each handshake decides on its own whether the lead is worth crossing for,
                // and a loop whose segments disagree is no harder to balance than one where
                // they all hoist.
                StinkyInstruction* head = findEnclosingLoopHead(trigger);
                // kRule3CrossLoop false: maxHops=0, climb stays in-segment. true: one hop.
                const int maxSegmentHops = cluster_barrier::kRule3CrossLoop ? kMaxSegmentHops : 0;
                Rule3SignalAnchor found = findRule3SignalAnchorByCycleLead(
                    trigger, tSegBegin, /*defaultAnchor=*/trigger, cycleMap, kRule3SignalLeadCycles,
                    kRule3SignalMaxLeadCycles, priorWaitAnchors, maxSegmentHops, head);
                // Read the exit label and climb the preheader now: once the handshakes go
                // in, the body is full of this pass's own skip branches and barriers, and
                // neither the loop's real exit nor an unspoken-for stretch of preheader is
                // easy to tell apart from them.
                if (found.hops > 0) {  // kRule3CrossLoop true only: cross-segment compensation
                    const auto [slot, isNew] = hoistedHeads.emplace(head, hoistedLoops.size());
                    if (isNew) hoistedLoops.push_back({head, findLoopExitLabelName(head), {}});
                    // Only a signal that crossed the back edge feeds the next trip instead of
                    // its own, so only that leaves the first trip with nothing to wait on and
                    // asks for a signal in the preheader.
                    LoopCompensation& comp = hoistedLoops[slot->second];
                    if (found.crossedLoopHead && comp.preLoopSignal.anchor == nullptr) {
                        comp.preLoopSignal = findPreLoopSignalAnchor(head);
                        // Crossing the back edge and posting a signal in the preheader are one
                        // decision, not two: the crossing is what hands this signal to the
                        // next trip, and the preheader signal is what the first trip waits on
                        // instead. A preheader with no spot to put it -- a live SCC range
                        // covering the whole run-up, say -- cannot honour half the bargain, so
                        // the signal stays inside the loop where its own wait can reach it.
                        if (comp.preLoopSignal.anchor == nullptr) {
                            found = findRule3SignalAnchorByCycleLead(
                                trigger, tSegBegin, /*defaultAnchor=*/trigger, cycleMap,
                                kRule3SignalLeadCycles, kRule3SignalMaxLeadCycles, priorWaitAnchors,
                                /*maxHops=*/0, head);
                        }
                    }
                }
                pending.emplace_back(trigger, found.anchor, static_cast<IRBase*>(trigger));
            }

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
            if (tailTL != nullptr && isImmediatelyPrecededByClusterBarrierWait(tailTL)) {
                tailTL = nullptr;
            }
            if (tailWait != nullptr) {
                StinkyInstruction* tailPairedSignal =
                    findPrecedingWorkgroupBarrierSignalInSegment(bb.begin(), tailWait);
                bool conflictsWithRule3 = false;
                for (const auto& [trigger, _sig, _wait] : pending) {
                    if (tailPairedSignal != nullptr && trigger == tailPairedSignal) {
                        conflictsWithRule3 = true;
                        break;
                    }
                }
                if (conflictsWithRule3 || isFollowedByClusterBarrierHandshakeOrSignal(tailWait)) {
                    tailWait = nullptr;
                }
            }

            if (pending.empty() && tailTL == nullptr && tailWait == nullptr) continue;

            AsmIRBuilder irBuilder(bb, archId);
            for (const auto& [trigger, signalAnchor, waitAnchor] : pending) {
                insertRule3HandshakeBefore(signalAnchor, waitAnchor, irBuilder, archId);
                (void)trigger;
            }
            // kRule3CrossLoop true only: drain / skipCBWait for hoisted loops.
            for (const LoopCompensation& comp : hoistedLoops) {
                emitLoopCarriedCompensation(comp.head, comp.exitLabel, comp.preLoopSignal, bb, func,
                                            archId);
            }
            // Producer-side drain right after each cooperative tensor_load
            // group (retire the async cooperative load before the back-edge so
            // the next loop-head barrier publishes it coherently).
            for (IRBase* postAnchor : producerDrainAnchors) {
                insertProducerTensorDrainBefore(postAnchor, irBuilder, archId);
            }
            if (tailWait != nullptr) {
                IRBase* anchor =
                    (tailWaitNextIt != bb.end()) ? tailWaitNextIt.getNodePtr() : nullptr;
                insertClusterBarrierSignalOnlyBefore(anchor, irBuilder, archId);
            }
            if (tailTL != nullptr) {
                insertClusterBarrierWaitBefore(tailTL, "cluster barrier wait", irBuilder, archId);
            }
        }

        return PreservedAnalyses::none();
    }

   private:
    const bool streamKMulticast_ = false;
    const int pgrValue_ = 1;
};

char InsertClusterBarrierPassImpl::ID = 0;

}  // namespace

std::unique_ptr<Pass> createInsertClusterBarrierPass(bool streamKMulticast, int pgrValue) {
    return std::make_unique<InsertClusterBarrierPassImpl>(streamKMulticast, pgrValue);
}

}  // namespace stinkytofu
