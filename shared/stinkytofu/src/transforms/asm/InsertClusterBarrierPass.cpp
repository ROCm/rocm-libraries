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
#include <cstring>
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
constexpr const char* kWaveIdxSymbol = "sgprWaveIdx";
constexpr const char* kLoopCounterLSymbol = "sgprLoopCounterL";
constexpr size_t kHashLen = 16;
constexpr const char* kGSU1LabelName = "label_GSU_1";
constexpr const char* kTailLoopMarker = "Tail Loop";

/// Estimated cycles the Rule 3 signal is planted ahead of its paired wait.
/// Set to 0 to co-locate the signal with the wait.
constexpr int kRule3SignalLeadCycles = 500;

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

StinkyInstruction* findLiveRestorableSccCmpUpstream(StinkyInstruction* anchor) {
    BasicBlock* parent = anchor->getParent();
    if (parent == nullptr) return nullptr;
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
            for (const auto& dst : inst->getDestRegs()) {
                if (dst.isRegister() && isAllocatableReg(dst.reg.type)) return nullptr;
            }
            for (const auto& src : inst->getSrcRegs()) {
                if (!src.isRegister()) continue;
                for (const auto& w : clobbered) {
                    if (src.isOverlap(w)) return nullptr;
                }
            }
            return inst;
        }
        for (const auto& dst : inst->getDestRegs()) {
            if (dst.isRegister()) clobbered.push_back(dst);
        }
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
        GFX::LABEL, GFX::LABEL, 0, 0, 0, 0, "LABEL", makeFlagSet({InstFlag::IF_HasSideEffect})};
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
        GFX::LABEL, GFX::LABEL, 0, 0, 0, 0, "LABEL", makeFlagSet({InstFlag::IF_HasSideEffect})};
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

void insertRule3SccRestore(IRBase* anchor, AsmIRBuilder& irBuilder,
                           StinkyInstruction* sccRestoreCmp) {
    const HwInstDesc* restoreDesc = sccRestoreCmp->getHwInstDesc();
    StinkyInstruction* restoreInst = irBuilder.create(restoreDesc, anchor);
    for (const auto& src : sccRestoreCmp->getSrcRegs()) restoreInst->addSrcReg(src);
    restoreInst->addModifier<CommentData>(
        CommentData{"restore SCC for downstream cbranch (Rule 3)"});
}

void insertRule3HandshakeBefore(IRBase* signalAnchor, IRBase* waitAnchor, AsmIRBuilder& irBuilder,
                                GfxArchID archId, StinkyInstruction* sccRestoreCmp) {
    insertClusterBarrierSignalOnlyBefore(signalAnchor, irBuilder, archId);
    if (sccRestoreCmp != nullptr) {
        insertRule3SccRestore(signalAnchor, irBuilder, sccRestoreCmp);
    }
    insertClusterBarrierWaitBefore(waitAnchor, "cluster barrier wait", irBuilder, archId);
}

/// Emit `s_wait_tensorcnt 0` immediately before \p anchor.
void insertProducerTensorDrainBefore(IRBase* anchor, AsmIRBuilder& irBuilder, GfxArchID archId,
                                     const char* comment) {
    const HwInstDesc* waitDesc = getMCIDByUOp(GFX::s_wait_tensorcnt, archId);
    assert(waitDesc && "s_wait_tensorcnt opcode is not supported on this architecture");
    StinkyInstruction* w = irBuilder.create(waitDesc, anchor);
    w->addSrcReg(StinkyRegister(0));
    SWaitTensorCntData d;
    d.tlcnt = 0;
    w->addModifier<SWaitTensorCntData>(d);
    w->addModifier<CommentData>(CommentData{comment});
}

bool isLabelNamed(const StinkyInstruction& inst, const char* name) {
    if (!isLabel(inst)) return false;
    const auto* labelData = inst.getModifier<LabelData>();
    return labelData != nullptr && labelData->label == name;
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
    for (StinkyInstruction* afterSig = firstRealInstAfter(wgSignal); afterSig != nullptr;
         afterSig = firstRealInstAfter(afterSig)) {
        if (isWorkgroupBarrierWait(*afterSig)) {
            StinkyInstruction* afterWgWait = firstRealInstAfter(afterSig);
            return (afterWgWait != nullptr) ? static_cast<IRBase*>(afterWgWait) : defaultAnchor;
        }
        if (isWorkgroupBarrierSignal(*afterSig)) break;
    }
    return defaultAnchor;
}

/// Forward scan from ``afterWait`` for the next workgroup barrier handshake
/// (signal then wait).  Returns the first real instruction after that wait, or
/// ``defaultAnchor`` when the pair is missing or incomplete.
IRBase* anchorAfterWorkgroupBarrierFollowing(StinkyInstruction* afterWait, IRBase* defaultAnchor) {
    for (StinkyInstruction* fwd = firstRealInstAfter(afterWait); fwd != nullptr;
         fwd = firstRealInstAfter(fwd)) {
        if (!isWorkgroupBarrierSignal(*fwd)) continue;
        return anchorAfterWorkgroupBarrierPair(fwd, defaultAnchor);
    }
    return defaultAnchor;
}

IRBase* findRule3SignalAnchorByCycleLead(
    StinkyInstruction* referenceAnchor, BasicBlock::iterator segBegin, IRBase* defaultAnchor,
    const std::unordered_map<const StinkyInstruction*, uint32_t>& cycleMap, int leadCycles,
    const std::unordered_set<StinkyInstruction*>& priorWaitAnchors) {
    if (leadCycles <= 0) return defaultAnchor;
    auto refIt = cycleMap.find(referenceAnchor);
    if (refIt == cycleMap.end()) return defaultAnchor;

    // target may be <= 0 when the wait anchor is fewer than leadCycles from
    // function entry; cycle matching then fails and the barrier/segment
    // fallbacks below decide the anchor.
    const int64_t target = static_cast<int64_t>(refIt->second) - leadCycles;
    auto it = BasicBlock::iterator(referenceAnchor);
    while (it != segBegin) {
        --it;
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (isClusterBarrierWait(*inst)) {
            return anchorAfterWorkgroupBarrierFollowing(inst, defaultAnchor);
        }
        if (isSegmentBoundary(*inst)) return segBegin.getNodePtr();
        if (isWorkgroupBarrierSignal(*inst)) {
            if (priorWaitAnchors.count(inst) != 0) {
                return anchorAfterWorkgroupBarrierPair(inst, defaultAnchor);
            }
            continue;
        }
        if (isWorkgroupBarrierWait(*inst)) continue;
        auto cycleIt = cycleMap.find(inst);
        if (cycleIt == cycleMap.end()) continue;
        if (static_cast<int64_t>(cycleIt->second) <= target) return inst;
    }
    return segBegin.getNodePtr();
}

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

/// Hoist a cluster-wait insertion point above the run of wait-cnt instructions
/// immediately preceding \p anchor.
///
/// StinkyWaitCntInsertionPass runs before this pass and anchors its counter
/// drains on the very instructions this pass targets (barriers and
/// tensor_loads), so the slot right before the anchor is typically already
/// occupied by `s_wait_tensorcnt` / `s_wait_dscnt` / ... Inserting the cluster
/// wait there would emit
///
///     s_wait_tensorcnt N
///     s_barrier_wait -3
///
/// Both orders are correct. The inverted one measured materially slower, which
/// is the whole justification for this hoist -- the mechanism is NOT
/// established. Both instructions are blocking waits on independent conditions
/// (a per-wave local counter; peer arrival at the barrier), and two such waits
/// commute, so no timing argument offered so far survives scrutiny. Treat this
/// as measurement-driven until someone explains it.
///
/// Wait-cnt instructions never write SCC, so hoisting past them does not
/// disturb the Rule 3 SCC restore.
/// Any counter drain the wait-cnt pass may have parked ahead of an anchor.
/// `s_wait_tensorcnt` carries IF_WaitTensorCnt, disjoint from IF_WaitCnt, so
/// `isWaitCnt()` alone misses it -- and it is the drain that matters most here.
/// Same idiom as WaitAwareScheduleRepairPass and StinkyRemoveWaitCntPass.
bool isAnyCounterDrain(const StinkyInstruction& inst) {
    return isWaitCnt(inst) || inst.is(InstFlag::IF_WaitTensorCnt);
}

IRBase* hoistAboveLeadingWaitCnts(StinkyInstruction* anchor) {
    BasicBlock* parent = anchor->getParent();
    if (parent == nullptr) return anchor;
    IRBase* hoisted = anchor;
    auto it = BasicBlock::iterator(anchor);
    while (it != parent->begin()) {
        --it;
        auto* prev = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (prev == nullptr) continue;
        if (isPseudoInst(prev)) continue;
        if (!isAnyCounterDrain(*prev)) break;
        hoisted = prev;
    }
    return hoisted;
}

/// Idempotency check for the cluster wait. Skips the same wait-cnt run that
/// hoistAboveLeadingWaitCnts walks over, so a re-run recognizes a handshake
/// this pass already planted above those waits and does not duplicate it.
bool isImmediatelyPrecededByClusterBarrierWait(StinkyInstruction* anchor) {
    BasicBlock* parent = anchor->getParent();
    if (parent == nullptr) return false;
    auto it = BasicBlock::iterator(anchor);
    while (it != parent->begin()) {
        --it;
        auto* prev = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (prev == nullptr) continue;
        if (isPseudoInst(prev)) continue;
        if (isAnyCounterDrain(*prev)) continue;
        return isClusterBarrierWait(*prev);
    }
    return false;
}

bool isSXorB32(const StinkyInstruction& inst) {
    const HwInstDesc* desc = inst.getHwInstDesc();
    return desc != nullptr && desc->mnemonic != nullptr &&
           std::strcmp(desc->mnemonic, "s_xor_b32") == 0;
}

bool isWaitTensorCnt(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_WaitTensorCnt);
}

bool isImmediatelyPrecededByWaitTensorCnt(StinkyInstruction* anchor) {
    BasicBlock* parent = anchor->getParent();
    if (parent == nullptr) return false;
    auto it = BasicBlock::iterator(anchor);
    while (it != parent->begin()) {
        --it;
        auto* prev = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (prev == nullptr) continue;
        if (isPseudoInst(prev)) continue;
        return isWaitTensorCnt(*prev);
    }
    return false;
}

/// First ``s_xor_b32`` after \p start in the same segment (TDM LDS ping-pong).
StinkyInstruction* findXorInSegmentAfter(StinkyInstruction* start) {
    for (StinkyInstruction* n = firstRealInstAfter(start); n != nullptr;
         n = firstRealInstAfter(n)) {
        if (isSegmentBoundary(*n)) return nullptr;
        if (isSXorB32(*n)) return n;
    }
    return nullptr;
}

bool hasWaitTensorCntBetween(StinkyInstruction* start, StinkyInstruction* endExclusive) {
    for (StinkyInstruction* n = firstRealInstAfter(start); n != nullptr && n != endExclusive;
         n = firstRealInstAfter(n)) {
        if (isWaitTensorCnt(*n)) return true;
    }
    return false;
}

/// Ends a cooperative A+MXSA tensor_load group. SIA=4 can split the two loads
/// with TDM increments / ds_load; those are not terminators.
bool isCoopLoadGroupTerminator(const StinkyInstruction& inst) {
    return isSegmentBoundary(inst) || isSXorB32(inst) || isWaitTensorCnt(inst) || isBarrier(inst);
}

/// Instruction after the last tensor_load of the cooperative group that starts
/// at \p tlIt. Walking only adjacent tensor_loads planted the PGR=1 SIA=4
/// producer drain between A and MXSA (confirmed hang vs SIA=0).
BasicBlock::iterator afterCooperativeTensorLoadGroup(BasicBlock::iterator tlIt, BasicBlock& bb) {
    auto lastTlNext = std::next(tlIt);
    for (auto postIt = lastTlNext; postIt != bb.end(); ++postIt) {
        auto* pinst = dyn_cast<StinkyInstruction>(postIt.getNodePtr());
        if (pinst == nullptr || isPseudoInst(pinst)) continue;
        if (isCoopLoadGroupTerminator(*pinst)) break;
        if (isTensorLoad(*pinst)) lastTlNext = std::next(postIt);
    }
    return lastTlNext;
}

class InsertClusterBarrierPassImpl : public Pass {
   public:
    static char ID;

    InsertClusterBarrierPassImpl(StreamKMulticastMode streamKMulticast)
        : streamKMulticast_(streamKMulticast) {}

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

        for (BasicBlock& bb : func) {
            std::vector<std::tuple<StinkyInstruction*, IRBase*, IRBase*, StinkyInstruction*>>
                pending;
            std::unordered_set<StinkyInstruction*> seenTriggers;
            // Anchors (instruction right after each cooperative tensor_load
            // group) for the producer-side drain; see
            // insertProducerTensorDrainBefore.
            std::vector<IRBase*> producerDrainAnchors;

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
                // Emit the cluster wait above the drains the wait-cnt pass
                // already anchored on this workgroup signal.
                IRBase* waitAnchor = hoistAboveLeadingWaitCnts(trigger);
                // Measure the lead from where the wait actually lands, not from
                // the trigger, so the hoist does not eat into the guaranteed
                // signal->wait distance.
                auto* hoistedInst = dyn_cast<StinkyInstruction>(waitAnchor);
                StinkyInstruction* waitAnchorInst =
                    (hoistedInst != nullptr) ? hoistedInst : trigger;

                std::unordered_set<StinkyInstruction*> priorWaitAnchors;
                for (const auto& [priorTrigger, _sig, _wait, _live] : pending) {
                    (void)_sig;
                    (void)_wait;
                    (void)_live;
                    priorWaitAnchors.insert(priorTrigger);
                }

                IRBase* signalAnchor = findRule3SignalAnchorByCycleLead(
                    waitAnchorInst, segBegin, /*defaultAnchor=*/waitAnchor, cycleMap,
                    kRule3SignalLeadCycles, priorWaitAnchors);
                StinkyInstruction* signalAnchorInst =
                    (signalAnchor != nullptr) ? dyn_cast<StinkyInstruction>(signalAnchor) : nullptr;

                StinkyInstruction* sccRestoreCmp = nullptr;
                if (signalAnchorInst != nullptr &&
                    !anySccWriterInRange(signalAnchorInst, waitAnchor)) {
                    sccRestoreCmp = findLiveRestorableSccCmpUpstream(signalAnchorInst);
                }
                pending.emplace_back(trigger, signalAnchor, waitAnchor, sccRestoreCmp);

                // Drain after the cooperative tensor_load group (SIA=4 may split A/MXSA).
                if (streamKMulticast_ >= kStreamKMulticastOn) {
                    auto postIt = afterCooperativeTensorLoadGroup(it, bb);
                    producerDrainAnchors.push_back((postIt != bb.end()) ? postIt.getNodePtr()
                                                                        : nullptr);
                }
            }

            std::vector<IRBase*> gsu1Anchors;
            for (auto it = bb.begin(); it != bb.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst == nullptr) continue;
                if (!isLabelNamed(*inst, kGSU1LabelName)) continue;
                if (isFollowedByClusterBarrierHandshakeOrSignal(inst)) continue;
                auto nextIt = std::next(it);
                gsu1Anchors.push_back((nextIt != bb.end()) ? nextIt.getNodePtr() : nullptr);
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
                for (const auto& [trigger, _sig, _wait, _live] : pending) {
                    if (tailPairedSignal != nullptr && trigger == tailPairedSignal) {
                        conflictsWithRule3 = true;
                        break;
                    }
                }
                if (conflictsWithRule3 || isFollowedByClusterBarrierHandshakeOrSignal(tailWait)) {
                    tailWait = nullptr;
                }
            }

            if (pending.empty() && gsu1Anchors.empty() && tailTL == nullptr && tailWait == nullptr)
                continue;

            AsmIRBuilder irBuilder(bb, archId);
            // Multicast-on: skip loop -3 (mixed LoopCounter); keep producer drain.
            if (streamKMulticast_ < kStreamKMulticastOn) {
                for (const auto& [trigger, signalAnchor, waitAnchor, sccRestoreCmp] : pending) {
                    insertRule3HandshakeBefore(signalAnchor, waitAnchor, irBuilder, archId,
                                               sccRestoreCmp);
                    (void)trigger;
                }
            }
            // Producer-side drain right after each cooperative tensor_load
            // group (retire the async cooperative load before the back-edge so
            // the next loop-head barrier publishes it coherently).
            for (IRBase* postAnchor : producerDrainAnchors) {
                insertProducerTensorDrainBefore(
                    postAnchor, irBuilder, archId,
                    "retire cooperative tensor_load_to_lds before back-edge");
            }
            if (streamKMulticast_ < kStreamKMulticastOn) {
                for (IRBase* anchor : gsu1Anchors) {
                    insertRule1ClusterBarrierSignalBefore(anchor, irBuilder, archId);
                }
                if (tailWait != nullptr) {
                    IRBase* anchor =
                        (tailWaitNextIt != bb.end()) ? tailWaitNextIt.getNodePtr() : nullptr;
                    insertClusterBarrierSignalOnlyBefore(anchor, irBuilder, archId);
                }
                if (tailTL != nullptr) {
                    insertClusterBarrierWaitBefore(hoistAboveLeadingWaitCnts(tailTL),
                                                   "cluster barrier wait", irBuilder, archId);
                }
            }
        }

        StinkyInstruction* firstTL = findFirstTensorLoadInFunc(func);
        // Rule 2: wait before first tensor_load. Multicast-on already waited prologue -3.
        if (streamKMulticast_ < kStreamKMulticastOn && firstTL != nullptr &&
            !isImmediatelyPrecededByClusterBarrierWait(firstTL)) {
            BasicBlock* parent = firstTL->getParent();
            AsmIRBuilder irBuilder(*parent, archId);
            insertClusterBarrierWaitBefore(hoistAboveLeadingWaitCnts(firstTL),
                                           "cluster_barrier wait", irBuilder, archId);
        }

        // Drain prologue TDM before the LDS ping-pong XOR of the descriptor.
        if (streamKMulticast_ >= kStreamKMulticastOn && firstTL != nullptr) {
            StinkyInstruction* pingPongXor = findXorInSegmentAfter(firstTL);
            if (pingPongXor != nullptr && !hasWaitTensorCntBetween(firstTL, pingPongXor) &&
                !isImmediatelyPrecededByWaitTensorCnt(pingPongXor)) {
                BasicBlock* parent = pingPongXor->getParent();
                AsmIRBuilder irBuilder(*parent, archId);
                insertProducerTensorDrainBefore(pingPongXor, irBuilder, archId,
                                                "retire TDM before LDS ping-pong XOR");
            }
        }

        return PreservedAnalyses::none();
    }

   private:
    const StreamKMulticastMode streamKMulticast_ = kStreamKMulticastOff;
};

char InsertClusterBarrierPassImpl::ID = 0;

}  // namespace

std::unique_ptr<Pass> createInsertClusterBarrierPass(StreamKMulticastMode streamKMulticast) {
    return std::make_unique<InsertClusterBarrierPassImpl>(streamKMulticast);
}

}  // namespace stinkytofu
