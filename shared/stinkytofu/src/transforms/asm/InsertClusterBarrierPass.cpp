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
constexpr const char* kWaveIdxSymbol = "sgprWaveIdx";
constexpr const char* kLoopCounterLSymbol = "sgprLoopCounterL";
constexpr size_t kHashLen = 16;
constexpr const char* kGSU1LabelName = "label_GSU_1";
constexpr const char* kTailLoopMarker = "Tail Loop";

/// Estimated cycles the Rule 3 signal is planted ahead of its paired wait.
/// Set to 0 to co-locate the signal with the wait.
constexpr int kRule3SignalLeadCycles = 500;

/// Ceiling on how far ahead of its wait the signal may end up after climbing out of a live
/// SCC range. The climb has to clear the whole range, so its cost is the length of that
/// range, not a constant; past this ceiling it buys correctness at more overlap than it is
/// worth. The anchor then drops below the range instead, which lands it closer than
/// kRule3SignalLeadCycles rather than further away.
constexpr int kRule3SignalMaxLeadCycles = 900;

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
StinkyInstruction* findSccDeadPointBelow(StinkyInstruction* from, const IRBase* limit) {
    BasicBlock* parent = from->getParent();
    if (parent == nullptr) return nullptr;
    for (auto it = BasicBlock::iterator(from); it != parent->end(); ++it) {
        if (it.getNodePtr() == limit) return nullptr;
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (isSegmentBoundary(*inst) || isClusterBarrierWait(*inst)) return nullptr;
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
    int maxLeadCycles, const std::unordered_set<StinkyInstruction*>& priorWaitAnchors) {
    if (leadCycles <= 0) return defaultAnchor;
    auto refIt = cycleMap.find(referenceAnchor);
    if (refIt == cycleMap.end()) return defaultAnchor;

    // target may be <= 0 when the wait anchor is fewer than leadCycles from
    // function entry; cycle matching then fails and the barrier/segment
    // fallbacks below decide the anchor.
    const int64_t target = static_cast<int64_t>(refIt->second) - leadCycles;

    // The handshake opens with `s_cmp_eq_u32 sgprWaveIdx, 0` and is planted in front of
    // whatever this scan returns, so the anchor may only land where SCC holds nothing
    // live. `sccLive` is maintained backwards to describe the point in front of the
    // instruction being looked at, and once the cycle lead is met the scan keeps climbing
    // until that point is clear -- which, for a lead that falls inside a def..reader
    // range, means coming to rest in front of the def. The boundary returns below still
    // win: a cluster wait, a segment edge or a prior handshake's barrier cannot be
    // crossed just to find a better spot.
    bool sccLive = isSccLiveBefore(referenceAnchor);
    bool targetMet = false;
    StinkyInstruction* leadPoint = nullptr;

    // A climb that ends up further than maxLeadCycles from the wait has cleared a range too
    // long to be worth it. Fall the other way instead: down from the lead point to the first
    // spot below the range, which is nearer the wait than the lead asked for.
    // A boundary-forced anchor is a lower bound -- the scan may not go above it -- so when
    // it lands inside a live range the only legal correction is to drop below the range.
    // Failing that the whole segment from the def down to the wait is live and there is no
    // safe spot at all; the caller's default (co-locating with the wait) is then no worse
    // than anything else this pass could pick.
    auto clearScc = [&](IRBase* anchor) -> IRBase* {
        auto* anchorInst = dyn_cast<StinkyInstruction>(anchor);
        if (anchorInst == nullptr || !isSccLiveBefore(anchorInst)) return anchor;
        StinkyInstruction* below = findSccDeadPointBelow(anchorInst, referenceAnchor);
        return (below != nullptr) ? static_cast<IRBase*>(below) : defaultAnchor;
    };

    auto settle = [&](IRBase* climbed) -> IRBase* {
        if (leadPoint == nullptr || climbed == leadPoint) return climbed;
        int64_t climbedCycle = 0;
        if (auto* climbedInst = dyn_cast<StinkyInstruction>(climbed)) {
            auto climbedIt = cycleMap.find(climbedInst);
            if (climbedIt != cycleMap.end()) climbedCycle = static_cast<int64_t>(climbedIt->second);
        }
        if (static_cast<int64_t>(refIt->second) - climbedCycle <= maxLeadCycles) return climbed;
        StinkyInstruction* below = findSccDeadPointBelow(leadPoint, referenceAnchor);
        return (below != nullptr) ? static_cast<IRBase*>(below) : defaultAnchor;
    };

    auto it = BasicBlock::iterator(referenceAnchor);
    while (it != segBegin) {
        --it;
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (isClusterBarrierWait(*inst)) {
            return clearScc(anchorAfterWorkgroupBarrierFollowing(inst, defaultAnchor));
        }
        if (isSegmentBoundary(*inst)) return clearScc(segBegin.getNodePtr());
        if (isWorkgroupBarrierSignal(*inst) && priorWaitAnchors.count(inst) != 0) {
            return clearScc(anchorAfterWorkgroupBarrierPair(inst, defaultAnchor));
        }

        // live-before(inst) = reads(inst) | (live-after(inst) & !writes(inst))
        sccLive = readsScc(*inst) || (sccLive && !writesScc(*inst));

        if (isWorkgroupBarrierSignal(*inst) || isWorkgroupBarrierWait(*inst)) continue;

        auto cycleIt = cycleMap.find(inst);
        if (cycleIt != cycleMap.end() && static_cast<int64_t>(cycleIt->second) <= target) {
            if (!targetMet) leadPoint = inst;
            targetMet = true;
        }
        if (targetMet && !sccLive) return settle(inst);
    }
    // Running out of segment can leave the anchor inside a range that starts above it.
    return clearScc(settle(segBegin.getNodePtr()));
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

        for (BasicBlock& bb : func) {
            std::vector<std::tuple<StinkyInstruction*, IRBase*, IRBase*>> pending;
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

                IRBase* waitAnchor = trigger;
                StinkyInstruction* waitAnchorInst = trigger;
                if (isImmediatelyPrecededByClusterBarrierWait(waitAnchorInst)) continue;

                std::unordered_set<StinkyInstruction*> priorWaitAnchors;
                for (const auto& [priorTrigger, _sig, _wait] : pending) {
                    (void)_sig;
                    (void)_wait;
                    priorWaitAnchors.insert(priorTrigger);
                }

                IRBase* signalAnchor = findRule3SignalAnchorByCycleLead(
                    waitAnchorInst, segBegin, /*defaultAnchor=*/waitAnchor, cycleMap,
                    kRule3SignalLeadCycles, kRule3SignalMaxLeadCycles, priorWaitAnchors);
                pending.emplace_back(trigger, signalAnchor, waitAnchor);

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

            if (pending.empty() && gsu1Anchors.empty() && tailTL == nullptr && tailWait == nullptr)
                continue;

            AsmIRBuilder irBuilder(bb, archId);
            for (const auto& [trigger, signalAnchor, waitAnchor] : pending) {
                insertRule3HandshakeBefore(signalAnchor, waitAnchor, irBuilder, archId);
                (void)trigger;
            }
            // Producer-side drain right after each cooperative tensor_load
            // group (retire the async cooperative load before the back-edge so
            // the next loop-head barrier publishes it coherently).
            for (IRBase* postAnchor : producerDrainAnchors) {
                insertProducerTensorDrainBefore(postAnchor, irBuilder, archId);
            }
            for (IRBase* anchor : gsu1Anchors) {
                insertRule1ClusterBarrierSignalBefore(anchor, irBuilder, archId);
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

        StinkyInstruction* firstTL = findFirstTensorLoadInFunc(func);
        if (firstTL != nullptr && !isImmediatelyPrecededByClusterBarrierWait(firstTL)) {
            BasicBlock* parent = firstTL->getParent();
            AsmIRBuilder irBuilder(*parent, archId);
            insertClusterBarrierWaitBefore(firstTL, "cluster_barrier wait", irBuilder, archId);
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
