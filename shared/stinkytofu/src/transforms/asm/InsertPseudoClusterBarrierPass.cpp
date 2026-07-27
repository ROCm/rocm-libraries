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
#include "stinkytofu/transforms/asm/InsertPseudoClusterBarrierPass.hpp"

#include <iterator>
#include <unordered_set>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {
namespace {

/// True if `inst` is a workgroup-scope barrier completion: `s_barrier_wait -1`
/// (the LDS publication point the cluster handshake anchors on). The cluster
/// `-3` wait our expansion synthesizes is intentionally not matched here.
bool isWorkgroupBarrierWait(const StinkyInstruction& inst) {
    return isBarrierWait(inst) && isSplitBarrierAllWave(inst);
}

/// True if `inst` is a workgroup-scope barrier arrival: `s_barrier_signal -1`.
bool isWorkgroupBarrierSignal(const StinkyInstruction& inst) {
    return isBarrierSignal(inst) && isSplitBarrierAllWave(inst);
}

/// A segment boundary is either a label (control-flow entry point) or a branch
/// (control-flow exit point). This pass runs at kernel scope, where Tensile has
/// lowered the whole kernel into a single flat entry basic block with inline
/// label pseudos and branches instead of a real CFG. Treating both as
/// boundaries recovers per-CFG-basic-block segmentation on that flat IR, which
/// matters for unrolled loops where iter 1/2 and iter 2/2 share one
/// `label_LoopBeginL` segment but are split by the odd-exit `s_cbranch`.
bool isSegmentBoundary(const StinkyInstruction& inst) {
    return isLabel(inst) || isBranch(inst);
}

/// Walk backward from \p anchor (exclusive) toward \p segmentBegin (inclusive)
/// to find the nearest preceding `s_barrier_wait -1`. Stops as soon as a
/// segment boundary is crossed so the trigger always lives in the same segment
/// as \p anchor (never crossing a label/branch, i.e. a CFG edge).
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

/// Walk backward from \p wait (exclusive) to find the nearest preceding
/// workgroup `s_barrier_signal -1` that pairs with it. Stops at a segment
/// boundary so the signal always lives in the same CFG segment as its wait.
StinkyInstruction* findMatchingWorkgroupBarrierSignal(StinkyInstruction* wait) {
    BasicBlock* parent = wait->getParent();
    if (parent == nullptr) return nullptr;
    auto it = BasicBlock::iterator(wait);
    while (it != parent->begin()) {
        --it;
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (isSegmentBoundary(*inst)) return nullptr;
        if (isWorkgroupBarrierSignal(*inst)) return inst;
    }
    return nullptr;
}

/// True if a PSEUDO_CLUSTER_BARRIER already sits immediately before \p signal
/// (idempotency guard for pipeline re-runs).
bool signalAlreadyGated(StinkyInstruction* signal) {
    BasicBlock* parent = signal->getParent();
    if (parent == nullptr) return false;
    auto it = BasicBlock::iterator(signal);
    if (it == parent->begin()) return false;
    --it;
    auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
    return inst != nullptr && isPseudoClusterBarrier(*inst);
}

/// Find the loop-counter SGPR actually used in \p bb and return it as a bare
/// single-dword register (real physical idx). The pseudo-cluster handshake reads the
/// loop counter, but the placeholder carries no operands, so nothing in the DAG keeps it
/// ordered against the loop-tail `dec counterL`. Attaching this register as a placeholder
/// src lets the reg-dependency builder add the RAW edge (read the right definition) and
/// the WAR edge (the decrement must stay after the pseudo). The DAG keys registers by
/// (type, idx) — not by symbolic name — so we must reuse the loop body's real register,
/// not a fresh idx-0 symbolic one. Returns false when no loop counter is present.
bool findLoopCounterReg(BasicBlock& bb, StinkyRegister& out) {
    auto match = [&](const StinkyRegister& r) -> bool {
        if (!r.isRegister() || r.reg.type != RegType::S) return false;
        if (r.getSymbolicName().find("LoopCounterL") == std::string::npos) return false;
        out = StinkyRegister(RegType::S, r.reg.idx, /*regNum=*/1u);
        out.setSymbolicName(r.getSymbolicName());
        return true;
    };
    for (IRBase& ir : bb) {
        auto* inst = dyn_cast<StinkyInstruction>(&ir);
        if (inst == nullptr) continue;
        for (const StinkyRegister& r : inst->getDestRegs()) if (match(r)) return true;
        for (const StinkyRegister& r : inst->getSrcRegs()) if (match(r)) return true;
    }
    return false;
}

/// True if a PSEUDO_CLUSTER_BARRIER already sits immediately after \p wait
/// (idempotency guard for pipeline re-runs).
bool waitAlreadyGated(StinkyInstruction* wait, BasicBlock& bb) {
    auto next = BasicBlock::iterator(wait);
    ++next;
    if (next == bb.end()) return false;
    auto* inst = dyn_cast<StinkyInstruction>(next.getNodePtr());
    return inst != nullptr && isPseudoClusterBarrier(*inst);
}

class InsertPseudoClusterBarrierPassImpl : public Pass {
   public:
    static char ID;

    const char* getName() const override {
        return "Insert Pseudo Cluster Barrier";
    }

    Pass::ID getPassID() const override {
        return &InsertPseudoClusterBarrierPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        const auto& arch = passCtx.getGemmTileConfig().arch;
        const GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);

        for (BasicBlock& bb : func) {
            // Collect the anchoring `s_barrier_wait -1` for each tensor load,
            // deduplicated by identity so loads sharing one wait yield exactly
            // one placeholder. Segment the flat kernel BB by labels/branches so
            // each load's backward scan for its anchor wait stays within its own
            // CFG segment. Gather first, insert after, to avoid mutating the
            // block while iterating it.
            std::unordered_set<StinkyInstruction*> seen;
            std::vector<StinkyInstruction*> anchors;
            auto segBegin = bb.begin();
            for (auto it = bb.begin(); it != bb.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst == nullptr) continue;
                if (isSegmentBoundary(*inst)) {
                    // The boundary belongs to neither side; the next segment
                    // starts right after it.
                    segBegin = std::next(it);
                    continue;
                }
                if (!isTensorLoad(*inst)) continue;
                StinkyInstruction* wait = findPrecedingWorkgroupBarrierWaitInSegment(segBegin, inst);
                if (wait != nullptr && seen.insert(wait).second) anchors.push_back(wait);
            }

            AsmIRBuilder irBuilder(bb, archId);
            StinkyRegister loopCounterReg;
            const bool hasLoopCounter = findLoopCounterReg(bb, loopCounterReg);
            for (StinkyInstruction* wait : anchors) {
                // Plant two placeholders sandwiching the workgroup barrier pair, in the
                // required program order:
                //
                //   pseudo(SignalOnly)     <- cluster signal, BEFORE the workgroup signal
                //   s_barrier_signal -1
                //   ...                       (latency-hiding window)
                //   s_barrier_wait -1
                //   pseudo(WaitOnly)       <- cluster wait, AFTER the workgroup wait
                //
                // A scheduling stick chain (StickChainData) keeps each placeholder glued
                // to its workgroup barrier without an EXEC_GROUP, so the barrier keeps its
                // own forced-barrier threshold. The chain is triggered by the workgroup
                // barrier's threshold and issues from member 0, so the SignalOnly chain
                // [placeholder, signal] emits the placeholder just before the signal, and
                // the WaitOnly chain [wait, placeholder] emits the placeholder just after
                // the wait. Each placeholder copies its barrier's MemTokenData to stay
                // ordered next to it.

                // WaitOnly right after the anchor wait.
                if (!waitAlreadyGated(wait, bb)) {
                    auto after = BasicBlock::iterator(wait);
                    ++after;
                    IRBase* insertBefore = (after != bb.end()) ? after.getNodePtr() : nullptr;
                    StinkyInstruction* waitPseudo = irBuilder.createPseudoClusterBarrier(
                        PseudoClusterBarrierData::Kind::WaitOnly, insertBefore);
                    if (const auto* mt = wait->getModifier<MemTokenData>()) {
                        waitPseudo->addModifier<MemTokenData>(MemTokenData{mt->tokens});
                    }
                    // Model the loop-counter read so the DAG keeps the placeholder ordered
                    // against the loop-tail decrement (WAR).
                    if (hasLoopCounter) waitPseudo->addSrcReg(loopCounterReg);
                }

                // SignalOnly right before the matching workgroup signal.
                StinkyInstruction* signal = findMatchingWorkgroupBarrierSignal(wait);
                if (signal != nullptr && !signalAlreadyGated(signal)) {
                    StinkyInstruction* signalPseudo = irBuilder.createPseudoClusterBarrier(
                        PseudoClusterBarrierData::Kind::SignalOnly, /*insertBefore=*/signal);
                    if (const auto* mt = signal->getModifier<MemTokenData>()) {
                        signalPseudo->addModifier<MemTokenData>(MemTokenData{mt->tokens});
                    }
                    // Model the loop-counter read so the DAG keeps the placeholder ordered
                    // against the loop-tail decrement (WAR).
                    if (hasLoopCounter) signalPseudo->addSrcReg(loopCounterReg);
                }
            }
        }

        return PreservedAnalyses::none();
    }
};

char InsertPseudoClusterBarrierPassImpl::ID = 0;

}  // namespace

std::unique_ptr<Pass> createInsertPseudoClusterBarrierPass() {
    return std::make_unique<InsertPseudoClusterBarrierPassImpl>();
}

}  // namespace stinkytofu
