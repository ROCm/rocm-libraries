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

/// True if the instruction immediately following \p wait is already a
/// PSEUDO_CLUSTER_BARRIER (idempotency guard for pipeline re-runs).
bool isAlreadyGated(StinkyInstruction* wait, BasicBlock& bb) {
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
            for (StinkyInstruction* wait : anchors) {
                if (isAlreadyGated(wait, bb)) continue;

                // Insert immediately after the workgroup wait: the placeholder
                // goes before whatever currently follows the wait.
                auto after = BasicBlock::iterator(wait);
                ++after;
                IRBase* insertBefore = (after != bb.end()) ? after.getNodePtr() : nullptr;

                // Insert the movable placeholder right after the anchor wait.
                // It models the future SCC clobber (IF_ImplicitWriteSCC) so the
                // scheduler never splits an SCC def->use pair across it, and we
                // copy the anchor wait's MemTokenData onto it so the DAG orders it
                // immediately after that wait (adjacency) via matching LDS
                // pseudo-regs materialized by BuildImplicitDependency.
                StinkyInstruction* pseudo = irBuilder.createPseudoClusterBarrier(
                    PseudoClusterBarrierData::Kind::SignalWait, insertBefore);
                if (const auto* mt = wait->getModifier<MemTokenData>()) {
                    pseudo->addModifier<MemTokenData>(MemTokenData{mt->tokens});
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
