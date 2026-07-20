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

#include <unordered_set>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {
namespace {

/// Workgroup-scope split-barrier literal id. `s_barrier_wait -1` is the
/// workgroup completion the cluster handshake anchors on; the cluster-scope
/// `-3` wait our expansion synthesizes is intentionally not matched here.
constexpr int kWorkgroupBarrierId = -1;

/// True if `inst` is a workgroup-scope barrier completion: `s_barrier_wait -1`.
bool isWorkgroupBarrierWait(const StinkyInstruction& inst) {
    return isBarrierWait(inst) && isSplitBarrierAllWave(inst);
}

/// Walk backward from \p anchor (exclusive) toward the containing basic block's
/// entry to find the nearest preceding `s_barrier_wait -1`. Stops at the BB
/// boundary so the trigger never crosses a CFG edge.
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
            // one placeholder. Gather first, insert after, to avoid mutating the
            // block while iterating it.
            std::unordered_set<StinkyInstruction*> seen;
            std::vector<StinkyInstruction*> anchors;
            for (auto it = bb.begin(); it != bb.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst == nullptr || !isTensorLoad(*inst)) continue;
                StinkyInstruction* wait = findPrecedingWorkgroupBarrierWaitInBB(inst);
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

                // The placeholder carries IF_HasSideEffect, so the DAG scheduler
                // treats it as a non-movable region boundary: its position is
                // strictly preserved right here, immediately after the anchor
                // `s_barrier_wait -1`. Being a boundary also stops the scheduler
                // from moving any instruction across it, so no SCC (or other)
                // def->use chain is split around the future expansion point.
                irBuilder.createPseudoClusterBarrier(PseudoClusterBarrierData::Kind::SignalWait,
                                                     insertBefore);
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
