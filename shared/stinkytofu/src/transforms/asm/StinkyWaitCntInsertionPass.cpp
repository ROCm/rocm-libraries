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

// StinkyWaitCntInsertionPass
//
// Inserts s_wait_dscnt / s_wait_loadcnt / s_wait_tensorcnt so that
// asynchronous memory operations complete before their results are
// consumed.
//
// Pipeline:
//   1. buildUseDefChain(includePseudo=true) so memtoken pseudo-registers
//      become first-class SSA edges (the implicit-dependency pass must
//      have already materialised them as pseudo-reg operands).
//   2. WaitDataflow.solve() computes a sound per-consumer wait plan
//      via forward dataflow with per-pred queues.
//   3. ShallowPredPromotion (and any other WaitPlanOptimizer) may relax
//      anchor waits by recording predecessor tail drains.
//   4. finalizePlan() replays blocks against the final plan (all counters)
//      with tail-drain-aware entry state so later anchors stay correct.
//   5. emitWaits() materialises the plan as s_wait_* IR nodes.
//   6. removePHIs() strips the PHI pseudo-instructions.

#include "stinkytofu/transforms/asm/StinkyWaitCntInsertionPass.hpp"

#include <memory>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/BBIndexAnalysis.hpp"
#include "stinkytofu/analysis/controlflow/DominanceAnalysis.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"
#include "stinkytofu/transforms/asm/waitcnt/ShallowPredPromotion.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitDataflow.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitPlan.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitPlanOptimizer.hpp"

#define DEBUG_TYPE "StinkyWaitCntInsertionPass"

namespace {
using namespace stinkytofu;
using namespace stinkytofu::waitcnt;

class StinkyWaitCntInsertionPass : public StinkyInstPass {
   public:
    static char ID;

    explicit StinkyWaitCntInsertionPass(WaitCntInsertionOptions options) : options(options) {}

    const char* getName() const override {
        return "StinkyWaitCntInsertionPass";
    }
    Pass::ID getPassID() const override {
        return &StinkyWaitCntInsertionPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        GfxArchID arch =
            getGfxArchID(passCtx.getGemmTileConfig().arch[0], passCtx.getGemmTileConfig().arch[1],
                         passCtx.getGemmTileConfig().arch[2]);

        const auto& domInfo = AM.getResult<DominanceAnalysis>(func);
        buildUseDefChain(func, domInfo, /*clearExisting=*/true, /*includePseudo=*/true);
        const auto& rpo = AM.getResult<BBIndexAnalysis>(func).rpo;

        // Dataflow must see every block so a skipped pred still contributes
        // its in-flight state to successors. PassContext gating only applies
        // to IR mutation below.
        WaitDataflow df(func, domInfo, rpo);
        df.setLoopCarriedTokenDepsEnabled(options.enableLoopCarriedTokenDeps);

        // Tensor counter drains only at barriers or when there is a single wave.
        const auto numWaves = passCtx.getGemmTileConfig().NumWaves;
        df.setRawNeedsWait(CK_Tensor, [numWaves](const StinkyInstruction& i) {
            return isBarrier(i) || numWaves == 1;
        });

        df.solve();
        WaitInsertionPlan plan = df.materializePlan();

        ShallowPredPromotion shallowPred;
        std::vector<WaitPlanOptimizer*> optimizers = {&shallowPred};
        for (auto* opt : optimizers) opt->rewrite(plan, df.getResult(), func);

        df.finalizePlan(plan);

        emitWaits(func, passCtx, arch, plan);
        removePHIs(passCtx, rpo);
        return preserveCFGAnalyses();
    }

   private:
    WaitCntInsertionOptions options;

    void emitWaits(Function& func, PassContext& passCtx, GfxArchID arch,
                   const WaitInsertionPlan& plan) {
        // Anchor waits: walk blocks/instructions in program order so the
        // insertion is deterministic when multiple anchors live in the
        // same block.
        for (BasicBlock& bb : func) {
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;
            AsmIRBuilder builder(bb, arch);
            for (IRBase& ir : bb) {
                auto* inst = dyn_cast<StinkyInstruction>(&ir);
                if (inst == nullptr) continue;
                auto it = plan.anchorWaits.find(inst);
                if (it == plan.anchorWaits.end()) continue;
                emitOneSpec(builder, arch, inst, it->second);
            }
        }

        // Tail drains: one block-relative anchor (the terminator branch, if
        // any) per requesting predecessor.
        for (const auto& drain : plan.tailDrains) {
            BasicBlock* pred = drain.predBB;
            if (pred == nullptr || !passCtx.shouldProcessBasicBlock(*pred)) continue;
            AsmIRBuilder builder(*pred, arch);
            IRBase* term = pred->getTerminator();
            StinkyInstruction* termInst = term ? dyn_cast<StinkyInstruction>(term) : nullptr;
            StinkyInstruction* anchor =
                (termInst != nullptr && isBranch(*termInst)) ? termInst : nullptr;
            emitOneSpec(builder, arch, anchor, drain.spec);
        }
    }

    void emitOneSpec(AsmIRBuilder& builder, GfxArchID arch, StinkyInstruction* anchor,
                     const WaitCountSpec& spec) {
        if (spec.dsCount != WaitCountSpec::kUnused) {
            StinkyInstruction* w = builder.create(getMCIDByUOp(GFX::s_wait_dscnt, arch), anchor);
            w->addSrcReg(StinkyRegister(spec.dsCount));
            SWaitCntData d;
            d.dlcnt = spec.dsCount;
            w->addModifier<SWaitCntData>(d);
        }
        if (spec.bufferCount != WaitCountSpec::kUnused) {
            StinkyInstruction* w = builder.create(getMCIDByUOp(GFX::s_wait_loadcnt, arch), anchor);
            w->addSrcReg(StinkyRegister(spec.bufferCount));
            SWaitCntData d;
            d.vlcnt = spec.bufferCount;
            w->addModifier<SWaitCntData>(d);
        }
        if (spec.kmCount != WaitCountSpec::kUnused) {
            StinkyInstruction* w = builder.create(getMCIDByUOp(GFX::s_wait_kmcnt, arch), anchor);
            w->addSrcReg(StinkyRegister(spec.kmCount));
            SWaitCntData d;
            d.kmcnt = spec.kmCount;
            w->addModifier<SWaitCntData>(d);
        }
        if (spec.tensorCount != WaitCountSpec::kUnused) {
            StinkyInstruction* w =
                builder.create(getMCIDByUOp(GFX::s_wait_tensorcnt, arch), anchor);
            w->addSrcReg(StinkyRegister(spec.tensorCount));
            SWaitTensorCntData d;
            d.tlcnt = spec.tensorCount;
            w->addModifier<SWaitTensorCntData>(d);
        }
    }

    void removePHIs(PassContext& passCtx, const std::vector<BasicBlock*>& rpo) {
        for (auto* bb : rpo) {
            if (!passCtx.shouldProcessBasicBlock(*bb)) continue;
            for (auto it = bb->begin(); it != bb->end();) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst && inst->getUnifiedOpcode() == GFX::PHI) {
                    it = bb->eraseIR(it);
                } else {
                    ++it;
                }
            }
        }
    }
};

char StinkyWaitCntInsertionPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createStinkyWaitCntInsertionPass(WaitCntInsertionOptions options) {
    return std::make_unique<StinkyWaitCntInsertionPass>(options);
}
}  // namespace stinkytofu
