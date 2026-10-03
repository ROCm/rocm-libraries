/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "stinkytofu/transforms/asm/GirWaitCntInsertionPass.hpp"

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/asm/GirFrameAnalysis.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/transforms/asm/waitcnt/GirFrameCounterFlow.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitPlan.hpp"

namespace {
using namespace stinkytofu;
using namespace stinkytofu::waitcnt;

void emitOneSpec(AsmIRBuilder& builder, GfxArchID arch, StinkyInstruction* anchor,
                 const WaitCountSpec& spec) {
    if (spec.dsCount != WaitCountSpec::kUnused) {
        StinkyInstruction* wait = builder.create(getMCIDByUOp(GFX::s_wait_dscnt, arch), anchor);
        wait->addSrcReg(StinkyRegister(spec.dsCount));
        SWaitCntData data;
        data.dlcnt = spec.dsCount;
        wait->addModifier<SWaitCntData>(data);
    }
    if (spec.tensorCount != WaitCountSpec::kUnused) {
        StinkyInstruction* wait = builder.create(getMCIDByUOp(GFX::s_wait_tensorcnt, arch), anchor);
        wait->addSrcReg(StinkyRegister(spec.tensorCount));
        SWaitTensorCntData data;
        data.tlcnt = spec.tensorCount;
        wait->addModifier<SWaitTensorCntData>(data);
        if (!spec.tensorTokens.empty())
            wait->addModifier<MemTokenData>(MemTokenData{spec.tensorTokens});
    }
}

void emitPlan(Function& function, PassContext& passCtx, GfxArchID arch,
              const WaitInsertionPlan& plan) {
    for (BasicBlock& block : function) {
        if (!passCtx.shouldProcessBasicBlock(block)) continue;
        AsmIRBuilder builder(block, arch);
        for (IRBase& node : block) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (!inst) continue;
            auto wait = plan.anchorWaits.find(inst);
            if (wait != plan.anchorWaits.end()) emitOneSpec(builder, arch, inst, wait->second);
        }
    }
    for (const auto& drain : plan.tailDrains) {
        BasicBlock* block = drain.predBB;
        if (!block || !passCtx.shouldProcessBasicBlock(*block)) continue;
        AsmIRBuilder builder(*block, arch);
        IRBase* terminator = block->getTerminator();
        auto* anchor = terminator ? dyn_cast<StinkyInstruction>(terminator) : nullptr;
        if (anchor && !isBranch(*anchor)) anchor = nullptr;
        emitOneSpec(builder, arch, anchor, drain.spec);
    }
}

class GirWaitCntInsertionPass final : public StinkyInstPass {
   public:
    static char ID;

    const char* getName() const override {
        return "GirWaitCntInsertionPass";
    }
    PassID getPassID() const override {
        return &ID;
    }

    PreservedAnalyses run(Function& function, PassContext& passCtx, AnalysisManager& AM) override {
        const auto& frames = AM.getResult<GirFrameAnalysis>(function);
        const auto& hazards = AM.getResult<GirFrameHazardAnalysis>(function);
        if (frames.empty() || hazards.empty()) return PreservedAnalyses::all();

        WaitInsertionPlan plan =
            buildGirFrameWaitPlan(function, frames, hazards, [&passCtx](const BasicBlock& block) {
                return passCtx.shouldProcessBasicBlock(const_cast<BasicBlock&>(block));
            });
        if (plan.anchorWaits.empty() && plan.tailDrains.empty()) return PreservedAnalyses::all();

        const GfxArchID arch =
            getGfxArchID(passCtx.getGemmTileConfig().arch[0], passCtx.getGemmTileConfig().arch[1],
                         passCtx.getGemmTileConfig().arch[2]);
        emitPlan(function, passCtx, arch, plan);
        return preserveCFGAnalyses();
    }
};

char GirWaitCntInsertionPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createGirWaitCntInsertionPass() {
    return std::make_unique<GirWaitCntInsertionPass>();
}
}  // namespace stinkytofu
