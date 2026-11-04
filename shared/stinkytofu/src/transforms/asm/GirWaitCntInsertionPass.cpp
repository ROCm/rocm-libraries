/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "stinkytofu/transforms/asm/GirWaitCntInsertionPass.hpp"

#include <set>
#include <string>
#include <utility>

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

/// `GIR RAW`, `GIR WAR`, or every kind that required this one wait.
std::string girWaitStamp(const std::set<GirHazardKind>* kinds) {
    if (!kinds || kinds->empty()) return "GIR";
    std::string result = "GIR ";
    bool first = true;
    for (GirHazardKind kind : *kinds) {
        if (!first) result += ",";
        result += kind == GirHazardKind::RAW ? "RAW" : kind == GirHazardKind::WAR ? "WAR" : "WAW";
        first = false;
    }
    return result;
}

void emitOneSpec(AsmIRBuilder& builder, GfxArchID arch, StinkyInstruction* anchor,
                 const WaitCountSpec& spec, const std::set<GirHazardKind>* dsKinds,
                 const std::set<GirHazardKind>* tensorKinds) {
    if (spec.dsCount != WaitCountSpec::kUnused) {
        StinkyInstruction* wait = builder.create(getMCIDByUOp(GFX::s_wait_dscnt, arch), anchor);
        wait->addSrcReg(StinkyRegister(spec.dsCount));
        SWaitCntData data;
        data.dlcnt = spec.dsCount;
        wait->addModifier<SWaitCntData>(data);
        wait->addModifier<CommentData>(CommentData{girWaitStamp(dsKinds)});
    }
    if (spec.tensorCount != WaitCountSpec::kUnused) {
        StinkyInstruction* wait = builder.create(getMCIDByUOp(GFX::s_wait_tensorcnt, arch), anchor);
        wait->addSrcReg(StinkyRegister(spec.tensorCount));
        SWaitTensorCntData data;
        data.tlcnt = spec.tensorCount;
        wait->addModifier<SWaitTensorCntData>(data);
        if (!spec.tensorTokens.empty())
            wait->addModifier<MemTokenData>(MemTokenData{spec.tensorTokens});
        wait->addModifier<CommentData>(CommentData{girWaitStamp(tensorKinds)});
    }
}

void emitPlan(Function& function, PassContext& passCtx, GfxArchID arch,
              const WaitInsertionPlan& plan, const GirWaitKinds& kinds) {
    const auto lookup = [](const auto& table, const auto& key) {
        auto found = table.find(key);
        return found == table.end() ? nullptr : &found->second;
    };
    for (BasicBlock& block : function) {
        if (!passCtx.shouldProcessBasicBlock(block)) continue;
        AsmIRBuilder builder(block, arch);
        for (IRBase& node : block) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (!inst) continue;
            auto wait = plan.anchorWaits.find(inst);
            if (wait == plan.anchorWaits.end()) continue;
            emitOneSpec(builder, arch, inst, wait->second,
                        lookup(kinds.anchors, std::pair{inst, CK_DS}),
                        lookup(kinds.anchors, std::pair{inst, CK_Tensor}));
        }
    }
    for (const auto& drain : plan.tailDrains) {
        BasicBlock* block = drain.predBB;
        if (!block || !passCtx.shouldProcessBasicBlock(*block)) continue;
        AsmIRBuilder builder(*block, arch);
        IRBase* terminator = block->getTerminator();
        auto* anchor = terminator ? dyn_cast<StinkyInstruction>(terminator) : nullptr;
        if (anchor && !isBranch(*anchor)) anchor = nullptr;
        emitOneSpec(
            builder, arch, anchor, drain.spec,
            lookup(kinds.tails, std::pair{static_cast<const BasicBlock*>(block), CK_DS}),
            lookup(kinds.tails, std::pair{static_cast<const BasicBlock*>(block), CK_Tensor}));
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

        const auto covers = [&passCtx](const BasicBlock& block) {
            return passCtx.shouldProcessBasicBlock(const_cast<BasicBlock&>(block));
        };
        GirWaitKinds kinds;
        WaitInsertionPlan plan =
            buildGirFrameWaitPlan(function, frames, hazards, covers, {}, &kinds);
        if (plan.anchorWaits.empty() && plan.tailDrains.empty()) return PreservedAnalyses::all();

        // Generic register waits are emitted first so GIR ranks can credit them. A stronger GIR
        // drain can in turn make a later generic wait stale; reconcile that one-way effect against
        // the complete virtual plan before materialising the GIR waits.
        removeObservedDsWaitsRedundantWithPlan(function, plan, covers);

        const GfxArchID arch =
            getGfxArchID(passCtx.getGemmTileConfig().arch[0], passCtx.getGemmTileConfig().arch[1],
                         passCtx.getGemmTileConfig().arch[2]);
        emitPlan(function, passCtx, arch, plan, kinds);
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
