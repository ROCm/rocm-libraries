// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
//      anchor waits by recording predecessor tail drains; WmmaRunWaitMerge
//      (when enabled) moves the waits of a WMMA run onto its first WMMA.
//   4. finalizePlan() replays blocks against the final plan (all counters)
//      with tail-drain-aware entry state so later anchors stay correct.
//   5. emitWaits() materialises the plan as s_wait_* IR nodes.
//   6. removePHIs() strips the PHI pseudo-instructions.
//   7. Optionally, the frame-contract dscnt/tensorcnt waits, or the
//      ds-proximity dscnt pruning of a scheduled loop.

#include "stinkytofu/transforms/asm/StinkyWaitCntInsertionPass.hpp"

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/BBIndexAnalysis.hpp"
#include "stinkytofu/analysis/asm/GirFrameAnalysis.hpp"
#include "stinkytofu/analysis/controlflow/DominanceAnalysis.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"
#include "stinkytofu/transforms/asm/RemoveDscntPass.hpp"
#include "stinkytofu/transforms/asm/waitcnt/GirFrameCounterFlow.hpp"
#include "stinkytofu/transforms/asm/waitcnt/ShallowPredPromotion.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitDataflow.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitPlan.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitPlanOptimizer.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WmmaRunWaitMerge.hpp"

#define DEBUG_TYPE "StinkyWaitCntInsertionPass"

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

void emitGirOneSpec(AsmIRBuilder& builder, GfxArchID arch, StinkyInstruction* anchor,
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

void emitGirPlan(Function& function, PassContext& passCtx, GfxArchID arch,
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
            emitGirOneSpec(builder, arch, inst, wait->second,
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
        emitGirOneSpec(
            builder, arch, anchor, drain.spec,
            lookup(kinds.tails, std::pair{static_cast<const BasicBlock*>(block), CK_DS}),
            lookup(kinds.tails, std::pair{static_cast<const BasicBlock*>(block), CK_Tensor}));
    }
}

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
        WmmaRunWaitMerge wmmaRunMerge;
        std::vector<WaitPlanOptimizer*> optimizers = {&shallowPred};
        if (options.mergeWaitsInWmmaRuns) optimizers.push_back(&wmmaRunMerge);
        for (auto* opt : optimizers) opt->rewrite(plan, df.getResult(), func);

        df.finalizePlan(plan);

        emitWaits(func, passCtx, arch, plan);
        removePHIs(passCtx, rpo);

        if (options.girFrameWaits) {
            AM.invalidate(func, preserveCFGAnalyses());
            insertGirFrameWaits(func, passCtx, AM);
        }
        if (options.pruneDscnt) {
            AM.invalidate(func, preserveCFGAnalyses());
            return createRemoveDscntPass()->run(func, passCtx, AM);
        }
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
        if (spec.loadCount != WaitCountSpec::kUnused) {
            StinkyInstruction* w = builder.create(getMCIDByUOp(GFX::s_wait_loadcnt, arch), anchor);
            w->addSrcReg(StinkyRegister(spec.loadCount));
            SWaitCntData d;
            d.vlcnt = spec.loadCount;
            w->addModifier<SWaitCntData>(d);
        }
        if (spec.kmCount != WaitCountSpec::kUnused) {
            StinkyInstruction* w = builder.create(getMCIDByUOp(GFX::s_wait_kmcnt, arch), anchor);
            w->addSrcReg(StinkyRegister(spec.kmCount));
            SWaitCntData d;
            d.kmcnt = spec.kmCount;
            w->addModifier<SWaitCntData>(d);
        }
        if (spec.tensorCount != WaitCountSpec::kUnused && !options.disableTensorcntInsertion) {
            StinkyInstruction* w =
                builder.create(getMCIDByUOp(GFX::s_wait_tensorcnt, arch), anchor);
            w->addSrcReg(StinkyRegister(spec.tensorCount));
            SWaitTensorCntData d;
            d.tlcnt = spec.tensorCount;
            w->addModifier<SWaitTensorCntData>(d);
            // Tag the wait with the drained loads' memory tokens so downstream passes
            // (e.g. TDMLoadWaveSyncPass) can identify the drained wait group. The
            // tlcnt above is what the hardware waits on.
            if (!spec.tensorTokens.empty()) {
                w->addModifier<MemTokenData>(MemTokenData{spec.tensorTokens});
            }
        }
        if (spec.asyncCount != WaitCountSpec::kUnused) {
            StinkyInstruction* w = builder.create(getMCIDByUOp(GFX::s_wait_asynccnt, arch), anchor);
            w->addSrcReg(StinkyRegister(spec.asyncCount));
            SWaitAsyncCntData d;
            d.asynccnt = spec.asyncCount;
            w->addModifier<SWaitAsyncCntData>(d);
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

bool insertGirFrameWaits(Function& function, PassContext& passCtx, AnalysisManager& AM) {
    const auto& frames = AM.getResult<GirFrameAnalysis>(function);
    const auto& hazards = AM.getResult<GirFrameHazardAnalysis>(function);
    if (frames.empty() || hazards.empty()) return false;

    const auto covers = [&passCtx](const BasicBlock& block) {
        return passCtx.shouldProcessBasicBlock(const_cast<BasicBlock&>(block));
    };
    GirWaitKinds kinds;
    WaitInsertionPlan plan = buildGirFrameWaitPlan(function, frames, hazards, covers, {}, &kinds);
    if (plan.anchorWaits.empty() && plan.tailDrains.empty()) return false;

    // Generic register waits are emitted first so GIR ranks can credit them. A stronger GIR
    // drain can in turn make a later generic wait stale; reconcile that one-way effect against
    // the complete virtual plan before materialising the GIR waits.
    removeObservedDsWaitsRedundantWithPlan(function, plan, covers);

    const GfxArchID arch =
        getGfxArchID(passCtx.getGemmTileConfig().arch[0], passCtx.getGemmTileConfig().arch[1],
                     passCtx.getGemmTileConfig().arch[2]);
    emitGirPlan(function, passCtx, arch, plan, kinds);
    return true;
}
}  // namespace stinkytofu
