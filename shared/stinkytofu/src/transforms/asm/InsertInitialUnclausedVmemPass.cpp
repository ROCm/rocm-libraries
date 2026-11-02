// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/InsertInitialUnclausedVmemPass.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/transforms/asm/ra/RegisterBudget.hpp"

#define DEBUG_TYPE "InsertInitialUnclausedVmemPass"

namespace {
using namespace stinkytofu;

// Default SADDR pair when compact has not rewritten SGPRs: s[64:65]. The
// hardware does not initialize these at wave launch, so they hold nothing
// useful at kernel entry and can be zeroed without clobbering live state.
// After SGPR compact those indexes may already hold values, so the pair is
// taken from the final numbering instead.
constexpr uint32_t kPrologueSaddrIdx = 64;
constexpr uint32_t kPrologueSaddrWidth = 2;

class InsertInitialUnclausedVmemPass : public Pass {
   public:
    static char ID;

    const char* getName() const override {
        return "Insert Initial Unclaused Vmem";
    }

    Pass::ID getPassID() const override {
        return &InsertInitialUnclausedVmemPass::ID;
    }

    void setAfterSgprCompact(bool on) {
        m_afterSgprCompact = on;
    }

    // Runs on the entry function. Callable functions have been merged into the
    // entry by FlattenCalleesPass by the time this pass runs, so the entry's
    // first real instruction is the kernel's first executed instruction.
    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        const auto arch = passCtx.getGemmTileConfig().arch;

        // gfx1250-only. The pass is wired into the gfx1250 pipeline, but it is
        // also registered in stinkytofu-opt where it can be invoked with any
        // --arch. No-op on other architectures so it never emits
        // gfx1250-specific opcodes on a target that lacks them.
        if (arch != std::array<int, 3>{12, 5, 0}) return preserveCFGAnalyses();

        const GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);

        // All three opcodes exist on gfx1250; guard defensively so a missing
        // descriptor no-ops instead of passing nullptr into create().
        const HwInstDesc* movDesc = getMCIDByUOp(GFX::s_mov_b64, archId);
        const HwInstDesc* prefetchDesc = getMCIDByUOp(GFX::global_prefetch_b8, archId);
        const HwInstDesc* nopDesc = getMCIDByUOp(GFX::v_nop, archId);
        assert(movDesc && prefetchDesc && nopDesc &&
               "s_mov_b64/global_prefetch_b8/v_nop unavailable on gfx1250");
        if (!movDesc || !prefetchDesc || !nopDesc) return preserveCFGAnalyses();

        for (BasicBlock& bb : func) {
            for (auto it = bb.begin(); it != bb.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (!inst || isPseudoInst(inst)) continue;

                // First real instruction found: prepend, in this order,
                //   s_mov_b64 s[base:base+1], 0
                //   v_nop
                //   global_prefetch_b8 v0, [sbase, sbase+1] scope:SCOPE_SE th:TH_LOAD_RT
                // so the emitted order is MOV, NOP, PREFETCH, <first inst>. The
                // v_nop covers the write-to-use delay between writing the SADDR
                // pair and the global op that reads it.
                AsmIRBuilder irBuilder(bb, archId);
                IRBase* insertBefore = it.getNodePtr();

                // The prologue runs before anything else, so it can reuse a pair
                // from inside the kernel's own range: nothing above the
                // dispatch-filled line holds a value yet. It takes the top such
                // pair, leaving the block below for the abs-prefetch burst that
                // follows it (see absPrefetchEntrySgprBase).
                uint32_t saddrIdx = kPrologueSaddrIdx;
                if (m_afterSgprCompact)
                    saddrIdx = reusableEvenSgprBase(func, kPrologueSaddrWidth)
                                   .value_or(nextEvenRegisterBase(func, RegType::S));
                const StinkyRegister saddr(RegType::S, saddrIdx, kPrologueSaddrWidth);

                StinkyInstruction* mov = irBuilder.create(movDesc, insertBefore);
                mov->addDestReg(saddr);
                mov->addSrcReg(StinkyRegister(0));

                irBuilder.create(nopDesc, insertBefore);

                StinkyInstruction* prefetch = irBuilder.create(prefetchDesc, insertBefore);
                prefetch->addSrcReg(StinkyRegister(RegType::V, 0, 1));
                prefetch->addSrcReg(saddr);
                prefetch->addModifier<GLOBALModifiers>(
                    GLOBALModifiers(/*offset=*/0, TemporalHint::TH_RT, MUBUFScope::SCOPE_SE));

                PASS_DEBUG(std::cerr << "[InsertInitialUnclausedVmemPass] inserted "
                                     << "s_mov_b64/v_nop/global_prefetch_b8 prologue in bb=\""
                                     << bb.getLabel() << "\"\n");
                return preserveCFGAnalyses();
            }
        }
        return preserveCFGAnalyses();
    }

   private:
    bool m_afterSgprCompact = false;
};

char InsertInitialUnclausedVmemPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createInsertInitialUnclausedVmemPass(bool afterSgprCompact) {
    auto p = std::make_unique<InsertInitialUnclausedVmemPass>();
    p->setAfterSgprCompact(afterSgprCompact);
    return p;
}
}  // namespace stinkytofu
