// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/InsertInitialUnclausedVmemPass.hpp"

#include <array>
#include <cassert>
#include <iostream>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"

#define DEBUG_TYPE "InsertInitialUnclausedVmemPass"

namespace {
using namespace stinkytofu;

class InsertInitialUnclausedVmemPass : public Pass {
   public:
    static char ID;

    const char* getName() const override {
        return "Insert Initial Unclaused Vmem";
    }

    Pass::ID getPassID() const override {
        return &InsertInitialUnclausedVmemPass::ID;
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

        // Both opcodes exist on gfx1250; guard defensively so a missing
        // descriptor no-ops instead of passing nullptr into create().
        const HwInstDesc* prefetchDesc = getMCIDByUOp(GFX::global_prefetch_b8, archId);
        const HwInstDesc* nopDesc = getMCIDByUOp(GFX::v_nop, archId);
        assert(prefetchDesc && nopDesc && "global_prefetch_b8/v_nop unavailable on gfx1250");
        if (!prefetchDesc || !nopDesc) return preserveCFGAnalyses();

        for (BasicBlock& bb : func) {
            for (auto it = bb.begin(); it != bb.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (!inst || isPseudoInst(inst)) continue;

                // First real instruction found: prepend
                // `global_prefetch_b8 v0, [s0, s1] scope:SCOPE_SE th:TH_LOAD_RT`
                // then `v_nop` so the emitted order is PREFETCH, NOP, <first inst>.
                AsmIRBuilder irBuilder(bb, archId);
                IRBase* insertBefore = it.getNodePtr();

                StinkyInstruction* prefetch = irBuilder.create(prefetchDesc, insertBefore);
                prefetch->addSrcReg(StinkyRegister(RegType::V, 0, 1));
                prefetch->addSrcReg(StinkyRegister(RegType::S, 0, 2));
                prefetch->addModifier<GLOBALModifiers>(
                    GLOBALModifiers(/*offset=*/0, TemporalHint::TH_RT, MUBUFScope::SCOPE_SE));

                irBuilder.create(nopDesc, insertBefore);

                PASS_DEBUG(std::cerr << "[InsertInitialUnclausedVmemPass] inserted "
                                     << "global_prefetch_b8/v_nop prologue in bb=\""
                                     << bb.getLabel() << "\"\n");
                return preserveCFGAnalyses();
            }
        }
        return preserveCFGAnalyses();
    }
};

char InsertInitialUnclausedVmemPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createInsertInitialUnclausedVmemPass() {
    return std::make_unique<InsertInitialUnclausedVmemPass>();
}
}  // namespace stinkytofu
