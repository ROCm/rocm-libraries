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
        const HwInstDesc* wbDesc = getMCIDByUOp(GFX::global_wb, archId);
        const HwInstDesc* nopDesc = getMCIDByUOp(GFX::v_nop, archId);
        assert(wbDesc && nopDesc && "global_wb/v_nop unavailable on gfx1250");
        if (!wbDesc || !nopDesc) return preserveCFGAnalyses();

        for (BasicBlock& bb : func) {
            for (auto it = bb.begin(); it != bb.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (!inst || isPseudoInst(inst)) continue;

                // First real instruction found: prepend `global_wb SCOPE:SCOPE_CU`
                // then `v_nop` so the emitted order is WB, NOP, <first inst>.
                AsmIRBuilder irBuilder(bb, archId);
                IRBase* insertBefore = it.getNodePtr();

                StinkyInstruction* wb = irBuilder.create(wbDesc, insertBefore);
                wb->addModifier<CacheScopeModifiers>(CacheScopeModifiers(MUBUFScope::SCOPE_CU));

                irBuilder.create(nopDesc, insertBefore);

                PASS_DEBUG(std::cerr << "[InsertInitialUnclausedVmemPass] inserted global_wb/v_nop "
                                     << "prologue in bb=\"" << bb.getLabel() << "\"\n");
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
