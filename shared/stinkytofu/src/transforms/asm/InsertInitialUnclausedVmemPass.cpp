// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/InsertInitialUnclausedVmemPass.hpp"

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
        const GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);

        for (BasicBlock& bb : func) {
            for (auto it = bb.begin(); it != bb.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (!inst || isPseudoInst(inst)) continue;

                // First real instruction found: prepend `global_wb SCOPE:SCOPE_CU`
                // then `v_nop` so the emitted order is WB, NOP, <first inst>.
                AsmIRBuilder irBuilder(bb, archId);
                IRBase* insertBefore = it.getNodePtr();

                StinkyInstruction* wb =
                    irBuilder.create(getMCIDByUOp(GFX::global_wb, archId), insertBefore);
                wb->addModifier<CacheScopeModifiers>(CacheScopeModifiers(MUBUFScope::SCOPE_CU));

                irBuilder.create(getMCIDByUOp(GFX::v_nop, archId), insertBefore);

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
