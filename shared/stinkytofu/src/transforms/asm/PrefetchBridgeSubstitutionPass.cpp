// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/PrefetchBridgeSubstitutionPass.hpp"

#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/PrefetchBridgePlanner.hpp"

#define DEBUG_TYPE "PrefetchBridgeSubstitutionPass"

namespace {
using namespace stinkytofu;

int rewriteToFlat(const std::unordered_set<const StinkyInstruction*>& chosen,
                  const HwInstDesc* flatDesc) {
    int substituted = 0;
    for (const StinkyInstruction* chosenPf : chosen) {
        // The planner only reads; the instructions are this function's own.
        auto* pf = const_cast<StinkyInstruction*>(chosenPf);
        pf->updateHwInstDesc(flatDesc);
        // A null saddr is spelled "off" in the GLOBAL syntax and omitted in the FLAT
        // one, so the operand has to go with the opcode. The encoding is the same;
        // only the spelling differs. An SGPR saddr is kept: FLAT takes it on gfx1250.
        std::vector<StinkyRegister> srcs;
        for (const StinkyRegister& src : pf->getSrcRegs()) {
            if (src.dataType == StinkyRegister::Type::LiteralString && src.literalValue == "off")
                continue;
            srcs.push_back(src);
        }
        pf->setSrcRegs(srcs);
        ++substituted;
    }
    return substituted;
}

class PrefetchBridgeSubstitutionPass : public Pass {
   public:
    static char ID;

    const char* getName() const override {
        return "Prefetch Bridge Substitution";
    }

    Pass::ID getPassID() const override {
        return &PrefetchBridgeSubstitutionPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        const int required = passCtx.getHWModel().waitHide.vmVsrcBridge;
        if (required <= 0) return preserveCFGAnalyses();

        const auto archTriple = passCtx.getGemmTileConfig().arch;
        const GfxArchID arch = getGfxArchID(archTriple[0], archTriple[1], archTriple[2]);
        const HwInstDesc* flatDesc = getMCIDByUOp(GFX::flat_prefetch_b8, arch);
        if (flatDesc == nullptr) return preserveCFGAnalyses();

        std::vector<std::vector<const StinkyInstruction*>> orders;
        orders.reserve(func.size());
        std::vector<BlockOrder> layout;
        for (BasicBlock& bb : func) {
            auto& order = orders.emplace_back();
            for (IRBase& node : bb)
                if (auto* inst = dyn_cast<StinkyInstruction>(&node)) order.push_back(inst);
            layout.push_back({&bb, &order});
        }
        const int substituted = rewriteToFlat(planPrefetchBridge(layout, required), flatDesc);

        PASS_DEBUG(std::cerr << "[PrefetchBridge] substituted " << substituted
                             << " prefetch(es)\n");
        return preserveCFGAnalyses();
    }
};

char PrefetchBridgeSubstitutionPass::ID = 0;

}  // namespace

namespace stinkytofu {

std::unique_ptr<Pass> createPrefetchBridgeSubstitutionPass() {
    return std::make_unique<PrefetchBridgeSubstitutionPass>();
}

}  // namespace stinkytofu
