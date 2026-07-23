// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// WaitCntValidationCheckPass
//
// Thin, read-only Pass wrapper around WaitCntValidator. It:
//   1. builds the SSA def-use chain (includePseudo=true) so register RAW and
//      LDS memtoken dependencies both appear as def-use edges,
//   2. runs the WaitCntValidator over the RPO block list, and
//   3. on any missing wait, prints a diagnostic and aborts via
//      report_fatal_error (the same failure model as
//      MemTokenConsistencyCheckPass).
//
// The def-use build materialises PHI pseudo-instructions; they are stripped
// again at the end so the check leaves the observable IR unchanged.

#include "stinkytofu/transforms/asm/WaitCntValidationCheckPass.hpp"

#include <iostream>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/BBIndexAnalysis.hpp"
#include "stinkytofu/analysis/controlflow/DominanceAnalysis.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitCntValidator.hpp"

namespace {
using namespace stinkytofu;

class WaitCntValidationCheckPass : public StinkyInstPass {
   public:
    static char ID;

    const char* getName() const override {
        return "WaitCntValidationCheckPass";
    }

    PassID getPassID() const override {
        return &WaitCntValidationCheckPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        const auto& domInfo = AM.getResult<DominanceAnalysis>(func);
        buildUseDefChain(func, domInfo, /*clearExisting=*/true, /*includePseudo=*/true);
        const auto& rpo = AM.getResult<BBIndexAnalysis>(func).rpo;

        const unsigned numWaves = passCtx.getGemmTileConfig().NumWaves;
        waitcnt::WaitCntValidator validator(numWaves);
        auto violations = validator.validate(func, rpo);

        // Strip the PHIs materialised by buildUseDefChain so the check does not
        // perturb the IR that downstream printers / passes observe.
        removePHIs(func);

        if (!violations.empty()) {
            std::cerr << "[WaitCntValidationCheck] ERROR: " << violations.size()
                      << " missing s_wait_* dependency violation(s):\n";
            for (const auto& v : violations) {
                std::cerr << "  MISSING: " << v.message << "\n";
            }
            report_fatal_error("missing s_wait_* for in-flight async memory dependency");
        }

        return preserveCFGAnalyses();
    }

   private:
    static void removePHIs(Function& func) {
        for (BasicBlock& bb : func) {
            for (auto it = bb.begin(); it != bb.end();) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst != nullptr && inst->getUnifiedOpcode() == GFX::PHI) {
                    it = bb.eraseIR(it);
                } else {
                    ++it;
                }
            }
        }
    }
};

char WaitCntValidationCheckPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createWaitCntValidationCheckPass() {
    return std::make_unique<WaitCntValidationCheckPass>();
}
}  // namespace stinkytofu
