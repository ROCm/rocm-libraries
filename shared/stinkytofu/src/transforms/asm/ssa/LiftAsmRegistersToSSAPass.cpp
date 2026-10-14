/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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
#include "stinkytofu/transforms/asm/ssa/LiftAsmRegistersToSSAPass.hpp"

#include <memory>
#include <string>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/controlflow/Dominance.hpp"
#include "stinkytofu/analysis/controlflow/DominanceAnalysis.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/support/OptimizationRemark.hpp"

#define DEBUG_TYPE "LiftAsmRegistersToSSAPass"

namespace stinkytofu {
namespace {

constexpr const char* kPassName = "LiftAsmRegistersToSSA";

class LiftAsmRegistersToSSAPassImpl : public Pass {
   public:
    static char ID;

    explicit LiftAsmRegistersToSSAPassImpl(const LiftAsmRegistersToSSAOptions& options)
        : options_(options) {}

    const char* getName() const override {
        return "Lift Asm Registers to SSA";
    }

    PassID getPassID() const override {
        return &LiftAsmRegistersToSSAPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        // Re-lifting replaces any prior attached SSA state.
        func.clearAttachedSSA();

        if (const BasicBlock* excluded = findExcludedBlock(func, passCtx)) {
            const std::string reason = "basic-block filtering excludes ^" + excluded->getLabel() +
                                       "; SSA lifting needs the whole function";
            missed(passCtx, reason);
            return preserveCFGAnalyses();
        }

        const DominanceInfo& dominance = AM.getResult<DominanceAnalysis>(func);
        Expected<LiftAttachedSSAResult> lifted =
            liftAsmRegistersToAttachedSSA(func, dominance, options_);
        if (lifted.hasError()) {
            const std::string& reason = lifted.getError();
            PASS_DEBUG(std::cerr << "LiftAsmRegistersToSSA: " << reason << "\n");
            missed(passCtx, reason);
            return preserveCFGAnalyses();
        }

        const size_t values = lifted->valueCount;
        const size_t blockArgs = lifted->blockArgumentCount;

        emitRemark(passCtx,
                   {OptimizationRemark::Kind::Passed, kPassName, "LiftedToSSA",
                    "@" + func.getName() + ": lifted " + std::to_string(values) +
                        " SSA value(s) and " + std::to_string(blockArgs) + " block argument(s)"});
        return preserveCFGAnalyses();
    }

   private:
    /// First block the pipeline asked us to skip, or null when all are in scope.
    static const BasicBlock* findExcludedBlock(const Function& func, const PassContext& passCtx) {
        for (const BasicBlock& bb : func) {
            if (!passCtx.shouldProcessBasicBlock(bb)) return &bb;
        }
        return nullptr;
    }

    static void missed(const PassContext& passCtx, const std::string& message) {
        emitRemark(passCtx, {OptimizationRemark::Kind::Missed, kPassName, "NotLifted", message});
    }

    LiftAsmRegistersToSSAOptions options_;
};

char LiftAsmRegistersToSSAPassImpl::ID = 0;

}  // namespace

bool kernelHasCallSites(const std::vector<const Function*>& functions) {
    for (const Function* function : functions) {
        if (function == nullptr) continue;
        for (const BasicBlock& bb : *function) {
            for (const IRBase& ir : bb) {
                const auto* instruction = dyn_cast<StinkyInstruction>(&ir);
                if (instruction == nullptr || instruction->getHwInstDesc() == nullptr) continue;
                if (isCall(*instruction)) return true;
            }
        }
    }
    return false;
}

std::unique_ptr<Pass> createLiftAsmRegistersToSSAPass(const LiftAsmRegistersToSSAOptions& options) {
    return std::make_unique<LiftAsmRegistersToSSAPassImpl>(options);
}

}  // namespace stinkytofu
