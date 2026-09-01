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
#include "stinkytofu/transforms/ssa/ReplayLegacyColoringPass.hpp"

#include <memory>
#include <string>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/ssa/SSAAllocation.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/support/OptimizationRemark.hpp"
#include "stinkytofu/transforms/ssa/SSADestruction.hpp"

#define DEBUG_TYPE "ReplayLegacyColoringPass"

namespace stinkytofu {
namespace {

constexpr const char* kPassName = "ReplayLegacyColoring";

class ReplayLegacyColoringPassImpl : public Pass {
   public:
    static char ID;

    const char* getName() const override {
        return "Replay Legacy Coloring";
    }

    PassID getPassID() const override {
        return &ReplayLegacyColoringPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager&) override {
        if (!func.hasAttachedSSA()) {
            emitRemark(passCtx, {OptimizationRemark::Kind::Missed, kPassName, "NoAttachedSSA",
                                 "@" + func.getName() + ": no attached SSA; nothing to lower"});
            return preserveCFGAnalyses();
        }

        const SSADestructionResult result = replayLegacyColoring(func);
        if (!result.ok()) {
            PASS_DEBUG(std::cerr << "ReplayLegacyColoring: " << result.toString() << "\n");
            emitRemark(passCtx, {OptimizationRemark::Kind::Missed, kPassName, "NotLowered",
                                 result.toString()});
            return preserveCFGAnalyses();
        }

        emitRemark(passCtx, {OptimizationRemark::Kind::Passed, kPassName, "ReplayedLegacyColoring",
                             "@" + func.getName() +
                                 ": lowered attached SSA back to its original registers"});
        return preserveCFGAnalyses();
    }
};

char ReplayLegacyColoringPassImpl::ID = 0;

}  // namespace

SSADestructionResult replayLegacyColoring(Function& function) {
    return destroyAttachedSSA(function, createLegacyColoring(function));
}

std::unique_ptr<Pass> createReplayLegacyColoringPass() {
    return std::make_unique<ReplayLegacyColoringPassImpl>();
}

}  // namespace stinkytofu
