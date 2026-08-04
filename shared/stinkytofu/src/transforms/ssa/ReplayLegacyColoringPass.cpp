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
#include "stinkytofu/analysis/ssa/CanonicalSSA.hpp"
#include "stinkytofu/analysis/ssa/CanonicalSSAAllocation.hpp"
#include "stinkytofu/analysis/ssa/CanonicalSSAAnalysis.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/support/OptimizationRemark.hpp"
#include "stinkytofu/transforms/ssa/CanonicalSSADestruction.hpp"

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

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        // The cached result only: lowering must apply the graph an allocator
        // already saw, so lifting one here on demand would be answering a
        // different question.
        const CanonicalSSAAnalysis::Result* cached = AM.getCachedResult<CanonicalSSAAnalysis>();
        if (cached == nullptr || cached->hasError()) {
            const std::string why = cached == nullptr
                                        ? "no canonical SSA attached; nothing to lower"
                                        : "not lifted: " + cached->getError();
            emitRemark(passCtx, {OptimizationRemark::Kind::Missed, kPassName, "NoCanonicalSSA",
                                 "@" + func.getName() + ": " + why});
            return preserveCFGAnalyses();
        }

        const SSADestructionResult result = replayLegacyColoring(func, **cached);
        if (!result.ok()) {
            PASS_DEBUG(std::cerr << "ReplayLegacyColoring: " << result.toString() << "\n");
            emitRemark(passCtx, {OptimizationRemark::Kind::Missed, kPassName, "NotLowered",
                                 result.toString()});
            return preserveCFGAnalyses();
        }

        emitRemark(passCtx, {OptimizationRemark::Kind::Passed, kPassName, "ReplayedLegacyColoring",
                             "@" + func.getName() +
                                 ": lowered canonical SSA back to its original registers"});
        // The graph describes pre-rewrite operands, so not preserving it here is
        // what discards it.
        return preserveCFGAnalyses();
    }
};

char ReplayLegacyColoringPassImpl::ID = 0;

}  // namespace

SSADestructionResult replayLegacyColoring(Function& function, const CanonicalSSA& ssa) {
    return destroyCanonicalSSA(function, ssa, createLegacyColoring(ssa));
}

std::unique_ptr<Pass> createReplayLegacyColoringPass() {
    return std::make_unique<ReplayLegacyColoringPassImpl>();
}

}  // namespace stinkytofu
