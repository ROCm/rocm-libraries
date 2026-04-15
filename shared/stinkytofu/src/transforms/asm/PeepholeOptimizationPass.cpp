/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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
#include "stinkytofu/transforms/asm/PeepholeOptimizationPass.hpp"

#include <iostream>
#include <optional>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/controlflow/DominanceAnalysis.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/DefUseChainUpdater.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"

// Include generated pattern matchers
#include "PeepholePatterns.inc"

namespace {
using namespace stinkytofu;

/// Owns the def-map and use-count data that MatchContext references.
/// Must outlive any MatchContext constructed from it.
struct MatchData {
    std::unordered_map<StinkyRegister, StinkyInstruction*> defMap;
    std::unordered_map<StinkyRegister, int> useCount;
};

/// Build match data for pattern matching from the def-use chains
/// that buildUseDefChain() has already populated on each instruction.
///
/// defMap:   register -> defining instruction  (from getSources())
/// useCount: register -> number of users       (from getUsers().size())
MatchData buildMatchData(StinkyInstruction* inst) {
    MatchData data;
    for (StinkyInstruction* src : inst->getSources()) {
        if (!src) continue;
        for (const auto& destReg : src->getDestRegs()) {
            if (destReg.isRegister())
                data.defMap[StinkyRegister(destReg.reg.type, destReg.reg.idx, 1)] = src;
        }
    }

    for (const auto& [reg, defInst] : data.defMap)
        data.useCount[reg] = static_cast<int>(defInst->getUsers().size());

    return data;
}

/// Peephole optimization pass implementation.
///
/// Applies declarative rewrite patterns defined in PeepholePatterns.pattern
/// to optimize instruction sequences. Patterns are compiled at build-time
/// by TableGen into efficient C++ matcher code.
class PeepholeOptimizationPassImpl : public Pass {
   public:
    static char ID;

    const char* getName() const override {
        return "Peephole Optimization";
    }

    Pass::ID getPassID() const override {
        return &PeepholeOptimizationPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        GfxArchID arch =
            getGfxArchID(passCtx.getGemmTileConfig().arch[0], passCtx.getGemmTileConfig().arch[1],
                         passCtx.getGemmTileConfig().arch[2]);
        int totalFusions = 0;

        const auto& domInfo = AM.getResult<DominanceAnalysis>(func);
        buildUseDefChain(func, domInfo, true);

        // Apply patterns iteratively until no more patterns match.
        // Rebuild def-use chains once per iteration over the whole function
        // so that cross-block chains stay consistent after rewrites.
        bool changed = true;
        while (changed) {
            changed = false;

            for (BasicBlock& bb : func) {
                if (!passCtx.shouldProcessBasicBlock(bb)) continue;

                int fusionsInBB = runOnBasicBlock(bb, arch);
                totalFusions += fusionsInBB;

                if (fusionsInBB > 0) changed = true;
            }
        }

        std::cout << "Peephole Optimization: Applied " << totalFusions << " fusion(s)\n";
        return preserveCFGAnalyses();
    }

   private:
    /// Pattern matchers (lazily initialized)
    std::vector<std::unique_ptr<patterns::PatternMatcher>> patternMatchers;

    /// Initialize pattern matchers
    void initPatterns() {
        if (patternMatchers.empty()) {
            patternMatchers = patterns::createAllPatterns();
            std::cout << "[PeepholePass] Loaded " << patternMatchers.size() << " pattern(s)\n";
        }
    }

    /// Run peephole optimizations on a single BasicBlock
    int runOnBasicBlock(BasicBlock& bb, GfxArchID arch) {
        initPatterns();

        struct RemovalEntry {
            StinkyInstruction* instToRemove;
            StinkyInstruction* replacement;
        };

        int numRewrites = 0;
        std::vector<RemovalEntry> toRemove;

        for (IRBase& irNode : bb) {
            if (irNode.getType() != IRBase::IRType::StinkyTofu) continue;

            auto* inst = cast<StinkyInstruction>(&irNode);
            auto data = buildMatchData(inst);
            patterns::PatternMatcher::MatchContext context{data.defMap, data.useCount, arch};

            // Try all patterns in order (first match wins)
            for (auto& pattern : patternMatchers) {
                // Try matching with original operand order
                if (auto result = pattern->tryMatchAndRewrite(inst, context)) {
                    std::cout << "  [Peephole] Applied " << pattern->getName() << "\n";

                    for (auto* inst : result->instructionsToRemove)
                        toRemove.push_back({inst, result->replacementInst});
                    numRewrites++;
                    break;  // Stop after first match
                }

                // For commutative binary operations, try swapping operands if the first match
                // failed This allows patterns to match regardless of operand order for commutative
                // ops
                bool isCommutative = inst->is(IF_Commutative);
                if (isCommutative && inst->getSrcRegs().size() >= 2) {
                    // Swap first two source operands
                    StinkyRegister temp = inst->getSrcReg(0);
                    inst->setSrcReg(0, inst->getSrcReg(1));
                    inst->setSrcReg(1, temp);

                    // Try matching again with swapped operands
                    if (auto result = pattern->tryMatchAndRewrite(inst, context)) {
                        std::cout << "  [Peephole] Applied " << pattern->getName()
                                  << " (commutative match)\n";

                        for (auto* inst : result->instructionsToRemove)
                            toRemove.push_back({inst, result->replacementInst});
                        numRewrites++;
                        // Keep operands swapped since the rewrite was applied
                        break;  // Stop after first match
                    }

                    // Restore original operand order if pattern didn't match
                    inst->setSrcReg(1, inst->getSrcReg(0));
                    inst->setSrcReg(0, temp);
                }
            }
        }

        // Transfer users to replacement instructions, then erase
        for (auto& [instToRemove, replacement] : toRemove) {
            if (replacement && !instToRemove->getUsers().empty())
                DefUseChainUpdater::replaceAllUsesWith(instToRemove, replacement);
            DefUseChainUpdater::eraseAndUnlink(instToRemove);
        }

        return numRewrites;
    }
};

char PeepholeOptimizationPassImpl::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createPeepholeOptimizationPass() {
    return std::make_unique<PeepholeOptimizationPassImpl>();
}
}  // namespace stinkytofu
