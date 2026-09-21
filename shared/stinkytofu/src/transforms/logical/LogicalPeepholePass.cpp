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

#include "stinkytofu/transforms/logical/LogicalPeepholePass.hpp"

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/logical/LogicalInstructions.hpp"
#include "stinkytofu/support/Casting.hpp"

// Include generated pattern matchers
#include "LogicalIRPatterns.inc"

namespace {
using namespace stinkytofu;

/// Implementation of the LogicalPeepholePass using unified Pass infrastructure
class LogicalPeepholePassImpl : public Pass {
   public:
    static constexpr const char* PassName = "LogicalPeepholePass";
    static char ID;

    PassID getPassID() const override {
        return &ID;
    }

    const char* getName() const override {
        return PassName;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        optimizationCount = 0;

        // Create pattern matcher registry
        std::vector<std::unique_ptr<hlir_patterns::PatternMatcher>> patterns =
            hlir_patterns::createAllPatterns();

        // Process all basic blocks
        for (BasicBlock& bb : func) {
            // Skip filtered basic blocks
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;

            runOnBasicBlock(bb, patterns);
        }
        return preserveCFGAnalyses();
    }

    size_t getOptimizationCount() const {
        return optimizationCount;
    }

   private:
    size_t optimizationCount = 0;

    void runOnBasicBlock(BasicBlock& bb,
                         std::vector<std::unique_ptr<hlir_patterns::PatternMatcher>>& patterns) {
        // Collect all LogicalInstructions from the IRList
        std::vector<LogicalInstruction*> instructions;
        for (IRBase& irNode : bb) {
            if (irNode.getType() == IRBase::IRType::StinkyTofu) {
                // LogicalInstruction inherits from IRBase with type StinkyTofu
                instructions.push_back(cast<LogicalInstruction>(&irNode));
            }
        }

        // Build def-use chains for the IR
        std::unordered_map<StinkyRegister, LogicalInstruction*> defMap;
        std::unordered_map<StinkyRegister, int> useCount;

        auto buildDefUseChains = [&]() {
            defMap.clear();
            useCount.clear();

            for (LogicalInstruction* inst : instructions) {
                for (const auto& dest : inst->dests) {
                    defMap[dest] = inst;
                }
            }

            for (LogicalInstruction* inst : instructions) {
                for (const auto& src : inst->srcs) {
                    useCount[src]++;
                }
            }
        };

        buildDefUseChains();

        // Iterate over all instructions and try to apply patterns
        for (size_t i = 0; i < instructions.size(); ++i) {
            auto* inst = instructions[i];

            // Create match context
            hlir_patterns::PatternMatcher::MatchContext context{defMap, useCount};

            // Try each pattern in order
            bool matched = false;
            for (auto& pattern : patterns) {
                // Try matching with original operand order
                auto result = pattern->tryMatchAndRewrite(inst, context);
                if (result && result->applied) {
                    // Pattern matched! Remove instructions marked for removal
                    for (auto* toRemove : result->instructionsToRemove) {
                        // Remove from IRList
                        bb.removeIR(static_cast<IRBase*>(toRemove));

                        // Remove from instructions vector
                        auto it = std::find(instructions.begin(), instructions.end(), toRemove);
                        if (it != instructions.end()) {
                            instructions.erase(it);
                        }
                    }

                    optimizationCount++;
                    matched = true;

                    // Rebuild def-use chains after modification
                    buildDefUseChains();

                    // Retry on same position for cascading optimizations
                    if (i > 0) --i;
                    break;  // Don't try other patterns
                }
            }

            // If no pattern matched with original order, try swapping operands
            // for commutative instructions
            if (!matched && inst->isCommutative() && inst->srcs.size() >= 2) {
                std::swap(inst->srcs[0], inst->srcs[1]);

                for (auto& pattern : patterns) {
                    auto result = pattern->tryMatchAndRewrite(inst, context);
                    if (result && result->applied) {
                        // Pattern matched! Remove instructions marked for removal
                        for (auto* toRemove : result->instructionsToRemove) {
                            // Remove from IRList
                            bb.removeIR(static_cast<IRBase*>(toRemove));

                            // Remove from instructions vector
                            auto it = std::find(instructions.begin(), instructions.end(), toRemove);
                            if (it != instructions.end()) {
                                instructions.erase(it);
                            }
                        }

                        optimizationCount++;
                        // Keep operands swapped since the rewrite was applied

                        // Rebuild def-use chains after modification
                        buildDefUseChains();

                        // Retry on same position for cascading optimizations
                        if (i > 0) --i;
                        break;  // Don't try other patterns
                    }
                }

                // If no pattern matched with swapped operands, restore original order
                if (!matched) {
                    std::swap(inst->srcs[0], inst->srcs[1]);
                }
            }
        }
    }
};

char LogicalPeepholePassImpl::ID = 0;

}  // anonymous namespace

namespace stinkytofu {
std::unique_ptr<Pass> createLogicalPeepholePass() {
    return std::make_unique<LogicalPeepholePassImpl>();
}

}  // namespace stinkytofu
