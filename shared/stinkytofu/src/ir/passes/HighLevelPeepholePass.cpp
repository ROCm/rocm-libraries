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

#include "ir/passes/HighLevelPeepholePass.hpp"
#include "ir/IRModule.hpp"
#include "ir/StinkyInstructions.hpp"

// Include generated pattern matchers
#include "HighLevelIRPatterns.inc"

namespace stinkytofu
{

    bool HighLevelPeepholePass::run(IRModule& module)
    {
        bool changed      = false;
        optimizationCount = 0;

        // Create pattern matcher registry
        std::vector<std::unique_ptr<hlir_patterns::PatternMatcher>> patterns
            = hlir_patterns::createAllPatterns();

        // Step 1: Build def-use chains for the IR
        std::unordered_map<StinkyRegister, IRInstruction*> defMap;
        std::unordered_map<StinkyRegister, int>            useCount;

        // Build definition map (which instruction defines each register)
        for(const auto& inst : module.getInstructions())
        {
            for(const auto& dest : inst->dests)
            {
                defMap[dest] = inst.get();
            }
        }

        // Count uses of each register
        for(const auto& inst : module.getInstructions())
        {
            for(const auto& src : inst->srcs)
            {
                useCount[src]++;
            }
        }

        // Step 2: Iterate over all instructions and try to apply patterns
        auto& instructions = module.getMutableInstructions();
        for(size_t i = 0; i < instructions.size(); ++i)
        {
            auto* inst = instructions[i].get();

            // Create match context
            hlir_patterns::PatternMatcher::MatchContext context{defMap, useCount};

            // Try each pattern in order
            bool matched = false;
            for(auto& pattern : patterns)
            {
                // Try matching with original operand order
                auto result = pattern->tryMatchAndRewrite(inst, context);
                if(result && result->applied)
                {
                    // Pattern matched! Remove instructions marked for removal
                    for(auto* toRemove : result->instructionsToRemove)
                    {
                        auto it = std::find_if(
                            instructions.begin(), instructions.end(), [toRemove](const auto& ptr) {
                                return ptr.get() == toRemove;
                            });
                        if(it != instructions.end())
                        {
                            instructions.erase(it);
                        }
                    }

                    changed = true;
                    optimizationCount++;
                    matched = true;

                    // Rebuild def-use chains after modification
                    defMap.clear();
                    useCount.clear();
                    for(const auto& inst : module.getInstructions())
                    {
                        for(const auto& dest : inst->dests)
                        {
                            defMap[dest] = inst.get();
                        }
                    }
                    for(const auto& inst : module.getInstructions())
                    {
                        for(const auto& src : inst->srcs)
                        {
                            useCount[src]++;
                        }
                    }

                    // Retry on same position for cascading optimizations
                    if(i > 0)
                        --i;
                    break; // Don't try other patterns, restart from this position
                }
            }

            // For commutative binary operations, try swapping operands if no match yet
            if(!matched && inst->isCommutative() && inst->srcs.size() >= 2)
            {
                // Swap first two source operands
                StinkyRegister temp = inst->srcs[0];
                inst->srcs[0]       = inst->srcs[1];
                inst->srcs[1]       = temp;

                // Try each pattern again with swapped operands
                for(auto& pattern : patterns)
                {
                    auto result = pattern->tryMatchAndRewrite(inst, context);
                    if(result && result->applied)
                    {
                        // Pattern matched! Remove instructions marked for removal
                        for(auto* toRemove : result->instructionsToRemove)
                        {
                            auto it = std::find_if(
                                instructions.begin(),
                                instructions.end(),
                                [toRemove](const auto& ptr) { return ptr.get() == toRemove; });
                            if(it != instructions.end())
                            {
                                instructions.erase(it);
                            }
                        }

                        changed = true;
                        optimizationCount++;
                        // Keep operands swapped since the rewrite was applied

                        // Rebuild def-use chains after modification
                        defMap.clear();
                        useCount.clear();
                        for(const auto& inst : module.getInstructions())
                        {
                            for(const auto& dest : inst->dests)
                            {
                                defMap[dest] = inst.get();
                            }
                        }
                        for(const auto& inst : module.getInstructions())
                        {
                            for(const auto& src : inst->srcs)
                            {
                                useCount[src]++;
                            }
                        }

                        // Retry on same position for cascading optimizations
                        if(i > 0)
                            --i;
                        break; // Don't try other patterns
                    }
                }

                // If no pattern matched with swapped operands, restore original order
                if(!matched)
                {
                    inst->srcs[1] = inst->srcs[0];
                    inst->srcs[0] = temp;
                }
            }
        }

        return changed;
    }

} // namespace stinkytofu
