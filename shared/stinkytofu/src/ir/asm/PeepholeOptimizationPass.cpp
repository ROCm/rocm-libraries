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
#include "ir/asm/PeepholeOptimizationPass.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include "isa/ArchHelper.hpp"
#include "support/Casting.hpp"

#include <climits>
#include <iostream>
#include <map>
#include <optional>
#include <unordered_set>
#include <vector>

// Include generated pattern matchers
#include "PeepholePatterns.inc"

namespace
{
    using namespace stinkytofu;

    /// Def-Use analysis helper for peephole optimizations.
    ///
    /// This class builds and maintains def-use chains for registers within
    /// a BasicBlock, which are essential for pattern matching and rewriting.
    ///
    /// Key features:
    /// - Tracks instruction order to handle in-place operations (dest == src)
    /// - Finds definitions that occur BEFORE a given instruction
    /// - Handles multiple definitions of the same register correctly
    class DefUseAnalysis
    {
    public:
        /// Analyzes a BasicBlock and builds def-use chains
        void analyze(BasicBlock& bb)
        {
            defMap.clear();
            useMap.clear();
            useCount.clear();
            instPosition.clear();

            IRList& irList = bb.getIR();
            int     pos    = 0;

            // First pass: build instruction position map
            for(IRBase& irNode : irList)
            {
                if(irNode.getType() != IRBase::IRType::StinkyTofu)
                    continue;

                auto* inst         = cast<StinkyInstruction>(&irNode);
                instPosition[inst] = pos++;
                instructions.push_back(inst);
            }

            // Second pass: build def-use chains
            for(auto* inst : instructions)
            {
                // Track definitions: map each destination register to its defining instruction
                // Store ALL definitions (not just the last one) with their positions
                for(StinkyRegister& destReg : inst->destRegs)
                {
                    if(destReg.isRegister())
                    {
                        // For simplicity, use the first register in the range as the key
                        StinkyRegister key(destReg.reg.type, destReg.reg.idx, 1);
                        defMap[key].push_back(inst);
                    }
                }

                // Track uses: map each source register to the using instruction
                for(StinkyRegister& srcReg : inst->srcRegs)
                {
                    if(srcReg.isRegister())
                    {
                        StinkyRegister key(srcReg.reg.type, srcReg.reg.idx, 1);
                        useMap[key].insert(inst);
                        useCount[key]++;
                    }
                }
            }
        }

        /// Returns the instruction that defines the given register BEFORE the specified instruction.
        /// This is crucial for handling in-place operations where dest == src.
        ///
        /// For example:
        ///   v_fma_f32 v0, v1, v2, 1.0   // pos=0, defines v0
        ///   v_add_f32 v0, 1.0, v0       // pos=1, uses v0 (should find pos=0), defines v0
        ///
        /// When matching v_add at pos=1, we need to find the v_fma at pos=0 that defined v0.
        StinkyInstruction* getDefiningInstBefore(const StinkyRegister& reg,
                                                 StinkyInstruction*    beforeInst) const
        {
            auto it = defMap.find(reg);
            if(it == defMap.end())
                return nullptr;

            auto beforePosIt = instPosition.find(beforeInst);
            if(beforePosIt == instPosition.end())
                return nullptr;

            int beforePos = beforePosIt->second;

            // Find the most recent definition that comes BEFORE beforeInst
            StinkyInstruction* mostRecentDef = nullptr;
            int                mostRecentPos = -1;

            for(auto* defInst : it->second)
            {
                auto defPosIt = instPosition.find(defInst);
                if(defPosIt == instPosition.end())
                    continue;

                int defPos = defPosIt->second;

                // Only consider definitions that come before the current instruction
                if(defPos < beforePos && defPos > mostRecentPos)
                {
                    mostRecentDef = defInst;
                    mostRecentPos = defPos;
                }
            }

            return mostRecentDef;
        }

        /// Returns all instructions that use the given register
        const std::unordered_set<StinkyInstruction*>& getUsingInsts(const StinkyRegister& reg) const
        {
            auto                                                it = useMap.find(reg);
            static const std::unordered_set<StinkyInstruction*> empty;
            return (it != useMap.end()) ? it->second : empty;
        }

        /// Returns the number of uses for a SPECIFIC DEFINITION of a register.
        /// Counts only uses between this definition and the next redefinition.
        ///
        /// Example with register reuse:
        ///   v10 = fma(...)      // defInst (pos=9)
        ///   v10 = add(v10, ...) // pos=10, uses defInst's v10, then REDEFINES v10
        ///   v10 = mul(v10, ...) // pos=11, uses pos=10's v10 (NOT defInst!)
        ///   v1  = mul(v10, ...) // pos=12, uses pos=11's v10 (NOT defInst!)
        ///
        /// getUseCountForDef(defInst at pos=9) returns 1 (only pos=10)
        int getUseCountForDef(StinkyInstruction* defInst, const StinkyRegister& reg) const
        {
            auto defPosIt = instPosition.find(defInst);
            if(defPosIt == instPosition.end())
                return 0;
            int defPos = defPosIt->second;

            // Find the next redefinition after defInst
            int  nextDefPos = INT_MAX;
            auto defMapIt   = defMap.find(reg);
            if(defMapIt != defMap.end())
            {
                for(auto* laterDefInst : defMapIt->second)
                {
                    auto laterPosIt = instPosition.find(laterDefInst);
                    if(laterPosIt == instPosition.end())
                        continue;
                    int laterPos = laterPosIt->second;

                    // First redefinition AFTER our definition ends our live range
                    if(laterPos > defPos && laterPos < nextDefPos)
                    {
                        nextDefPos = laterPos;
                    }
                }
            }

            // Count uses in the live range [defPos+1, nextDefPos]
            // Note: Uses AT nextDefPos are included because in-place operations
            // (e.g., v10 = add(v10, ...)) use the register before redefining it.
            int  count    = 0;
            auto useMapIt = useMap.find(reg);
            if(useMapIt != useMap.end())
            {
                for(auto* useInst : useMapIt->second)
                {
                    auto usePosIt = instPosition.find(useInst);
                    if(usePosIt == instPosition.end())
                        continue;
                    int usePos = usePosIt->second;

                    // Count uses after definition UP TO AND INCLUDING the next redefinition
                    // (the redefining instruction uses the value before overwriting it)
                    if(usePos > defPos && usePos <= nextDefPos)
                    {
                        count++;
                    }
                }
            }

            return count;
        }

        /// Returns the number of uses for the given register (all definitions)
        int getUseCount(const StinkyRegister& reg) const
        {
            auto it = useCount.find(reg);
            return (it != useCount.end()) ? it->second : 0;
        }

        /// Checks if the register has exactly one use (common constraint in fusion patterns)
        bool hasOneUse(const StinkyRegister& reg) const
        {
            return getUseCount(reg) == 1;
        }

        /// Build a context-aware def map for a specific instruction.
        /// Only includes definitions that come BEFORE the given instruction.
        std::unordered_map<StinkyRegister, StinkyInstruction*>
            getDefMapBefore(StinkyInstruction* beforeInst) const
        {
            std::unordered_map<StinkyRegister, StinkyInstruction*> result;

            for(const auto& entry : defMap)
            {
                const StinkyRegister& reg = entry.first;
                if(auto* defInst = getDefiningInstBefore(reg, beforeInst))
                {
                    result[reg] = defInst;
                }
            }

            return result;
        }

        /// Get a context-aware use count map for pattern matching.
        /// For each register in the defMap, returns the use count for that SPECIFIC definition,
        /// not the total use count across all definitions.
        ///
        /// This is critical for patterns with register reuse:
        ///   v10 = fma(...)      // def #1
        ///   v10 = add(v10, ...) // def #2, uses def #1
        ///   v10 = mul(v10, ...) // def #3, uses def #2
        ///
        /// When checking if def #1 has one use, we should return 1 (not 2).
        std::unordered_map<StinkyRegister, int> getUseCountMapForDefs(
            const std::unordered_map<StinkyRegister, StinkyInstruction*>& defs) const
        {
            std::unordered_map<StinkyRegister, int> result;

            for(const auto& entry : defs)
            {
                const StinkyRegister& reg     = entry.first;
                StinkyInstruction*    defInst = entry.second;

                // Get use count for this specific definition
                result[reg] = getUseCountForDef(defInst, reg);
            }

            return result;
        }

        /// Get the use count map as unordered_map (for pattern matchers) - LEGACY
        std::unordered_map<StinkyRegister, int> getUseCountMap() const
        {
            return std::unordered_map<StinkyRegister, int>(useCount.begin(), useCount.end());
        }

    private:
        // Map from register to ALL instructions that define it (in program order)
        std::map<StinkyRegister, std::vector<StinkyInstruction*>> defMap;

        // Map from register to all instructions that use it
        std::map<StinkyRegister, std::unordered_set<StinkyInstruction*>> useMap;

        // Map from register to number of uses
        std::map<StinkyRegister, int> useCount;

        // Map from instruction to its position in the BasicBlock (for ordering)
        std::unordered_map<StinkyInstruction*, int> instPosition;

        // List of all instructions in order
        std::vector<StinkyInstruction*> instructions;
    };

    /// Peephole optimization pass implementation.
    ///
    /// This pass applies declarative rewrite patterns defined in PeepholePatterns.pattern
    /// to optimize instruction sequences. Patterns are compiled at build-time by TableGen
    /// into efficient C++ matcher code.
    class PeepholeOptimizationPassImpl : public Pass
    {
    public:
        static char ID;

        const char* getName() const override
        {
            return "Peephole Optimization";
        }

        Pass::ID getPassID() const override
        {
            return &PeepholeOptimizationPassImpl::ID;
        }

        void run(Function& func, PassContext& passCtx) override
        {
            GfxArchID arch         = getGfxArchID(passCtx.getGemmTileConfig().arch[0],
                                          passCtx.getGemmTileConfig().arch[1],
                                          passCtx.getGemmTileConfig().arch[2]);
            int       totalFusions = 0;

            // Process each BasicBlock
            for(BasicBlock& bb : func)
            {
                // Skip BasicBlocks that don't pass the global filter
                if(!passCtx.shouldProcessBasicBlock(bb))
                    continue;

                // Apply patterns iteratively until no more patterns match
                // This handles cases where one fusion creates new fusion opportunities
                bool changed = true;
                while(changed)
                {
                    // Analyze def-use chains for this BasicBlock
                    DefUseAnalysis defUse;
                    defUse.analyze(bb);

                    // Apply patterns within this BasicBlock
                    int fusionsInBB = runOnBasicBlock(bb, defUse, arch);
                    totalFusions += fusionsInBB;

                    // Continue iterating if we made progress
                    changed = (fusionsInBB > 0);
                }
            }

            std::cout << "Peephole Optimization: Applied " << totalFusions << " fusion(s)\n";
        }

    private:
        /// Pattern matchers (lazily initialized)
        std::vector<std::unique_ptr<patterns::PatternMatcher>> patternMatchers;

        /// Initialize pattern matchers
        void initPatterns()
        {
            if(patternMatchers.empty())
            {
                patternMatchers = patterns::createAllPatterns();
                std::cout << "[PeepholePass] Loaded " << patternMatchers.size() << " pattern(s)\n";
            }
        }

        /// Run peephole optimizations on a single BasicBlock
        int runOnBasicBlock(BasicBlock& bb, DefUseAnalysis& defUse, GfxArchID arch)
        {
            initPatterns();

            int                             numRewrites = 0;
            std::vector<StinkyInstruction*> toRemove;

            IRList& irList = bb.getIR();
            for(IRBase& irNode : irList)
            {
                if(irNode.getType() != IRBase::IRType::StinkyTofu)
                    continue;

                auto* inst = cast<StinkyInstruction>(&irNode);

                // Build context-aware def map for this specific instruction
                // This ensures we only see definitions that come BEFORE this instruction
                auto defMapBefore = defUse.getDefMapBefore(inst);

                // Build context-aware use count map for those specific definitions
                // This ensures we count only uses within each definition's live range
                auto useCountMap = defUse.getUseCountMapForDefs(defMapBefore);

                // Prepare pattern match context with position-aware def and use maps
                patterns::PatternMatcher::MatchContext context{defMapBefore, useCountMap, arch};

                // Try all patterns in order (first match wins)
                for(auto& pattern : patternMatchers)
                {
                    // Try matching with original operand order
                    if(auto result = pattern->tryMatchAndRewrite(inst, context))
                    {
                        std::cout << "  [Peephole] Applied " << pattern->getName() << "\n";

                        // Collect instructions to remove
                        toRemove.insert(toRemove.end(),
                                        result->instructionsToRemove.begin(),
                                        result->instructionsToRemove.end());
                        numRewrites++;
                        break; // Stop after first match
                    }

                    // For commutative binary operations, try swapping operands if the first match failed
                    // This allows patterns to match regardless of operand order for commutative ops
                    bool isCommutative = inst->is(IF_Commutative);
                    if(isCommutative && inst->srcRegs.size() >= 2)
                    {
                        // Swap first two source operands
                        std::swap(inst->srcRegs[0], inst->srcRegs[1]);

                        // Try matching again with swapped operands
                        if(auto result = pattern->tryMatchAndRewrite(inst, context))
                        {
                            std::cout << "  [Peephole] Applied " << pattern->getName()
                                      << " (commutative match)\n";

                            // Collect instructions to remove
                            toRemove.insert(toRemove.end(),
                                            result->instructionsToRemove.begin(),
                                            result->instructionsToRemove.end());
                            numRewrites++;
                            // Keep operands swapped since the rewrite was applied
                            break; // Stop after first match
                        }

                        // Restore original operand order if pattern didn't match
                        std::swap(inst->srcRegs[0], inst->srcRegs[1]);
                    }
                }
            }

            // Remove dead instructions
            for(auto* instToRemove : toRemove)
            {
                irList.remove(instToRemove);
            }

            return numRewrites;
        }
    };

    char PeepholeOptimizationPassImpl::ID = 0;

} // namespace

namespace stinkytofu
{
    std::unique_ptr<Pass> createPeepholeOptimizationPass()
    {
        return std::make_unique<PeepholeOptimizationPassImpl>();
    }
} // namespace stinkytofu
