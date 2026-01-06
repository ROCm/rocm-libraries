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
#include "ir/asm/DuplicateEliminationPass.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include "support/Casting.hpp"

#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>

namespace
{
    using namespace stinkytofu;

    /// Represents the signature of an instruction for CSE purposes
    struct InstructionSignature
    {
        uint32_t                    opcode; // Unified opcode
        std::vector<StinkyRegister> srcRegs; // Source operands

        bool operator==(const InstructionSignature& other) const
        {
            if(opcode != other.opcode)
                return false;
            if(srcRegs.size() != other.srcRegs.size())
                return false;
            for(size_t i = 0; i < srcRegs.size(); ++i)
            {
                if(!(srcRegs[i] == other.srcRegs[i]))
                    return false;
            }
            return true;
        }
    };

    /// Hash function for InstructionSignature to use in unordered_map
    struct InstructionSignatureHash
    {
        size_t operator()(const InstructionSignature& sig) const
        {
            size_t hash = std::hash<uint32_t>{}(sig.opcode);
            for(const auto& reg : sig.srcRegs)
            {
                // Simple hash combination
                hash ^= reg.hash() + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            return hash;
        }
    };

    /// Analysis for duplicate instruction elimination
    class DuplicateAnalysis
    {
    public:
        void analyze(BasicBlock& bb)
        {
            instructions.clear();
            instPosition.clear();
            killedRegs.clear();

            IRList& irList = bb.getIR();
            int     pos    = 0;

            // Collect all instructions with position tracking
            for(IRBase& irNode : irList)
            {
                if(irNode.getType() != IRBase::IRType::StinkyTofu)
                    continue;

                auto* inst         = cast<StinkyInstruction>(&irNode);
                instPosition[inst] = pos++;
                instructions.push_back(inst);
            }
        }

        /// Finds duplicate instructions and returns a map from duplicate to original
        std::unordered_map<StinkyInstruction*, StinkyInstruction*> findDuplicates()
        {
            std::unordered_map<InstructionSignature, StinkyInstruction*, InstructionSignatureHash>
                                                                       signatureMap;
            std::unordered_map<StinkyInstruction*, StinkyInstruction*> duplicates;

            for(StinkyInstruction* inst : instructions)
            {
                // Skip instructions that must be preserved (use centralized check)
                if(mustPreserveInstruction(*inst))
                    continue;

                // Skip instructions with no destinations
                if(inst->getDestRegs().empty())
                    continue;

                // Build signature
                InstructionSignature sig;
                sig.opcode = inst->getUnifiedOpcode();
                sig.srcRegs.insert(
                    sig.srcRegs.end(), inst->getSrcRegs().begin(), inst->getSrcRegs().end());

                // Check if we've seen this signature before
                auto it = signatureMap.find(sig);
                if(it != signatureMap.end())
                {
                    StinkyInstruction* original = it->second;

                    // Verify that source operands haven't been modified between original and duplicate
                    if(areOperandsUnmodified(original, inst))
                    {
                        duplicates[inst] = original;
                    }
                    else
                    {
                        // Source was modified, this is now the "new" original for this signature
                        signatureMap[sig] = inst;
                    }
                }
                else
                {
                    // First occurrence of this signature
                    signatureMap[sig] = inst;
                }

                // Track that this instruction's destinations are "killed" (redefined)
                for(const StinkyRegister& destReg : inst->getDestRegs())
                {
                    if(destReg.isRegister())
                    {
                        StinkyRegister key(destReg.reg.type, destReg.reg.idx, 1);
                        killedRegs[key] = inst;
                    }
                }
            }

            return duplicates;
        }

        const std::vector<StinkyInstruction*>& getInstructions() const
        {
            return instructions;
        }

    private:
        /// Checks if source operands haven't been modified between two instructions
        bool areOperandsUnmodified(StinkyInstruction* original, StinkyInstruction* duplicate) const
        {
            int origPos = instPosition.at(original);
            int dupPos  = instPosition.at(duplicate);

            // Check each source operand of the duplicate
            for(const StinkyRegister& srcReg : duplicate->getSrcRegs())
            {
                if(!srcReg.isRegister())
                    continue;

                StinkyRegister key(srcReg.reg.type, srcReg.reg.idx, 1);

                // Check if this register was redefined between original and duplicate
                auto killIt = killedRegs.find(key);
                if(killIt != killedRegs.end())
                {
                    int killPos = instPosition.at(killIt->second);
                    // If register was redefined between original and duplicate, operands are modified
                    if(killPos > origPos && killPos < dupPos)
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        std::vector<StinkyInstruction*>                        instructions;
        std::unordered_map<StinkyInstruction*, int>            instPosition;
        std::unordered_map<StinkyRegister, StinkyInstruction*> killedRegs;
    };

    /// Implementation of the Duplicate Elimination pass
    class DuplicateEliminationPassImpl : public Pass
    {
    public:
        static constexpr const char* PassName = "DuplicateEliminationPass";
        static char                  ID;

        PassID getPassID() const override
        {
            return &ID;
        }

        const char* getName() const override
        {
            return PassName;
        }

        void run(Function& func, PassContext& passCtx) override
        {
            int totalEliminated = 0;

            // Process all basic blocks
            for(BasicBlock& bb : func)
            {
                // Skip filtered basic blocks
                if(!passCtx.shouldProcessBasicBlock(bb))
                    continue;

                int eliminated = runOnBasicBlock(bb);
                totalEliminated += eliminated;
            }
        }

    private:
        int runOnBasicBlock(BasicBlock& bb)
        {
            DuplicateAnalysis analysis;
            analysis.analyze(bb);

            auto duplicates = analysis.findDuplicates();
            if(duplicates.empty())
                return 0;

            IRList& irList = bb.getIR();

            // For each duplicate, replace uses with original and mark for deletion
            for(const auto& [duplicate, original] : duplicates)
            {
                // Get the destination register of the duplicate and original
                if(duplicate->getDestRegs().empty() || original->getDestRegs().empty())
                    continue;

                StinkyRegister dupDest  = duplicate->getDestRegs()[0];
                StinkyRegister origDest = original->getDestRegs()[0];

                // Replace all uses of dupDest with origDest in subsequent instructions
                replaceRegisterUses(irList, duplicate, dupDest, origDest);
            }

            // Remove duplicate instructions (they are now dead)
            for(const auto& [duplicate, original] : duplicates)
            {
                irList.remove(duplicate);
                delete duplicate;
            }

            return duplicates.size();
        }

        /// Replaces uses of oldReg with newReg in all instructions after startInst
        void replaceRegisterUses(IRList&               irList,
                                 StinkyInstruction*    startInst,
                                 const StinkyRegister& oldReg,
                                 const StinkyRegister& newReg)
        {
            bool foundStart = false;

            for(IRBase& irNode : irList)
            {
                if(irNode.getType() != IRBase::IRType::StinkyTofu)
                    continue;

                auto* inst = cast<StinkyInstruction>(&irNode);

                // Skip until we find the start instruction
                if(!foundStart)
                {
                    if(inst == startInst)
                        foundStart = true;
                    continue;
                }

                // Replace uses in source registers
                auto srcRegs  = inst->getSrcRegs();
                bool modified = false;
                for(StinkyRegister& srcReg : srcRegs)
                {
                    if(srcReg.isRegister() && srcReg == oldReg)
                    {
                        srcReg   = newReg;
                        modified = true;
                    }
                }
                if(modified)
                {
                    inst->setSrcRegs(srcRegs);
                }
            }
        }
    };

    char DuplicateEliminationPassImpl::ID = 0;

} // namespace

namespace stinkytofu
{
    std::unique_ptr<Pass> createDuplicateEliminationPass()
    {
        return std::make_unique<DuplicateEliminationPassImpl>();
    }
} // namespace stinkytofu
