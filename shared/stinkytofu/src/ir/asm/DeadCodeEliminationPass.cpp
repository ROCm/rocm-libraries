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
#include "ir/asm/DeadCodeEliminationPass.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include "support/Casting.hpp"

#include <set>
#include <vector>

namespace
{
    using namespace stinkytofu;

    /// Implementation of the Dead Code Elimination pass
    ///
    /// This pass eliminates dead stores using backward dataflow analysis:
    /// - Scan backward through instructions
    /// - Track "live" registers (will be used later)
    /// - If an instruction defines a register that's not live, it's a dead store
    ///
    /// Time complexity: O(n) - single backward pass
    /// Space complexity: O(r) - set of live registers
    class DeadCodeEliminationPassImpl : public Pass
    {
    public:
        static constexpr const char* PassName = "DeadCodeEliminationPass";
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
            int totalRemoved = 0;

            // Process all basic blocks
            for(BasicBlock& bb : func)
            {
                // Skip filtered basic blocks
                if(!passCtx.shouldProcessBasicBlock(bb))
                    continue;

                // Iteratively remove dead stores until fixpoint
                int  removedInBB = 0;
                bool changed     = true;
                while(changed)
                {
                    int removed = runOnBasicBlock(bb, func);
                    removedInBB += removed;
                    changed = (removed > 0);
                }

                totalRemoved += removedInBB;
            }
        }

    private:
        /// Collect all instructions from a function in order
        void collectAllInstructions(Function& func, std::vector<StinkyInstruction*>& instructions)
        {
            for(BasicBlock& bb : func)
            {
                for(IRBase& irNode : bb.getIR())
                {
                    if(irNode.getType() == IRBase::IRType::StinkyTofu)
                    {
                        instructions.push_back(cast<StinkyInstruction>(&irNode));
                    }
                }
            }
        }

        int runOnBasicBlock(BasicBlock& bb, Function& func)
        {
            // Collect ALL instructions from entire function
            std::vector<StinkyInstruction*> allInstructions;
            collectAllInstructions(func, allInstructions);

            // Track registers that are redefined and used later (backward scan)
            std::set<StinkyRegister> redefinedRegs; // Registers redefined later
            std::set<StinkyRegister> usedRegs; // Registers used later

            // Mark instructions to remove
            std::set<StinkyInstruction*> toRemove;

            // Backward scan through ALL instructions (O(n))
            for(auto it = allInstructions.rbegin(); it != allInstructions.rend(); ++it)
            {
                StinkyInstruction* inst = *it;

                // Always preserve side-effecting instructions
                if(mustPreserveInstruction(*inst))
                {
                    // Record that sources are used
                    for(const StinkyRegister& srcReg : inst->getSrcRegs())
                    {
                        if(srcReg.isRegister())
                        {
                            usedRegs.insert(srcReg);
                        }
                    }
                    // Record that destinations are redefined
                    for(const StinkyRegister& destReg : inst->getDestRegs())
                    {
                        if(destReg.isRegister())
                        {
                            redefinedRegs.insert(destReg);
                        }
                    }
                    continue;
                }

                // Check if this is a dead store
                bool isDeadStore = false;
                if(!inst->getDestRegs().empty())
                {
                    // A dead store is when a destination is:
                    // 1. Redefined later (in redefinedRegs)
                    // 2. NOT used before that redefinition (NOT in usedRegs)
                    for(const StinkyRegister& destReg : inst->getDestRegs())
                    {
                        if(destReg.isRegister())
                        {
                            if(redefinedRegs.count(destReg) > 0 && usedRegs.count(destReg) == 0)
                            {
                                // Overwritten before use -> dead store!
                                isDeadStore = true;
                                break;
                            }
                        }
                    }
                }

                if(isDeadStore)
                {
                    // Mark for removal (don't update tracking sets)
                    toRemove.insert(inst);
                }
                else
                {
                    // Not a dead store - update tracking sets
                    // Record sources as used
                    for(const StinkyRegister& srcReg : inst->getSrcRegs())
                    {
                        if(srcReg.isRegister())
                        {
                            usedRegs.insert(srcReg);
                        }
                    }
                    // Record destinations as redefined
                    // IMPORTANT: Only clear from usedRegs if NOT also a source (in-place ops!)
                    for(const StinkyRegister& destReg : inst->getDestRegs())
                    {
                        if(destReg.isRegister())
                        {
                            redefinedRegs.insert(destReg);

                            // Check if this destination is also a source (in-place operation)
                            bool isAlsoSource = false;
                            for(const StinkyRegister& srcReg : inst->getSrcRegs())
                            {
                                if(srcReg.isRegister() && srcReg.reg.type == destReg.reg.type
                                   && srcReg.reg.idx == destReg.reg.idx)
                                {
                                    isAlsoSource = true;
                                    break;
                                }
                            }

                            // Only erase if NOT in-place (destination must be used first)
                            if(!isAlsoSource)
                            {
                                usedRegs.erase(destReg);
                            }
                        }
                    }
                }
            }

            // Now remove dead instructions that belong to this basic block
            IRList& irList      = bb.getIR();
            int     removedInBB = 0;

            for(StinkyInstruction* inst : toRemove)
            {
                // Check if this instruction belongs to current basic block
                bool inThisBlock = false;
                for(IRBase& irNode : irList)
                {
                    if(&irNode == inst)
                    {
                        inThisBlock = true;
                        break;
                    }
                }

                if(inThisBlock)
                {
                    irList.remove(inst);
                    delete inst;
                    removedInBB++;
                }
            }

            return removedInBB;
        }
    };

    char DeadCodeEliminationPassImpl::ID = 0;

} // namespace

namespace stinkytofu
{
    std::unique_ptr<Pass> createDeadCodeEliminationPass()
    {
        return std::make_unique<DeadCodeEliminationPassImpl>();
    }
} // namespace stinkytofu
