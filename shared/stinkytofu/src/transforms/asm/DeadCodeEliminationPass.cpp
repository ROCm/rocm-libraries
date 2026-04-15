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
#include "stinkytofu/transforms/asm/DeadCodeEliminationPass.hpp"

#include <set>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/ir/asm/DefUseChainUpdater.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace {
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
class DeadCodeEliminationPassImpl : public Pass {
   public:
    static constexpr const char* PassName = "DeadCodeEliminationPass";
    static char ID;

    PassID getPassID() const override {
        return &ID;
    }

    const char* getName() const override {
        return PassName;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        int totalRemoved = 0;

        // Process all basic blocks
        for (BasicBlock& bb : func) {
            // Skip filtered basic blocks
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;

            // Iteratively remove dead stores until fixpoint
            int removedInBB = 0;
            bool changed = true;
            while (changed) {
                int removed = runOnBasicBlock(bb, func);
                removedInBB += removed;
                changed = (removed > 0);
            }

            totalRemoved += removedInBB;
        }
        return preserveCFGAnalyses();
    }

   private:
    /// Collect all instructions from a function in order
    void collectAllInstructions(Function& func, std::vector<StinkyInstruction*>& instructions) {
        for (BasicBlock& bb : func) {
            for (IRBase& irNode : bb) {
                if (irNode.getType() == IRBase::IRType::StinkyTofu) {
                    instructions.push_back(cast<StinkyInstruction>(&irNode));
                }
            }
        }
    }

    int runOnBasicBlock(BasicBlock& bb, Function& func) {
        // Simple DCE with two rules:
        // Rule 1: If only one assignment to a register in this block -> DON'T delete (next block
        // might use it) Rule 2: If assigned again, and register NOT used between assignments ->
        // DELETE first assignment

        std::vector<StinkyInstruction*> blockInstructions;
        for (IRBase& irNode : bb) {
            if (irNode.getType() == IRBase::IRType::StinkyTofu) {
                blockInstructions.push_back(cast<StinkyInstruction>(&irNode));
            }
        }

        std::set<StinkyInstruction*> toRemove;

        // For each instruction that writes to a register
        for (size_t i = 0; i < blockInstructions.size(); ++i) {
            StinkyInstruction* inst = blockInstructions[i];

            // Never remove side-effecting instructions (barriers, branches, memory ops, etc.)
            if (mustPreserveInstruction(*inst)) continue;

            // Never remove instructions with no explicit destination registers
            // (these include: comparisons, waits, delays, etc.)
            if (inst->getDestRegs().empty()) continue;

            // Never remove instructions that write to dummy registers (dependency tracking only)
            bool hasDummyDest = false;
            for (const StinkyRegister& destReg : inst->getDestRegs()) {
                if (destReg.reg.type == RegType::SCC || isPseudoReg(destReg)) {
                    hasDummyDest = true;
                    break;
                }
            }
            if (hasDummyDest) continue;

            // Skip in-place operations (ANY dest overlaps a source)
            // e.g., s_add_u32 s10, s10, s20 - these are NOT dead stores!
            bool isInPlace = false;
            for (const StinkyRegister& destReg : inst->getDestRegs()) {
                if (!destReg.isRegister()) continue;
                for (const StinkyRegister& srcReg : inst->getSrcRegs()) {
                    if (srcReg.isOverlap(destReg)) {
                        isInPlace = true;
                        break;
                    }
                }
                if (isInPlace) break;
            }
            if (isInPlace) continue;  // Skip entire instruction

            // Check each destination register
            for (const StinkyRegister& destReg : inst->getDestRegs()) {
                if (!destReg.isRegister()) continue;

                // First, count how many times this register is assigned in this block
                int assignmentCount = 0;
                for (size_t k = 0; k < blockInstructions.size(); ++k) {
                    for (const StinkyRegister& kDestReg : blockInstructions[k]->getDestRegs()) {
                        if (kDestReg.isOverlap(destReg)) {
                            assignmentCount++;
                            break;
                        }
                    }
                }

                // Rule 1: If only one assignment in this block, DON'T delete
                if (assignmentCount <= 1) continue;

                // Rule 2: Multiple assignments exist, check if register is used before next
                // assignment
                bool foundNextAssignment = false;
                bool usedBeforeNextAssignment = false;

                for (size_t j = i + 1; j < blockInstructions.size(); ++j) {
                    StinkyInstruction* laterInst = blockInstructions[j];

                    // Check if this register is used as a source
                    for (const StinkyRegister& srcReg : laterInst->getSrcRegs()) {
                        if (srcReg.isOverlap(destReg)) {
                            usedBeforeNextAssignment = true;
                            break;
                        }
                    }

                    // Check if this register is reassigned
                    for (const StinkyRegister& laterDestReg : laterInst->getDestRegs()) {
                        if (laterDestReg.isOverlap(destReg)) {
                            foundNextAssignment = true;
                            break;
                        }
                    }

                    if (foundNextAssignment) break;
                }

                // Delete if: found next assignment AND register NOT used between
                if (foundNextAssignment && !usedBeforeNextAssignment) {
                    toRemove.insert(inst);
                    break;  // No need to check other destinations of this instruction
                }
            }
        }

        // Now remove dead instructions that belong to this basic block
        int removedInBB = 0;

        for (StinkyInstruction* inst : toRemove) {
            // Check if this instruction belongs to current basic block
            bool inThisBlock = inst->getParentBlock() == &bb;

            if (inThisBlock) {
                DefUseChainUpdater::eraseAndUnlink(inst);
                removedInBB++;
            }
        }

        return removedInBB;
    }
};

char DeadCodeEliminationPassImpl::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createDeadCodeEliminationPass() {
    return std::make_unique<DeadCodeEliminationPassImpl>();
}
}  // namespace stinkytofu
