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
#include "stinkytofu/transforms/asm/RedundantMovEliminationPass.hpp"

#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/ir/asm/DefUseChainUpdater.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace {
using namespace stinkytofu;

/// Helper to check if an instruction should be considered for redundant mov elimination
bool isEligibleForRedundantMovElimination(const StinkyInstruction& inst) {
    // Only consider specific instruction types that are safe to eliminate
    // Currently: v_mov_b32 and s_mov_b32
    // Easy to extend by adding more checks here

    // Check if it's a mov instruction by looking at the opcode
    // This is a simple implementation - you may want to check specific opcode values
    // or use instruction flags to identify mov instructions

    // For now, we'll use a simple heuristic: check if instruction has exactly one dest and one
    // source and no side effects (typical for mov instructions)
    if (inst.getDestRegs().size() != 1 || inst.getSrcRegs().size() != 1) return false;

    if (mustPreserveInstruction(inst)) return false;

    // TODO: Add specific opcode checks for v_mov_b32, s_mov_b32
    // For now, accept any single-dest, single-source, non-side-effect instruction
    return true;
}

/// Checks if two instructions are identical (same opcode, same operands)
bool areInstructionsIdentical(const StinkyInstruction* inst1, const StinkyInstruction* inst2) {
    // Must have same opcode
    if (inst1->getUnifiedOpcode() != inst2->getUnifiedOpcode()) return false;

    // Must have same number of destinations
    if (inst1->getDestRegs().size() != inst2->getDestRegs().size()) return false;

    // Must have same number of sources
    if (inst1->getSrcRegs().size() != inst2->getSrcRegs().size()) return false;

    // Check all destinations match
    for (size_t i = 0; i < inst1->getDestRegs().size(); ++i) {
        if (!(inst1->getDestRegs()[i] == inst2->getDestRegs()[i])) return false;
    }

    // Check all sources match
    for (size_t i = 0; i < inst1->getSrcRegs().size(); ++i) {
        if (!(inst1->getSrcRegs()[i] == inst2->getSrcRegs()[i])) return false;
    }

    return true;
}

/// Implementation of the Duplicate Elimination pass
class RedundantMovEliminationPassImpl : public Pass {
   public:
    static constexpr const char* PassName = "RedundantMovEliminationPass";
    static char ID;

    PassID getPassID() const override {
        return &ID;
    }

    const char* getName() const override {
        return PassName;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        // Process all basic blocks
        for (BasicBlock& bb : func) {
            // Skip filtered basic blocks
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;

            runOnBasicBlock(bb);
        }
        return preserveCFGAnalyses();
    }

   private:
    int runOnBasicBlock(BasicBlock& bb) {
        // Simple duplicate elimination within a basic block:
        // Rule: If two instructions have same opcode + same operands (dest and src),
        //       remove the second one as it's redundant
        // Only applies to specific instruction types (v_mov_b32, s_mov_b32, etc.)

        std::vector<StinkyInstruction*> blockInstructions;

        // Collect all instructions
        for (IRBase& irNode : bb) {
            if (irNode.getType() == IRBase::IRType::StinkyTofu) {
                blockInstructions.push_back(cast<StinkyInstruction>(&irNode));
            }
        }

        // duplicate → original mapping
        std::vector<std::pair<StinkyInstruction*, StinkyInstruction*>> toReplace;

        // For each instruction, check if it's identical to any previous instruction
        for (size_t i = 0; i < blockInstructions.size(); ++i) {
            StinkyInstruction* inst = blockInstructions[i];

            // Only consider eligible instructions
            if (!isEligibleForRedundantMovElimination(*inst)) continue;

            // Look backward to find an identical instruction
            for (size_t j = 0; j < i; ++j) {
                StinkyInstruction* prevInst = blockInstructions[j];

                // Skip if previous instruction is not eligible
                if (!isEligibleForRedundantMovElimination(*prevInst)) continue;

                // Check if instructions are identical
                if (areInstructionsIdentical(inst, prevInst)) {
                    // Before marking as duplicate, verify source registers haven't been modified
                    bool srcModified = false;
                    const StinkyRegister& srcReg = inst->getSrcRegs()[0];

                    // Check if source was modified between j and i
                    for (size_t k = j + 1; k < i; ++k) {
                        StinkyInstruction* intermediateInst = blockInstructions[k];
                        for (const StinkyRegister& destReg : intermediateInst->getDestRegs()) {
                            if (destReg.isRegister() && destReg == srcReg) {
                                srcModified = true;
                                break;
                            }
                        }
                        if (srcModified) break;
                    }

                    if (!srcModified) {
                        toReplace.push_back({inst, prevInst});
                        break;
                    }
                }
            }
        }

        for (auto [duplicate, original] : toReplace) {
            DefUseChainUpdater::replaceAllUsesWith(duplicate, original);
            DefUseChainUpdater::eraseAndUnlink(duplicate);
        }

        return static_cast<int>(toReplace.size());
    }
};

char RedundantMovEliminationPassImpl::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createRedundantMovEliminationPass() {
    return std::make_unique<RedundantMovEliminationPassImpl>();
}
}  // namespace stinkytofu
