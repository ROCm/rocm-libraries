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

/// Only plain moves: repeating any other one-source op (v_not_b32, a conversion,
/// s_getreg) is not a no-op, and the second copy computes something new.
bool isEligibleForRedundantMovElimination(const StinkyInstruction& inst) {
    if (inst.getDestRegs().size() != 1 || inst.getSrcRegs().size() != 1) return false;
    if (mustPreserveInstruction(inst)) return false;
    switch (inst.getUnifiedOpcode()) {
        case GFX::v_mov_b32:
        case GFX::s_mov_b32:
        case GFX::s_mov_b64:
            return true;
        default:
            return false;
    }
}

bool writesExec(const StinkyInstruction& inst) {
    if (inst.is(InstFlag::IF_ImplicitWriteEXEC)) return true;
    for (const StinkyRegister& d : inst.getDestRegs()) {
        if (!d.isRegister()) continue;
        if (d.reg.type == RegType::EXEC || d.reg.type == RegType::EXEC_LO ||
            d.reg.type == RegType::EXEC_HI)
            return true;
    }
    return false;
}

/// Whether \p between leaves the value \p original put in its destination intact
/// for an identical move: nothing may write the destination or the source (by
/// overlap, not operand equality), and a VALU move also needs the same EXEC.
/// A null entry is IR this pass cannot read (e.g. a text block) and ends the run.
bool valueSurvives(const StinkyInstruction& original,
                   const std::vector<StinkyInstruction*>& between) {
    const StinkyRegister& dst = original.getDestRegs()[0];
    const StinkyRegister& src = original.getSrcRegs()[0];
    if (dst.isOverlap(src)) return false;
    const bool isValu = original.getUnifiedOpcode() == GFX::v_mov_b32;
    for (const StinkyInstruction* inst : between) {
        if (inst == nullptr || isLabel(*inst) || isBranch(*inst)) return false;
        if (isValu && writesExec(*inst)) return false;
        for (const StinkyRegister& d : inst->getDestRegs())
            if (d.isOverlap(dst) || d.isOverlap(src)) return false;
    }
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

        // Collect all instructions; null stands for IR that is not an instruction
        for (IRBase& irNode : bb) {
            blockInstructions.push_back(irNode.getType() == IRBase::IRType::StinkyTofu
                                            ? cast<StinkyInstruction>(&irNode)
                                            : nullptr);
        }

        // duplicate → original mapping
        std::vector<std::pair<StinkyInstruction*, StinkyInstruction*>> toReplace;

        // For each instruction, check if it's identical to any previous instruction
        for (size_t i = 0; i < blockInstructions.size(); ++i) {
            StinkyInstruction* inst = blockInstructions[i];

            // Only consider eligible instructions
            if (inst == nullptr || !isEligibleForRedundantMovElimination(*inst)) continue;

            // Look backward to find an identical instruction
            for (size_t j = 0; j < i; ++j) {
                StinkyInstruction* prevInst = blockInstructions[j];

                // Skip if previous instruction is not eligible
                if (prevInst == nullptr || !isEligibleForRedundantMovElimination(*prevInst))
                    continue;

                if (!areInstructionsIdentical(inst, prevInst)) continue;
                const std::vector<StinkyInstruction*> between(blockInstructions.begin() + j + 1,
                                                              blockInstructions.begin() + i);
                if (valueSurvives(*prevInst, between)) {
                    toReplace.push_back({inst, prevInst});
                    break;
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
