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

#include <cstring>
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
        // Process all basic blocks
        for (BasicBlock& bb : func) {
            // Skip filtered basic blocks
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;

            // Iteratively remove dead stores until fixpoint
            bool changed = true;
            while (changed) {
                int removed = runOnBasicBlock(bb, func);
                changed = (removed > 0);
            }
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

    static bool hasMnemonic(const StinkyInstruction& inst, const char* part) {
        const char* m = inst.getHwInstDesc()->mnemonic;
        return m != nullptr && std::strstr(m, part) != nullptr;
    }

    /// Whether \p inst may read registers its operands do not name: text blocks
    /// (null), M0-indexed moves, and anything past which the block's register
    /// view no longer holds (labels, branches, VGPR MSB changes).
    static bool mayReadAnything(const StinkyInstruction* inst) {
        return inst == nullptr || isLabel(*inst) || isBranch(*inst) || isCall(*inst) ||
               hasMnemonic(*inst, "movrel") || hasMnemonic(*inst, "s_set_vgpr_msb");
    }

    /// Whether \p inst overwrites every lane and bit of its destinations, so that it
    /// can end the previous value's lifetime. Modifiers (true16 halves, DPP, SDWA),
    /// lane writes, M0-indexed writes and memory results do not.
    static bool writesWholeDest(const StinkyInstruction& inst) {
        return inst.getModifiers().empty() && !mustPreserveInstruction(inst) &&
               !hasMnemonic(inst, "writelane") && !hasMnemonic(inst, "movrel") &&
               !hasMnemonic(inst, "d16");
    }

    static bool writesExec(const StinkyInstruction& inst) {
        if (inst.is(InstFlag::IF_ImplicitWriteEXEC)) return true;
        for (const StinkyRegister& d : inst.getDestRegs())
            if (d.isRegister() && (d.reg.type == RegType::EXEC || d.reg.type == RegType::EXEC_LO ||
                                   d.reg.type == RegType::EXEC_HI))
                return true;
        return false;
    }

    /// Whether every dword \p destReg receives from blockInstructions[i] is
    /// overwritten before anything reads it.
    static bool isDeadDef(const std::vector<StinkyInstruction*>& blockInstructions, size_t i,
                          const StinkyRegister& destReg) {
        std::vector<bool> covered(destReg.reg.num, false);
        size_t remaining = destReg.reg.num;
        const bool isVgpr = destReg.reg.type != RegType::S;
        for (size_t j = i + 1; j < blockInstructions.size(); ++j) {
            const StinkyInstruction* later = blockInstructions[j];
            if (mayReadAnything(later)) return false;
            for (const StinkyRegister& src : later->getSrcRegs())
                if (src.isOverlap(destReg)) return false;
            // Lanes disabled under a different EXEC keep the old value.
            if (isVgpr && writesExec(*later)) return false;
            if (!writesWholeDest(*later)) continue;
            for (const StinkyRegister& d : later->getDestRegs()) {
                if (!d.isOverlap(destReg)) continue;
                for (uint32_t r = d.reg.idx; r < d.reg.idx + d.reg.num; ++r) {
                    if (r < destReg.reg.idx || r >= destReg.reg.idx + destReg.reg.num) continue;
                    if (!covered[r - destReg.reg.idx]) {
                        covered[r - destReg.reg.idx] = true;
                        --remaining;
                    }
                }
            }
            if (remaining == 0) return true;
        }
        return false;
    }

    int runOnBasicBlock(BasicBlock& bb, Function& func) {
        // A def is dead only when each of its dwords is overwritten in this block
        // before any read; values reaching the block end may be used by a successor.

        std::vector<StinkyInstruction*> blockInstructions;
        for (IRBase& irNode : bb) {
            blockInstructions.push_back(irNode.getType() == IRBase::IRType::StinkyTofu
                                            ? cast<StinkyInstruction>(&irNode)
                                            : nullptr);
        }

        std::set<StinkyInstruction*> toRemove;

        // For each instruction that writes to a register
        for (size_t i = 0; i < blockInstructions.size(); ++i) {
            StinkyInstruction* inst = blockInstructions[i];
            if (inst == nullptr) continue;

            // Never remove side-effecting instructions (barriers, branches, memory ops, etc.)
            if (mustPreserveInstruction(*inst)) continue;

            // Never remove instructions with no explicit destination registers
            // (these include: comparisons, waits, delays, etc.)
            if (inst->getDestRegs().empty()) continue;

            // A def-use chain still names this as a source.
            if (!inst->getUsers().empty()) continue;

            // Implicit SCC/EXEC writes are not operands, so nothing below sees them die.
            if (inst->is(InstFlag::IF_ImplicitWriteSCC) || inst->is(InstFlag::IF_ImplicitWriteEXEC))
                continue;

            // Special registers (SCC, VCC, EXEC, M0, ...) have implicit readers the
            // operand lists do not show; pseudo registers only carry dependencies.
            bool hasDummyDest = false;
            for (const StinkyRegister& destReg : inst->getDestRegs()) {
                if (!destReg.isRegister() || !isAllocatableReg(destReg.reg.type) ||
                    isPseudoReg(destReg)) {
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

            bool allDead = true;
            for (const StinkyRegister& destReg : inst->getDestRegs()) {
                if (!isDeadDef(blockInstructions, i, destReg)) {
                    allDead = false;
                    break;
                }
            }
            if (allDead) toRemove.insert(inst);
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
