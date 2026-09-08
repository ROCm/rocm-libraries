// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/TieExecMaskedWritesPass.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/support/OptimizationRemark.hpp"
#include "stinkytofu/transforms/asm/ExecMaskGrouping.hpp"

#define DEBUG_TYPE "TieExecMaskedWritesPass"

namespace {
using namespace stinkytofu;

constexpr const char* kPassName = "TieExecMaskedWrites";

/// True for a destination whose lanes the exec mask gates. A scalar register
/// holds one value per wave, so no lane mask applies to it, and a pseudo
/// register is a dependency token rather than storage.
bool isLaneMaskedDest(const StinkyRegister& reg) {
    if (!reg.isRegister() || reg.isVirtualReg()) return false;
    if (isPseudoReg(reg)) return false;
    return reg.reg.type == RegType::V;
}

/// Appends \p reg as a source unless an equal operand is already there, which
/// is both the ToStinkyAsmPass normalization and what makes this idempotent. An
/// instruction that already reads its destination -- the producer's own
/// `v_add_nc_u32 v0, v0, 1` -- needs nothing.
bool appendPreservedRead(StinkyInstruction& instruction, const StinkyRegister& reg) {
    for (const StinkyRegister& src : instruction.getSrcRegs()) {
        if (src == reg) return false;
    }
    instruction.addSrcReg(reg);
    return true;
}

class TieExecMaskedWritesPassImpl : public Pass {
   public:
    static char ID;

    const char* getName() const override {
        return "TieExecMaskedWritesPass";
    }

    PassID getPassID() const override {
        return &TieExecMaskedWritesPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager&) override {
        const uint32_t wavefrontSize = passCtx.getWavefrontSize();
        size_t added = 0;
        size_t unmatchedBlocks = 0;

        // Deliberately not filtered by shouldProcessBasicBlock: the operand this
        // adds states a property of the instruction rather than an optimization
        // to bisect, and both the lift and allocation refuse outright when block
        // filtering is active, so a partially normalized function never reaches
        // an allocator anyway.
        for (BasicBlock& bb : func) {
            bool unmatched = false;
            const std::unordered_set<const StinkyInstruction*> masked =
                execMaskedInstructions(bb, wavefrontSize, &unmatched);
            if (unmatched) ++unmatchedBlocks;
            if (masked.empty()) continue;

            for (IRBase& ir : bb) {
                auto* instruction = dyn_cast<StinkyInstruction>(&ir);
                if (instruction == nullptr) continue;
                if (masked.count(instruction) == 0) continue;
                for (const StinkyRegister& dest : instruction->getDestRegs()) {
                    if (!isLaneMaskedDest(dest)) continue;
                    if (appendPreservedRead(*instruction, dest)) ++added;
                }
            }
        }

        PASS_DEBUG(std::cerr << "[TieExecMaskedWritesPass] @" << func.getName() << " added "
                             << added << " preserved-lane read(s)\n");

        if (added > 0) {
            emitRemark(passCtx, {OptimizationRemark::Kind::Passed, kPassName, "TiedWrites",
                                 "@" + func.getName() + ": gave " + std::to_string(added) +
                                     " exec-masked vector write(s) a read of the lanes they "
                                     "preserve"});
        }
        // A span that does not close in its own block leaves the mask narrow on
        // the way out, and this pass cannot say which successors it still covers.
        // Worth a remark rather than silence: a masked write missed there is the
        // very defect the pass exists to prevent.
        if (unmatchedBlocks > 0) {
            emitRemark(passCtx, {OptimizationRemark::Kind::Missed, kPassName, "UnmatchedExecSpan",
                                 "@" + func.getName() + ": " + std::to_string(unmatchedBlocks) +
                                     " block(s) end with a narrow exec mask; writes the mask still "
                                     "covers in their successors are not tied"});
        }

        return preserveCFGAnalyses();
    }
};

char TieExecMaskedWritesPassImpl::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createTieExecMaskedWritesPass() {
    return std::make_unique<TieExecMaskedWritesPassImpl>();
}
}  // namespace stinkytofu
