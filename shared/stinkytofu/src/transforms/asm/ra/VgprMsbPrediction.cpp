// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/ra/VgprMsbPrediction.hpp"

#include <span>
#include <unordered_map>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationConstraints.hpp"

namespace stinkytofu {

std::vector<LoopVgprMsbSwitches> predictLoopVgprMsbSwitches(const Function& function,
                                                            const AllocationResult& colouring,
                                                            const std::vector<Loop>& loops,
                                                            uint32_t vgprIndexCount,
                                                            VgprMsbPolicy policy) {
    std::vector<LoopVgprMsbSwitches> predicted(loops.size());
    std::unordered_map<const BasicBlock*, std::vector<size_t>> loopsOf;
    for (size_t index = 0; index < loops.size(); ++index) {
        if (loops[index].headerBB != nullptr)
            predicted[index].header = loops[index].headerBB->getLabel();
        predicted[index].switches = 0;
        for (const BasicBlock* bb : loops[index].bodyBBs) loopsOf[bb].push_back(index);
    }

    const RegClassSet& lifted = function.ssaArena().liftedClasses();
    for (const BasicBlock& bb : function) {
        // The pass starts every block from "no word", so a block outside every
        // loop cannot change what a loop's blocks count.
        const auto owners = loopsOf.find(&bb);
        if (owners == loopsOf.end()) continue;

        VgprMsbTracker tracker(policy);
        for (const IRBase& ir : bb) {
            const auto* inst = dyn_cast<StinkyInstruction>(&ir);
            if (inst == nullptr) continue;
            if (inst->getUnifiedOpcode() == GFX::LABEL) {
                tracker.label();
                continue;
            }
            if (isPseudoInst(inst)) continue;
            if (isCall(*inst)) {
                tracker.reset();
                continue;
            }
            if (inst->getUnifiedOpcode() == GFX::s_set_vgpr_msb) {
                tracker.observeSetter(vgprMsbSetterWord(*inst));
                continue;
            }

            const OperandGroups groups =
                inst->hasAttachedSSA() ? operandGroupsOf(*inst, lifted) : OperandGroups{};
            bool encodable = true;
            const auto bankOf = [&](const StinkyRegister& reg, size_t operand, bool isDest) {
                uint32_t idx = reg.reg.idx;
                const std::span<const SSAValueID> ids = groups.at(operand, isDest);
                if (!ids.empty() && ids.front() != kInvalidSSAValueID &&
                    colouring.isAssigned(ids.front()))
                    idx = colouring.assignmentOf(ids.front()).idx;
                if (idx >= vgprIndexCount) encodable = false;
                return static_cast<int>(idx / kVgprBankSize);
            };
            const VgprMsbRequirement required = requiredVgprMsbWith(*inst, bankOf);
            if (!encodable) {
                for (const size_t owner : owners->second) predicted[owner].switches.reset();
            }

            if (!tracker.step(required).emit) continue;
            for (const size_t owner : owners->second) {
                if (predicted[owner].switches.has_value()) ++*predicted[owner].switches;
            }
        }
    }
    return predicted;
}

}  // namespace stinkytofu
