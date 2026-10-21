// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/ra/LoopMsbCost.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {

LoopMsbCost::LoopMsbCost(const Function& function, const std::vector<Loop>& loops,
                         UnusedVgprMsbSlots unused)
    : unused_(unused) {
    std::unordered_map<const BasicBlock*, std::unordered_set<const BasicBlock*>> bodies;
    for (const Loop& loop : loops) {
        if (loop.headerBB == nullptr) continue;
        bodies[loop.headerBB].insert(loop.bodyBBs.begin(), loop.bodyBBs.end());
    }
    std::unordered_set<const BasicBlock*> walked;
    for (const auto& [header, body] : bodies) {
        const bool innermost = std::none_of(bodies.begin(), bodies.end(), [&](const auto& other) {
            return other.first != header && body.count(other.first) != 0;
        });
        if (innermost) walked.insert(body.begin(), body.end());
    }
    if (walked.empty()) return;

    // In function order, so a count does not depend on how the loops were found.
    const RegClassSet& lifted = function.ssaArena().liftedClasses();
    for (const BasicBlock& block : function) {
        if (walked.count(&block) == 0) continue;
        steps_.push_back({});
        for (const IRBase& ir : block) {
            const auto* instruction = dyn_cast<StinkyInstruction>(&ir);
            if (instruction == nullptr) continue;
            if (instruction->getUnifiedOpcode() == GFX::LABEL || isCall(*instruction)) {
                steps_.push_back({});
                continue;
            }
            if (isPseudoInst(instruction) || !carriesVgprMsb(*instruction)) continue;
            steps_.push_back({instruction, instruction->hasAttachedSSA()
                                               ? operandGroupsOf(*instruction, lifted)
                                               : OperandGroups{}});
        }
    }
    size_t runEnd = steps_.size();
    for (size_t i = steps_.size(); i-- > 0;) {
        if (steps_[i].instruction == nullptr) runEnd = i;
        steps_[i].runEnd = runEnd;
    }
}

uint32_t LoopMsbCost::of(const AllocationResult& colouring) const {
    return count([&](SSAValueID value, const StinkyRegister& reg) {
        return colouring.isAssigned(value) ? colouring.assignmentOf(value).idx : reg.reg.idx;
    });
}

uint32_t LoopMsbCost::ofProducer() const {
    return count([](SSAValueID, const StinkyRegister& reg) { return reg.reg.idx; });
}

}  // namespace stinkytofu
