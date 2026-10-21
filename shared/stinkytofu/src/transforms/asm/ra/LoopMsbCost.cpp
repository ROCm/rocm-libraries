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
    // Each block of an innermost loop, with the loop it belongs to.
    std::unordered_map<const BasicBlock*, const BasicBlock*> loopOf;
    for (const auto& [header, body] : bodies) {
        const bool innermost = std::none_of(bodies.begin(), bodies.end(), [&](const auto& other) {
            return other.first != header && body.count(other.first) != 0;
        });
        if (!innermost) continue;
        for (const BasicBlock* block : body) loopOf.emplace(block, header);
    }
    if (loopOf.empty()) return;

    // In function order, so a count does not depend on how the loops were found.
    const RegClassSet& lifted = function.ssaArena().liftedClasses();
    std::vector<const BasicBlock*> loopOfStep;
    for (const BasicBlock& block : function) {
        const auto found = loopOf.find(&block);
        if (found == loopOf.end()) continue;
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
        loopOfStep.resize(steps_.size(), found->second);
    }
    size_t runEnd = steps_.size();
    for (size_t i = steps_.size(); i-- > 0;) {
        if (steps_[i].instruction == nullptr) runEnd = i;
        steps_[i].runEnd = runEnd;
    }
    // The main loop is the innermost loop with the most instructions, the
    // first in function order on a tie. Weighting its switches above the count
    // of all the others makes mainLoopFirstCount lexicographic.
    std::unordered_map<const BasicBlock*, uint64_t> sizeOf;
    const BasicBlock* main = nullptr;
    for (size_t i = 0; i < steps_.size(); ++i) {
        if (steps_[i].instruction == nullptr) continue;
        const uint64_t size = ++sizeOf[loopOfStep[i]];
        if (main == nullptr || size > sizeOf[main]) main = loopOfStep[i];
    }
    const uint64_t mainWeight = steps_.size() + 1;
    for (size_t i = 0; i < steps_.size(); ++i)
        steps_[i].weight = loopOfStep[i] == main ? mainWeight : 1;
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
