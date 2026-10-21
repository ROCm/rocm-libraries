// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

// What a colouring costs in s_set_vgpr_msb inside loops, before it is applied.

#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"
#include "stinkytofu/ir/asm/ssa/AllocationResult.hpp"
#include "stinkytofu/support/LoopDetection.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationConstraints.hpp"

namespace stinkytofu {

/// The s_set_vgpr_msb a function's innermost loops pay per iteration, counted
/// by InsertVgprMsbPass's rule (vgprMsbNeed, VgprMsbState) for registers that
/// are not written yet.
///
/// Built once from the attached SSA: every instruction of an innermost loop,
/// with the values behind its operands. A count then walks those instructions
/// asking only where each value would sit, so scoring a colouring, or a choice
/// of bank per value, rewrites nothing. An innermost loop is one whose body
/// holds no other loop's header; loops sharing a header are one loop.
///
/// The walk is per block, as the pass's is: the byte in force is unknown where
/// a block starts and after a label or a call. Each loop block is walked once,
/// which is what one iteration pays.
class LoopMsbCost {
   public:
    LoopMsbCost(const Function& function, const std::vector<Loop>& loops);

    /// The count if each operand of a lifted class were at the VGPR index
    /// `indexOf(value, operand)` gives, where `value` is the SSA value in the
    /// operand's first register unit. An operand the lift left physical keeps
    /// its own register.
    template <typename IndexOf>
    uint32_t count(IndexOf&& indexOf) const;

    /// The count under \p colouring.
    uint32_t of(const AllocationResult& colouring) const;

    /// The count with every operand where the producer put it.
    uint32_t ofProducer() const;

    /// False for a function with no loop, where every count is 0.
    bool hasLoops() const {
        return !steps_.empty();
    }

   private:
    /// One loop instruction and the values behind its operands, or a point
    /// where the byte in force is forgotten: a block start, a label, a call.
    struct Step {
        const StinkyInstruction* instruction = nullptr;
        OperandGroups values;
    };
    std::vector<Step> steps_;
};

template <typename IndexOf>
uint32_t LoopMsbCost::count(IndexOf&& indexOf) const {
    static_assert(std::is_invocable_r_v<uint32_t, IndexOf, SSAValueID, const StinkyRegister&>,
                  "LoopMsbCost::count lookup must be callable as "
                  "(SSAValueID value, const StinkyRegister& operand) -> uint32_t");
    uint32_t emitted = 0;
    VgprMsbState state;
    for (const Step& step : steps_) {
        if (step.instruction == nullptr) {
            state.forget();
            continue;
        }
        const VgprMsbNeed need = vgprMsbNeed(
            *step.instruction,
            [&](const StinkyRegister& reg, size_t operand, bool isDest) -> uint32_t {
                const std::span<const SSAValueID> units = step.values.at(operand, isDest);
                return units.empty() ? reg.reg.idx : indexOf(units.front(), reg);
            });
        if (!state.needsSet(need)) continue;
        state.set(need);
        ++emitted;
    }
    return emitted;
}

}  // namespace stinkytofu
