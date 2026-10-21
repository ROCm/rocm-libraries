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
/// which is what one iteration pays. \p unused has to be the rule the pass will
/// emit by (AsmCapsConfig::vgprMsbKeepUnusedSlots), or the count predicts a
/// different program.
class LoopMsbCost {
   public:
    LoopMsbCost(const Function& function, const std::vector<Loop>& loops,
                UnusedVgprMsbSlots unused = UnusedVgprMsbSlots::Zero);

    /// The count if each operand of a lifted class were at the VGPR index
    /// `indexOf(value, operand)` gives, where `value` is the SSA value in the
    /// operand's first register unit. An operand the lift left physical keeps
    /// its own register.
    template <typename IndexOf>
    uint32_t count(IndexOf&& indexOf) const {
        return static_cast<uint32_t>(walk(indexOf, /*weighted=*/false));
    }

    /// count(), with the largest innermost loop -- a GEMM's main loop -- ranked
    /// first: one switch there outweighs every switch in the other loops, which
    /// run fewer times, such as its tail. A search on this does not buy a
    /// cheaper tail with a costlier main loop.
    template <typename IndexOf>
    uint64_t mainLoopFirstCount(IndexOf&& indexOf) const {
        return walk(indexOf, /*weighted=*/true);
    }

    /// The count under \p colouring.
    uint32_t of(const AllocationResult& colouring) const;

    /// The count with every operand where the producer put it.
    uint32_t ofProducer() const;

    /// False for a function with no loop, where every count is 0.
    bool hasLoops() const {
        return !steps_.empty();
    }

    /// Calls `visit(value)` for every operand of a lifted class that a loop
    /// instruction names through a bank slot, with the value in the operand's
    /// first register unit: the values whose bank the count depends on.
    template <typename Visit>
    void forEachSlotValue(Visit&& visit) const;

   private:
    /// One loop instruction and the values behind its operands, or a point
    /// where the byte in force is forgotten: a block start, a label, a call.
    struct Step {
        const StinkyInstruction* instruction = nullptr;
        OperandGroups values;
        /// One past the last step before the next forget point.
        size_t runEnd = 0;
        /// What a switch here counts for in mainLoopFirstCount().
        uint64_t weight = 1;
    };

    template <typename IndexOf>
    uint64_t walk(IndexOf&& indexOf, bool weighted) const;

    std::vector<Step> steps_;
    UnusedVgprMsbSlots unused_;
};

template <typename IndexOf>
uint64_t LoopMsbCost::walk(IndexOf&& indexOf, bool weighted) const {
    static_assert(std::is_invocable_r_v<uint32_t, IndexOf, SSAValueID, const StinkyRegister&>,
                  "LoopMsbCost::count lookup must be callable as "
                  "(SSAValueID value, const StinkyRegister& operand) -> uint32_t");
    // Every need first: under Keep, a new byte looks ahead to the later ones.
    std::vector<VgprMsbNeed> needs(steps_.size());
    for (size_t i = 0; i < steps_.size(); ++i) {
        const Step& step = steps_[i];
        if (step.instruction == nullptr) continue;
        needs[i] = vgprMsbNeed(
            *step.instruction,
            [&](const StinkyRegister& reg, size_t operand, bool isDest) -> uint32_t {
                const std::span<const SSAValueID> units = step.values.at(operand, isDest);
                return units.empty() ? reg.reg.idx : indexOf(units.front(), reg);
            });
    }
    uint64_t emitted = 0;
    VgprMsbState state(unused_);
    for (size_t i = 0; i < steps_.size(); ++i) {
        if (steps_[i].instruction == nullptr) {
            state.forget();
            continue;
        }
        if (!state.needsSet(needs[i])) continue;
        state.set(needs[i],
                  std::span<const VgprMsbNeed>(needs).subspan(i + 1, steps_[i].runEnd - i - 1));
        emitted += weighted ? steps_[i].weight : 1;
    }
    return emitted;
}

template <typename Visit>
void LoopMsbCost::forEachSlotValue(Visit&& visit) const {
    for (const Step& step : steps_) {
        if (step.instruction == nullptr) continue;
        forEachVgprOperandField(*step.instruction, [&](const StinkyRegister& reg, size_t operand,
                                                       bool isDest, int slot) {
            if (slot < 0 || getMsbFromVgpr(reg) < 0) return;
            const std::span<const SSAValueID> units = step.values.at(operand, isDest);
            if (!units.empty()) visit(units.front());
        });
    }
}

}  // namespace stinkytofu
