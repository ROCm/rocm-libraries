// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/VgprMsbPlanner.hpp"

#include <optional>

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"

namespace stinkytofu {
namespace {

enum VgprMsbState : int {
    NOT_REQUIRED = -1,
    LABEL_BEGIN = -2,
};

bool isMsbComputableClass(const StinkyInstruction& inst) {
    return !(inst.is(InstFlag::IF_SALU) || inst.is(InstFlag::IF_SMemLoad) ||
             inst.is(InstFlag::IF_SMemStore) || inst.is(InstFlag::IF_SMemAtomic) ||
             inst.is(InstFlag::IF_Branch) || inst.is(InstFlag::IF_Call) ||
             inst.is(InstFlag::IF_Barrier) || inst.is(InstFlag::IF_WaitCnt) ||
             inst.is(InstFlag::IF_HasSideEffect));
}

bool preferInsertAfter(const StinkyInstruction& inst) {
    return isVectorALU(inst) || (isScalarALU(inst) && !isBarrier(inst)) ||
           isMatrixInstruction(inst);
}

}  // namespace

std::vector<VgprMsbInsertion> planVgprMsb(const std::vector<const StinkyInstruction*>& insts,
                                          VgprMsbMode mode) {
    std::vector<VgprMsbInsertion> plan;
    int currentMsb = VgprMsbState::NOT_REQUIRED;
    // A switch needed later goes right after the last ALU or matrix op when only
    // instructions with no MSB opinion (SALU, waits, ...) sit in between.
    std::optional<size_t> preferredBefore;

    for (size_t i = 0; i < insts.size(); ++i) {
        const StinkyInstruction* inst = insts[i];
        if (inst->getUnifiedOpcode() == GFX::LABEL) {
            currentMsb = VgprMsbState::LABEL_BEGIN;
            preferredBefore.reset();
            continue;
        }
        if (isPseudoInst(inst)) continue;

        // A call (e.g. s_swappc_b64) transfers to a callee that may leave the VGPR MSB
        // hardware register in an unknown state: the next VGPR op re-establishes MSB, and
        // no deferred insertion anchor is carried across the call.
        if (isCall(*inst)) {
            currentMsb = VgprMsbState::NOT_REQUIRED;
            preferredBefore.reset();
            continue;
        }

        const size_t before = preferredBefore ? *preferredBefore : i;
        const auto [requiredMsb, hasVgpr] = computeRequiredMsb(inst);
        bool emitted = false;
        if (!hasVgpr || requiredMsb == currentMsb) {
            if (currentMsb == VgprMsbState::LABEL_BEGIN) currentMsb = VgprMsbState::NOT_REQUIRED;
        } else {
            VgprMsbInsertion insertion;
            insertion.before = before;
            insertion.requiredMsb = requiredMsb;
            insertion.nopFirst = currentMsb == VgprMsbState::LABEL_BEGIN;
            insertion.immediate = requiredMsb;
            if (mode == VgprMsbMode::Msb16 && currentMsb != VgprMsbState::NOT_REQUIRED &&
                currentMsb != VgprMsbState::LABEL_BEGIN)
                insertion.immediate += currentMsb << 8;
            plan.push_back(insertion);
            currentMsb = requiredMsb;
            emitted = true;
        }
        if (emitted || isMsbComputableClass(*inst)) preferredBefore.reset();
        if (preferInsertAfter(*inst)) {
            if (i + 1 < insts.size())
                preferredBefore = i + 1;
            else
                preferredBefore.reset();
        }
    }
    return plan;
}

}  // namespace stinkytofu
