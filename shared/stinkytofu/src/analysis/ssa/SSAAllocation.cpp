/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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
#include "stinkytofu/analysis/ssa/SSAAllocation.hpp"

#include <cassert>

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {
namespace {

const RegKey kUnassigned{RegType::UNKNOWN, 0, RegHalf::NONE};

void mixWord(uint64_t& hash, uint64_t word) {
    hash ^= word + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
}

void mixOperand(uint64_t& hash, const StinkyRegister& reg) {
    if (!reg.isRegister()) {
        mixWord(hash, 0);
        return;
    }
    mixWord(hash, 1);
    mixWord(hash, static_cast<uint64_t>(reg.reg.type));
    mixWord(hash, reg.reg.idx);
    mixWord(hash, reg.reg.num);
}

}  // namespace

uint64_t computeFunctionShape(const Function& function) {
    uint64_t hash = 0x27d4eb2f165667c5ULL;
    for (const BasicBlock& bb : function) {
        mixWord(hash, bb.getPredecessors().size());
        mixWord(hash, bb.getSuccessors().size());
        for (const IRBase& ir : bb) {
            const auto* instruction = dyn_cast<StinkyInstruction>(&ir);
            if (instruction == nullptr) continue;
            mixWord(hash, instruction->getUnifiedOpcode());
            for (const StinkyRegister& reg : instruction->getSrcRegs()) mixOperand(hash, reg);
            // Separator, so moving one operand from sources to destinations
            // cannot produce the same hash.
            mixWord(hash, 0);
            for (const StinkyRegister& reg : instruction->getDestRegs()) mixOperand(hash, reg);
        }
    }
    return hash == kUnstampedShape ? 1 : hash;
}

AllocationResult::AllocationResult(const Function& function)
    : byValue_(function.ssaArena().valueCount() + 1, kUnassigned),
      shape_(function.ssaArena().shape()) {}

void AllocationResult::assign(SSAValueID id, RegKey physical) {
    assert(id != kInvalidSSAValueID && id < byValue_.size() && "invalid SSA value ID");
    byValue_[id] = physical;
}

bool AllocationResult::isAssigned(SSAValueID id) const {
    if (id == kInvalidSSAValueID || id >= byValue_.size()) return false;
    return byValue_[id].type != RegType::UNKNOWN;
}

RegKey AllocationResult::assignmentOf(SSAValueID id) const {
    assert(isAssigned(id) && "value has no physical register");
    return byValue_[id];
}

size_t AllocationResult::valueCount() const {
    return byValue_.empty() ? 0 : byValue_.size() - 1;
}

uint64_t AllocationResult::shape() const {
    return shape_;
}

size_t AllocationResult::unassignedCount() const {
    size_t unassigned = 0;
    for (size_t id = 1; id < byValue_.size(); ++id) {
        if (byValue_[id].type == RegType::UNKNOWN) ++unassigned;
    }
    return unassigned;
}

AllocationResult createLegacyColoring(const Function& function) {
    AllocationResult result(function);
    for (StinkySSAValue* value : function.ssaArena().values()) {
        if (value == nullptr || !value->hasPhysicalBinding()) continue;
        const StinkySSAValue::PhysicalBinding& binding = value->physical();
        result.assign(value->valueId(), RegKey{binding.type, binding.idx, RegHalf::NONE});
    }
    return result;
}

}  // namespace stinkytofu
