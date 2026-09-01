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
#include "stinkytofu/analysis/asm/ssa/SSASlotIndexes.hpp"

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {
namespace {
/// Indexes per program point: one for reads, one for writes.
constexpr SlotIndex kPointsPerStep = 2;
}  // namespace

SlotIndex SSASlotIndexes::blockStart(const BasicBlock* block) const {
    auto it = byBlock_.find(block);
    return it == byBlock_.end() ? kInvalidSlot : it->second.start;
}

SlotIndex SSASlotIndexes::blockArgDef(const BasicBlock* block) const {
    const SlotIndex start = blockStart(block);
    return start == kInvalidSlot ? kInvalidSlot : start + 1;
}

SlotIndex SSASlotIndexes::blockEnd(const BasicBlock* block) const {
    auto it = byBlock_.find(block);
    return it == byBlock_.end() ? kInvalidSlot : it->second.end;
}

SlotIndex SSASlotIndexes::useSlot(const StinkyInstruction* instruction) const {
    auto it = byInstruction_.find(instruction);
    return it == byInstruction_.end() ? kInvalidSlot : it->second;
}

SlotIndex SSASlotIndexes::defSlot(const StinkyInstruction* instruction) const {
    const SlotIndex use = useSlot(instruction);
    return use == kInvalidSlot ? kInvalidSlot : use + 1;
}

SSASlotIndexes computeSSASlotIndexes(const Function& function) {
    SSASlotIndexes indexes;
    SlotIndex next = 0;

    for (const BasicBlock& block : function) {
        SSASlotIndexes::BlockRange range;
        range.start = next;
        next += kPointsPerStep;  // block arguments are defined at start + 1

        for (const IRBase& ir : block) {
            const auto* instruction = dyn_cast<StinkyInstruction>(&ir);
            if (instruction == nullptr) continue;
            indexes.byInstruction_.emplace(instruction, next);
            next += kPointsPerStep;
        }

        range.end = next;
        indexes.byBlock_.emplace(&block, range);
    }

    indexes.slotCount_ = next;
    return indexes;
}

}  // namespace stinkytofu
