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
#pragma once

// Program points for register allocation.
//
// Live intervals need a total order over the function with room to distinguish
// "reads its operands" from "writes its result" at the same instruction. Without
// that distinction a read-modify-write operand would look like two overlapping
// values and could never share a register, which would make even the legacy
// colouring fail verification.
//
// Each instruction therefore gets two consecutive indexes:
//
//   base + 0   use point   operands are read here
//   base + 1   def point   results are written here
//
// Every block also gets a leading pair whose def point is where block arguments
// are defined; a value live into the block starts at the block's first index.
// Indexes are assigned in block-list order, which is emission order, so a
// block's indexes are contiguous and adjacent blocks are numerically adjacent.

#include <cstdint>
#include <unordered_map>

namespace stinkytofu {

class BasicBlock;
class Function;
struct StinkyInstruction;

using SlotIndex = uint32_t;

class SSASlotIndexes {
   public:
    static constexpr SlotIndex kInvalidSlot = ~0u;

    /// First index of \p block, where a value live into the block starts.
    SlotIndex blockStart(const BasicBlock* block) const;

    /// Where \p block's arguments are defined.
    SlotIndex blockArgDef(const BasicBlock* block) const;

    /// One past \p block's last index, where a value live out of it ends.
    SlotIndex blockEnd(const BasicBlock* block) const;

    /// Where \p instruction reads its operands.
    SlotIndex useSlot(const StinkyInstruction* instruction) const;

    /// Where \p instruction writes its results.
    SlotIndex defSlot(const StinkyInstruction* instruction) const;

    /// One past the last index in the function.
    SlotIndex slotCount() const {
        return slotCount_;
    }

    size_t instructionCount() const {
        return byInstruction_.size();
    }

    size_t blockCount() const {
        return byBlock_.size();
    }

   private:
    friend SSASlotIndexes computeSSASlotIndexes(const Function& function);

    struct BlockRange {
        SlotIndex start = 0;
        SlotIndex end = 0;
    };

    std::unordered_map<const BasicBlock*, BlockRange> byBlock_;
    std::unordered_map<const StinkyInstruction*, SlotIndex> byInstruction_;
    SlotIndex slotCount_ = 0;
};

/// Number \p function in block-list order. Only StinkyInstructions are numbered.
SSASlotIndexes computeSSASlotIndexes(const Function& function);

}  // namespace stinkytofu
