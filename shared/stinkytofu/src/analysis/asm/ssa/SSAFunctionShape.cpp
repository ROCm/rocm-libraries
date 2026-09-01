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
#include "stinkytofu/analysis/asm/ssa/SSAFunctionShape.hpp"

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyRegister.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {
namespace {

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
            // Separator, so moving one register operand from sources to
            // destinations cannot produce the same hash. It shares its value
            // with the non-register marker above, so the same does not hold for
            // a non-register operand crossing that boundary.
            mixWord(hash, 0);
            for (const StinkyRegister& reg : instruction->getDestRegs()) mixOperand(hash, reg);
        }
    }
    return hash == kUnstampedShape ? 1 : hash;
}

}  // namespace stinkytofu
