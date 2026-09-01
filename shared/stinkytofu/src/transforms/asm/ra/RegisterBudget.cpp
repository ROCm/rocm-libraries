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
#include "stinkytofu/transforms/asm/ra/RegisterBudget.hpp"

#include <algorithm>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {
namespace {

/// One past the last index this operand covers, or 0 when it is not a physical
/// register of \p regClass. A multi-DWORD operand names only its base, so the
/// width is what decides the count. A virtual register carries kVirtualBit in
/// its index and would swamp the maximum, so it is skipped rather than counted.
uint32_t endOf(const StinkyRegister& reg, RegType regClass) {
    if (reg.dataType != StinkyRegister::Type::Register) return 0;
    if (reg.isVirtualReg()) return 0;
    if (reg.reg.type != regClass) return 0;
    const uint32_t width = std::max<uint16_t>(1, reg.reg.num);
    return reg.reg.idx + width;
}

}  // namespace

uint32_t highestRegisterCount(const Function& function, RegType regClass) {
    uint32_t count = 0;
    for (const BasicBlock& block : function) {
        for (const IRBase& ir : block) {
            const auto* instruction = dyn_cast<StinkyInstruction>(&ir);
            if (instruction == nullptr) continue;
            for (const StinkyRegister& reg : instruction->getDestRegs())
                count = std::max(count, endOf(reg, regClass));
            for (const StinkyRegister& reg : instruction->getSrcRegs())
                count = std::max(count, endOf(reg, regClass));
        }
    }
    return count;
}

uint32_t requiredSgprCount(const Function& function, int numSgprPreload,
                           const std::array<int, 3>& workgroupIds) {
    const uint32_t used = highestRegisterCount(function, RegType::S);

    // The kernarg segment pointer occupies two, matching the `numSgprPreload + 2`
    // the descriptor emits as .amdhsa_user_sgpr_count.
    uint32_t abi = numSgprPreload > 0 ? static_cast<uint32_t>(numSgprPreload) + 2u : 0u;
    for (int enabled : workgroupIds) {
        if (enabled > 0) ++abi;
    }
    return std::max(used, abi);
}

}  // namespace stinkytofu
