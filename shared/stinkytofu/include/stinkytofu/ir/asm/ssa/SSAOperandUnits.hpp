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

// How a physical operand maps onto attached SSA values.
//
// Lifting, SSA destruction, and the ssaForm printer all walk srcRegs/destRegs
// alongside AttachedSSA, so they must agree on how many value slots each
// operand consumes. Disagreeing by one silently shifts every later operand onto
// the wrong value, so the rule lives here once instead of in each of them.

#include <cstddef>

#include "stinkytofu/ir/asm/StinkyRegister.hpp"

namespace stinkytofu {

/// Classes whose registers become SSA values.
///
/// VCC and EXEC are their own register types rather than SGPR indices, so
/// widening to SGPRs cannot make a scalar operand alias a special register.
/// Accumulators stay out until their VGPR aliasing rules are modelled: on some
/// architectures an AGPR and a VGPR name the same storage, and two SSA values
/// over one physical register would be unsound.
inline bool isLiftableRegClass(RegType type) {
    return type == RegType::V || type == RegType::S;
}

/// Number of SSA value slots \p reg contributes to attached SSA, one per DWORD.
///
/// Zero means the operand is not lifted, and carries an immediate payload
/// instead: a literal, a hwreg, a special or pseudo register, an unresolved
/// template virtual register, or a register class outside VGPR/SGPR.
inline size_t liftedSSAUnits(const StinkyRegister& reg) {
    if (!reg.isRegister() || reg.isVirtualReg()) return 0;
    if (isPseudoReg(reg)) return 0;
    if (!isAllocatableReg(reg.reg.type) || !isLiftableRegClass(reg.reg.type)) return 0;
    return reg.reg.num;
}

}  // namespace stinkytofu
