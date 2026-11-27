// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"

namespace stinkytofu {

enum class OperandSide { Src, Dst };

/// The True16 half selector of the operand printed at `printedIndex` on `side`; NONE when the
/// instruction has no selector there. Selectors are indexed by PRINTED position, the convention
/// rocisa builds them with (ToStinkyTofuUtils attachTrue16ModifiersFromOperands) and the emitter
/// prints them with (StinkyAsmEmitter emitOperands): pseudo and implicit operands are skipped,
/// every other operand (VGPR, SGPR, constant, literal) takes one slot.
inline HighBitSel true16HalfAt(const True16Modifiers* mod, OperandSide side, size_t printedIndex) {
    if (mod == nullptr) return HighBitSel::NONE;
    if (side == OperandSide::Src) return mod->getSrc(printedIndex);
    if (printedIndex == 0) return mod->getDst0();
    if (printedIndex == 1) return mod->getDst1();
    return HighBitSel::NONE;
}

/// Calls fn(vgprIdx, half) for every VGPR of every VGPR operand on `side` of `inst`, with the
/// True16 half the emitter prints for that operand (NONE = the whole 32 bits). Read-modify-write
/// duplicates that the IR appends after the printed sources are visited too, as NONE.
template <typename Fn>
inline void forEachVgprHalf(const StinkyInstruction& inst, OperandSide side, Fn&& fn) {
    const std::vector<StinkyRegister>& regs =
        side == OperandSide::Src ? inst.getSrcRegs() : inst.getDestRegs();
    const True16Modifiers* mod = inst.getModifier<True16Modifiers>();
    size_t printed = 0;
    for (const StinkyRegister& reg : regs) {
        if (isPseudoReg(reg)) continue;
        if (side == OperandSide::Src ? isImplicitSrc(reg, inst) : isImplicitDest(reg, inst))
            continue;
        const size_t index = printed++;
        if (reg.dataType != StinkyRegister::Type::Register || reg.reg.type != RegType::V) continue;
        const HighBitSel half = true16HalfAt(mod, side, index);
        for (uint16_t off = 0; off < reg.reg.num; ++off) fn(reg.reg.idx + off, half);
    }
}

}  // namespace stinkytofu
