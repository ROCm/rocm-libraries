/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyRegister.hpp"

namespace stinkytofu {

/// Returns the VGPR-MSB slot (0=src0, 1=src1, 2=src2, 3=dst) for a given
/// instruction encoding field, or -1 if the field does not participate in
/// MSB selection.
///
/// This mapping is load-bearing for both directions:
///   - encode (InsertVgprMsbPass): figure out which 2-bit slot of the
///     s_set_vgpr_msb immediate to set when a high-MSB VGPR appears.
///   - decode (RaiseVgprMsbPass): figure out which slot's MSB to apply
///     when raising encoded operands back to full physical indices.
///
/// Drift between the two directions silently corrupts register identity, so
/// both passes must call this single function.
inline int encodeFieldToVgprOffSlot(EncodeField ef) {
    switch (ef) {
        case EncodeField::vdst:
        case EncodeField::vdata:
            return 3;
        case EncodeField::src0:
        case EncodeField::addr:
        case EncodeField::vaddr:
        case EncodeField::vaddr0:
            return 0;
        case EncodeField::src1:
        case EncodeField::vsrc1:
        case EncodeField::data0:
        case EncodeField::vsrc:
        case EncodeField::vaddr1:
            return 1;
        case EncodeField::src2:
        case EncodeField::data1:
        case EncodeField::vaddr2:
            return 2;
        default:
            return -1;
    }
}

/// Extract the 2-bit MSB field for `slot` from an s_set_vgpr_msb immediate.
/// Layout (low byte): [1:0]=src0, [3:2]=src1, [5:4]=src2, [7:6]=dst.
inline int decodeVgprMsbForSlot(int setVal, int slot) {
    return (setVal >> (slot * 2)) & 0x3;
}

/// Pack a 2-bit MSB value for `slot` into the s_set_vgpr_msb immediate layout.
/// Inverse of decodeVgprMsbForSlot. OR together the per-slot results to build
/// the full byte:
///   setVal = encodeVgprMsbForSlot(0, msbSrc0) |
///            encodeVgprMsbForSlot(1, msbSrc1) |
///            encodeVgprMsbForSlot(2, msbSrc2) |
///            encodeVgprMsbForSlot(3, msbDst);
inline int encodeVgprMsbForSlot(int slot, int msb) {
    return (msb & 0x3) << (slot * 2);
}

/// Registers one bank holds, which is what eight bits of operand index can name.
constexpr uint32_t kVgprBankSize = 256;

/// MSB (which VGPR bank) of a VGPR operand, or -1 for non-VGPR operands.
inline int getMsbFromVgpr(const StinkyRegister& reg) {
    if (reg.dataType != StinkyRegister::Type::Register || reg.reg.type != RegType::V) return -1;
    return static_cast<int>(reg.reg.idx) / static_cast<int>(kVgprBankSize);
}

/// The `reg.offset` a VGPR needs for the emitter to print its byte form: the
/// operand text is `idx + offset`, so v272 prints `v[272-256]`, which the
/// assembler reduces to v16 of the selected bank. Derive it, never store it --
/// a register moved between banks keeps a bias that no longer matches its
/// index, and `v[16-256]` is not a register.
inline int getMsbOffsetForVgpr(const StinkyRegister& reg) {
    const int msb = getMsbFromVgpr(reg);
    return msb <= 0 ? 0 : msb * -static_cast<int>(kVgprBankSize);
}

/// Calls \p callback once per register operand of \p inst: the operand, its
/// index into getDestRegs() or getSrcRegs(), which of the two, and the MSB slot
/// its encoding field selects. The slot is -1 when the field selects none.
///
/// One implementation, because pairing an operand with the wrong field gives it
/// another field's bank. The instruction then touches a different register and
/// nothing reports an error. encodeFieldToVgprOffSlot is one function for the
/// same reason.
///
/// A slotless field is reported, not skipped: bank selection ignores it, but
/// allocation needs it, since that operand reaches only the first kVgprBankSize
/// registers.
template <typename Callback>
inline void forEachVgprOperandField(const StinkyInstruction& inst, Callback&& callback) {
    static_assert(std::is_invocable_v<Callback, const StinkyRegister&, size_t, bool, int>,
                  "forEachVgprOperandField callback must be callable as "
                  "(const StinkyRegister& operand, size_t index, bool isDest, int slot)");

    const HwInstDesc* desc = inst.getHwInstDesc();
    if (desc == nullptr) return;

    const std::vector<StinkyRegister>& srcRegs = inst.getSrcRegs();
    const std::vector<StinkyRegister>& destRegs = inst.getDestRegs();

    size_t srcIdx = 0;
    size_t dstIdx = 0;
    for (const HwInstDesc::OperandFieldDesc& field : desc->operandFields) {
        const bool isDest = field.isDest || field.isReadWrite;
        const std::vector<StinkyRegister>& regs = isDest ? destRegs : srcRegs;
        size_t& cursor = isDest ? dstIdx : srcIdx;
        if (cursor >= regs.size()) continue;

        const size_t operandIndex = cursor++;
        callback(regs[operandIndex], operandIndex, isDest,
                 encodeFieldToVgprOffSlot(field.encodeField));
    }
}

/// Calls \p callback with each VGPR operand of \p inst that its own field cannot
/// name: the operand, its index, and whether it is a destination.
///
/// A field with no MSB slot has no bank selector, so it reaches one bank only.
/// An index past that still gets the bias above, and the assembler folds it back
/// into range. The instruction then uses a different register, nothing reports
/// an error, and the result is wrong.
///
/// Here rather than in a pass, because it is a property of the encoding: every
/// operand reaching the emitter must satisfy it, whatever chose the register.
template <typename Callback>
inline void forEachUnencodableVgprOperand(const StinkyInstruction& inst, Callback&& callback) {
    static_assert(std::is_invocable_v<Callback, const StinkyRegister&, size_t, bool>,
                  "forEachUnencodableVgprOperand callback must be callable as "
                  "(const StinkyRegister& operand, size_t index, bool isDest)");

    forEachVgprOperandField(inst,
                            [&](const StinkyRegister& reg, size_t operand, bool isDest, int slot) {
                                if (slot >= 0) return;
                                if (reg.dataType != StinkyRegister::Type::Register) return;
                                if (reg.reg.type != RegType::V) return;
                                // The whole range, so a tuple straddling the boundary is caught on
                                // the unit that crosses it rather than passing on its base.
                                const uint32_t width = reg.reg.num < 1u ? 1u : reg.reg.num;
                                if (reg.reg.idx + width <= kVgprBankSize) return;
                                callback(reg, operand, isDest);
                            });
}

/// Record the per-slot VGPR banks of \p inst; \p hasVgpr set if any VGPR is seen.
inline void collectVgprMsbSlots(const StinkyInstruction* inst, int msbSrc[3], int& msbDst,
                                bool& hasVgpr) {
    forEachVgprOperandField(*inst, [&](const StinkyRegister& reg, size_t, bool, int slot) {
        if (slot < 0) return;
        const int msb = getMsbFromVgpr(reg);
        if (msb < 0) return;

        hasVgpr = true;
        if (slot == 3)
            msbDst = msb;
        else
            msbSrc[slot] = msb;
    });
}

/// The s_set_vgpr_msb immediate \p inst needs for its VGPR operands; (setVal, hasVgpr)
/// with hasVgpr false / setVal -1 for ops that carry no VGPR MSB. Shared by the scheduler
/// (MSB-affinity tiebreak) and InsertVgprMsbPass (materialization) so they cannot drift.
inline std::pair<int, bool> computeRequiredMsb(const StinkyInstruction* inst) {
    if (inst->is(InstFlag::IF_SALU) || inst->is(InstFlag::IF_SMemLoad) ||
        inst->is(InstFlag::IF_SMemStore) || inst->is(InstFlag::IF_SMemAtomic) ||
        inst->is(InstFlag::IF_Branch) || inst->is(InstFlag::IF_Call) ||
        inst->is(InstFlag::IF_Barrier) || inst->is(InstFlag::IF_WaitCnt) ||
        inst->is(InstFlag::IF_HasSideEffect)) {
        return {-1, false};
    }

    int msbSrc[3] = {0, 0, 0};
    int msbDst = 0;
    bool hasVgpr = false;

    collectVgprMsbSlots(inst, msbSrc, msbDst, hasVgpr);

    if (!hasVgpr) return {-1, false};

    int setVal = encodeVgprMsbForSlot(0, msbSrc[0]) | encodeVgprMsbForSlot(1, msbSrc[1]) |
                 encodeVgprMsbForSlot(2, msbSrc[2]) | encodeVgprMsbForSlot(3, msbDst);
    return {setVal, true};
}

}  // namespace stinkytofu
