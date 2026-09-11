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

#include <utility>

#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

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

/// MSB (which 256-VGPR bank) of a VGPR operand, or -1 for non-VGPR operands.
inline int getMsbFromVgpr(const StinkyRegister& reg) {
    if (reg.dataType != StinkyRegister::Type::Register || reg.reg.type != RegType::V) return -1;
    return static_cast<int>(reg.reg.idx) / 256;
}

/// Record the per-slot VGPR banks of \p inst; \p hasVgpr set if any VGPR is seen.
inline void collectVgprMsbSlots(const StinkyInstruction* inst, int msbSrc[3], int& msbDst,
                                bool& hasVgpr) {
    const auto& fields = inst->getHwInstDesc()->operandFields;
    const auto& srcRegs = inst->getSrcRegs();
    const auto& destRegs = inst->getDestRegs();

    int srcIdx = 0, dstIdx = 0;
    for (const auto& field : fields) {
        const StinkyRegister* reg = nullptr;
        if (field.isDest || field.isReadWrite) {
            if (dstIdx < static_cast<int>(destRegs.size())) reg = &destRegs[dstIdx++];
        } else {
            if (srcIdx < static_cast<int>(srcRegs.size())) reg = &srcRegs[srcIdx++];
        }
        if (!reg) continue;

        int slot = encodeFieldToVgprOffSlot(field.encodeField);
        if (slot < 0) continue;

        int msb = getMsbFromVgpr(*reg);
        if (msb < 0) continue;

        hasVgpr = true;
        if (slot == 3)
            msbDst = msb;
        else
            msbSrc[slot] = msb;
    }
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
