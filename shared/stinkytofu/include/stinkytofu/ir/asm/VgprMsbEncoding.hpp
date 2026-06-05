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

#include "stinkytofu/hardware/GfxIsa.hpp"

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

}  // namespace stinkytofu
