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
#include "stinkytofu/ir/asm/ReadWriteOperands.hpp"

#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"

namespace stinkytofu {
namespace {

/// The destination field at \p slot, with the number of source fields the
/// instruction declares.
///
/// Slots count `isDest` fields, the same walk that indexes `getDestRegs()`
/// elsewhere, so one slot means one register everywhere.
struct DestField {
    const HwInstDesc::OperandFieldDesc* field = nullptr;
    size_t sourceCount = 0;
};

DestField destFieldAt(const StinkyInstruction& inst, size_t slot) {
    DestField found;
    const HwInstDesc* desc = inst.getHwInstDesc();
    if (desc == nullptr) return found;

    size_t seen = 0;
    for (const HwInstDesc::OperandFieldDesc& field : desc->operandFields) {
        // Sources declared after the match still count, because op_sel numbers
        // all of them ahead of the destination.
        if (!field.isDest)
            ++found.sourceCount;
        else if (seen++ == slot)
            found.field = &field;
    }
    return found;
}

/// The bits \p found leaves as they were, given the op_sel \p inst carries.
uint32_t keptBits(const StinkyInstruction& inst, const DestField& found, size_t slot) {
    if (found.field == nullptr) return 0;

    const uint32_t width = found.field->fieldSizeBits;
    if (width == 0 || width >= kRegisterBits) return 0;

    // op_sel is [src0, ..., srcN, dst], so the destination element follows the
    // sources. No modifier means no stated position, and that reads as a write
    // of the whole register.
    const VOP3PModifiers* vop3p = inst.getModifier<VOP3PModifiers>();
    if (vop3p == nullptr) return 0;

    const size_t element = found.sourceCount + slot;
    if (element >= vop3p->op_sel.size()) return 0;

    const int position = vop3p->op_sel[element];
    if (position < 0) return 0;

    const uint32_t offset = static_cast<uint32_t>(position) * width;
    if (offset >= kRegisterBits) return 0;

    return ~(((1u << width) - 1u) << offset);
}

}  // namespace

uint32_t keptDestinationBits(const StinkyInstruction& inst, size_t slot) {
    return keptBits(inst, destFieldAt(inst, slot), slot);
}

bool readsDestination(const StinkyInstruction& inst, size_t slot) {
    const DestField found = destFieldAt(inst, slot);
    if (found.field == nullptr) return false;
    // The flag states a property of the opcode, so it holds whatever op_sel says.
    return found.field->isReadWrite || keptBits(inst, found, slot) != 0;
}

}  // namespace stinkytofu
