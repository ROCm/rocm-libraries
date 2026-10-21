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
#include <span>
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

/// Whether \p inst reads the bank state at all. Scalar, branch, call, barrier,
/// wait and side-effect instructions never ask for a bank, so they neither need
/// an s_set_vgpr_msb nor change which byte is in force.
inline bool carriesVgprMsb(const StinkyInstruction& inst) {
    return !(inst.is(InstFlag::IF_SALU) || inst.is(InstFlag::IF_SMemLoad) ||
             inst.is(InstFlag::IF_SMemStore) || inst.is(InstFlag::IF_SMemAtomic) ||
             inst.is(InstFlag::IF_Branch) || inst.is(InstFlag::IF_Call) ||
             inst.is(InstFlag::IF_Barrier) || inst.is(InstFlag::IF_WaitCnt) ||
             inst.is(InstFlag::IF_HasSideEffect));
}

/// What an instruction needs the bank state to be when it issues.
struct VgprMsbNeed {
    /// The s_set_vgpr_msb byte: the bank of each slot's operand. A slot no
    /// operand selects reads bank 0.
    int value = 0;
    /// 0x3 at each slot an operand selects its bank through.
    int usedSlots = 0;

    /// False for an instruction that names no VGPR through a slot; the bank
    /// state is then none of its business.
    bool any() const {
        return usedSlots != 0;
    }
};

/// The bank state \p inst needs, with each VGPR operand at the index
/// `indexOf(operand, operandIndex, isDest)` gives it: the operand's own index
/// for the instruction as written, or the register a colouring not yet applied
/// would write there. A later operand in a slot overrides an earlier one, since
/// each slot has one selector.
///
/// InsertVgprMsbPass emits by this rule, so anything that predicts what it will
/// emit has to use the rule too.
template <typename IndexOf>
inline VgprMsbNeed vgprMsbNeed(const StinkyInstruction& inst, IndexOf&& indexOf) {
    static_assert(std::is_invocable_r_v<uint32_t, IndexOf, const StinkyRegister&, size_t, bool>,
                  "vgprMsbNeed index lookup must be callable as "
                  "(const StinkyRegister& operand, size_t index, bool isDest) -> uint32_t");

    VgprMsbNeed need;
    if (!carriesVgprMsb(inst)) return need;
    forEachVgprOperandField(
        inst, [&](const StinkyRegister& reg, size_t operand, bool isDest, int slot) {
            if (slot < 0 || getMsbFromVgpr(reg) < 0) return;
            const int bank = static_cast<int>(indexOf(reg, operand, isDest) / kVgprBankSize);
            const int field = encodeVgprMsbForSlot(slot, 0x3);
            need.value = (need.value & ~field) | encodeVgprMsbForSlot(slot, bank);
            need.usedSlots |= field;
        });
    return need;
}

/// The bank state \p inst needs as written.
inline VgprMsbNeed vgprMsbNeed(const StinkyInstruction& inst) {
    return vgprMsbNeed(inst, [](const StinkyRegister& reg, size_t, bool) { return reg.reg.idx; });
}

/// What the bank state has to hold in a slot an instruction does not use.
enum class UnusedVgprMsbSlots {
    /// Bank 0: the byte in force has to equal the instruction's byte in every
    /// slot. What InsertVgprMsbPass has always emitted.
    Zero,
    /// Anything. Only the slots an instruction uses have to match, and a new
    /// byte gives each slot it leaves unused the bank the next instruction in
    /// the block wants there. Correct only if the hardware ignores the selector
    /// of a slot the instruction does not encode, which the ISA has to confirm
    /// before this becomes a default.
    Keep,
};

/// The byte the last s_set_vgpr_msb put in force, followed along a block in
/// order. Unknown where a block starts, after a label and after a call, because
/// control can arrive there with any byte in force.
class VgprMsbState {
   public:
    explicit VgprMsbState(UnusedVgprMsbSlots unused = UnusedVgprMsbSlots::Zero) : unused_(unused) {}

    bool known() const {
        return current_ >= 0;
    }

    /// The byte in force; meaningful only when known().
    int current() const {
        return current_;
    }

    /// Whether an instruction with \p need has to be preceded by an s_set_vgpr_msb.
    bool needsSet(const VgprMsbNeed& need) const {
        if (!need.any()) return false;
        if (!known()) return true;
        const int differs = need.value ^ current_;
        return (unused_ == UnusedVgprMsbSlots::Keep ? differs & need.usedSlots : differs) != 0;
    }

    /// Put \p need in force and return the byte that does it. \p ahead is what
    /// the block's later instructions need, up to the next point the byte is
    /// forgotten; under Keep it fills the slots \p need leaves unused.
    int set(const VgprMsbNeed& need, std::span<const VgprMsbNeed> ahead = {}) {
        int byte = need.value;
        if (unused_ == UnusedVgprMsbSlots::Keep) {
            constexpr int kAllSlots = 0xff;
            int decided = need.usedSlots;
            for (const VgprMsbNeed& later : ahead) {
                if (decided == kAllSlots) break;
                const int open = later.usedSlots & ~decided;
                byte |= later.value & open;
                decided |= open;
            }
            if (known()) byte |= current_ & ~decided;
        }
        current_ = byte;
        return byte;
    }

    void forget() {
        current_ = -1;
    }

   private:
    UnusedVgprMsbSlots unused_;
    int current_ = -1;
};

/// The s_set_vgpr_msb immediate \p inst needs for its VGPR operands; (setVal,
/// hasVgpr) with hasVgpr false and setVal -1 for an instruction that carries no
/// VGPR MSB. The scheduler's MSB-affinity tiebreak reads this; it is
/// vgprMsbNeed() in another shape.
inline std::pair<int, bool> computeRequiredMsb(const StinkyInstruction* inst) {
    const VgprMsbNeed need = vgprMsbNeed(*inst);
    if (!need.any()) return {-1, false};
    return {need.value, true};
}

}  // namespace stinkytofu
