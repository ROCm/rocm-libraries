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
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "stinkytofu/core/Types.hpp"
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

/// Whether \p inst can carry a VGPR bank word at all. Scalar, branch, call,
/// barrier, wait and side-effect instructions never need an s_set_vgpr_msb,
/// whatever their operands.
inline bool carriesVgprMsb(const StinkyInstruction& inst) {
    return !(inst.is(InstFlag::IF_SALU) || inst.is(InstFlag::IF_SMemLoad) ||
             inst.is(InstFlag::IF_SMemStore) || inst.is(InstFlag::IF_SMemAtomic) ||
             inst.is(InstFlag::IF_Branch) || inst.is(InstFlag::IF_Call) ||
             inst.is(InstFlag::IF_Barrier) || inst.is(InstFlag::IF_WaitCnt) ||
             inst.is(InstFlag::IF_HasSideEffect));
}

/// Calls \p callback once per VGPR operand of \p inst that decides a slot of its
/// bank word, as (operand, index, isDest, slot), in field order. When two
/// operands share a slot, the later one is what the word holds.
template <typename Callback>
inline void forEachVgprMsbSlotOperand(const StinkyInstruction& inst, Callback&& callback) {
    forEachVgprOperandField(inst,
                            [&](const StinkyRegister& reg, size_t operand, bool isDest, int slot) {
                                if (slot < 0) return;
                                if (getMsbFromVgpr(reg) < 0) return;
                                callback(reg, operand, isDest, slot);
                            });
}

/// Every slot of a bank word.
constexpr int kVgprMsbFullMask = 0xFF;

/// What one instruction needs of the bank word in force.
struct VgprMsbRequirement {
    /// The word with each filled slot's bank and 0 in the slots no operand
    /// fills; -1 when the instruction needs no word at all.
    int word = -1;
    /// The bits of `word` the instruction depends on: 0b11 for each slot an
    /// operand fills, or every slot when a VGPR operand's field selects none,
    /// since which bank such a field reads is not settled.
    int mask = 0;

    /// Whether the instruction needs a word: it can carry one, and a VGPR
    /// operand fills a slot.
    bool needed() const {
        return word >= 0;
    }
};

/// requiredVgprMsb() with each slotted VGPR operand's bank supplied by \p bankOf,
/// called as bankOf(operand, operandIndex, isDest), so that a colouring can be
/// priced before it is written into the operands. One body for both, so a
/// predicted word and an emitted one cannot disagree about which instructions
/// carry an MSB or which field maps to which slot. When two operands share a
/// slot, the later one is what the word holds.
template <typename BankOf>
inline VgprMsbRequirement requiredVgprMsbWith(const StinkyInstruction& inst, BankOf&& bankOf) {
    VgprMsbRequirement required;
    if (!carriesVgprMsb(inst)) return required;

    uint32_t word = 0;
    uint32_t mask = 0;
    bool slotless = false;
    forEachVgprOperandField(
        inst, [&](const StinkyRegister& reg, size_t operand, bool isDest, int slot) {
            if (getMsbFromVgpr(reg) < 0) return;
            if (slot < 0) {
                slotless = true;
                return;
            }
            const uint32_t shift = 2U * static_cast<uint32_t>(slot);
            const uint32_t bank = static_cast<uint32_t>(bankOf(reg, operand, isDest)) & 0x3U;
            word = (word & ~(0x3U << shift)) | (bank << shift);
            mask |= 0x3U << shift;
        });
    if (mask == 0) return required;
    required.word = static_cast<int>(word);
    required.mask = slotless ? kVgprMsbFullMask : static_cast<int>(mask);
    return required;
}

/// The bank word \p inst needs for its VGPR operands, each in the bank it names.
inline VgprMsbRequirement requiredVgprMsb(const StinkyInstruction& inst) {
    return requiredVgprMsbWith(
        inst, [](const StinkyRegister& reg, size_t, bool) { return getMsbFromVgpr(reg); });
}

/// requiredVgprMsb() as (setVal, hasVgpr): the word \p inst needs for its VGPR
/// operands, or -1 with hasVgpr false for an instruction that needs none.
inline std::pair<int, bool> computeRequiredMsb(const StinkyInstruction* inst) {
    const VgprMsbRequirement required = requiredVgprMsb(*inst);
    return {required.word, required.needed()};
}

/// The word an s_set_vgpr_msb sets: the low byte of its literal, or -1 when it
/// has none.
inline int vgprMsbSetterWord(const StinkyInstruction& setter) {
    const std::vector<StinkyRegister>& srcs = setter.getSrcRegs();
    if (srcs.empty() || srcs[0].dataType != StinkyRegister::Type::LiteralInt) return -1;
    return static_cast<int>(static_cast<uint32_t>(srcs[0].getLiteralInt()) &
                            static_cast<uint32_t>(kVgprMsbFullMask));
}

/// The slot fields \p required depends on in which \p word holds another bank;
/// none when \p word serves it in every slot it uses.
inline uint32_t vgprMsbChangedFields(int word, const VgprMsbRequirement& required) {
    return (static_cast<uint32_t>(word) ^ static_cast<uint32_t>(required.word)) &
           static_cast<uint32_t>(required.mask);
}

/// Which bank words serve an instruction.
enum class VgprMsbPolicy {
    /// Only its own word, every slot compared, with 0 in the slots no operand
    /// fills. What InsertVgprMsbPass has always emitted.
    ExactWord,
    /// Any word that agrees on the slots the instruction depends on (its mask);
    /// the others are don't-care. A setter takes the banks a later instruction
    /// needs as long as nothing since it has read the slots that change, so
    /// within a block the setters are as few as the order allows.
    UnusedSlotsDontCare,
};

/// The policy \p caps asks for.
inline VgprMsbPolicy vgprMsbPolicyOf(const AsmCapsConfig& caps) {
    return caps.vgprMsbUnusedSlotsDontCare ? VgprMsbPolicy::UnusedSlotsDontCare
                                           : VgprMsbPolicy::ExactWord;
}

/// What has to happen before one instruction, as VgprMsbTracker::step() works it out.
struct VgprMsbStep {
    /// A new s_set_vgpr_msb goes before the instruction.
    bool emit = false;
    /// Instead of a new one, the last setter this tracker asked for takes `word`:
    /// nothing since it has read the slots that change. Never set with `emit`.
    bool rewrite = false;
    /// The new setter is the first since a label, which InsertVgprMsbPass
    /// precedes with an s_nop.
    bool firstAfterLabel = false;
    /// The word a new setter replaces, when one was set. Msb16 packs it into the
    /// high byte.
    std::optional<int> previousWord;
    /// The word to write: the new setter's, or the rewritten setter's low byte.
    int word = -1;
};

/// The bank word in force while instructions are walked in order.
///
/// An instruction needs a setter whenever the word in force does not serve it,
/// as the policy decides. Nothing is known at the start of a block, after a
/// call, or after a label, which may be reached from anywhere.
///
/// Under UnusedSlotsDontCare the tracker also remembers its last setter and the
/// slots read since: an instruction that word does not serve, but whose
/// changed slots nothing has read since, rewrites that setter rather than
/// needing a new one. The setters then split each block into the longest
/// stretches whose instructions agree on every slot they use.
///
/// InsertVgprMsbPass emits by this, and anything predicting what it will emit,
/// or ordering instructions to emit fewer, steps the same object, so they
/// cannot drift.
class VgprMsbTracker {
   public:
    explicit VgprMsbTracker(VgprMsbPolicy policy = VgprMsbPolicy::ExactWord) : policy_(policy) {}

    /// A label: no word is known, and the next setter follows an s_nop.
    void label() {
        current_ = kLabelBegin;
        setterOpen_ = false;
    }

    /// The start of a block, or a call, after which the callee's word is unknown.
    void reset() {
        current_ = kNotRequired;
        setterOpen_ = false;
    }

    /// An s_set_vgpr_msb already in the stream, setting \p word
    /// (vgprMsbSetterWord(), -1 when unknown): in force from here, but that
    /// setter is not this tracker's to rewrite.
    void observeSetter(int word) {
        current_ = word >= 0 ? word : kNotRequired;
        setterOpen_ = false;
    }

    /// Whether nothing has been stepped since a label, so a setter here, whoever
    /// placed it, directly follows the label and needs an s_nop before it.
    bool directlyAfterLabel() const {
        return current_ == kLabelBegin;
    }

    /// Whether the word in force serves \p required as it stands. True of an
    /// instruction that needs no word.
    bool satisfies(const VgprMsbRequirement& required) const {
        if (!required.needed()) return true;
        switch (policy_) {
            case VgprMsbPolicy::ExactWord:
                return required.word == current_;
            case VgprMsbPolicy::UnusedSlotsDontCare:
                return current_ >= 0 && vgprMsbChangedFields(current_, required) == 0;
        }
        return false;
    }

    /// Whether step(\p required) would emit a new s_set_vgpr_msb: neither the
    /// word in force nor a rewrite of the last setter serves it.
    bool needsSetter(const VgprMsbRequirement& required) const {
        return !satisfies(required) && !rewritable(required);
    }

    /// One instruction, needing \p required: whether a setter goes before it,
    /// or the last one is rewritten, and which word is written.
    VgprMsbStep step(const VgprMsbRequirement& required) {
        VgprMsbStep result;
        if (satisfies(required)) {
            // The next setter no longer directly follows the label, so it needs
            // no s_nop.
            if (current_ == kLabelBegin) current_ = kNotRequired;
            readSince_ |= static_cast<uint32_t>(required.mask);
            return result;
        }
        if (rewritable(required)) {
            current_ = merged(required);
            readSince_ |= static_cast<uint32_t>(required.mask);
            result.rewrite = true;
            result.word = current_;
            return result;
        }
        result.emit = true;
        result.firstAfterLabel = current_ == kLabelBegin;
        if (current_ >= 0) result.previousWord = current_;
        result.word = policy_ == VgprMsbPolicy::UnusedSlotsDontCare && current_ >= 0
                          ? merged(required)
                          : required.word;
        current_ = result.word;
        readSince_ = static_cast<uint32_t>(required.mask);
        setterOpen_ = true;
        return result;
    }

   private:
    /// Under UnusedSlotsDontCare, whether the last setter can take \p required's
    /// banks: it is this tracker's, and nothing since it has read a slot whose
    /// bank would change.
    bool rewritable(const VgprMsbRequirement& required) const {
        if (policy_ != VgprMsbPolicy::UnusedSlotsDontCare || !setterOpen_ || current_ < 0 ||
            !required.needed())
            return false;
        return (vgprMsbChangedFields(current_, required) & readSince_) == 0;
    }

    /// The word in force with the slots \p required depends on set to its banks.
    int merged(const VgprMsbRequirement& required) const {
        const uint32_t mask = static_cast<uint32_t>(required.mask);
        const uint32_t kept = static_cast<uint32_t>(current_) & ~mask;
        return static_cast<int>((kept | (static_cast<uint32_t>(required.word) & mask)) &
                                static_cast<uint32_t>(kVgprMsbFullMask));
    }

    static constexpr int kNotRequired = -1;
    static constexpr int kLabelBegin = -2;
    VgprMsbPolicy policy_;
    int current_ = kNotRequired;
    /// The slot fields read since this tracker's last setter.
    uint32_t readSince_ = 0;
    /// The last setter is this tracker's own and can still be rewritten.
    bool setterOpen_ = false;
};

}  // namespace stinkytofu
