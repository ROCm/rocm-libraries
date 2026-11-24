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
#include "stinkytofu/transforms/asm/InsertVgprMsbPass.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"
#include "stinkytofu/support/OptimizationRemark.hpp"

namespace stinkytofu {
namespace {
// Give each VGPR operand the offset its index calls for, so `v[idx + offset]`
// lands in 0-255. Assigned unconditionally, bank 0 included: the offset is
// derived from an index other passes rewrite, and skipping it there leaves a
// register that moved down out of bank 1 holding a bias it no longer earns.
void encodeVgprOperands(StinkyInstruction* inst) {
    auto rewrite = [](StinkyRegister& reg) {
        if (reg.dataType != StinkyRegister::Type::Register) return;
        if (reg.reg.type != RegType::V) return;
        // Nothing upstream carries the bias, so an operand arrives unbiased or
        // already agreeing with its index because this pass ran before.
        assert((reg.reg.offset == 0 || reg.reg.offset == getMsbOffsetForVgpr(reg)) &&
               "VGPR offset disagrees with its index; something stored a bias upstream");
        reg.reg.offset = static_cast<int16_t>(getMsbOffsetForVgpr(reg));
    };
    for (auto& src : const_cast<std::vector<StinkyRegister>&>(inst->getSrcRegs())) rewrite(src);
    for (auto& dst : const_cast<std::vector<StinkyRegister>&>(inst->getDestRegs())) rewrite(dst);
}

/// The text an operand of \p inst prints, for a diagnostic that has to name a
/// range rather than a base.
std::string vgprOperandText(const StinkyRegister& reg) {
    const uint32_t width = std::max<uint32_t>(1, reg.reg.num);
    const std::string base = std::to_string(reg.reg.idx);
    if (width == 1) return "v" + base;
    return "v[" + base + ":" + std::to_string(reg.reg.idx + width - 1) + "]";
}

/// Describe each operand of \p inst that its field cannot name.
///
/// Reported from here because this is where the bias is applied, so no operand
/// reaches the emitter without passing this point, whatever chose its register.
void collectUnbankableOperands(const StinkyInstruction& inst, const std::string& function,
                               std::vector<std::string>& found) {
    forEachUnencodableVgprOperand(
        inst, [&](const StinkyRegister& reg, size_t operand, bool isDest) {
            const HwInstDesc* desc = inst.getHwInstDesc();
            const char* mnemonic =
                desc != nullptr && desc->mnemonic != nullptr ? desc->mnemonic : "<unknown>";
            found.push_back("@" + function + ": " + (isDest ? "dest[" : "src[") +
                            std::to_string(operand) + "] of '" + mnemonic + "' is " +
                            vgprOperandText(reg) +
                            ", but that field has no VGPR bank selector and reaches v0-v" +
                            std::to_string(kVgprBankSize - 1) + " only");
        });
}

std::string vgprMsbComment(int word) {
    return "src0: " + std::to_string(decodeVgprMsbForSlot(word, 0)) +
           ", src1: " + std::to_string(decodeVgprMsbForSlot(word, 1)) +
           ", src2: " + std::to_string(decodeVgprMsbForSlot(word, 2)) +
           ", dst: " + std::to_string(decodeVgprMsbForSlot(word, 3));
}

/// The new s_set_vgpr_msb \p step asks for, at \p insertBefore.
StinkyInstruction* emitVgprMsb(const VgprMsbStep& step, AsmIRBuilder& irBuilder, GfxArchID archId,
                               IRBase* insertBefore, VgprMsbMode msbMode) {
    if (step.firstAfterLabel) {
        StinkyInstruction* nopInst =
            irBuilder.create(getMCIDByUOp(GFX::s_nop, archId), insertBefore);
        nopInst->addSrcReg(StinkyRegister(0));
    }

    int combinedSetVal = step.word;
    if (msbMode == VgprMsbMode::Msb16 && step.previousWord.has_value()) {
        combinedSetVal += (*step.previousWord << 8);
    }

    const HwInstDesc* desc = getMCIDByUOp(GFX::s_set_vgpr_msb, archId);
    assert(desc != nullptr && "s_set_vgpr_msb is not supported on this architecture");
    StinkyInstruction* msbInst = irBuilder.create(desc, insertBefore);
    msbInst->addSrcReg(StinkyRegister(combinedSetVal));
    msbInst->addModifier<CommentData>(CommentData{vgprMsbComment(step.word)});
    return msbInst;
}

/// Give \p setter the low byte \p word. Its Msb16 high byte stays: the word in
/// force before the setter does not change.
void rewriteVgprMsb(StinkyInstruction* setter, int word) {
    const uint32_t setVal = static_cast<uint32_t>(setter->getSrcRegs()[0].getLiteralInt());
    const uint32_t lowByte = static_cast<uint32_t>(kVgprMsbFullMask);
    setter->setSrcReg(0, StinkyRegister(static_cast<int>((setVal & ~lowByte) |
                                                         (static_cast<uint32_t>(word) & lowByte))));
    if (CommentData* comment = setter->getModifier<CommentData>())
        comment->comment = vgprMsbComment(word);
}

/// Whether \p inst names a VGPR through a field with a bank slot.
bool hasSlottedVgprOperand(const StinkyInstruction& inst) {
    bool found = false;
    forEachVgprMsbSlotOperand(inst,
                              [&](const StinkyRegister&, size_t, bool, int) { found = true; });
    return found;
}

/// \p inst's mnemonic, for a diagnostic.
const char* mnemonicOf(const StinkyInstruction& inst) {
    const HwInstDesc* desc = inst.getHwInstDesc();
    return desc != nullptr && desc->mnemonic != nullptr ? desc->mnemonic : "<unknown>";
}

bool preferInsertAfter(const StinkyInstruction& inst) {
    return isVectorALU(inst) || (isScalarALU(inst) && !isBarrier(inst)) ||
           isMatrixInstruction(inst);
}

class InsertVgprMsbPassImpl : public Pass {
   public:
    static char ID;

    const char* getName() const override {
        return "Insert VGPR MSB";
    }

    Pass::ID getPassID() const override {
        return &InsertVgprMsbPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        auto arch = passCtx.getGemmTileConfig().arch;
        GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);

        VgprMsbMode msbMode = passCtx.getAsmCapsConfig().vgprMsbMode;
        if (msbMode == VgprMsbMode::None) return preserveCFGAnalyses();

        // Reported, not asserted. A slotless field certainly cannot name a bank,
        // but which bank it then reads is not settled -- one kernel emits eight
        // such operands and validates -- so aborting here would stop a build on
        // something that may work. Whoever chose the register is upstream; this
        // says only that the choice is not encodable, and says it per operand so
        // a release build names them rather than emitting them in silence.
        std::vector<std::string> unbankable;
        runOnFunction(func, archId, msbMode, vgprMsbPolicyOf(passCtx.getAsmCapsConfig()),
                      unbankable);
        for (const std::string& operand : unbankable) {
            emitRemark(passCtx, {OptimizationRemark::Kind::Missed, "InsertVgprMsb",
                                 "UnbankableOperand", operand});
        }
        // Asserted, unlike the operands above: a mismatch is a word this pass
        // got wrong, so some instruction would touch a register it does not
        // name. StinkyTofu keeps assertions in Release builds (CMakeLists.txt),
        // so this stops kernel generation there too.
#ifndef NDEBUG
        const std::optional<std::string> mismatch = firstVgprMsbMismatch(func);
        if (mismatch) std::cerr << "[InsertVgprMsb] " << *mismatch << "\n";
        assert(!mismatch && "an instruction runs with a VGPR bank other than the one it names");
#endif
        return preserveCFGAnalyses();
    }

   private:
    static void runOnFunction(Function& func, GfxArchID archId, VgprMsbMode msbMode,
                              VgprMsbPolicy policy, std::vector<std::string>& unbankable) {
        for (auto bbIt = func.begin(); bbIt != func.end(); ++bbIt) {
            BasicBlock& bb = *bbIt;
            AsmIRBuilder irBuilder(bb, archId);
            VgprMsbTracker tracker(policy);
            // The setter the tracker last asked for, which a rewrite changes.
            StinkyInstruction* lastSetter = nullptr;
            IRBase* preferredInsertBefore = nullptr;
            auto findNextInstructionAnchor = [&](BasicBlock::iterator from) -> IRBase* {
                for (auto scanIt = from; scanIt != bb.end(); ++scanIt) {
                    if (dyn_cast<StinkyInstruction>(scanIt.getNodePtr()))
                        return scanIt.getNodePtr();
                }
                return nullptr;
            };

            for (auto it = bb.begin(); it != bb.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (!inst) continue;

                if (inst->getUnifiedOpcode() == GFX::LABEL) {
                    tracker.label();
                    lastSetter = nullptr;
                    preferredInsertBefore = nullptr;
                    continue;
                }

                if (isPseudoInst(inst)) continue;

                // A setter already in the stream sets the word like one of ours,
                // but it is not this pass's to rewrite, and none of ours may go
                // in front of it.
                if (inst->getUnifiedOpcode() == GFX::s_set_vgpr_msb) {
                    tracker.observeSetter(vgprMsbSetterWord(*inst));
                    lastSetter = nullptr;
                    preferredInsertBefore = nullptr;
                    continue;
                }

                // A call (e.g. s_swappc_b64) transfers to a callee that may leave
                // the VGPR MSB hardware register in an unknown state. Reset the
                // tracked value so the next VGPR op re-establishes MSB — matching
                // the single-function pipeline, which re-established MSB after the
                // call because the call ended a basic block.
                if (isCall(*inst)) {
                    tracker.reset();
                    lastSetter = nullptr;
                    // Never carry a deferred insertion anchor across call boundaries:
                    // call may clobber VGPR MSB state, so post-call rebuilds must stay post-call.
                    preferredInsertBefore = nullptr;
                    continue;
                }

                IRBase* insertBefore = preferredInsertBefore ? preferredInsertBefore : inst;

                const VgprMsbStep step = tracker.step(requiredVgprMsb(*inst));
                if (step.rewrite) {
                    assert(lastSetter != nullptr &&
                           "the tracker rewrites only a setter it asked for");
                    rewriteVgprMsb(lastSetter, step.word);
                } else if (step.emit) {
                    lastSetter = emitVgprMsb(step, irBuilder, archId, insertBefore, msbMode);
                }
                collectUnbankableOperands(*inst, func.getName(), unbankable);
                encodeVgprOperands(inst);
                if (step.emit || carriesVgprMsb(*inst)) preferredInsertBefore = nullptr;

                if (preferInsertAfter(*inst)) {
                    preferredInsertBefore = findNextInstructionAnchor(std::next(it));
                }
            }
        }
    }
};

char InsertVgprMsbPassImpl::ID = 0;

}  // anonymous namespace

std::unique_ptr<Pass> createInsertVgprMsbPass() {
    return std::make_unique<InsertVgprMsbPassImpl>();
}

std::optional<std::string> firstVgprMsbMismatch(const Function& function) {
    for (const BasicBlock& bb : function) {
        // The low byte of the last s_set_vgpr_msb, or -1 while none is known.
        int word = -1;
        size_t position = 0;
        for (const IRBase& ir : bb) {
            const auto* inst = dyn_cast<StinkyInstruction>(&ir);
            if (inst == nullptr) continue;
            ++position;
            if (inst->getUnifiedOpcode() == GFX::LABEL) {
                word = -1;
                continue;
            }
            if (isPseudoInst(inst)) continue;
            if (isCall(*inst)) {
                word = -1;
                continue;
            }
            if (inst->getUnifiedOpcode() == GFX::s_set_vgpr_msb) {
                word = vgprMsbSetterWord(*inst);
                continue;
            }

            const std::string where = "@" + function.getName() + " ^" + bb.getLabel() + " #" +
                                      std::to_string(position) + " '" + mnemonicOf(*inst) + "'";
            // A slot keeps whatever bank it last selected, so a class that is
            // given no word and read a VGPR through one would see a bank chosen
            // for someone else.
            if (!carriesVgprMsb(*inst)) {
                if (hasSlottedVgprOperand(*inst))
                    return where +
                           " names a VGPR through a bank slot, but its class is given no word";
                continue;
            }
            const VgprMsbRequirement required = requiredVgprMsb(*inst);
            if (!required.needed()) continue;
            if (word < 0) return where + " needs a bank word, but none is set";
            if (vgprMsbChangedFields(word, required) != 0)
                return where + " needs " + vgprMsbComment(required.word) + " (mask " +
                       std::to_string(required.mask) + "), but runs with " + vgprMsbComment(word);
        }
    }
    return std::nullopt;
}

}  // namespace stinkytofu
