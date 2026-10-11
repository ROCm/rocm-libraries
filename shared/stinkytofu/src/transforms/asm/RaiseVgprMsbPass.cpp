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
#include "stinkytofu/transforms/asm/RaiseVgprMsbPass.hpp"

#include <cstdint>
#include <iostream>
#include <map>
#include <optional>
#include <string>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"
#include "stinkytofu/transforms/asm/InsertWaitAluPass.hpp"

#define DEBUG_TYPE "RaiseVgprMsbPass"

namespace {
using namespace stinkytofu;

// Absorbs the slot's MSB into `reg`'s physical index. Post-conditions:
//   - reg.reg.idx is the full physical register index
//   - reg.reg.offset is 0
// Returns true iff the operand was a VGPR that got rewritten.
bool raiseVgprOperand(StinkyRegister& reg, int slot, int currentMsb) {
    if (reg.dataType != StinkyRegister::Type::Register) return false;
    if (reg.reg.type != RegType::V) return false;
    if (slot < 0) return false;

    int msb = decodeVgprMsbForSlot(currentMsb, slot);

    // The parser stores either:
    //   (a) the encoded byte (0..255) with offset=0 — typical compiler-emitted input
    //   (b) the encoded byte with offset = -msb*256 — from textual `v[X-256]`
    //       form that the parser's negative-index recovery branch normalizes
    // Both collapse to the same raised form: full phys idx, offset = 0.
    uint32_t fullPhys = static_cast<uint32_t>(static_cast<int>(reg.reg.idx) + msb * 256);
    reg.reg.idx = fullPhys;
    reg.reg.offset = 0;
    return true;
}

// Walks operandFields and applies the current MSB state to every VGPR operand
// of `inst` using the shared slot mapping.
void applyMsbToInst(StinkyInstruction* inst, int currentMsb) {
    const auto* desc = inst->getHwInstDesc();
    if (!desc) return;
    const auto& fields = desc->operandFields;

    auto& srcRegs = const_cast<std::vector<StinkyRegister>&>(inst->getSrcRegs());
    auto& destRegs = const_cast<std::vector<StinkyRegister>&>(inst->getDestRegs());

    int srcIdx = 0, dstIdx = 0;
    for (const auto& field : fields) {
        StinkyRegister* reg = nullptr;
        if (field.isDest || field.isReadWrite) {
            if (dstIdx < static_cast<int>(destRegs.size())) reg = &destRegs[dstIdx++];
        } else {
            if (srcIdx < static_cast<int>(srcRegs.size())) reg = &srcRegs[srcIdx++];
        }
        if (!reg) continue;

        int slot = encodeFieldToVgprOffSlot(field.encodeField);
        if (slot < 0) continue;

        raiseVgprOperand(*reg, slot, currentMsb);
    }
}

// The previous-mode byte of an s_set_vgpr_msb right after the label at `it` (past
// s_nop and pseudo instructions): the mode the input says every edge brings.
std::optional<int> declaredEntryMsb(BasicBlock& bb, BasicBlock::iterator it) {
    for (++it; it != bb.end(); ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!inst || isPseudoInst(inst) || inst->getUnifiedOpcode() == GFX::s_nop) continue;
        if (inst->getUnifiedOpcode() != GFX::s_set_vgpr_msb) return std::nullopt;
        const auto& srcs = inst->getSrcRegs();
        const auto imm = srcs.empty() ? std::nullopt : srcs[0].getIntegerValue();
        if (!imm) return std::nullopt;
        return static_cast<int>((*imm >> 8) & 0xFF);
    }
    return std::nullopt;
}

class RaiseVgprMsbPassImpl : public Pass {
   public:
    static char ID;

    const char* getName() const override {
        return "RaiseVgprMsbPass";
    }

    Pass::ID getPassID() const override {
        return &RaiseVgprMsbPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        // The canonical wait-repair pipeline raises and later re-inserts VGPR
        // MSB state around Remove/InsertWaitAlu. Detect the conservative
        // bundled-wait fallback before either transformation starts so the
        // complete four-pass pipeline preserves the input function.
        if (hasRegeneratedFieldWaitBundle(func, passCtx)) {
            func.setMetaData(kPreserveWaitAluScheduleMetaKey, 1);
            PASS_DEBUG(std::cerr << "[RaiseVgprMsb] @" << func.getName()
                                 << ": pre-existing wait bundle, preserve function schedule\n");
            return PreservedAnalyses::all();
        }
        unsigned absorbedSetters = 0;
        unsigned raisedInsts = 0;

        // At a label the MSB state is unknown across arbitrary predecessors;
        // assume MSB=0 (matching InsertVgprMsbPass's LABEL_BEGIN reset
        // convention, which re-emits s_set_vgpr_msb after labels when a
        // non-zero MSB is needed). Blocks without a label are reached only by
        // falling through from the previous block, so the state carries over.
        int currentMsb = 0;
        // Whether a setter fixed currentMsb since the last label (or the entry),
        // as opposed to the assumption above.
        bool setSinceLabel = false;
        // Region mode: the input's state at each label (see createRaiseVgprMsbPass).
        const bool region = func.getMetaData(kFuncIsRegionMetaKey).has_value();
        std::map<std::string, int> labelEntryMsb;
        bool fallsThrough = false;
        for (BasicBlock& bb : func) {
            const bool process = passCtx.shouldProcessBasicBlock(bb);

            PASS_DEBUG(std::cerr << "[RaiseVgprMsb] enter bb=\"" << bb.getLabel()
                                 << "\" ir_count=" << bb.size() << " msb=0x" << std::hex
                                 << currentMsb << std::dec << "\n");

            for (auto it = bb.begin(); it != bb.end();) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (!inst) {
                    ++it;
                    continue;
                }

                if (inst->getUnifiedOpcode() == GFX::LABEL) {
                    int entry = 0;
                    const auto* label = inst->getModifier<LabelData>();
                    if (region && label) {
                        if (auto declared = declaredEntryMsb(bb, it))
                            entry = *declared;
                        else if (fallsThrough)
                            entry = currentMsb;
                        else if (auto seen = labelEntryMsb.find(label->label);
                                 seen != labelEntryMsb.end())
                            entry = seen->second;
                        labelEntryMsb[label->label] = entry;
                    }
                    currentMsb = entry;
                    setSinceLabel = entry != 0;
                    fallsThrough = true;
                    ++it;
                    continue;
                }

                if (inst->getUnifiedOpcode() == GFX::s_set_vgpr_msb) {
                    const auto& srcs = inst->getSrcRegs();
                    const auto imm = srcs.empty() ? std::nullopt : srcs[0].getIntegerValue();
                    if (!process) {
                        // Skipped blocks keep their setters; follow the state only.
                        if (imm) {
                            currentMsb = static_cast<int>(*imm & 0xFF);
                            setSinceLabel = true;
                        }
                        ++it;
                        continue;
                    }
                    if (!imm) {
                        // Erasing a setter we cannot evaluate would silently remap
                        // every following high VGPR onto a low one.
                        std::cerr << "RaiseVgprMsbPass: s_set_vgpr_msb operand is not an "
                                     "integer literal in bb \""
                                  << bb.getLabel() << "\"\n";
                        STINKY_UNREACHABLE("RaiseVgprMsbPass: unevaluable s_set_vgpr_msb");
                    }
                    // hasVgprMsb16 packs the *previous* MSB state into bits
                    // [15:8] of the immediate; only the low byte encodes
                    // the new state for src0/src1/src2/dst.
                    currentMsb = static_cast<int>(*imm & 0xFF);
                    setSinceLabel = true;
                    PASS_DEBUG(std::cerr
                               << "[RaiseVgprMsb]   absorb s_set_vgpr_msb -> currentMsb=0x"
                               << std::hex << currentMsb << std::dec << "\n");
                    ++absorbedSetters;
                    it = bb.eraseIR(it);
                    continue;
                }

                if (isPseudoInst(inst)) {
                    ++it;
                    continue;
                }
                fallsThrough = !isUnconditionalBranch(*inst) && !isEndOfFunction(*inst);
                if (region && isBranch(*inst))
                    if (const std::string target = getBranchTarget(*inst); !target.empty())
                        labelEntryMsb.emplace(target, currentMsb);

                if (process && currentMsb != 0) {
                    applyMsbToInst(inst, currentMsb);
                    ++raisedInsts;
                }

                ++it;
            }
        }
        if (setSinceLabel)
            func.setMetaData(kFuncExitVgprMsbMetaKey, static_cast<uint64_t>(currentMsb));
        for (const auto& [label, msb] : labelEntryMsb)
            if (msb != 0)
                func.setMetaData(labelEntryVgprMsbMetaKey(label), static_cast<uint64_t>(msb));

        PASS_DEBUG(std::cerr << "[RaiseVgprMsb] done: absorbed_setters=" << absorbedSetters
                             << " raised_insts=" << raisedInsts << "\n");
        return preserveCFGAnalyses();
    }
};

char RaiseVgprMsbPassImpl::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createRaiseVgprMsbPass() {
    return std::make_unique<RaiseVgprMsbPassImpl>();
}
}  // namespace stinkytofu
