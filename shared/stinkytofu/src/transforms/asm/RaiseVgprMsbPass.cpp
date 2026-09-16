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

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"

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
        unsigned absorbedSetters = 0;
        unsigned raisedInsts = 0;

        for (BasicBlock& bb : func) {
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;

            PASS_DEBUG(std::cerr << "[RaiseVgprMsb] enter bb=\"" << bb.getLabel()
                                 << "\" ir_count=" << bb.size() << "\n");

            // MSB state at BB entry is unknown across arbitrary predecessors;
            // assume MSB=0 (matching InsertVgprMsbPass's LABEL_BEGIN reset
            // convention, which re-emits s_set_vgpr_msb after labels when a
            // non-zero MSB is needed).
            int currentMsb = 0;

            for (auto it = bb.begin(); it != bb.end();) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (!inst) {
                    ++it;
                    continue;
                }

                if (inst->getUnifiedOpcode() == GFX::LABEL) {
                    currentMsb = 0;
                    ++it;
                    continue;
                }

                if (inst->getUnifiedOpcode() == GFX::s_set_vgpr_msb) {
                    const auto& srcs = inst->getSrcRegs();
                    if (!srcs.empty() && srcs[0].dataType == StinkyRegister::Type::LiteralInt) {
                        // hasVgprMsb16 packs the *previous* MSB state into bits
                        // [15:8] of the immediate; only the low byte encodes
                        // the new state for src0/src1/src2/dst.
                        currentMsb = srcs[0].getLiteralInt() & 0xFF;
                    }
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

                if (currentMsb != 0) {
                    applyMsbToInst(inst, currentMsb);
                    ++raisedInsts;
                }

                ++it;
            }
        }

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
