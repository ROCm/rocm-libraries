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

#include <cassert>
#include <cstdint>
#include <string>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"

namespace stinkytofu {
namespace {
enum VgprMsbState : int {
    NOT_REQUIRED = -1,
    LABEL_BEGIN = -2,
};

int getMsbFromVgpr(const StinkyRegister& reg) {
    if (reg.dataType != StinkyRegister::Type::Register || reg.reg.type != RegType::V) return -1;
    return static_cast<int>(reg.reg.idx) / 256;
}

void collectVgprMsbSlots(const StinkyInstruction* inst, int msbSrc[3], int& msbDst, bool& hasVgpr) {
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

/// \return (setVal, hasVgpr). setVal is NOT_REQUIRED when the instruction
///         doesn't use VGPRs or is a non-VOP instruction type.
std::pair<int, bool> computeRequiredMsb(const StinkyInstruction* inst) {
    if (inst->is(InstFlag::IF_SALU) || inst->is(InstFlag::IF_SMemLoad) ||
        inst->is(InstFlag::IF_SMemStore) || inst->is(InstFlag::IF_SMemAtomic) ||
        inst->is(InstFlag::IF_Branch) || inst->is(InstFlag::IF_Barrier) ||
        inst->is(InstFlag::IF_WaitCnt) || inst->is(InstFlag::IF_HasSideEffect)) {
        return {VgprMsbState::NOT_REQUIRED, false};
    }

    int msbSrc[3] = {0, 0, 0};
    int msbDst = 0;
    bool hasVgpr = false;

    collectVgprMsbSlots(inst, msbSrc, msbDst, hasVgpr);

    if (!hasVgpr) return {VgprMsbState::NOT_REQUIRED, false};

    int setVal = encodeVgprMsbForSlot(0, msbSrc[0]) | encodeVgprMsbForSlot(1, msbSrc[1]) |
                 encodeVgprMsbForSlot(2, msbSrc[2]) | encodeVgprMsbForSlot(3, msbDst);
    return {setVal, true};
}

// Set offset = -msb*256 on each VGPR operand so the emitter prints byte form
// (`v[idx + offset]` evaluates to idx ≤ 255).
void encodeVgprOperands(StinkyInstruction* inst) {
    auto rewrite = [](StinkyRegister& reg) {
        if (reg.dataType != StinkyRegister::Type::Register) return;
        if (reg.reg.type != RegType::V) return;
        int msb = static_cast<int>(reg.reg.idx) / 256;
        if (msb == 0) return;  // already byte-form; nothing to do
        int wantOffset = -msb * 256;
        if (reg.reg.offset == wantOffset) return;  // already encoded (rocisa path)
        reg.reg.offset = static_cast<int16_t>(wantOffset);
    };
    for (auto& src : const_cast<std::vector<StinkyRegister>&>(inst->getSrcRegs())) rewrite(src);
    for (auto& dst : const_cast<std::vector<StinkyRegister>&>(inst->getDestRegs())) rewrite(dst);
}

void emitVgprMsbIfNeeded(int requiredSetVal, bool hasVgpr, int& currentMsb, AsmIRBuilder& irBuilder,
                         GfxArchID archId, IRBase* insertBefore, VgprMsbMode msbMode) {
    if (!hasVgpr || requiredSetVal == currentMsb) {
        if (currentMsb == VgprMsbState::LABEL_BEGIN) currentMsb = VgprMsbState::NOT_REQUIRED;
        return;
    }

    if (currentMsb == VgprMsbState::LABEL_BEGIN) {
        StinkyInstruction* nopInst =
            irBuilder.create(getMCIDByUOp(GFX::s_nop, archId), insertBefore);
        nopInst->addSrcReg(StinkyRegister(0));
    }

    int combinedSetVal = requiredSetVal;
    if (msbMode == VgprMsbMode::Msb16 && currentMsb != VgprMsbState::NOT_REQUIRED &&
        currentMsb != VgprMsbState::LABEL_BEGIN) {
        combinedSetVal += (currentMsb << 8);
    }

    const HwInstDesc* desc = getMCIDByUOp(GFX::s_set_vgpr_msb, archId);
    assert(desc != nullptr && "s_set_vgpr_msb is not supported on this architecture");
    StinkyInstruction* msbInst = irBuilder.create(desc, insertBefore);
    msbInst->addSrcReg(StinkyRegister(combinedSetVal));

    std::string msbComment = "src0: " + std::to_string(decodeVgprMsbForSlot(requiredSetVal, 0)) +
                             ", src1: " + std::to_string(decodeVgprMsbForSlot(requiredSetVal, 1)) +
                             ", src2: " + std::to_string(decodeVgprMsbForSlot(requiredSetVal, 2)) +
                             ", dst: " + std::to_string(decodeVgprMsbForSlot(requiredSetVal, 3));
    msbInst->addModifier<CommentData>(CommentData{msbComment});
    currentMsb = requiredSetVal;
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

        for (auto bbIt = func.begin(); bbIt != func.end(); ++bbIt) {
            BasicBlock& bb = *bbIt;
            AsmIRBuilder irBuilder(bb, archId);
            int currentMsb = VgprMsbState::NOT_REQUIRED;

            for (auto it = bb.begin(); it != bb.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (!inst) continue;

                if (inst->getUnifiedOpcode() == GFX::LABEL) {
                    currentMsb = VgprMsbState::LABEL_BEGIN;
                    continue;
                }

                if (isPseudoInst(inst)) continue;

                auto [requiredMsb, hasVgpr] = computeRequiredMsb(inst);
                emitVgprMsbIfNeeded(requiredMsb, hasVgpr, currentMsb, irBuilder, archId, inst,
                                    msbMode);
                encodeVgprOperands(inst);
            }
        }
        return preserveCFGAnalyses();
    }
};

char InsertVgprMsbPassImpl::ID = 0;

}  // anonymous namespace

std::unique_ptr<Pass> createInsertVgprMsbPass() {
    return std::make_unique<InsertVgprMsbPassImpl>();
}

}  // namespace stinkytofu
