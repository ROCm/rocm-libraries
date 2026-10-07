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
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"
#include "stinkytofu/transforms/asm/VgprMsbPlanner.hpp"

namespace stinkytofu {
namespace {
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

void emitVgprMsb(const VgprMsbInsertion& insertion, AsmIRBuilder& irBuilder, GfxArchID archId,
                 IRBase* insertBefore) {
    if (insertion.nopFirst) {
        StinkyInstruction* nopInst =
            irBuilder.create(getMCIDByUOp(GFX::s_nop, archId), insertBefore);
        nopInst->addSrcReg(StinkyRegister(0));
    }

    const HwInstDesc* desc = getMCIDByUOp(GFX::s_set_vgpr_msb, archId);
    assert(desc != nullptr && "s_set_vgpr_msb is not supported on this architecture");
    StinkyInstruction* msbInst = irBuilder.create(desc, insertBefore);
    msbInst->addSrcReg(StinkyRegister(insertion.immediate));

    const int requiredSetVal = insertion.requiredMsb;
    std::string msbComment = "src0: " + std::to_string(decodeVgprMsbForSlot(requiredSetVal, 0)) +
                             ", src1: " + std::to_string(decodeVgprMsbForSlot(requiredSetVal, 1)) +
                             ", src2: " + std::to_string(decodeVgprMsbForSlot(requiredSetVal, 2)) +
                             ", dst: " + std::to_string(decodeVgprMsbForSlot(requiredSetVal, 3));
    msbInst->addModifier<CommentData>(CommentData{msbComment});
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

        runOnFunction(func, archId, msbMode);
        return preserveCFGAnalyses();
    }

   private:
    // The plan comes from planVgprMsb (shared with the co-issue repair's insertion model);
    // this pass only emits it and re-encodes the operands.
    static void runOnFunction(Function& func, GfxArchID archId, VgprMsbMode msbMode) {
        for (BasicBlock& bb : func) {
            std::vector<StinkyInstruction*> insts;
            for (IRBase& node : bb)
                if (auto* inst = dyn_cast<StinkyInstruction>(&node)) insts.push_back(inst);
            const std::vector<const StinkyInstruction*> view(insts.begin(), insts.end());
            const std::vector<VgprMsbInsertion> plan = planVgprMsb(view, msbMode);

            AsmIRBuilder irBuilder(bb, archId);
            for (const VgprMsbInsertion& insertion : plan)
                emitVgprMsb(insertion, irBuilder, archId, insts[insertion.before]);
            for (StinkyInstruction* inst : insts) {
                if (inst->getUnifiedOpcode() == GFX::LABEL || isPseudoInst(inst) || isCall(*inst))
                    continue;
                encodeVgprOperands(inst);
            }
        }
    }
};

char InsertVgprMsbPassImpl::ID = 0;

}  // anonymous namespace

std::unique_ptr<Pass> createInsertVgprMsbPass() {
    return std::make_unique<InsertVgprMsbPassImpl>();
}

}  // namespace stinkytofu
