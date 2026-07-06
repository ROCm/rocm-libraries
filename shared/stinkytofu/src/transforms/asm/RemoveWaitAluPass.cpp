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

#include "stinkytofu/transforms/asm/RemoveWaitAluPass.hpp"

#include <iostream>
#include <string>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HwReg.hpp"
#include "stinkytofu/hardware/HwRegHelpers.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"

#define DEBUG_TYPE "RemoveWaitAluPass"

namespace {
using namespace stinkytofu;

// We only own the DEP_MODE sub-field of SCHED_MODE (the new InsertWaitAluPass
// takes over its lifecycle); DISABLE_XDL_ARB_STALL and any other sub-field
// of SCHED_MODE is orthogonal and must NOT be stripped by this pass.

// Scrubs mode1 wait_alu state ahead of InsertWaitAluPass, which re-inserts as mode2:
// strips s_setreg(SCHED_MODE) outright and clears the six GPR hazard fields of
// every s_wait_alu. hold_cnt is left untouched because it encodes constraints
// the wait_alu pass does not model; a hold_cnt-only s_wait_alu survives.
class RemoveWaitAluPassImpl : public Pass {
   public:
    static char ID;

    const char* getName() const override {
        return "RemoveWaitAluPass";
    }

    Pass::ID getPassID() const override {
        return &RemoveWaitAluPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        using F = SWaitAluData::Field;
        static constexpr F kGprHazardFields[6] = {F::VA_VDST, F::VM_VSRC, F::VA_SDST,
                                                  F::VA_SSRC, F::VA_VCC,  F::SA_SDST};

        auto archTriple = passCtx.getGemmTileConfig().arch;
        const GfxArchID arch = getGfxArchID(archTriple[0], archTriple[1], archTriple[2]);
        const uint16_t schedModeId = HwReg::schedModeId(arch);
        const HwReg::SubField depMode = HwReg::schedModeDepMode(arch);

        unsigned strippedSetreg = 0;
        unsigned strippedWaitAlu = 0;
        unsigned clearedWaitAlu = 0;
        unsigned holdCntSurvivors = 0;

        for (BasicBlock& bb : func) {
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;

            PASS_DEBUG(std::cerr << "[RemoveWaitAlu] enter bb=\"" << bb.getLabel()
                                 << "\" ir_count=" << bb.size() << "\n");

            for (auto it = bb.begin(); it != bb.end();) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (!inst) {
                    ++it;
                    continue;
                }

                if (HwReg::isSetregTo(*inst, schedModeId, depMode)) {
                    PASS_DEBUG(std::cerr << "[RemoveWaitAlu]   strip setreg(SCHED_MODE): "
                                         << inst->getHwInstDesc()->mnemonic << "\n");
                    ++strippedSetreg;
                    it = bb.eraseIR(it);
                    continue;
                }

                if (inst->getUnifiedOpcode() == GFX::s_wait_alu) {
                    auto* data = inst->getModifier<SWaitAluData>();
                    if (data) {
                        for (F field : kGprHazardFields) data->clearField(field);
                    }
                    // Delete the instruction only when no field (including
                    // hold_cnt) remains. hold_cnt survivors stay in place;
                    // their position is stale after scheduling but silently
                    // deleting them could drop a real hold constraint.
                    if (!data || data->empty()) {
                        PASS_DEBUG(
                            std::cerr
                            << "[RemoveWaitAlu]   strip s_wait_alu (all-zero after clear)\n");
                        ++strippedWaitAlu;
                        it = bb.eraseIR(it);
                        continue;
                    }
                    PASS_DEBUG(std::cerr << "[RemoveWaitAlu]   clear gpr fields, keep s_wait_alu "
                                            "(hold_cnt survivor)\n");
                    ++clearedWaitAlu;
                    ++holdCntSurvivors;
                }

                ++it;
            }
        }

        PASS_DEBUG(std::cerr << "[RemoveWaitAlu] done: stripped_setreg=" << strippedSetreg
                             << " stripped_wait_alu=" << strippedWaitAlu
                             << " cleared_wait_alu=" << clearedWaitAlu
                             << " hold_cnt_survivors=" << holdCntSurvivors << "\n");
        return preserveCFGAnalyses();
    }
};

char RemoveWaitAluPassImpl::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createRemoveWaitAluPass() {
    return std::make_unique<RemoveWaitAluPassImpl>();
}
}  // namespace stinkytofu
