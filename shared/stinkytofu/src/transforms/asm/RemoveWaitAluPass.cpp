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
#include <utility>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HwReg.hpp"
#include "stinkytofu/hardware/HwRegHelpers.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/transforms/asm/InsertWaitAluPass.hpp"

#define DEBUG_TYPE "RemoveWaitAluPass"

namespace stinkytofu {
bool hasRegeneratedFieldWaitBundle(Function& func, PassContext& passCtx) {
    using F = SWaitAluData::Field;
    for (BasicBlock& bb : func) {
        if (!passCtx.shouldProcessBasicBlock(bb)) continue;
        for (auto it = bb.begin(); it != bb.end(); ++it) {
            auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (!inst || inst->getUnifiedOpcode() != GFX::s_wait_alu) continue;
            auto* data = inst->getModifier<SWaitAluData>();
            if (!data || (!data->hasField(F::VA_VDST) && !data->hasField(F::VM_VSRC))) continue;
            auto next = it;
            ++next;
            auto* nextInst =
                next != bb.end() ? dyn_cast<StinkyInstruction>(next.getNodePtr()) : nullptr;
            if (nextInst && (isWaitCnt(*nextInst) || nextInst->is(InstFlag::IF_WaitTensorCnt)))
                return true;
        }
    }
    return false;
}
}  // namespace stinkytofu

namespace {
using namespace stinkytofu;

// We only own the DEP_MODE sub-field of SCHED_MODE (the new InsertWaitAluPass
// takes over its lifecycle); DISABLE_XDL_ARB_STALL and any other sub-field
// of SCHED_MODE is orthogonal and must NOT be stripped by this pass.

// Scrubs mode1 wait_alu state ahead of InsertWaitAluPass, which re-inserts as mode2:
// strips s_setreg(SCHED_MODE) outright and clears the va_vdst / vm_vsrc fields of
// every s_wait_alu, the two InsertWaitAlu regenerates. va_sdst, va_ssrc, va_vcc,
// sa_sdst and hold_cnt are live expert-mode counters nothing re-inserts, so they
// stay; an s_wait_alu left with any of them survives.
//
// A pre-existing s_wait_alu immediately followed by a waitcnt is different: it
// deliberately drains two scoreboards at one placement point. Under per-pipe
// VA_VDST semantics the remove/insert pair cannot prove that splitting or moving
// that bundle is profitable. Mark the function and leave its wait schedule alone;
// InsertWaitAlu observes the marker and is also a no-op.
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
        auto archTriple = passCtx.getGemmTileConfig().arch;
        const GfxArchID arch = getGfxArchID(archTriple[0], archTriple[1], archTriple[2]);
        const uint16_t schedModeId = HwReg::schedModeId(arch);
        const HwReg::SubField depMode = HwReg::schedModeDepMode(arch);
        if (hasRegeneratedFieldWaitBundle(func, passCtx)) {
            func.setMetaData(kPreserveWaitAluScheduleMetaKey, 1);
            PASS_DEBUG(std::cerr << "[RemoveWaitAlu] @" << func.getName()
                                 << ": pre-existing wait bundle, preserve function schedule\n");
            return preserveCFGAnalyses();
        }
        // InsertWaitAlu re-enables mode 2 only at a kernel entry. In a region cut
        // from the middle of a kernel nothing would put a stripped mode switch back,
        // so the region keeps the ones it has.
        const bool keepSetreg = func.getMetaData(kFuncStartsAtKernelEntryMetaKey) == uint64_t{0};
        // Stripping leaves the function in mode 0, which is safe whether or not
        // InsertWaitAlu runs next, only if every DEP_MODE write goes. One that
        // also covers other SCHED_MODE bits cannot be deleted, so it could keep
        // mode 2 on with its waits gone: leave such a function alone.
        if (!keepSetreg && hasUnstrippableDepModeWrite(func, passCtx, schedModeId, depMode)) {
            PASS_DEBUG(std::cerr << "[RemoveWaitAlu] @" << func.getName()
                                 << ": SCHED_MODE write wider than DEP_MODE, nothing removed\n");
            return preserveCFGAnalyses();
        }
        removeInFunction(func, passCtx, schedModeId, depMode, keepSetreg);
        return preserveCFGAnalyses();
    }

   private:
    static bool hasUnstrippableDepModeWrite(Function& func, PassContext& passCtx,
                                            uint16_t schedModeId, HwReg::SubField depMode) {
        if (schedModeId == 0 || depMode.size == 0) return false;
        for (BasicBlock& bb : func) {
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;
            for (IRBase& ir : bb) {
                auto* inst = dyn_cast<StinkyInstruction>(&ir);
                if (!inst) continue;
                const uint16_t op = inst->getUnifiedOpcode();
                if (op != GFX::s_setreg_IMM32_b32 && op != GFX::s_setreg_b32) continue;
                if (HwReg::isSetregTo(*inst, schedModeId, depMode)) continue;
                for (const auto& r : inst->getDestRegs()) {
                    if (r.dataType != StinkyRegister::Type::HwReg || r.hwreg.id != schedModeId)
                        continue;
                    const unsigned lo = r.hwreg.offset, hi = lo + r.hwreg.size;
                    if (lo < unsigned(depMode.offset + depMode.size) && depMode.offset < hi)
                        return true;
                }
            }
        }
        return false;
    }

    static void removeInFunction(Function& func, PassContext& passCtx, uint16_t schedModeId,
                                 HwReg::SubField depMode, bool keepSetreg) {
        using F = SWaitAluData::Field;
        static constexpr F kRegeneratedFields[2] = {F::VA_VDST, F::VM_VSRC};

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

                if (!keepSetreg && HwReg::isSetregTo(*inst, schedModeId, depMode)) {
                    PASS_DEBUG(std::cerr << "[RemoveWaitAlu]   strip setreg(SCHED_MODE): "
                                         << inst->getHwInstDesc()->mnemonic << "\n");
                    ++strippedSetreg;
                    it = bb.eraseIR(it);
                    continue;
                }

                if (inst->getUnifiedOpcode() == GFX::s_wait_alu) {
                    auto* data = inst->getModifier<SWaitAluData>();
                    if (data) {
                        for (F field : kRegeneratedFields) data->clearField(field);
                    }
                    // Delete the instruction only when no field remains.
                    // Survivors stay in place;
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
                    PASS_DEBUG(std::cerr << "[RemoveWaitAlu]   clear va_vdst/vm_vsrc, keep "
                                            "s_wait_alu (other fields survive)\n");
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
    }
};

char RemoveWaitAluPassImpl::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createRemoveWaitAluPass() {
    return std::make_unique<RemoveWaitAluPassImpl>();
}
}  // namespace stinkytofu
