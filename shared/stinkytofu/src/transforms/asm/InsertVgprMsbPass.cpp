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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"
#include "stinkytofu/transforms/asm/InsertWaitAluPass.hpp"
#include "stinkytofu/transforms/asm/RaiseVgprMsbPass.hpp"

namespace stinkytofu {
namespace {
enum VgprMsbState : int {
    NOT_REQUIRED = -1,
    LABEL_BEGIN = -2,
};

bool isMsbComputableClass(const StinkyInstruction& inst) {
    return !(inst.is(InstFlag::IF_SALU) || inst.is(InstFlag::IF_SMemLoad) ||
             inst.is(InstFlag::IF_SMemStore) || inst.is(InstFlag::IF_SMemAtomic) ||
             inst.is(InstFlag::IF_Branch) || inst.is(InstFlag::IF_Call) ||
             inst.is(InstFlag::IF_Barrier) || inst.is(InstFlag::IF_WaitCnt) ||
             inst.is(InstFlag::IF_HasSideEffect));
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

// `labelEntryMsb` is the mode a label is entered in, the previous mode of the
// first setter after it.
bool emitVgprMsbIfNeeded(int requiredSetVal, bool hasVgpr, int& currentMsb, AsmIRBuilder& irBuilder,
                         GfxArchID archId, IRBase* insertBefore, VgprMsbMode msbMode,
                         int labelEntryMsb = 0) {
    if (!hasVgpr || requiredSetVal == currentMsb) {
        if (currentMsb == VgprMsbState::LABEL_BEGIN)
            currentMsb = labelEntryMsb != 0 ? labelEntryMsb : VgprMsbState::NOT_REQUIRED;
        return false;
    }

    if (currentMsb == VgprMsbState::LABEL_BEGIN) {
        StinkyInstruction* nopInst =
            irBuilder.create(getMCIDByUOp(GFX::s_nop, archId), insertBefore);
        nopInst->addSrcReg(StinkyRegister(0));
    }

    int combinedSetVal = requiredSetVal;
    if (msbMode == VgprMsbMode::Msb16 && currentMsb != VgprMsbState::NOT_REQUIRED) {
        combinedSetVal += (currentMsb == VgprMsbState::LABEL_BEGIN ? labelEntryMsb : currentMsb)
                          << 8;
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
    return true;
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
        if (func.getMetaData(kPreserveWaitAluScheduleMetaKey) == uint64_t{1})
            return PreservedAnalyses::all();
        auto arch = passCtx.getGemmTileConfig().arch;
        GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);

        VgprMsbMode msbMode = passCtx.getAsmCapsConfig().vgprMsbMode;
        if (msbMode == VgprMsbMode::None) return preserveCFGAnalyses();

        runOnFunction(func, archId, msbMode);
        return preserveCFGAnalyses();
    }

   private:
    static void runOnFunction(Function& func, GfxArchID archId, VgprMsbMode msbMode) {
        // A labeled block starts with its LABEL, which resets the state; a block
        // without one is reached only by fall-through and keeps it.
        //
        // Every edge into a label carries the label's entry mode, 0 unless
        // RaiseVgprMsbPass recorded the input's (a region's labels can be entered
        // from outside it): s_set_vgpr_msb's high byte is the previous mode, so a
        // setter after a join with different incoming modes has no correct
        // encoding. The mode is set to the target's entry mode before each branch
        // and before falling into a label (as the generators do), and back to 0
        // before s_endpgm, since the mode is MODE register state.
        int currentMsb = VgprMsbState::NOT_REQUIRED;
        int labelEntryMsb = 0;
        // The function's entry is reached from outside, already in the label's mode.
        bool fallsThrough = false;
        auto entryMsbOf = [&](const std::string& label) {
            const auto msb =
                label.empty() ? std::nullopt : func.getMetaData(labelEntryVgprMsbMetaKey(label));
            return msb ? static_cast<int>(*msb) : 0;
        };
        auto setModeTo = [&](int mode, AsmIRBuilder& irBuilder, IRBase* insertBefore) {
            const int now = currentMsb == VgprMsbState::LABEL_BEGIN ? labelEntryMsb : currentMsb;
            if (now == mode || (mode == 0 && now < 0)) return;
            emitVgprMsbIfNeeded(mode, true, currentMsb, irBuilder, archId, insertBefore, msbMode,
                                labelEntryMsb);
        };
        for (auto bbIt = func.begin(); bbIt != func.end(); ++bbIt) {
            BasicBlock& bb = *bbIt;
            AsmIRBuilder irBuilder(bb, archId);
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
                    const auto* label = inst->getModifier<LabelData>();
                    const int entry = entryMsbOf(label ? label->label : std::string());
                    if (fallsThrough) setModeTo(entry, irBuilder, inst);
                    currentMsb = VgprMsbState::LABEL_BEGIN;
                    labelEntryMsb = entry;
                    preferredInsertBefore = nullptr;
                    fallsThrough = true;
                    continue;
                }

                if (isPseudoInst(inst)) continue;

                fallsThrough = !isUnconditionalBranch(*inst) && !isEndOfFunction(*inst);
                if (inst->is(InstFlag::IF_Branch) || isEndOfFunction(*inst)) {
                    setModeTo(isBranch(*inst) ? entryMsbOf(getBranchTarget(*inst)) : 0, irBuilder,
                              inst);
                    preferredInsertBefore = nullptr;
                    continue;
                }

                // A call (e.g. s_swappc_b64) transfers to a callee that may leave
                // the VGPR MSB hardware register in an unknown state. Reset the
                // tracked value so the next VGPR op re-establishes MSB — matching
                // the single-function pipeline, which re-established MSB after the
                // call because the call ended a basic block.
                if (isCall(*inst)) {
                    currentMsb = VgprMsbState::NOT_REQUIRED;
                    // Never carry a deferred insertion anchor across call boundaries:
                    // call may clobber VGPR MSB state, so post-call rebuilds must stay post-call.
                    preferredInsertBefore = nullptr;
                    continue;
                }

                IRBase* insertBefore = preferredInsertBefore ? preferredInsertBefore : inst;

                auto [requiredMsb, hasVgpr] = computeRequiredMsb(inst);
                bool emittedVgprMsb =
                    emitVgprMsbIfNeeded(requiredMsb, hasVgpr, currentMsb, irBuilder, archId,
                                        insertBefore, msbMode, labelEntryMsb);
                encodeVgprOperands(inst);
                if (emittedVgprMsb || isMsbComputableClass(*inst)) preferredInsertBefore = nullptr;

                if (preferInsertAfter(*inst)) {
                    preferredInsertBefore = findNextInstructionAnchor(std::next(it));
                }
            }
        }
        restoreExitMsb(func, archId, msbMode, currentMsb, labelEntryMsb);
    }

    // Code after a region was written for the MSB state the input left at the
    // region's fall-through exit (RaiseVgprMsbPass records it). Put it back.
    static void restoreExitMsb(Function& func, GfxArchID archId, VgprMsbMode msbMode,
                               int currentMsb, int labelEntryMsb) {
        const auto exitMsb = func.getMetaData(kFuncExitVgprMsbMetaKey);
        if (!exitMsb || func.empty()) return;
        const int want = static_cast<int>(*exitMsb);
        if (currentMsb == want) return;
        if (currentMsb == VgprMsbState::LABEL_BEGIN && labelEntryMsb != 0 && labelEntryMsb == want)
            return;

        BasicBlock* lastBB = nullptr;
        for (BasicBlock& b : func) lastBB = &b;
        BasicBlock& bb = *lastBB;
        StinkyInstruction* last = nullptr;
        for (auto it = bb.begin(); it != bb.end(); ++it)
            if (auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                inst && !isPseudoInst(inst))
                last = inst;
        if (last && last->getUnifiedOpcode() == GFX::s_endpgm) return;
        // A trailing branch is the back-edge or exit of the region: the fall-through
        // path leaves the region after it, so restore before it. Its taken target
        // starts with a label and re-establishes its own state.
        IRBase* insertBefore = last && last->is(InstFlag::IF_Branch) ? last : nullptr;
        AsmIRBuilder irBuilder(bb, archId);
        emitVgprMsbIfNeeded(want, true, currentMsb, irBuilder, archId, insertBefore, msbMode,
                            labelEntryMsb);
    }
};

char InsertVgprMsbPassImpl::ID = 0;

}  // anonymous namespace

std::unique_ptr<Pass> createInsertVgprMsbPass() {
    return std::make_unique<InsertVgprMsbPassImpl>();
}

}  // namespace stinkytofu
