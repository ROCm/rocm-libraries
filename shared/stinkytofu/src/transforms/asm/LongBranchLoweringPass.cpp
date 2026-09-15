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

#include "stinkytofu/transforms/asm/LongBranchLoweringPass.hpp"

#include <optional>
#include <string>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace {
using namespace stinkytofu;

// Returns true if \p reg is a scalar register pair (S, num == 2) starting at
// the given index. The long-branch idiom carries the PC in such a pair.
bool isSGPRPair(const StinkyRegister& reg, uint32_t startIdx) {
    if (reg.dataType != StinkyRegister::Type::Register) return false;
    if (reg.reg.type != RegType::S) return false;
    if (reg.reg.num != 2) return false;
    return reg.reg.idx == startIdx;
}

// Returns true if \p reg is a single SGPR with the given index.
bool isSGPRScalar(const StinkyRegister& reg, uint32_t idx) {
    if (reg.dataType != StinkyRegister::Type::Register) return false;
    if (reg.reg.type != RegType::S) return false;
    if (reg.reg.num != 1) return false;
    return reg.reg.idx == idx;
}

// Returns true if \p reg is a literal int equal to \p value.
bool isLiteralInt(const StinkyRegister& reg, int value) {
    return reg.dataType == StinkyRegister::Type::LiteralInt && reg.literalInt == value;
}

// If \p inst is the s_add_i32 anchor of the long-branch idiom (an
// "s_add_i32 ?, LBL, +/-4" with a LiteralString label and an integer offset
// of +/-4), returns the label name; otherwise std::nullopt.
//
// The +4 (or -4) immediate compensates for the size of s_getpc_b64; rocisa
// emits 4 unconditionally. We accept either sign so that future tweaks to
// the rocisa helper are not silent matching failures here.
std::optional<std::string> matchAddI32Anchor(const StinkyInstruction& inst) {
    if (inst.getUnifiedOpcode() != GFX::s_add_i32) return std::nullopt;
    if (inst.getNumSrcRegs() != 2) return std::nullopt;
    const StinkyRegister& src0 = inst.getSrcReg(0);
    const StinkyRegister& src1 = inst.getSrcReg(1);
    if (src0.dataType != StinkyRegister::Type::LiteralString) return std::nullopt;
    if (src1.dataType != StinkyRegister::Type::LiteralInt) return std::nullopt;
    if (src1.literalInt != 4 && src1.literalInt != -4) return std::nullopt;
    return src0.getLiteralString();
}

// Walks backward from \p setpcInst (which must be an s_setpc_b64) within its
// basic block, looking for the rocisa long-branch idiom. Returns the target
// label name if the full def chain matches; std::nullopt otherwise.
//
// The full chain we verify (positive form):
//   s_getpc_b64                 s[D:D+1]    (optional anchor)
//   s_add_i32                   sX, LBL, 4
//   s_add_u32  / s_sub_u32      sD,   sD,   sX
//   s_addc_u32 / s_subb_u32     sD+1, sD+1, 0
//   s_setpc_b64                 s[D:D+1]
//
// We tolerate intervening s_abs_i32 sX, sX (negative arm) and unrelated
// instructions that touch other registers; we stop the search when we hit
// another branch (so we never cross a previous long-branch boundary).
std::optional<std::string> matchLongBranchIdiom(const BasicBlock& bb,
                                                BasicBlock::const_iterator setpcIt) {
    auto* setpcInst = dyn_cast<StinkyInstruction>(setpcIt.getNodePtr());
    if (!setpcInst) return std::nullopt;
    if (setpcInst->getUnifiedOpcode() != GFX::s_setpc_b64) return std::nullopt;
    if (setpcInst->getNumSrcRegs() < 1) return std::nullopt;

    // The setpc consumes an SGPR pair s[D:D+1] holding the target PC.
    const StinkyRegister& pcReg = setpcInst->getSrcReg(0);
    if (pcReg.dataType != StinkyRegister::Type::Register) return std::nullopt;
    if (pcReg.reg.type != RegType::S) return std::nullopt;
    if (pcReg.reg.num != 2) return std::nullopt;
    const uint32_t D = pcReg.reg.idx;

    // State machine:
    //   need_addc  : looking for s_addc_u32/s_subb_u32 sD+1, sD+1, 0
    //   need_addlo : looking for s_add_u32/s_sub_u32   sD,   sD,   sX
    //   need_addi  : looking for s_add_i32             sX, LBL, +/-4
    enum class Stage { NeedAddC, NeedAddLo, NeedAddI };
    Stage stage = Stage::NeedAddC;
    std::optional<uint32_t> X;  // index of the offset SGPR sX, learned from add_lo

    // Walk backward starting from the instruction immediately before setpc.
    // begin() is the first IR; if setpcIt == begin(), no idiom is possible.
    if (setpcIt == bb.begin()) return std::nullopt;
    auto it = setpcIt;
    do {
        --it;
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!inst) continue;

        // Bail if we cross another branch -- we should never reach into a
        // prior long-branch region or fall off the basic block boundary.
        if (isBranch(*inst) && inst != setpcInst) return std::nullopt;

        const uint16_t uop = inst->getUnifiedOpcode();

        if (stage == Stage::NeedAddC) {
            // s_addc_u32 sD+1, sD+1, 0   (positive arm)
            // s_subb_u32 sD+1, sD+1, 0   (negative arm)
            if (uop != GFX::s_addc_u32 && uop != GFX::s_subb_u32) continue;
            if (inst->getNumDestRegs() < 1 || inst->getNumSrcRegs() < 2) continue;
            if (!isSGPRScalar(inst->getDestReg(0), D + 1)) continue;
            if (!isSGPRScalar(inst->getSrcReg(0), D + 1)) continue;
            if (!isLiteralInt(inst->getSrcReg(1), 0)) continue;
            stage = Stage::NeedAddLo;
            continue;
        }

        if (stage == Stage::NeedAddLo) {
            // s_add_u32 sD, sD, sX  (positive arm)
            // s_sub_u32 sD, sD, sX  (negative arm)
            if (uop != GFX::s_add_u32 && uop != GFX::s_sub_u32) continue;
            if (inst->getNumDestRegs() < 1 || inst->getNumSrcRegs() < 2) continue;
            if (!isSGPRScalar(inst->getDestReg(0), D)) continue;
            if (!isSGPRScalar(inst->getSrcReg(0), D)) continue;
            const StinkyRegister& src1 = inst->getSrcReg(1);
            if (src1.dataType != StinkyRegister::Type::Register) continue;
            if (src1.reg.type != RegType::S || src1.reg.num != 1) continue;
            X = src1.reg.idx;
            stage = Stage::NeedAddI;
            continue;
        }

        // Stage::NeedAddI -- look for s_add_i32 sX, LBL, +/-4 producing sX.
        // Allow an optional s_abs_i32 sX, sX in between (negative arm).
        if (uop == GFX::s_abs_i32) {
            if (inst->getNumDestRegs() < 1 || inst->getNumSrcRegs() < 1) continue;
            if (!X || !isSGPRScalar(inst->getDestReg(0), *X)) continue;
            // s_abs is a passthrough: keep waiting for the producing s_add_i32.
            continue;
        }
        if (uop != GFX::s_add_i32) continue;
        if (inst->getNumDestRegs() < 1) continue;
        if (!X || !isSGPRScalar(inst->getDestReg(0), *X)) continue;

        std::optional<std::string> label = matchAddI32Anchor(*inst);
        if (!label) return std::nullopt;

        // Optional sanity check: confirm the s_getpc_b64 sometime earlier in
        // the block defines s[D:D+1]. If not present we still accept the
        // pattern -- the s_add_i32 + add_lo + addc_hi + setpc fingerprint is
        // already very specific to the long-branch idiom.
        for (auto bk = it; bk != bb.begin();) {
            --bk;
            auto* cand = dyn_cast<StinkyInstruction>(bk.getNodePtr());
            if (!cand) continue;
            if (cand->getUnifiedOpcode() != GFX::s_getpc_b64) continue;
            if (cand->getNumDestRegs() < 1) continue;
            if (isSGPRPair(cand->getDestReg(0), D)) break;
        }

        return label;

    } while (it != bb.begin());

    return std::nullopt;
}

// LongBranchLoweringPassImpl
//
// For every s_setpc_b64 missing intra-Function branch metadata, attempt to
// recover the static target label from the surrounding rocisa long-branch
// idiom and stamp it as a LabelData modifier. After this pass the
// CFGBuilderPass treats the s_setpc_b64 as a normal direct branch.
class LongBranchLoweringPassImpl : public Pass {
   public:
    static char ID;

    const char* getName() const override {
        return "LongBranchLoweringPass";
    }

    Pass::ID getPassID() const override {
        return &LongBranchLoweringPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        for (BasicBlock& bb : func) {
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;

            for (auto it = bb.begin(); it != bb.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (!inst) continue;
                if (inst->getUnifiedOpcode() != GFX::s_setpc_b64) continue;

                // Skip if the converter or a prior pass already stamped a target.
                if (inst->getModifier<LabelData>() != nullptr) continue;

                std::optional<std::string> label = matchLongBranchIdiom(bb, it);
                if (!label) continue;
                inst->addModifier(LabelData{*label});
            }
        }

        // We only attach modifiers to existing instructions; the CFG (if any
        // was built before this pass) is structurally unchanged. Subsequent
        // CFGBuilder runs will see the new LabelData and rebuild accordingly.
        return preserveCFGAnalyses();
    }
};

char LongBranchLoweringPassImpl::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createLongBranchLoweringPass() {
    return std::make_unique<LongBranchLoweringPassImpl>();
}
}  // namespace stinkytofu
