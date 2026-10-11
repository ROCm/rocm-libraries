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
#include <vector>

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
    return reg.getIntegerValue() == value;
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
    const auto offset = src1.getIntegerValue();
    if (offset != 4 && offset != -4) return std::nullopt;
    return src0.getLiteralString();
}

bool writesSGPR(const StinkyInstruction& inst, uint32_t idx) {
    for (unsigned i = 0; i < inst.getNumDestRegs(); ++i) {
        const StinkyRegister& r = inst.getDestReg(i);
        if (r.dataType != StinkyRegister::Type::Register || r.reg.type != RegType::S) continue;
        if (idx >= r.reg.idx && idx < r.reg.idx + r.reg.num) return true;
    }
    return false;
}

// Instruction-by-instruction backward walk over one basic block, skipping
// non-instruction nodes (comments, directives).
class BackwardScan {
   public:
    BackwardScan(const BasicBlock& bb, BasicBlock::const_iterator from) : bb_(bb), it_(from) {}

    const StinkyInstruction* prev() {
        while (it_ != bb_.begin()) {
            --it_;
            if (auto* inst = dyn_cast<StinkyInstruction>(it_.getNodePtr())) return inst;
        }
        return nullptr;
    }

    // Scans back to the nearest instruction satisfying \p pred. Fails (returns
    // nullptr, with \p stop set to the culprit) at a label or branch, at an
    // instruction writing one of \p guarded, or, with \p guardScc, at an SCC
    // writer: any of those breaks the def chain or lets another path in.
    template <typename Pred>
    const StinkyInstruction* findDef(Pred pred, const std::vector<uint32_t>& guarded, bool guardScc,
                                     const StinkyInstruction** stop = nullptr) {
        while (const StinkyInstruction* inst = prev()) {
            if (pred(*inst)) return inst;
            bool blocks = isLabel(*inst) || isBranch(*inst) ||
                          (guardScc && inst->is(InstFlag::IF_ImplicitWriteSCC));
            for (uint32_t r : guarded) blocks = blocks || writesSGPR(*inst, r);
            if (blocks) {
                if (stop) *stop = inst;
                return nullptr;
            }
        }
        return nullptr;
    }

   private:
    const BasicBlock& bb_;
    BasicBlock::const_iterator it_;
};

bool isCmpGeZero(const StinkyInstruction* inst, uint32_t x) {
    return inst && inst->getUnifiedOpcode() == GFX::s_cmp_ge_i32 && inst->getNumSrcRegs() == 2 &&
           isSGPRScalar(inst->getSrcReg(0), x) && isLiteralInt(inst->getSrcReg(1), 0);
}

// Walks backward from \p setpcIt (an s_setpc_b64) within its basic block,
// looking for one of the rocisa long-branch idioms. Returns the target label
// if the full def chain matches; std::nullopt otherwise.
//
// The chain (positive arm / negative arm):
//   s_getpc_b64                 s[D:D+1]
//   s_add_i32                   sX, LBL, +/-4
//   [s_cmp_ge_i32 sX, 0; s_cbranch_scc1 POS]     dispatching SLongBranch only
//   [s_abs_i32 sX, sX]                           negative arm only
//   s_add_u32  / s_sub_u32      sD,   sD,   sX
//   s_addc_u32 / s_subb_u32     sD+1, sD+1, 0
//   s_setpc_b64                 s[D:D+1]
//
// Unrelated instructions may sit between the links as long as they do not
// write sD, sD+1 or sX (or SCC between the lo/hi pair). The positive arm of the
// dispatching form starts at POS, right after the negative arm's s_setpc_b64:
// the walk crosses that arm, which is not on this path, and resumes at the
// s_cbranch_scc1 POS.
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

    BackwardScan scan(bb, setpcIt);

    const StinkyInstruction* hi = scan.findDef(
        [&](const StinkyInstruction& inst) {
            const uint16_t uop = inst.getUnifiedOpcode();
            return (uop == GFX::s_addc_u32 || uop == GFX::s_subb_u32) &&
                   inst.getNumDestRegs() >= 1 && inst.getNumSrcRegs() >= 2 &&
                   isSGPRScalar(inst.getDestReg(0), D + 1) &&
                   isSGPRScalar(inst.getSrcReg(0), D + 1) && isLiteralInt(inst.getSrcReg(1), 0);
        },
        {D, D + 1}, false);
    if (!hi) return std::nullopt;
    const bool negative = hi->getUnifiedOpcode() == GFX::s_subb_u32;
    const uint16_t loOp = negative ? GFX::s_sub_u32 : GFX::s_add_u32;

    const StinkyInstruction* lo = scan.findDef(
        [&](const StinkyInstruction& inst) {
            if (inst.getUnifiedOpcode() != loOp || inst.getNumDestRegs() < 1 ||
                inst.getNumSrcRegs() < 2)
                return false;
            const StinkyRegister& src1 = inst.getSrcReg(1);
            return isSGPRScalar(inst.getDestReg(0), D) && isSGPRScalar(inst.getSrcReg(0), D) &&
                   src1.dataType == StinkyRegister::Type::Register && src1.reg.type == RegType::S &&
                   src1.reg.num == 1;
        },
        {D, D + 1}, true);
    if (!lo) return std::nullopt;
    const uint32_t X = lo->getSrcReg(1).reg.idx;
    if (X == D || X == D + 1) return std::nullopt;

    const std::vector<uint32_t> guarded = {D, D + 1, X};
    auto isAnchor = [&](const StinkyInstruction& inst) {
        return inst.getUnifiedOpcode() == GFX::s_add_i32 && inst.getNumDestRegs() >= 1 &&
               isSGPRScalar(inst.getDestReg(0), X);
    };
    auto isAbs = [&](const StinkyInstruction& inst) {
        return inst.getUnifiedOpcode() == GFX::s_abs_i32 && inst.getNumDestRegs() >= 1 &&
               inst.getNumSrcRegs() >= 1 && isSGPRScalar(inst.getDestReg(0), X) &&
               isSGPRScalar(inst.getSrcReg(0), X);
    };

    const StinkyInstruction* stop = nullptr;
    const StinkyInstruction* anchor = nullptr;
    if (negative) {
        anchor = scan.findDef(
            [&](const StinkyInstruction& inst) { return isAnchor(inst) || isAbs(inst); }, guarded,
            false, &stop);
        if (anchor && isAbs(*anchor)) anchor = scan.findDef(isAnchor, guarded, false, &stop);
        if (!anchor && stop && stop->getUnifiedOpcode() == GFX::s_cbranch_scc1) {
            // Negative arm of the dispatching form: falls through when sX < 0.
            if (!isCmpGeZero(scan.prev(), X)) return std::nullopt;
            anchor = scan.findDef(isAnchor, guarded, false);
        }
    } else {
        anchor = scan.findDef(isAnchor, guarded, false, &stop);
        if (!anchor && stop && isLabel(*stop)) {
            // Positive arm of the dispatching form: POS is entered only from
            // s_cbranch_scc1 POS, because the negative arm above ends in its
            // own s_setpc_b64.
            const auto* posLabel = stop->getModifier<LabelData>();
            const StinkyInstruction* negSetpc = scan.prev();
            if (!posLabel || !negSetpc || negSetpc->getUnifiedOpcode() != GFX::s_setpc_b64)
                return std::nullopt;
            const StinkyInstruction* cbr = scan.prev();
            while (cbr && !isLabel(*cbr) && !isBranch(*cbr)) cbr = scan.prev();
            if (!cbr || cbr->getUnifiedOpcode() != GFX::s_cbranch_scc1 ||
                getBranchTarget(*cbr) != posLabel->label)
                return std::nullopt;
            if (!isCmpGeZero(scan.prev(), X)) return std::nullopt;
            anchor = scan.findDef(isAnchor, guarded, false);
        }
    }
    if (!anchor) return std::nullopt;

    std::optional<std::string> label = matchAddI32Anchor(*anchor);
    if (!label) return std::nullopt;

    // The offset is relative to the PC that s_getpc_b64 captured.
    const StinkyInstruction* getpc = scan.findDef(
        [&](const StinkyInstruction& inst) {
            return inst.getUnifiedOpcode() == GFX::s_getpc_b64 && inst.getNumDestRegs() >= 1 &&
                   isSGPRPair(inst.getDestReg(0), D);
        },
        {D, D + 1}, false);
    if (!getpc) return std::nullopt;

    return label;
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
            annotateLongBranches(bb);
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
void annotateLongBranches(BasicBlock& bb) {
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

std::unique_ptr<Pass> createLongBranchLoweringPass() {
    return std::make_unique<LongBranchLoweringPassImpl>();
}
}  // namespace stinkytofu
