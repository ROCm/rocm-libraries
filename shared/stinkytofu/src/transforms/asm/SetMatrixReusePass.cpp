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
#include "stinkytofu/transforms/asm/SetMatrixReusePass.hpp"

#include <iostream>
#include <optional>
#include <string_view>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"

#define DEBUG_TYPE "SetMatrixReusePass"

namespace stinkytofu {
namespace {

struct MatrixOperands {
    const StinkyRegister* a = nullptr;
    const StinkyRegister* b = nullptr;
};

// Match rocisa reuse gating: non-scale WMMA f8f6f4 and MX f4-family never reuse.
static bool supportsMatrixReuse(const StinkyInstruction& inst) {
    const std::string_view m(inst.getHwInstDesc()->mnemonic);
    if (isWMMA(inst) && !isMXWMMA(inst) && m.find("_f8f6f4") != std::string_view::npos)
        return false;
    if (isMXWMMA(inst) && m.ends_with("_f4")) return false;
    return true;
}

std::optional<MatrixOperands> getMatrixOperands(const StinkyInstruction& inst) {
    const HwInstDesc* desc = inst.getHwInstDesc();
    if (!desc) return std::nullopt;

    MatrixOperands ops;
    const auto& fields = desc->operandFields;
    const auto& srcRegs = inst.getSrcRegs();
    unsigned srcIdx = 0;

    for (const auto& field : fields) {
        if (field.isDest || field.isReadWrite) continue;
        if (srcIdx >= srcRegs.size()) break;

        if (field.encodeField == EncodeField::src0)
            ops.a = &srcRegs[srcIdx];
        else if (field.encodeField == EncodeField::src1)
            ops.b = &srcRegs[srcIdx];
        ++srcIdx;
    }

    if (ops.a && ops.b) return ops;
    if (srcRegs.size() >= 2) {
        ops.a = &srcRegs[0];
        ops.b = &srcRegs[1];
        return ops;
    }
    return std::nullopt;
}

MFMAModifiers* getOrCreateMfmaModifiers(StinkyInstruction& inst) {
    if (MFMAModifiers* mod = inst.getModifier<MFMAModifiers>()) return mod;
    inst.addModifier<MFMAModifiers>(MFMAModifiers{});
    return inst.getModifier<MFMAModifiers>();
}

void clearMatrixReuse(StinkyInstruction& inst) {
    if (MFMAModifiers* mod = inst.getModifier<MFMAModifiers>()) {
        mod->reuseA = false;
        mod->reuseB = false;
    }
}

void applyReuseFromNext(StinkyInstruction& prev, const StinkyInstruction& next) {
    const auto prevOps = getMatrixOperands(prev);
    const auto nextOps = getMatrixOperands(next);
    if (!prevOps || !nextOps) return;

    MFMAModifiers* mod = getOrCreateMfmaModifiers(prev);
    mod->reuseA = (*prevOps->a == *nextOps->a);
    mod->reuseB = (*prevOps->b == *nextOps->b);

    PASS_DEBUG(std::cerr << "[SetMatrixReusePass] prev=" << prev.getHwInstDesc()->mnemonic
                         << " reuseA=" << mod->reuseA << " reuseB=" << mod->reuseB << "\n");
}

void setMatrixReuseInFunction(Function& func) {
    StinkyInstruction* pending = nullptr;

    for (BasicBlock& bb : func) {
        for (auto it = bb.begin(); it != bb.end(); ++it) {
            auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (!inst || !isMatrixInstruction(*inst)) continue;

            clearMatrixReuse(*inst);
            if (pending && supportsMatrixReuse(*pending)) applyReuseFromNext(*pending, *inst);
            pending = inst;
        }
    }
}

class SetMatrixReusePassImpl : public StinkyInstPass {
   public:
    static char ID;

    const char* getName() const override {
        return "SetMatrixReusePass";
    }

    PassID getPassID() const override {
        return &SetMatrixReusePassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& /*passCtx*/,
                          AnalysisManager& /*AM*/) override {
        setMatrixReuseInFunction(func);
        return preserveCFGAnalyses();
    }
};

char SetMatrixReusePassImpl::ID = 0;

}  // anonymous namespace

std::unique_ptr<Pass> createSetMatrixReusePass() {
    return std::make_unique<SetMatrixReusePassImpl>();
}

}  // namespace stinkytofu
