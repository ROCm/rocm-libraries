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

#include "stinkytofu/transforms/asm/StinkyRemoveWaitCntPass.hpp"

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace {
using namespace stinkytofu;

bool tryRemoveTensorWaitCnt(StinkyInstruction* stinkyInst) {
    if (stinkyInst && stinkyInst->is(InstFlag::IF_WaitTensorCnt)) {
        return true;
    }
    return false;
}

void removeWaitCntsInBlock(BasicBlock& bb, bool removeTensorWaitCnt) {
    for (auto it = bb.begin(); it != bb.end();) {
        auto* stinkyInst = dyn_cast<StinkyInstruction>(it.getNodePtr());

        if (stinkyInst && (isWaitCnt(*stinkyInst) ||
                           (removeTensorWaitCnt && tryRemoveTensorWaitCnt(stinkyInst)))) {
            it = bb.eraseIR(it);
        } else {
            ++it;
        }
    }
}

class StinkyRemoveWaitCntPass : public StinkyInstPass {
   public:
    StinkyRemoveWaitCntPass(bool removeTensorWaitCnt) : removeTensorWaitCnt(removeTensorWaitCnt) {}

    static char ID;

    const char* getName() const override {
        return "StinkyRemoveWaitCntPass";
    }

    PassID getPassID() const override {
        return &StinkyRemoveWaitCntPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        for (BasicBlock& bb : func) {
            if (passCtx.shouldProcessBasicBlock(bb)) {
                removeWaitCntsInBlock(bb, removeTensorWaitCnt);
            }
        }
        return preserveCFGAnalyses();
    }

   private:
    bool removeTensorWaitCnt;
};

char StinkyRemoveWaitCntPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createStinkyRemoveWaitCntPass(bool removeTensorWaitCnt) {
    return std::make_unique<StinkyRemoveWaitCntPass>(removeTensorWaitCnt);
}
}  // namespace stinkytofu
