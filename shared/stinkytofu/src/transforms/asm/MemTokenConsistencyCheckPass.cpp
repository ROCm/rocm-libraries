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
#include "stinkytofu/transforms/asm/MemTokenConsistencyCheckPass.hpp"

#include <iostream>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"

namespace {
using namespace stinkytofu;

static bool isMemTokenCandidate(const StinkyInstruction& inst) {
    return isTensorLoad(inst) || isDSWrite(inst) || isDSRead(inst);
}

static const char* memTokenCandidateKind(const StinkyInstruction& inst) {
    if (isTensorLoad(inst)) return "tensor_load";
    if (isDSWrite(inst)) return "ds_store";
    if (isDSRead(inst)) return "ds_load";
    return "unknown";
}

static void checkConsistentMemTokens(const BasicBlock& bb) {
    bool hasWithToken = false;
    bool hasWithoutToken = false;

    for (auto it = bb.begin(); it != bb.end(); ++it) {
        const auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!inst || !isMemTokenCandidate(*inst)) continue;

        if (inst->getModifier<MemTokenData>())
            hasWithToken = true;
        else
            hasWithoutToken = true;
    }

    if (!hasWithToken || !hasWithoutToken) return;

    std::cerr << "[MemTokenConsistencyCheck] ERROR: BB \"" << bb.getLabel()
              << "\" has inconsistent memory tokens — some ds_load/ds_store/tensor_load"
                 " instructions have MemTokenData while others do not:\n";

    for (auto it = bb.begin(); it != bb.end(); ++it) {
        const auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!inst || !isMemTokenCandidate(*inst)) continue;

        const bool hasToken = inst->getModifier<MemTokenData>() != nullptr;
        std::cerr << "  " << (hasToken ? "[has token]   " : "[NO TOKEN]    ")
                  << memTokenCandidateKind(*inst) << " (" << inst->getHwInstDesc()->mnemonic
                  << ")\n";
    }

    report_fatal_error(
        "inconsistent MemTokenData across ds_load/ds_store/tensor_load in basic block");
}

class MemTokenConsistencyCheckPass : public Pass {
   public:
    static char ID;

    const char* getName() const override {
        return "MemTokenConsistencyCheckPass";
    }

    PassID getPassID() const override {
        return &MemTokenConsistencyCheckPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& /*passCtx*/,
                          AnalysisManager& /*AM*/) override {
        for (const BasicBlock& bb : func) {
            checkConsistentMemTokens(bb);
        }
        return PreservedAnalyses::all();
    }
};

char MemTokenConsistencyCheckPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createMemTokenConsistencyCheckPass() {
    return std::make_unique<MemTokenConsistencyCheckPass>();
}
}  // namespace stinkytofu
