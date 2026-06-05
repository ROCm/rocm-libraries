// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/StinkyRemoveNopPass.hpp"

#include <iostream>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

#define DEBUG_TYPE "StinkyRemoveNopPass"

namespace {
using namespace stinkytofu;

bool isNop(const StinkyInstruction& inst) {
    const auto opcode = inst.getUnifiedOpcode();
    return opcode == GFX::s_nop || opcode == GFX::v_nop;
}

size_t removeNopsInBlock(BasicBlock& bb) {
    size_t removed = 0;
    for (auto it = bb.begin(); it != bb.end();) {
        auto* stinkyInst = dyn_cast<StinkyInstruction>(it.getNodePtr());

        if (stinkyInst && isNop(*stinkyInst)) {
            it = bb.eraseIR(it);
            ++removed;
        } else {
            ++it;
        }
    }
    return removed;
}

class StinkyRemoveNopPass : public StinkyInstPass {
   public:
    static char ID;

    const char* getName() const override {
        return "StinkyRemoveNopPass";
    }

    PassID getPassID() const override {
        return &StinkyRemoveNopPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        for (BasicBlock& bb : func) {
            if (passCtx.shouldProcessBasicBlock(bb)) {
                const size_t removed = removeNopsInBlock(bb);
                PASS_DEBUG(std::cerr << "[StinkyRemoveNopPass] bb=\"" << bb.getLabel()
                                     << "\" removed_nops=" << removed << "\n");
            }
        }
        return preserveCFGAnalyses();
    }
};

char StinkyRemoveNopPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createStinkyRemoveNopPass() {
    return std::make_unique<StinkyRemoveNopPass>();
}
}  // namespace stinkytofu
