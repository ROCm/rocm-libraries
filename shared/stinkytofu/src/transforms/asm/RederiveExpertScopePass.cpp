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

#include "stinkytofu/transforms/asm/RederiveExpertScopePass.hpp"

#include <string>
#include <utility>

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/IRBase.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/IntrusiveList.hpp"

#define DEBUG_TYPE "RederiveExpertScopePass"

namespace {
using namespace stinkytofu;

class RederiveExpertScopePassImpl : public Pass {
   public:
    static char ID;

    RederiveExpertScopePassImpl(StinkyAsmModule& module, std::string scope, std::string startLabel,
                                std::string endGroup)
        : module_(module),
          scope_(std::move(scope)),
          startLabel_(std::move(startLabel)),
          endGroup_(std::move(endGroup)) {}

    const char* getName() const override {
        return "RederiveExpertScopePass";
    }

    Pass::ID getPassID() const override {
        return &RederiveExpertScopePassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& /*passCtx*/,
                          AnalysisManager& /*AM*/) override {
        if (!module_.hasGroup(scope_)) return PreservedAnalyses::all();

        // Find the start label in the live IR for the range begin. It precedes
        // the mutated region, so it is never deleted by the earlier adaptor.
        IRBase* first = nullptr;
        for (BasicBlock& bb : func) {
            for (auto it = bb.begin(); it != bb.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst && isLabel(*inst)) {
                    const auto* ld = inst->getModifier<LabelData>();
                    if (ld && ld->label == startLabel_) {
                        first = it.getNodePtr();
                        break;
                    }
                }
            }
            if (first) break;
        }

        // The end boundary is the last instruction of endGroup, whose range the
        // earlier multi-region adaptor just refreshed. Fall back to end of BB.
        IRBase* last = nullptr;
        if (auto endRange = module_.findGroupRange(endGroup_)) {
            IRBase* candidate = endRange->second.getNodePtr();  // one-past-last
            // findGroupRange end() == ++last; step back to the real last node.
            if (candidate && candidate->getPrev() && candidate->getPrev()->isInList()) {
                last = candidate->getPrev();
            } else if (candidate && candidate->isInList()) {
                last = candidate;
            }
        }
        if (!last) {
            for (BasicBlock& bb : func) {
                if (bb.begin() != bb.end()) last = bb.rbegin().getNodePtr();
            }
        }

        if (first && last && first->isInList() && last->isInList()) {
            module_.setGroupRange(scope_, IntrusiveListIterator<IRBase>(first),
                                  IntrusiveListIterator<IRBase>(last));
        } else {
            // No valid landmarks: clear so findGroupRange() returns nullopt and
            // the downstream adaptor no-ops instead of dereferencing a dangling
            // range.
            module_.setGroupRange(scope_, IntrusiveListIterator<IRBase>(),
                                  IntrusiveListIterator<IRBase>());
        }
        return PreservedAnalyses::all();
    }

   private:
    StinkyAsmModule& module_;
    std::string scope_;
    std::string startLabel_;
    std::string endGroup_;
};

char RederiveExpertScopePassImpl::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createRederiveExpertScopePass(StinkyAsmModule& module, std::string scope,
                                                    std::string startLabel, std::string endGroup) {
    return std::make_unique<RederiveExpertScopePassImpl>(
        module, std::move(scope), std::move(startLabel), std::move(endGroup));
}
}  // namespace stinkytofu
