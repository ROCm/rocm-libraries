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
#include "stinkytofu/transforms/asm/FlattenCalleesPass.hpp"

#include <utility>
#include <vector>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/IRBase.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"

#define DEBUG_TYPE "FlattenCalleesPass"

namespace stinkytofu {
namespace {

/// First CALLEE_BODY marker in \p func (program order), or nullptr.
StinkyInstruction* findFirstMarker(Function& func) {
    for (BasicBlock& bb : func) {
        for (IRBase& ir : bb) {
            auto* inst = dyn_cast<StinkyInstruction>(&ir);
            if (inst && isCalleeBody(*inst)) return inst;
        }
    }
    return nullptr;
}

/// Callee Function in \p functions whose name matches \p name, or nullptr.
Function* findFunctionByName(const std::vector<Function*>& functions, const std::string& name) {
    for (Function* f : functions)
        if (f != nullptr && f->getName() == name) return f;
    return nullptr;
}

/// Move the named callee's instructions in program order to right after the
/// marker, then erase the marker. Callee name comes from the marker's LabelData.
/// Instructions are moved, not cloned (insertIR relinks each node).
void spliceCalleeAtMarker(StinkyInstruction* marker, const std::vector<Function*>& functions) {
    BasicBlock* hostBlock = marker->getParent();
    IRList::iterator markerIt(marker);

    // Anchor: node after the marker (may be end()); inserting before it keeps the
    // callee body between the marker and what originally followed.
    IRList::iterator insertPos = markerIt;
    ++insertPos;

    const auto* nameMod = marker->getModifier<LabelData>();
    Function* callee = nameMod != nullptr ? findFunctionByName(functions, nameMod->label) : nullptr;
    if (callee != nullptr) {
        for (BasicBlock& calleeBlock : *callee) {
            for (auto it = calleeBlock.begin(); it != calleeBlock.end();) {
                IRBase* node = it.getNodePtr();
                ++it;  // advance before the move detaches the node
                hostBlock->insertIR(insertPos, node);
            }
        }
    }

    hostBlock->eraseIR(markerIt);
}

class FlattenCalleesPassImpl : public Pass {
   public:
    static char ID;

    explicit FlattenCalleesPassImpl(std::vector<Function*> functions)
        : functions(std::move(functions)) {}

    const char* getName() const override {
        return "FlattenCalleesPass";
    }

    Pass::ID getPassID() const override {
        return &FlattenCalleesPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& /*passCtx*/,
                          AnalysisManager& /*AM*/) override {
        // One marker per iteration (each splice erases one, so this terminates).
        // Markers from a spliced-in callee land in the stream and resolve on a
        // later iteration (nested callees); a marker for an already-emptied
        // callee inlines nothing and is just erased.
        while (StinkyInstruction* marker = findFirstMarker(func)) {
            spliceCalleeAtMarker(marker, functions);
        }
        return PreservedAnalyses::none();
    }

   private:
    std::vector<Function*> functions;
};

char FlattenCalleesPassImpl::ID = 0;

}  // namespace

std::unique_ptr<Pass> createFlattenCalleesPass(std::vector<Function*> functions) {
    return std::make_unique<FlattenCalleesPassImpl>(std::move(functions));
}

}  // namespace stinkytofu
