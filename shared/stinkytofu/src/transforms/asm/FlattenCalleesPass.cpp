// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
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

/// First function ASM placement marker in \p func (program order), or nullptr.
StinkyInstruction* findFirstMarker(Function& func) {
    for (BasicBlock& bb : func) {
        for (IRBase& ir : bb) {
            auto* inst = dyn_cast<StinkyInstruction>(&ir);
            if (inst && isFunctionAsmPlacementMarker(*inst)) return inst;
        }
    }
    return nullptr;
}

/// Function in \p functions whose name matches \p name, or nullptr.
Function* findFunctionByName(const std::vector<Function*>& functions, const std::string& name) {
    for (Function* f : functions)
        if (f != nullptr && f->getName() == name) return f;
    return nullptr;
}

/// Move the named function's instructions in program order to right after the
/// marker, then erase the marker. Function name comes from the marker's LabelData.
/// Instructions are moved, not cloned (insertIR relinks each node).
void spliceFunctionAtMarker(StinkyInstruction* marker, const std::vector<Function*>& functions) {
    BasicBlock* hostBlock = marker->getParent();
    IRList::iterator markerIt(marker);

    // Anchor: node after the marker (may be end()); inserting before it keeps the
    // function body between the marker and what originally followed.
    IRList::iterator insertPos = markerIt;
    ++insertPos;

    const auto* nameMod = marker->getModifier<LabelData>();
    Function* target = nameMod != nullptr ? findFunctionByName(functions, nameMod->label) : nullptr;
    if (target != nullptr) {
        for (BasicBlock& targetBlock : *target) {
            for (auto it = targetBlock.begin(); it != targetBlock.end();) {
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
        // Markers from a spliced-in callable function land in the stream and
        // resolve on a later iteration (nested callables); a marker for an
        // already-emptied function inlines nothing and is just erased.
        while (StinkyInstruction* marker = findFirstMarker(func)) {
            spliceFunctionAtMarker(marker, functions);
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
