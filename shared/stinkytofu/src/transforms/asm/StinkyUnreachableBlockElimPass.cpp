/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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
#include "stinkytofu/transforms/asm/StinkyUnreachableBlockElimPass.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/support/CFGTraversal.hpp"
#include "stinkytofu/support/OptimizationRemark.hpp"

#define DEBUG_TYPE "StinkyUnreachableBlockElimPass"

namespace {
using namespace stinkytofu;

constexpr const char* kPassName = "StinkyUnreachableBlockElim";

const BasicBlock* findExcludedBlock(const Function& func, const PassContext& passCtx) {
    for (const BasicBlock& bb : func) {
        if (!passCtx.shouldProcessBasicBlock(bb)) return &bb;
    }
    return nullptr;
}

class StinkyUnreachableBlockElimPassImpl : public Pass {
   public:
    static char ID;

    explicit StinkyUnreachableBlockElimPassImpl(StinkyAsmModule* module) : module_(module) {}

    const char* getName() const override {
        return "StinkyUnreachableBlockElimPass";
    }

    PassID getPassID() const override {
        return &StinkyUnreachableBlockElimPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager&) override {
        if (func.empty() || func.getEntryBlock() == nullptr) return preserveCFGAnalyses();

        if (const BasicBlock* excluded = findExcludedBlock(func, passCtx)) {
            const std::string reason = "basic-block filtering excludes ^" + excluded->getLabel() +
                                       "; unreachable-block elimination needs the whole function";
            emitRemark(passCtx,
                       {OptimizationRemark::Kind::Missed, kPassName, "NotEliminated", reason});
            return preserveCFGAnalyses();
        }

        std::unordered_set<BasicBlock*> reachable;
        traverseCFGInRPO(func, [&](BasicBlock* bb) { reachable.insert(bb); });

        std::vector<BasicBlock*> dead;
        for (BasicBlock& bb : func) {
            if (!reachable.contains(&bb)) dead.push_back(&bb);
        }
        if (dead.empty()) return preserveCFGAnalyses();

        // Drop attached SSA before the CFG changes: incoming lists and the
        // arena shape would describe the function that still had these blocks.
        if (func.hasAttachedSSA()) func.clearAttachedSSA();

        // Let the module drop group ranges bounded by instructions in these
        // blocks while those instructions are still alive to be identified.
        if (module_ != nullptr) {
            for (const std::string& groupName : module_->invalidateGroupsInBlocks(dead)) {
                emitRemark(passCtx,
                           {OptimizationRemark::Kind::Missed, kPassName, "ClearedGroupRange",
                            "group '" + groupName +
                                "' was bounded by an unreachable instruction; its range "
                                "is cleared and downstream region passes will skip it"});
            }
        }

        for (BasicBlock* bb : dead) {
            func.removeSuccessorEdges(*bb);
            func.removePredecessorEdges(*bb);
        }
        for (BasicBlock* bb : dead) bb->erase();

        PASS_DEBUG(std::cerr << "[StinkyUnreachableBlockElimPass] @" << func.getName() << " erased "
                             << dead.size() << " unreachable block(s)\n");
        emitRemark(passCtx, {OptimizationRemark::Kind::Passed, kPassName, "EliminatedUnreachable",
                             "@" + func.getName() + ": erased " + std::to_string(dead.size()) +
                                 " unreachable block(s)"});
        return PreservedAnalyses::none();
    }

   private:
    StinkyAsmModule* module_ = nullptr;
};

char StinkyUnreachableBlockElimPassImpl::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createStinkyUnreachableBlockElimPass() {
    return std::make_unique<StinkyUnreachableBlockElimPassImpl>(nullptr);
}

std::unique_ptr<Pass> createStinkyUnreachableBlockElimPass(StinkyAsmModule& module) {
    return std::make_unique<StinkyUnreachableBlockElimPassImpl>(&module);
}
}  // namespace stinkytofu
