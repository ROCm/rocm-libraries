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
#include "stinkytofu/transforms/asm/FlattenCFGPass.hpp"

#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/support/OptimizationRemark.hpp"

#define DEBUG_TYPE "FlattenCFGPass"

namespace {
using namespace stinkytofu;

constexpr const char* kPassName = "FlattenCFG";

const BasicBlock* findExcludedBlock(const Function& func, const PassContext& passCtx) {
    for (const BasicBlock& bb : func) {
        if (!passCtx.shouldProcessBasicBlock(bb)) return &bb;
    }
    return nullptr;
}

class FlattenCFGPassImpl : public Pass {
   public:
    static char ID;

    const char* getName() const override {
        return "FlattenCFGPass";
    }

    PassID getPassID() const override {
        return &FlattenCFGPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager&) override {
        BasicBlock* entry = func.getEntryBlock();
        if (entry == nullptr || func.size() <= 1) return preserveCFGAnalyses();

        if (const BasicBlock* excluded = findExcludedBlock(func, passCtx)) {
            const std::string reason = "basic-block filtering excludes ^" + excluded->getLabel() +
                                       "; flattening needs the whole function";
            emitRemark(passCtx,
                       {OptimizationRemark::Kind::Missed, kPassName, "NotFlattened", reason});
            return preserveCFGAnalyses();
        }

        // The arena's block arguments and slot indexes describe the block
        // structure this pass dissolves.
        if (func.hasAttachedSSA()) func.clearAttachedSSA();

        // List order is program order: CFGBuilderPass creates one block per label
        // as it scans the flat stream forward.
        std::vector<BasicBlock*> merged;
        for (auto bbIt = std::next(func.begin()); bbIt != func.end(); ++bbIt) {
            BasicBlock& bb = *bbIt;
            for (auto irIt = bb.begin(); irIt != bb.end();) {
                IRBase* node = irIt.getNodePtr();
                ++irIt;
                bb.removeIR(node);
                entry->appendIR(node);
            }
            merged.push_back(&bb);
        }

        // Detach from both sides before erasing, and leave the entry with no
        // edges: one block with a dangling successor list is not a CFG.
        for (BasicBlock* bb : merged) {
            func.removeSuccessorEdges(*bb);
            func.removePredecessorEdges(*bb);
        }
        func.removeSuccessorEdges(*entry);
        func.removePredecessorEdges(*entry);

        for (BasicBlock* bb : merged) bb->erase();

        PASS_DEBUG(std::cerr << "[FlattenCFGPass] @" << func.getName() << " merged "
                             << merged.size() << " block(s) into ^" << entry->getLabel() << "\n");
        emitRemark(passCtx, {OptimizationRemark::Kind::Passed, kPassName, "Flattened",
                             "@" + func.getName() + ": merged " + std::to_string(merged.size()) +
                                 " block(s) into the entry"});
        return PreservedAnalyses::none();
    }
};

char FlattenCFGPassImpl::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createFlattenCFGPass() {
    return std::make_unique<FlattenCFGPassImpl>();
}
}  // namespace stinkytofu
