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
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/transforms/asm/CFGBuilderPass.hpp"
#include "stinkytofu/transforms/asm/FlattenCFGPass.hpp"
#include "stinkytofu/transforms/asm/ssa/LiftAsmRegistersToSSAPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

constexpr GfxArchID kArch = GfxArchID::Gfx1250;

class FlattenCFGPassTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kArch);
        entry = func->createBasicBlock("entry");
    }

    void runFlatten() {
        auto pass = createFlattenCFGPass();
        pass->run(*func, passCtx, am);
    }

    void runCFGBuilder() {
        auto pass = createCFGBuilderPass();
        pass->run(*func, passCtx, am);
    }

    StinkyInstruction* createLabelInst(BasicBlock* bb, const std::string& name) {
        AsmIRBuilder builder(*bb, kArch);
        return builder.createLabel(name);
    }

    /// Dest register of every instruction in the one flat block, in list order.
    /// Labels have no dest and contribute -1, so this captures order and identity
    /// of the whole stream, not just its length.
    std::vector<int> streamShape() {
        std::vector<int> shape;
        for (BasicBlock& bb : *func) {
            for (IRBase& ir : bb) {
                auto* inst = dyn_cast<StinkyInstruction>(&ir);
                if (inst == nullptr) continue;
                shape.push_back(inst->getDestRegs().empty()
                                    ? -1
                                    : static_cast<int>(inst->getDestRegs().front().reg.idx));
            }
        }
        return shape;
    }

    size_t countLabels() {
        size_t labels = 0;
        for (BasicBlock& bb : *func) {
            for (IRBase& ir : bb) {
                auto* inst = dyn_cast<StinkyInstruction>(&ir);
                if (inst != nullptr && inst->getUnifiedOpcode() == GFX::LABEL) labels++;
            }
        }
        return labels;
    }

    /// Three labelled sections, so CFGBuilderPass has something to split on.
    void buildLabelledStream() {
        createVAddInBlock(entry, kArch, 2, 0, 1);
        createLabelInst(entry, "middle");
        createVAddInBlock(entry, kArch, 3, 0, 1);
        createLabelInst(entry, "tail");
        createVAddInBlock(entry, kArch, 4, 0, 1);
    }

    std::unique_ptr<Function> func;
    BasicBlock* entry = nullptr;
    PassContext passCtx;
    AnalysisManager am;
};

// The round trip is the contract: a stream that was split on labels comes back
// byte-for-byte in one block, so a group range captured before the split is a
// walkable single-list range again afterwards.
TEST_F(FlattenCFGPassTest, RoundTripPreservesStreamOrder) {
    buildLabelledStream();
    const std::vector<int> before = streamShape();
    ASSERT_EQ(before.size(), 5u);

    runCFGBuilder();
    ASSERT_GT(func->size(), 1u) << "CFGBuilderPass must split for this test to mean anything";

    runFlatten();

    EXPECT_EQ(func->size(), 1u);
    EXPECT_EQ(func->getEntryBlock(), entry);
    EXPECT_EQ(streamShape(), before);
}

// Labels are moved, never dropped: the CFGBuilderPass later in the pipeline
// splits on them again to give RegionClonePass its start blocks.
TEST_F(FlattenCFGPassTest, KeepsLabelInstructions) {
    buildLabelledStream();
    ASSERT_EQ(countLabels(), 2u);

    runCFGBuilder();
    runFlatten();

    EXPECT_EQ(countLabels(), 2u);
}

// Flattening is reversible, which is what lets the pipeline build a CFG for
// lifting and still hand a flat kernel to the region adaptor.
TEST_F(FlattenCFGPassTest, SplitAfterFlattenRebuildsTheSameBlocks) {
    buildLabelledStream();

    runCFGBuilder();
    const size_t blocksFirstBuild = func->size();
    std::vector<std::string> labelsFirstBuild;
    for (BasicBlock& bb : *func) labelsFirstBuild.push_back(bb.getLabel());

    runFlatten();
    runCFGBuilder();

    EXPECT_EQ(func->size(), blocksFirstBuild);
    std::vector<std::string> labelsSecondBuild;
    for (BasicBlock& bb : *func) labelsSecondBuild.push_back(bb.getLabel());
    EXPECT_EQ(labelsSecondBuild, labelsFirstBuild);
}

// Merging blocks leaves no CFG, so no stale edge may point at an erased block.
TEST_F(FlattenCFGPassTest, LeavesNoEdges) {
    buildLabelledStream();
    runCFGBuilder();
    ASSERT_FALSE(func->getEntryBlock()->getSuccessors().empty());

    runFlatten();

    EXPECT_TRUE(func->getEntryBlock()->getSuccessors().empty());
    EXPECT_TRUE(func->getEntryBlock()->getPredecessors().empty());
}

// A shadow-mode register allocation upstream leaves its arena attached. Its block
// arguments and slot indexes describe the block structure being dissolved, so
// carrying it past this pass would hand later passes a stale SSA view.
TEST_F(FlattenCFGPassTest, ClearsAttachedSSA) {
    buildLabelledStream();
    runCFGBuilder();

    Expected<LiftAttachedSSAResult> lifted = liftAsmRegistersToAttachedSSA(*func);
    ASSERT_TRUE(lifted.hasValue()) << (lifted.hasValue() ? "" : lifted.getError());
    ASSERT_TRUE(func->hasAttachedSSA());

    runFlatten();

    EXPECT_FALSE(func->hasAttachedSSA());
}

// Nothing to do on a kernel that was never split, and in particular the entry
// must not be disturbed.
TEST_F(FlattenCFGPassTest, LeavesAlreadyFlatFunctionAlone) {
    buildLabelledStream();
    const std::vector<int> before = streamShape();

    runFlatten();

    EXPECT_EQ(func->size(), 1u);
    EXPECT_EQ(func->getEntryBlock(), entry);
    EXPECT_EQ(streamShape(), before);
}

// Restructuring blocks is whole-function work. A region adaptor's temporary has a
// different entry, so a filtered run must decline rather than merge into it.
TEST_F(FlattenCFGPassTest, RefusesToRunWhenBlockFilteringExcludesABlock) {
    buildLabelledStream();
    runCFGBuilder();
    const size_t blocksBefore = func->size();
    passCtx.setBasicBlockFilter(BasicBlockFilterBuilder::byLabels({"entry"}));

    runFlatten();

    EXPECT_EQ(func->size(), blocksBefore);
}

}  // namespace
