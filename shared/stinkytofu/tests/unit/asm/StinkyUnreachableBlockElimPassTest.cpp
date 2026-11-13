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

#include "TestHelpers.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/transforms/asm/StinkyUnreachableBlockElimPass.hpp"
#include "stinkytofu/transforms/asm/ssa/LiftAsmRegistersToSSAPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

constexpr GfxArchID kArch = GfxArchID::Gfx1250;

bool hasBlock(const Function& func, const std::string& label) {
    for (const BasicBlock& bb : func) {
        if (bb.getLabel() == label) return true;
    }
    return false;
}

BasicBlock* findBlock(Function& func, const std::string& label) {
    for (BasicBlock& bb : func) {
        if (bb.getLabel() == label) return &bb;
    }
    return nullptr;
}

class StinkyUnreachableBlockElimPassTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kArch);
        entry = func->createBasicBlock("entry");
    }

    void runPass() {
        auto pass = createStinkyUnreachableBlockElimPass();
        pass->run(*func, passCtx, am);
    }

    std::unique_ptr<Function> func;
    BasicBlock* entry = nullptr;
    PassContext passCtx;
    AnalysisManager am;
};

TEST_F(StinkyUnreachableBlockElimPassTest, ErasesOrphanBlock) {
    func->createBasicBlock("orphan");
    ASSERT_EQ(func->size(), 2u);

    runPass();

    EXPECT_EQ(func->size(), 1u);
    EXPECT_TRUE(hasBlock(*func, "entry"));
    EXPECT_FALSE(hasBlock(*func, "orphan"));
}

TEST_F(StinkyUnreachableBlockElimPassTest, KeepsReachableDiamond) {
    BasicBlock* left = func->createBasicBlock("left");
    BasicBlock* right = func->createBasicBlock("right");
    BasicBlock* join = func->createBasicBlock("join");
    func->addEdge(entry, left);
    func->addEdge(entry, right);
    func->addEdge(left, join);
    func->addEdge(right, join);

    runPass();

    EXPECT_EQ(func->size(), 4u);
    EXPECT_TRUE(hasBlock(*func, "entry"));
    EXPECT_TRUE(hasBlock(*func, "left"));
    EXPECT_TRUE(hasBlock(*func, "right"));
    EXPECT_TRUE(hasBlock(*func, "join"));
}

TEST_F(StinkyUnreachableBlockElimPassTest, ErasesUnreachableComponentWithInternalEdges) {
    BasicBlock* a = func->createBasicBlock("dead_a");
    BasicBlock* b = func->createBasicBlock("dead_b");
    func->addEdge(a, b);
    func->addEdge(b, a);

    runPass();

    EXPECT_EQ(func->size(), 1u);
    EXPECT_FALSE(hasBlock(*func, "dead_a"));
    EXPECT_FALSE(hasBlock(*func, "dead_b"));
}

TEST_F(StinkyUnreachableBlockElimPassTest, DetachesDeadPredecessorOfReachableBlock) {
    BasicBlock* join = func->createBasicBlock("join");
    BasicBlock* orphan = func->createBasicBlock("orphan");
    func->addEdge(entry, join);
    func->addEdge(orphan, join);

    runPass();

    EXPECT_FALSE(hasBlock(*func, "orphan"));
    BasicBlock* kept = findBlock(*func, "join");
    ASSERT_NE(kept, nullptr);
    ASSERT_EQ(kept->getPredecessors().size(), 1u);
    EXPECT_EQ(kept->getPredecessors().front(), entry);
}

TEST_F(StinkyUnreachableBlockElimPassTest, LeavesSingleReachableBlockAlone) {
    runPass();

    EXPECT_EQ(func->size(), 1u);
    EXPECT_EQ(func->getEntryBlock(), entry);
}

TEST_F(StinkyUnreachableBlockElimPassTest, RefusesToRunWhenBlockFilteringExcludesABlock) {
    func->createBasicBlock("orphan");
    passCtx.setBasicBlockFilter(BasicBlockFilterBuilder::byLabels({"entry"}));

    runPass();

    EXPECT_EQ(func->size(), 2u);
    EXPECT_TRUE(hasBlock(*func, "orphan"));
}

TEST_F(StinkyUnreachableBlockElimPassTest, ClearsAttachedSSAWhenErasing) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    Expected<LiftAttachedSSAResult> lifted = liftAsmRegistersToAttachedSSA(*func);
    ASSERT_TRUE(lifted.hasValue()) << (lifted.hasValue() ? "" : lifted.getError());
    ASSERT_TRUE(func->hasAttachedSSA());

    func->createBasicBlock("orphan");
    runPass();

    EXPECT_FALSE(hasBlock(*func, "orphan"));
    EXPECT_FALSE(func->hasAttachedSSA());
}

TEST_F(StinkyUnreachableBlockElimPassTest, MakesPreviouslyRejectedFunctionLiftable) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    func->createBasicBlock("orphan");

    Expected<LiftAttachedSSAResult> before = liftAsmRegistersToAttachedSSA(*func);
    ASSERT_TRUE(before.hasError());
    EXPECT_NE(before.getError().find("unreachable from the entry"), std::string::npos);

    runPass();

    Expected<LiftAttachedSSAResult> after = liftAsmRegistersToAttachedSSA(*func);
    EXPECT_TRUE(after.hasValue()) << (after.hasValue() ? "" : after.getError());
    EXPECT_TRUE(func->hasAttachedSSA());
}

// A module records each instruction group as a pair of raw boundary nodes, so
// erasing the block that holds one would leave the range dangling with nothing
// able to detect it afterwards. Given the module, the pass clears such a group
// while the nodes are still alive.
//
// Both groups are checked in one run because the property that matters is that
// the guard is per-group: a group in a surviving block keeps its range. A guard
// that cleared everything on any deletion would pass the first half alone.
TEST(StinkyUnreachableBlockElimGroupGuardTest, ClearsOnlyGroupsInErasedBlocks) {
    StinkyAsmModule::ModuleOptions opts{};
    opts.OptLevel = 0;
    StinkyAsmModule module("test", {12, 5, 0}, opts);
    module.addGroup("live");
    module.addGroup("dead");

    Function& func = module.getFunction();
    BasicBlock* entry = func.getEntryBlock();
    BasicBlock* orphan = func.createBasicBlock("orphan");

    StinkyInstruction* liveFirst = createVAddInBlock(entry, kArch, 2, 0, 1);
    StinkyInstruction* liveLast = createVAddInBlock(entry, kArch, 3, 0, 1);
    StinkyInstruction* deadFirst = createVAddInBlock(orphan, kArch, 4, 0, 1);
    StinkyInstruction* deadLast = createVAddInBlock(orphan, kArch, 5, 0, 1);

    module.setGroupRange("live", IntrusiveListIterator<IRBase>(liveFirst),
                         IntrusiveListIterator<IRBase>(liveLast));
    module.setGroupRange("dead", IntrusiveListIterator<IRBase>(deadFirst),
                         IntrusiveListIterator<IRBase>(deadLast));
    ASSERT_TRUE(module.findGroupRange("live").has_value());
    ASSERT_TRUE(module.findGroupRange("dead").has_value());

    PassContext passCtx;
    AnalysisManager am;
    createStinkyUnreachableBlockElimPass(module)->run(func, passCtx, am);

    ASSERT_FALSE(hasBlock(func, "orphan"));
    EXPECT_FALSE(module.findGroupRange("dead").has_value());
    EXPECT_TRUE(module.findGroupRange("live").has_value());
}

}  // namespace
