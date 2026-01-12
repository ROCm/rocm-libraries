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
#include <gtest/gtest.h>

#include "stinkytofu.hpp"

using namespace stinkytofu;

// Dummy IRBuilder for testing
class TestIRBuilder : public IRBuilder
{
public:
    static IRBuilder::ID ID;

    TestIRBuilder(IRList& irlist)
        : IRBuilder(irlist)
    {
    }

    // Method to verify which IRList this builder points to
    IRList* getCurrentIRList() const
    {
        return irlist;
    }
};
IRBuilder::ID TestIRBuilder::ID = &TestIRBuilder::ID;

// Test fixture for IRBuilder tests
class IRBuilderTest : public ::testing::Test
{
protected:
    Function    func;
    PassContext passCtx;
    BasicBlock* bb1;
    BasicBlock* bb2;
    BasicBlock* bb3;

    void SetUp() override
    {
        // Create a function with three BasicBlocks
        bb1 = func.createBasicBlock("bb1");
        bb2 = func.createBasicBlock("bb2");
        bb3 = func.createBasicBlock("bb3");
        func.setEntryBlock(bb1);
    }

    void TearDown() override
    {
        // Clean up is automatic via Function destructor
    }
};

// Test that IRBuilder correctly points to different IRLists
// This test would FAIL with the old cached implementation where
// the builder would still point to the first IRList
TEST_F(IRBuilderTest, BuilderPointsToCorrectIRListAcrossBasicBlocks)
{
    // Get builder for first BasicBlock
    auto builder1 = passCtx.getIRBuilder<TestIRBuilder>(bb1->getIR());
    EXPECT_EQ(builder1.getCurrentIRList(), &bb1->getIR());

    // Get builder for second BasicBlock
    auto builder2 = passCtx.getIRBuilder<TestIRBuilder>(bb2->getIR());
    EXPECT_EQ(builder2.getCurrentIRList(), &bb2->getIR());

    // Get builder for third BasicBlock
    auto builder3 = passCtx.getIRBuilder<TestIRBuilder>(bb3->getIR());
    EXPECT_EQ(builder3.getCurrentIRList(), &bb3->getIR());

    // Verify each builder still points to correct IRList
    EXPECT_EQ(builder1.getCurrentIRList(), &bb1->getIR());
    EXPECT_EQ(builder2.getCurrentIRList(), &bb2->getIR());
    EXPECT_EQ(builder3.getCurrentIRList(), &bb3->getIR());

    // Verify they point to different IRLists
    EXPECT_NE(builder1.getCurrentIRList(), builder2.getCurrentIRList());
    EXPECT_NE(builder2.getCurrentIRList(), builder3.getCurrentIRList());
    EXPECT_NE(builder1.getCurrentIRList(), builder3.getCurrentIRList());
}

// Test that getting builders in a loop works correctly
// This simulates the common pattern of iterating over BasicBlocks
TEST_F(IRBuilderTest, BuilderInLoopPointsToCurrentBasicBlock)
{
    std::vector<IRList*> expectedIRLists = {&bb1->getIR(), &bb2->getIR(), &bb3->getIR()};
    std::vector<IRList*> actualIRLists;

    size_t index = 0;
    for(BasicBlock& bb : func)
    {
        // Get builder for current BasicBlock
        auto builder = passCtx.getIRBuilder<TestIRBuilder>(bb.getIR());

        // Verify builder points to current BasicBlock's IRList
        EXPECT_EQ(builder.getCurrentIRList(), &bb.getIR());
        EXPECT_EQ(builder.getCurrentIRList(), expectedIRLists[index]);

        actualIRLists.push_back(builder.getCurrentIRList());
        index++;
    }

    EXPECT_EQ(actualIRLists.size(), 3);
    EXPECT_EQ(actualIRLists, expectedIRLists);
}

// Test that builder can update insertion point
TEST_F(IRBuilderTest, BuilderCanUpdateInsertionPoint)
{
    // Create builder pointing to first BasicBlock
    auto builder = passCtx.getIRBuilder<TestIRBuilder>(bb1->getIR());
    EXPECT_EQ(builder.getCurrentIRList(), &bb1->getIR());

    // Update insertion point to second BasicBlock
    builder.setInsertionPoint(bb2->getIR());
    EXPECT_EQ(builder.getCurrentIRList(), &bb2->getIR());

    // Update insertion point to third BasicBlock
    builder.setInsertionPoint(bb3->getIR());
    EXPECT_EQ(builder.getCurrentIRList(), &bb3->getIR());
}

// Test that multiple builders created in sequence don't interfere
TEST_F(IRBuilderTest, MultipleSequentialBuildersAreIndependent)
{
    // Create multiple builders for different BasicBlocks
    auto builder1 = passCtx.getIRBuilder<TestIRBuilder>(bb1->getIR());
    auto builder2 = passCtx.getIRBuilder<TestIRBuilder>(bb2->getIR());
    auto builder3 = passCtx.getIRBuilder<TestIRBuilder>(bb3->getIR());

    // All builders should point to their respective IRLists
    EXPECT_EQ(builder1.getCurrentIRList(), &bb1->getIR());
    EXPECT_EQ(builder2.getCurrentIRList(), &bb2->getIR());
    EXPECT_EQ(builder3.getCurrentIRList(), &bb3->getIR());

    // Create another builder for bb1
    auto builder4 = passCtx.getIRBuilder<TestIRBuilder>(bb1->getIR());
    EXPECT_EQ(builder4.getCurrentIRList(), &bb1->getIR());

    // Original builders should still be correct
    EXPECT_EQ(builder1.getCurrentIRList(), &bb1->getIR());
    EXPECT_EQ(builder2.getCurrentIRList(), &bb2->getIR());
    EXPECT_EQ(builder3.getCurrentIRList(), &bb3->getIR());
}

// Regression test: This would fail with the old cached implementation
// where calling getIRBuilder multiple times would return a reference
// to a cached builder that still pointed to the first IRList
TEST_F(IRBuilderTest, RegressionTestForCachedBuilderBug)
{
    // Simulate the bug scenario: get builder for each BB in sequence
    // With old implementation: all would point to bb1's IRList
    // With new implementation: each points to correct IRList

    auto    builder_for_bb1 = passCtx.getIRBuilder<TestIRBuilder>(bb1->getIR());
    IRList* irlist1         = builder_for_bb1.getCurrentIRList();

    auto    builder_for_bb2 = passCtx.getIRBuilder<TestIRBuilder>(bb2->getIR());
    IRList* irlist2         = builder_for_bb2.getCurrentIRList();

    auto    builder_for_bb3 = passCtx.getIRBuilder<TestIRBuilder>(bb3->getIR());
    IRList* irlist3         = builder_for_bb3.getCurrentIRList();

    // The bug would cause all three to point to bb1's IRList
    // Verify this is not the case
    EXPECT_EQ(irlist1, &bb1->getIR()) << "Builder 1 should point to BB1's IRList";
    EXPECT_EQ(irlist2, &bb2->getIR()) << "Builder 2 should point to BB2's IRList (BUG: would point "
                                         "to BB1 with old implementation)";
    EXPECT_EQ(irlist3, &bb3->getIR()) << "Builder 3 should point to BB3's IRList (BUG: would point "
                                         "to BB1 with old implementation)";

    // Verify they're all different
    EXPECT_NE(irlist1, irlist2) << "IRLists should be different";
    EXPECT_NE(irlist2, irlist3) << "IRLists should be different";
    EXPECT_NE(irlist1, irlist3) << "IRLists should be different";
}
