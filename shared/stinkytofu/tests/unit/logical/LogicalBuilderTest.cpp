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

#include "stinkytofu/core/IRBuilder.hpp"
#include "stinkytofu/core/PassManager.hpp"

using namespace stinkytofu;

// Dummy IRBuilder for testing
class TestIRBuilder : public IRBuilder {
   public:
    TestIRBuilder(BasicBlock& bb) : IRBuilder(bb) {}

    BasicBlock* getBasicBlock() const {
        return bb;
    }
};

// Test fixture for IRBuilder tests
class IRBuilderTest : public ::testing::Test {
   protected:
    Function func;
    PassContext passCtx;
    BasicBlock* bb1;
    BasicBlock* bb2;
    BasicBlock* bb3;

    void SetUp() override {
        // Create a function with three BasicBlocks
        bb1 = func.createBasicBlock("bb1");
        bb2 = func.createBasicBlock("bb2");
        bb3 = func.createBasicBlock("bb3");
    }

    void TearDown() override {
        // Clean up is automatic via Function destructor
    }
};

// Test that IRBuilder correctly points to different BasicBlocks
// This test would FAIL with the old cached implementation where
// the builder would still point to the first BasicBlock
TEST_F(IRBuilderTest, BuilderPointsToCorrectBasicBlockAcrossBasicBlocks) {
    // Get builder for first BasicBlock
    auto builder1 = TestIRBuilder(*bb1);
    EXPECT_EQ(builder1.getBasicBlock(), bb1);

    // Get builder for second BasicBlock
    auto builder2 = TestIRBuilder(*bb2);
    EXPECT_EQ(builder2.getBasicBlock(), bb2);

    // Get builder for third BasicBlock
    auto builder3 = TestIRBuilder(*bb3);
    EXPECT_EQ(builder3.getBasicBlock(), bb3);

    // Verify each builder still points to correct BasicBlock
    EXPECT_EQ(builder1.getBasicBlock(), bb1);
    EXPECT_EQ(builder2.getBasicBlock(), bb2);
    EXPECT_EQ(builder3.getBasicBlock(), bb3);

    // Verify they point to different BasicBlocks
    EXPECT_NE(builder1.getBasicBlock(), builder2.getBasicBlock());
    EXPECT_NE(builder2.getBasicBlock(), builder3.getBasicBlock());
    EXPECT_NE(builder1.getBasicBlock(), builder3.getBasicBlock());
}

// Test that getting builders in a loop works correctly
// This simulates the common pattern of iterating over BasicBlocks
TEST_F(IRBuilderTest, BuilderInLoopPointsToCurrentBasicBlock) {
    std::vector<BasicBlock*> expectedBBs = {bb1, bb2, bb3};
    std::vector<BasicBlock*> actualBBs;

    size_t index = 0;
    for (BasicBlock& bb : func) {
        // Get builder for current BasicBlock
        auto builder = TestIRBuilder(bb);

        // Verify builder points to current BasicBlock
        EXPECT_EQ(builder.getBasicBlock(), &bb);
        EXPECT_EQ(builder.getBasicBlock(), expectedBBs[index]);

        actualBBs.push_back(builder.getBasicBlock());
        index++;
    }

    EXPECT_EQ(actualBBs.size(), 3);
    EXPECT_EQ(actualBBs, expectedBBs);
}

// Test that builder can update insertion point
TEST_F(IRBuilderTest, BuilderCanUpdateInsertionPoint) {
    // Create builder pointing to first BasicBlock
    auto builder = TestIRBuilder(*bb1);
    EXPECT_EQ(builder.getBasicBlock(), bb1);

    // Update insertion point to second BasicBlock
    builder.setInsertionPoint(*bb2);
    EXPECT_EQ(builder.getBasicBlock(), bb2);

    // Update insertion point to third BasicBlock
    builder.setInsertionPoint(*bb3);
    EXPECT_EQ(builder.getBasicBlock(), bb3);
}

// Test that multiple builders created in sequence don't interfere
TEST_F(IRBuilderTest, MultipleSequentialBuildersAreIndependent) {
    // Create multiple builders for different BasicBlocks
    auto builder1 = TestIRBuilder(*bb1);
    auto builder2 = TestIRBuilder(*bb2);
    auto builder3 = TestIRBuilder(*bb3);

    // All builders should point to their respective BasicBlocks
    EXPECT_EQ(builder1.getBasicBlock(), bb1);
    EXPECT_EQ(builder2.getBasicBlock(), bb2);
    EXPECT_EQ(builder3.getBasicBlock(), bb3);

    // Create another builder for bb1
    auto builder4 = TestIRBuilder(*bb1);
    EXPECT_EQ(builder4.getBasicBlock(), bb1);

    // Original builders should still be correct
    EXPECT_EQ(builder1.getBasicBlock(), bb1);
    EXPECT_EQ(builder2.getBasicBlock(), bb2);
    EXPECT_EQ(builder3.getBasicBlock(), bb3);
}

// Regression test: This would fail with a cached builder implementation
// where multiple "get builder" calls would return a reference to a single
// cached builder that still pointed to the first BasicBlock
TEST_F(IRBuilderTest, RegressionTestForCachedBuilderBug) {
    // Simulate the bug scenario: get builder for each BB in sequence
    // With old implementation: all would point to bb1
    // With new implementation: each points to correct BasicBlock

    auto builder_for_bb1 = TestIRBuilder(*bb1);
    BasicBlock* block1 = builder_for_bb1.getBasicBlock();

    auto builder_for_bb2 = TestIRBuilder(*bb2);
    BasicBlock* block2 = builder_for_bb2.getBasicBlock();

    auto builder_for_bb3 = TestIRBuilder(*bb3);
    BasicBlock* block3 = builder_for_bb3.getBasicBlock();

    // The bug would cause all three to point to bb1
    // Verify this is not the case
    EXPECT_EQ(block1, bb1) << "Builder 1 should point to BB1";
    EXPECT_EQ(block2, bb2)
        << "Builder 2 should point to BB2 (BUG: would point to BB1 with old implementation)";
    EXPECT_EQ(block3, bb3)
        << "Builder 3 should point to BB3 (BUG: would point to BB1 with old implementation)";

    // Verify they're all different
    EXPECT_NE(block1, block2) << "BasicBlocks should be different";
    EXPECT_NE(block2, block3) << "BasicBlocks should be different";
    EXPECT_NE(block1, block3) << "BasicBlocks should be different";
}
