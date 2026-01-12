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

#include "ir/passes/HighLevelPeepholePass.hpp"
#include "TestHelpers.hpp"
#include "ir/IRModule.hpp"
#include "ir/StinkyInstructions.hpp"
#include <gtest/gtest.h>

using namespace stinkytofu;
using namespace stinkytofu::test;

class HighLevelPeepholePassTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create a test module
        module = std::make_unique<IRModule>("test_module");
    }

    void TearDown() override
    {
        module.reset();
    }

    std::unique_ptr<IRModule> module;
};

// Test: Pass can be instantiated and run
TEST_F(HighLevelPeepholePassTest, PassInstantiation)
{
    HighLevelPeepholePass pass;

    EXPECT_STREQ(pass.name(), "HighLevelPeepholePass");
    EXPECT_EQ(pass.getOptimizationCount(), 0);
}

// Test: Pass runs successfully on empty module
TEST_F(HighLevelPeepholePassTest, EmptyModule)
{
    HighLevelPeepholePass pass;

    bool changed = pass.run(*module);

    // Pass should not modify empty module
    EXPECT_FALSE(changed);
    EXPECT_EQ(pass.getOptimizationCount(), 0);
}

// Test: Pass runs successfully on module with simple instructions
TEST_F(HighLevelPeepholePassTest, SimpleInstructions)
{
    HighLevelPeepholePass pass;

    // Create a simple basic block with a few instructions
    // NOTE: Since pattern matching is not yet implemented,
    // this just verifies the pass doesn't crash
    StinkyRegister v0 = vgpr(0);
    StinkyRegister v1 = vgpr(1);
    StinkyRegister v2 = vgpr(2);

    // v1 = v_add_f32(v0, v0)
    module->add(std::make_shared<VAddF32>(v1, v0, v0));

    // v2 = v_mul_f32(v1, 2.0)
    module->add(std::make_shared<VMulF32>(v2, v1, 2.0));

    bool changed = pass.run(*module);

    // Since pattern matching is not implemented, pass should not modify anything
    EXPECT_FALSE(changed);
    EXPECT_EQ(pass.getOptimizationCount(), 0);
}

// Test: Pass can run multiple times without issues
TEST_F(HighLevelPeepholePassTest, MultipleRuns)
{
    HighLevelPeepholePass pass;

    StinkyRegister v0 = vgpr(0);
    StinkyRegister v1 = vgpr(1);

    module->add(std::make_shared<VAddF32>(v1, v0, v0));

    // Run pass multiple times
    bool changed1 = pass.run(*module);
    bool changed2 = pass.run(*module);
    bool changed3 = pass.run(*module);

    EXPECT_FALSE(changed1);
    EXPECT_FALSE(changed2);
    EXPECT_FALSE(changed3);
    EXPECT_EQ(pass.getOptimizationCount(), 0);
}

// TODO: Add tests for actual pattern matching when implementation is complete
// Future tests should include:
//   - TEST_F(HighLevelPeepholePassTest, MulMulFusion) { ... }
//   - TEST_F(HighLevelPeepholePassTest, AddFMAFusion) { ... }
//   - TEST_F(HighLevelPeepholePassTest, MovPropagation) { ... }
//   - TEST_F(HighLevelPeepholePassTest, AlgebraicSimplification) { ... }

// Test: MUL+MUL fusion optimization
TEST_F(HighLevelPeepholePassTest, MulMulFusion)
{
    HighLevelPeepholePass pass;

    StinkyRegister v0 = vgpr(0);
    StinkyRegister v1 = vgpr(1);
    StinkyRegister v2 = vgpr(2);

    // Create: v1 = v_mul_f32(2.0, v0)  // Pattern expects constant first
    //         v2 = v_mul_f32(3.0, v1)
    module->add(std::make_shared<VMulF32>(v1, 2.0, v0));
    module->add(std::make_shared<VMulF32>(v2, 3.0, v1));

    // Expected after optimization:
    //         v2 = v_mul_f32(6.0, v0)  (folds 2.0 * 3.0 = 6.0)

    bool changed = pass.run(*module);

    // Should fuse the two multiplications
    EXPECT_TRUE(changed);
    EXPECT_EQ(pass.getOptimizationCount(), 1);

    // Verify the result: should have only one instruction left
    EXPECT_EQ(module->size(), 1);

    // Verify the remaining instruction has the correct operands
    const auto& instructions = module->getInstructions();
    ASSERT_EQ(instructions.size(), 1);

    auto* mulInst = instructions[0].get();
    EXPECT_EQ(mulInst->getOpcode(), HLIR::VMulF32);
    EXPECT_EQ(mulInst->dests[0], v2); // Destination should be v2
    // First source should be the folded constant 6.0
    EXPECT_EQ(mulInst->srcs[0].dataType, StinkyRegister::Type::LiteralDouble);
    EXPECT_NEAR(mulInst->srcs[0].literalDouble, 6.0, 0.001);
    EXPECT_EQ(mulInst->srcs[1], v0); // Second source should be v0
}

// Test: ADD+FMA fusion optimization
TEST_F(HighLevelPeepholePassTest, AddFMAFusion)
{
    HighLevelPeepholePass pass;

    StinkyRegister v0 = vgpr(0);
    StinkyRegister v1 = vgpr(1);
    StinkyRegister v2 = vgpr(2);
    StinkyRegister v3 = vgpr(3);

    // Create: v1 = v_fma_f32(v0, v2, 1.0)
    //         v3 = v_add_f32(2.0, v1)  // Note: constant first for pattern match
    module->add(std::make_shared<VFmaF32>(v1, v0, v2, 1.0));
    module->add(std::make_shared<VAddF32>(v3, 2.0, v1));

    // Expected after optimization:
    //         v3 = v_fma_f32(v0, v2, 3.0)  (folds 1.0 + 2.0 = 3.0)

    bool changed = pass.run(*module);

    // Should fuse ADD into FMA
    EXPECT_TRUE(changed);
    EXPECT_EQ(pass.getOptimizationCount(), 1);

    // Verify the result: should have only one instruction left
    EXPECT_EQ(module->size(), 1);

    // Verify the remaining instruction has the correct operands
    const auto& instructions = module->getInstructions();
    ASSERT_EQ(instructions.size(), 1);

    auto* fmaInst = instructions[0].get();
    EXPECT_EQ(fmaInst->getOpcode(), HLIR::VFmaF32);
    EXPECT_EQ(fmaInst->dests[0], v3); // Destination should be v3
    EXPECT_EQ(fmaInst->srcs[0], v0); // First source should be v0
    EXPECT_EQ(fmaInst->srcs[1], v2); // Second source should be v2
    // Third source should be the folded constant 3.0
    EXPECT_EQ(fmaInst->srcs[2].dataType, StinkyRegister::Type::LiteralDouble);
    EXPECT_NEAR(fmaInst->srcs[2].literalDouble, 3.0, 0.001);
}
