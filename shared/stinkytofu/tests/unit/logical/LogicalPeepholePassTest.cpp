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

#include "TestHelpers.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/logical/LogicalInstructions.hpp"
#include "stinkytofu/transforms/logical/LogicalPeepholePass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

class LogicalPeepholePassTest : public ::testing::Test {
   protected:
    void SetUp() override {
        passCtx = std::make_unique<PassContext>();
        arch = GfxArchID::Gfx1250;

        bb = func.createBasicBlock("entry");
    }

    void TearDown() override {
        passCtx.reset();
    }

    void runPass() {
        auto pass = createLogicalPeepholePass();
        pass->run(func, *passCtx, am);
    }

    Function func{"kernel"};
    std::unique_ptr<PassContext> passCtx;
    BasicBlock* bb;
    GfxArchID arch;
    AnalysisManager am;
};

// Test: Pass can be instantiated and run
TEST_F(LogicalPeepholePassTest, PassInstantiation) {
    auto pass = createLogicalPeepholePass();

    EXPECT_STREQ(pass->getName(), "LogicalPeepholePass");
}

// Test: Pass runs successfully on empty module
TEST_F(LogicalPeepholePassTest, EmptyModule) {
    // Run pass on empty function - should not crash
    runPass();

    // Verify the function is still valid and empty
    EXPECT_TRUE(bb->empty());
}

// Test: Pass runs successfully on module with simple instructions
TEST_F(LogicalPeepholePassTest, SimpleInstructions) {
    // Create instructions using factory functions (returns raw pointers)
    StinkyRegister v0 = vgpr(0);
    StinkyRegister v1 = vgpr(1);
    StinkyRegister v2 = vgpr(2);

    // v1 = v_add_f32(v0, v0) - use factory function
    bb->appendIR(static_cast<IRBase*>(VAddF32(v1, v0, v0, std::nullopt, std::nullopt, "")));

    // v2 = v_mul_f32(v1, 2.0)
    bb->appendIR(static_cast<IRBase*>(VMulF32(v2, v1, vgpr(2), std::nullopt, std::nullopt, "")));

    // Run pass - should not crash
    runPass();

    // Verify instructions are still there (no optimizations expected for this pattern)
    EXPECT_FALSE(bb->empty());

    // Count instructions
    size_t count = 0;
    for (auto& ir : *bb) {
        (void)ir;
        count++;
    }
    EXPECT_EQ(count, 2);
}

// Test: Pass can be run multiple times
TEST_F(LogicalPeepholePassTest, MultipleRuns) {
    StinkyRegister v0 = vgpr(0);
    StinkyRegister v1 = vgpr(1);

    bb->appendIR(static_cast<IRBase*>(VAddF32(v1, v0, v0, std::nullopt, std::nullopt, "")));

    // Run pass multiple times - should not crash and produce same results
    runPass();

    // Count instructions after first run
    size_t count1 = 0;
    for (auto& ir : *bb) {
        (void)ir;
        count1++;
    }

    runPass();

    // Count instructions after second run - should be the same
    size_t count2 = 0;
    for (auto& ir : *bb) {
        (void)ir;
        count2++;
    }

    EXPECT_EQ(count1, count2) << "Multiple runs should produce consistent results";
}

// Test: Mul-Mul fusion pattern (when implemented)
TEST_F(LogicalPeepholePassTest, MulMulFusion) {
    // Create a pattern that could be optimized: mul(mul(x, c1), c2) -> mul(x, c1*c2)
    StinkyRegister v0 = vgpr(0);
    StinkyRegister v1 = vgpr(1);
    StinkyRegister v2 = vgpr(2);

    // v1 = v_mul_f32(v0, 2.0)
    bb->appendIR(static_cast<IRBase*>(VMulF32(v1, v0, vgpr(10), std::nullopt, std::nullopt, "")));

    // v2 = v_mul_f32(v1, 3.0)
    bb->appendIR(static_cast<IRBase*>(VMulF32(v2, v1, vgpr(11), std::nullopt, std::nullopt, "")));

    // Verify we start with 2 instructions
    size_t countBefore = 0;
    for (auto& ir : *bb) {
        (void)ir;
        countBefore++;
    }
    EXPECT_EQ(countBefore, 2);

    // Run pass
    runPass();

    // Verify instructions after pass
    size_t countAfter = 0;
    for (auto& ir : *bb) {
        (void)ir;
        countAfter++;
    }

    // TODO: When pattern matching is implemented, verify optimization:
    // - v1 definition should be removed
    // - v2 should be v_mul_f32(v0, 6.0)
    // For now, just verify no crashes and IR is still valid
    EXPECT_GT(countAfter, 0) << "IR should not be empty after pass";
}

// Test: Add+Mul -> FMA fusion pattern (when implemented)
TEST_F(LogicalPeepholePassTest, AddFMAFusion) {
    // Create a pattern that could be optimized: add(mul(a, b), c) -> fma(a, b, c)
    StinkyRegister v0 = vgpr(0);
    StinkyRegister v1 = vgpr(1);
    StinkyRegister v2 = vgpr(2);
    StinkyRegister v3 = vgpr(3);

    // v2 = v_mul_f32(v0, v1)
    bb->appendIR(static_cast<IRBase*>(VMulF32(v2, v0, v1, std::nullopt, std::nullopt, "")));

    // v3 = v_add_f32(v2, v0)
    bb->appendIR(static_cast<IRBase*>(VAddF32(v3, v2, v0, std::nullopt, std::nullopt, "")));

    // Verify we start with 2 instructions
    size_t countBefore = 0;
    for (auto& ir : *bb) {
        (void)ir;
        countBefore++;
    }
    EXPECT_EQ(countBefore, 2);

    // Run pass
    runPass();

    // Verify instructions after pass
    size_t countAfter = 0;
    for (auto& ir : *bb) {
        (void)ir;
        countAfter++;
    }

    // TODO: When pattern matching is implemented, verify optimization:
    // - v2 definition should be removed
    // - v3 should be v_fma_f32(v0, v1, v0)
    // For now, just verify no crashes and IR is still valid
    EXPECT_GT(countAfter, 0) << "IR should not be empty after pass";
}
