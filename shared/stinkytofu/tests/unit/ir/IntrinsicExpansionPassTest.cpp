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

/**
 * @file IntrinsicExpansionPassTest.cpp
 * @brief Test the IntrinsicExpansionPass
 *
 * This test verifies that IntrinsicCall instructions are correctly expanded
 * into concrete LogicalInstructions using definitions from intrinsics.st.bc.
 */

#include <gtest/gtest.h>

#include <cstring>

#include "stinkytofu/core/IRBase.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/logical/IntrinsicCall.hpp"
#include "stinkytofu/ir/logical/IntrinsicRegistry.hpp"
#include "stinkytofu/transforms/logical/IntrinsicExpansionPass.hpp"

using namespace stinkytofu;

class IntrinsicExpansionPassTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Verify intrinsic registry is initialized
        auto& registry = IntrinsicRegistry::instance();
        ASSERT_TRUE(registry.isInitialized())
            << "IntrinsicRegistry not initialized. Make sure intrinsics.st.bc is loaded.";

        // Initialize PassContext with default config
        passCtx.setGemmTileConfig(defaultConfig);
    }

    GemmTileConfig defaultConfig = {{12, 5, 0}, 128, 128, 64, 0, 0, 0};
    PassContext passCtx;
    AnalysisManager am;
};

TEST_F(IntrinsicExpansionPassTest, ExpandSimpleIntrinsic) {
    Function func("kernel");
    BasicBlock* bb = func.createBasicBlock("entry");

    // Create an IntrinsicCall for "ReluF32"
    // ReluF32 signature: (dest, src)
    StinkyRegister v0(RegType::V, 0, 1);
    StinkyRegister v1(RegType::V, 1, 1);

    std::vector<StinkyRegister> args1 = {v0, v1};
    auto* ir = IRBase::createIR<IntrinsicCall>("ReluF32", args1);
    bb->insertIR(bb->end(), ir);

    // Verify initial state: 1 instruction (IntrinsicCall)
    ASSERT_EQ(bb->size(), 1);

    // Run IntrinsicExpansionPass
    auto pass = createIntrinsicExpansionPass();
    pass->run(func, passCtx, am);

    // After expansion, IntrinsicCall should be replaced with expanded instructions
    // ReluF32 expands to 1 instruction (v_max_f32)
    size_t numInsts = bb->size();
    EXPECT_EQ(numInsts, 1) << "IntrinsicCall should be expanded into 1 instruction (v_max_f32)";

    // Verify no IntrinsicCall remains
    bool hasIntrinsicCall = false;
    for (auto& ir : *bb) {
        if (ir.getType() == IRBase::IRType::LogicalIR) {
            auto* logical = static_cast<LogicalInstruction*>(&ir);
            if (std::strcmp(logical->getLogicalName(), "IntrinsicCall") == 0) {
                hasIntrinsicCall = true;
                break;
            }
        }
    }
    EXPECT_FALSE(hasIntrinsicCall) << "IntrinsicCall should be removed after expansion";

    // Dump expanded instructions for debugging
    std::cout << "Expanded ReluF32 into " << numInsts << " instructions:\n";
    for (auto& ir : *bb) {
        if (ir.getType() == IRBase::IRType::LogicalIR) {
            auto* logical = static_cast<LogicalInstruction*>(&ir);
            std::cout << "  - " << logical->getLogicalName() << "\n";
        }
    }
}

TEST_F(IntrinsicExpansionPassTest, UnknownIntrinsicFails) {
    Function func("kernel");
    BasicBlock* bb = func.createBasicBlock("entry");

    // Create an IntrinsicCall for non-existent intrinsic
    StinkyRegister v0(RegType::V, 0, 1);
    std::vector<StinkyRegister> args2 = {v0};
    auto* ir = IRBase::createIR<IntrinsicCall>("NonExistentIntrinsic", args2);
    bb->insertIR(bb->end(), ir);

    // Run IntrinsicExpansionPass - should fail gracefully
    auto pass = createIntrinsicExpansionPass();
    pass->run(func, passCtx, am);

    // IntrinsicCall should still be there (expansion failed)
    bool hasIntrinsicCall = false;
    for (auto& ir : *bb) {
        if (ir.getType() == IRBase::IRType::LogicalIR) {
            auto* logical = static_cast<LogicalInstruction*>(&ir);
            if (std::strcmp(logical->getLogicalName(), "IntrinsicCall") == 0) {
                hasIntrinsicCall = true;
                break;
            }
        }
    }
    EXPECT_TRUE(hasIntrinsicCall) << "IntrinsicCall should remain when expansion fails";
}

TEST_F(IntrinsicExpansionPassTest, MultipleIntrinsicCalls) {
    Function func("kernel");
    BasicBlock* bb = func.createBasicBlock("entry");

    // Add multiple IntrinsicCalls
    StinkyRegister v0(RegType::V, 0, 1);
    StinkyRegister v1(RegType::V, 1, 1);
    StinkyRegister v2(RegType::V, 2, 1);
    StinkyRegister v3(RegType::V, 3, 1);

    std::vector<StinkyRegister> args3a = {v0, v1}, args3b = {v2, v3};
    auto* ir1 = IRBase::createIR<IntrinsicCall>("ReluF32", args3a);
    bb->insertIR(bb->end(), ir1);
    auto* ir2 = IRBase::createIR<IntrinsicCall>("ReluF32", args3b);
    bb->insertIR(bb->end(), ir2);

    ASSERT_EQ(bb->size(), 2);

    // Run IntrinsicExpansionPass
    auto pass = createIntrinsicExpansionPass();
    pass->run(func, passCtx, am);

    // Should have 2 instructions (each ReluF32 expands to 1 v_max_f32)
    EXPECT_EQ(bb->size(), 2) << "Both IntrinsicCalls should be expanded to v_max_f32";

    // Verify no IntrinsicCalls remain
    size_t intrinsicCallCount = 0;
    for (auto& ir : *bb) {
        if (ir.getType() == IRBase::IRType::LogicalIR) {
            auto* logical = static_cast<LogicalInstruction*>(&ir);
            if (std::strcmp(logical->getLogicalName(), "IntrinsicCall") == 0) {
                intrinsicCallCount++;
            }
        }
    }
    EXPECT_EQ(intrinsicCallCount, 0) << "All IntrinsicCalls should be removed after expansion";
}
