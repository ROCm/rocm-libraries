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
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/StinkyRemoveWaitCntPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

class StinkyRemoveWaitCntPassTest : public ::testing::Test {
   protected:
    static constexpr GfxArchID kArch = GfxArchID::Gfx1250;

    void SetUp() override {
        func = std::make_unique<Function>("test");
        setFunctionArch(*func, kArch);
        bb = func->createBasicBlock("entry");
        registerAllAnalyses(am);
    }

    StinkyInstruction* addInst(GFX uop) {
        AsmIRBuilder builder(*bb, kArch);
        return builder.create(getMCIDByUOp(uop, kArch));
    }

    void runPass(bool removeTensor = true) {
        PassContext ctx;
        auto pass = createStinkyRemoveWaitCntPass(removeTensor);
        pass->run(*func, ctx, am);
    }

    int countInstructions() {
        int n = 0;
        for (auto& b : *func)
            for (auto& ir : b)
                if (ir.getType() == IRBase::IRType::StinkyTofu) n++;
        return n;
    }

    std::unique_ptr<Function> func;
    BasicBlock* bb = nullptr;
    AnalysisManager am;
};

TEST_F(StinkyRemoveWaitCntPassTest, RemovesStandardWaitCntInstruction) {
    addInst(GFX::s_wait_dscnt);
    ASSERT_EQ(countInstructions(), 1);
    runPass();
    EXPECT_EQ(countInstructions(), 0);
}

TEST_F(StinkyRemoveWaitCntPassTest, RemovesTensorWaitCntWhenEnabled) {
    addInst(GFX::s_wait_tensorcnt);
    ASSERT_EQ(countInstructions(), 1);
    runPass(/*removeTensor=*/true);
    EXPECT_EQ(countInstructions(), 0);
}

TEST_F(StinkyRemoveWaitCntPassTest, PreservesTensorWaitCntWhenDisabled) {
    addInst(GFX::s_wait_tensorcnt);
    ASSERT_EQ(countInstructions(), 1);
    runPass(/*removeTensor=*/false);
    EXPECT_EQ(countInstructions(), 1);
}

TEST_F(StinkyRemoveWaitCntPassTest, PreservesNonWaitInstructions) {
    addInst(GFX::s_nop);
    addInst(GFX::s_wait_dscnt);
    ASSERT_EQ(countInstructions(), 2);
    runPass();
    EXPECT_EQ(countInstructions(), 1);
    IRBase& ir = *bb->begin();
    auto* inst = dyn_cast<StinkyInstruction>(&ir);
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(inst->getHwInstDesc()->unifiedOpcode, GFX::s_nop);
}

TEST_F(StinkyRemoveWaitCntPassTest, RemovesMultipleWaitCntsAcrossBlock) {
    addInst(GFX::s_wait_dscnt);
    addInst(GFX::s_nop);
    addInst(GFX::s_wait_tensorcnt);
    ASSERT_EQ(countInstructions(), 3);
    runPass(/*removeTensor=*/true);
    EXPECT_EQ(countInstructions(), 1);
}

TEST_F(StinkyRemoveWaitCntPassTest, EmptyFunctionIsNoOp) {
    ASSERT_EQ(countInstructions(), 0);
    runPass();
    EXPECT_EQ(countInstructions(), 0);
}
