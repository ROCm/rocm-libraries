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

#include "AllocationTestUtils.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/transforms/asm/ra/RegisterBudget.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

class RegisterBudgetTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kRaTestArch);
        entry = func->createBasicBlock("entry");
    }

    /// sX = s_mov_b32(sY), with widths so tuple spans can be exercised.
    void mov(uint32_t dst, uint32_t src, uint16_t dstWidth = 1, uint16_t srcWidth = 1) {
        AsmIRBuilder builder(*entry, kRaTestArch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_mov_b32, kRaTestArch));
        inst->addDestReg(StinkyRegister("s", dst, dstWidth));
        inst->addSrcReg(StinkyRegister("s", src, srcWidth));
    }

    std::unique_ptr<Function> func;
    BasicBlock* entry = nullptr;
};

}  // namespace

TEST_F(RegisterBudgetTest, CountsOnePastTheHighestIndexUsed) {
    mov(/*dst=*/7, /*src=*/3);
    EXPECT_EQ(highestRegisterCount(*func, RegType::S), 8u);
    // A class the function never names needs nothing declared.
    EXPECT_EQ(highestRegisterCount(*func, RegType::V), 0u);
}

TEST_F(RegisterBudgetTest, CountsTheWholeSpanOfATupleOperand) {
    // s[8:11] names only its base, so the count has to follow the width.
    mov(/*dst=*/8, /*src=*/0, /*dstWidth=*/4);
    EXPECT_EQ(highestRegisterCount(*func, RegType::S), 12u);
}

TEST_F(RegisterBudgetTest, SourcesCountAsWellAsDestinations) {
    mov(/*dst=*/1, /*src=*/40);
    EXPECT_EQ(highestRegisterCount(*func, RegType::S), 41u);
}

TEST_F(RegisterBudgetTest, UsageWinsWhenItExceedsWhatTheAbiFills) {
    mov(/*dst=*/60, /*src=*/0);
    // 4 preloaded + 2 for the kernarg pointer + 3 workgroup ids = 9, well under.
    EXPECT_EQ(requiredSgprCount(*func, /*numSgprPreload=*/4, {1, 1, 1}), 61u);
}

TEST_F(RegisterBudgetTest, NeverDeclaresFewerThanTheHardwareFills) {
    // The whole point of the floor: a kernel can compact its own registers far
    // below the preloaded arguments, which the dispatch still writes whether or
    // not any operand names them.
    mov(/*dst=*/1, /*src=*/0);
    EXPECT_EQ(highestRegisterCount(*func, RegType::S), 2u);
    EXPECT_EQ(requiredSgprCount(*func, /*numSgprPreload=*/27, {1, 1, 1}), 32u);
}

TEST_F(RegisterBudgetTest, EachEnabledWorkgroupIdCostsOneRegister) {
    mov(/*dst=*/1, /*src=*/0);
    EXPECT_EQ(requiredSgprCount(*func, /*numSgprPreload=*/27, {1, 0, 0}), 30u);
    EXPECT_EQ(requiredSgprCount(*func, /*numSgprPreload=*/27, {0, 0, 0}), 29u);
}

TEST_F(RegisterBudgetTest, NoPreloadMeansNoKernargPointerToAccountFor) {
    mov(/*dst=*/1, /*src=*/0);
    // numSgprPreload of 0 suppresses the .amdhsa_user_sgpr_count line entirely,
    // so there is no +2 to carry either.
    EXPECT_EQ(requiredSgprCount(*func, /*numSgprPreload=*/0, {1, 1, 1}), 3u);
}
