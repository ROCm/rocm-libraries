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
#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/transforms/asm/SetMatrixReusePass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

class SetMatrixReusePassTest : public ::testing::Test {
   protected:
    GfxArchID arch = GfxArchID::Gfx1250;
    std::unique_ptr<Function> func;
    BasicBlock* bb = nullptr;
    std::unique_ptr<Pass> pass;
    AnalysisManager am;

    void SetUp() override {
        func = std::make_unique<Function>("set_matrix_reuse_test");
        setFunctionArch(*func, arch);
        bb = func->createBasicBlock("entry");
        pass = createSetMatrixReusePass();
        registerAllAnalyses(am);
    }

    static const MFMAModifiers* mfmaMod(const StinkyInstruction& inst) {
        return inst.getModifier<MFMAModifiers>();
    }

    StinkyInstruction* createMatrixInst(GFX op, int destStart, int src0Start, int src1Start) {
        AsmIRBuilder builder(*bb, arch);
        const HwInstDesc* desc = getMCIDByUOp(op, arch);
        if (!desc) return nullptr;
        StinkyInstruction* inst = builder.create(desc);
        inst->addDestReg(StinkyRegister("a", destStart, 8));
        inst->addSrcReg(StinkyRegister("v", src0Start, 8));
        inst->addSrcReg(StinkyRegister("v", src1Start, 8));
        inst->addSrcReg(StinkyRegister("a", destStart, 8));
        MFMAModifiers mod;
        mod.reuseA = true;
        mod.reuseB = true;
        inst->addModifier<MFMAModifiers>(mod);
        return inst;
    }

    StinkyInstruction* createWMMA(int destStart, int src0Start, int src1Start) {
        return createMatrixInst(GFX::v_wmma_f32_16x16x32_bf16, destStart, src0Start, src1Start);
    }

    void runPass() {
        PassContext ctx;
        pass->run(*func, ctx, am);
    }
};

TEST_F(SetMatrixReusePassTest, SameBOperand_SetsReuseBOnFirst) {
    const HwInstDesc* wmmaDesc = getMCIDByUOp(GFX::v_wmma_f32_16x16x32_bf16, arch);
    if (!wmmaDesc) GTEST_SKIP() << "v_wmma_f32_16x16x32_bf16 unavailable";

    StinkyInstruction* w0 = createWMMA(/*dest=*/0, /*src0=*/8, /*src1=*/16);
    StinkyInstruction* w1 = createWMMA(/*dest=*/32, /*src0=*/24, /*src1=*/16);
    ASSERT_NE(w0, nullptr);
    ASSERT_NE(w1, nullptr);

    runPass();

    ASSERT_NE(mfmaMod(*w0), nullptr);
    EXPECT_FALSE(mfmaMod(*w0)->reuseA);
    EXPECT_TRUE(mfmaMod(*w0)->reuseB);
    ASSERT_NE(mfmaMod(*w1), nullptr);
    EXPECT_FALSE(mfmaMod(*w1)->reuseA);
    EXPECT_FALSE(mfmaMod(*w1)->reuseB);
}

TEST_F(SetMatrixReusePassTest, NonMatrixBetweenWmmas_StillChainsReuse) {
    const HwInstDesc* wmmaDesc = getMCIDByUOp(GFX::v_wmma_f32_16x16x32_bf16, arch);
    if (!wmmaDesc) GTEST_SKIP() << "v_wmma_f32_16x16x32_bf16 unavailable";

    StinkyInstruction* w0 = createWMMA(/*dest=*/0, /*src0=*/8, /*src1=*/16);
    createDsReadB128InBlock(bb, arch, /*destReg=*/64, /*addrReg=*/72);
    StinkyInstruction* w1 = createWMMA(/*dest=*/32, /*src0=*/8, /*src1=*/16);
    ASSERT_NE(w0, nullptr);
    ASSERT_NE(w1, nullptr);

    runPass();

    ASSERT_NE(mfmaMod(*w0), nullptr);
    EXPECT_TRUE(mfmaMod(*w0)->reuseA);
    EXPECT_TRUE(mfmaMod(*w0)->reuseB);
}

TEST_F(SetMatrixReusePassTest, ClearsStaleReuseOnLastMma) {
    const HwInstDesc* wmmaDesc = getMCIDByUOp(GFX::v_wmma_f32_16x16x32_bf16, arch);
    if (!wmmaDesc) GTEST_SKIP() << "v_wmma_f32_16x16x32_bf16 unavailable";

    StinkyInstruction* w0 = createWMMA(/*dest=*/0, /*src0=*/8, /*src1=*/16);
    ASSERT_NE(w0, nullptr);

    runPass();

    ASSERT_NE(mfmaMod(*w0), nullptr);
    EXPECT_FALSE(mfmaMod(*w0)->reuseA);
    EXPECT_FALSE(mfmaMod(*w0)->reuseB);
}

TEST_F(SetMatrixReusePassTest, SkipsReuseOnF8f6f4Wmma) {
    if (!getMCIDByUOp(GFX::v_wmma_f32_16x16x128_f8f6f4, arch)) {
        GTEST_SKIP() << "v_wmma_f32_16x16x128_f8f6f4 unavailable";
    }

    StinkyInstruction* w0 =
        createMatrixInst(GFX::v_wmma_f32_16x16x128_f8f6f4, /*dest=*/0, /*src0=*/8, /*src1=*/16);
    StinkyInstruction* w1 =
        createMatrixInst(GFX::v_wmma_f32_16x16x128_f8f6f4, /*dest=*/32, /*src0=*/24, /*src1=*/16);
    ASSERT_NE(w0, nullptr);
    ASSERT_NE(w1, nullptr);

    runPass();

    ASSERT_NE(mfmaMod(*w0), nullptr);
    EXPECT_FALSE(mfmaMod(*w0)->reuseA);
    EXPECT_FALSE(mfmaMod(*w0)->reuseB);
}

TEST_F(SetMatrixReusePassTest, SkipsReuseOnMxF4ScaleWmma) {
    if (!getMCIDByUOp(GFX::v_wmma_scale_f32_32x16x128_f4, arch)) {
        GTEST_SKIP() << "v_wmma_scale_f32_32x16x128_f4 unavailable";
    }

    StinkyInstruction* w0 =
        createMatrixInst(GFX::v_wmma_scale_f32_32x16x128_f4, /*dest=*/0, /*src0=*/8, /*src1=*/16);
    StinkyInstruction* w1 =
        createMatrixInst(GFX::v_wmma_scale_f32_32x16x128_f4, /*dest=*/32, /*src0=*/24, /*src1=*/16);
    ASSERT_NE(w0, nullptr);
    ASSERT_NE(w1, nullptr);

    runPass();

    ASSERT_NE(mfmaMod(*w0), nullptr);
    EXPECT_FALSE(mfmaMod(*w0)->reuseA);
    EXPECT_FALSE(mfmaMod(*w0)->reuseB);
}

TEST_F(SetMatrixReusePassTest, AllowsReuseOnMxF8f6f4ScaleWmma) {
    if (!getMCIDByUOp(GFX::v_wmma_scale_f32_16x16x128_f8f6f4, arch)) {
        GTEST_SKIP() << "v_wmma_scale_f32_16x16x128_f8f6f4 unavailable";
    }

    StinkyInstruction* w0 =
        createMatrixInst(GFX::v_wmma_scale_f32_16x16x128_f8f6f4, /*dest=*/0, /*src0=*/8,
                         /*src1=*/16);
    StinkyInstruction* w1 =
        createMatrixInst(GFX::v_wmma_scale_f32_16x16x128_f8f6f4, /*dest=*/32, /*src0=*/24,
                         /*src1=*/16);
    ASSERT_NE(w0, nullptr);
    ASSERT_NE(w1, nullptr);

    runPass();

    ASSERT_NE(mfmaMod(*w0), nullptr);
    EXPECT_FALSE(mfmaMod(*w0)->reuseA);
    EXPECT_TRUE(mfmaMod(*w0)->reuseB);
}

}  // namespace
