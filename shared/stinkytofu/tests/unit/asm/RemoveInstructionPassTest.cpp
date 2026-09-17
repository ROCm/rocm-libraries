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
#include "stinkytofu/transforms/asm/RemoveInstructionPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

class RemoveInstructionPassTest : public ::testing::Test {
   protected:
    static constexpr GfxArchID kArch = GfxArchID::Gfx1250;

    void SetUp() override {
        func = std::make_unique<Function>("test");
        setFunctionArch(*func, kArch);
        bb = func->createBasicBlock("entry");
        registerAllAnalyses(am);
    }

    StinkyInstruction* addVAdd() {
        return createVAddInBlock(bb, kArch, /*dest=*/0, /*src0=*/1, /*src1=*/2);
    }

    StinkyInstruction* addTensorLoad() {
        return createTensorLoadInBlock(bb, kArch, /*src0=*/0, /*src1=*/10, /*memTokens=*/{1});
    }

    StinkyInstruction* addDsLoad() {
        return createDsReadB128InBlock(bb, kArch, /*dest=*/8, /*addr=*/32);
    }

    StinkyInstruction* addWmma() {
        AsmIRBuilder builder(*bb, kArch);
        StinkyInstruction* inst =
            builder.create(getMCIDByUOp(GFX::v_wmma_f32_16x16x16_bf16, kArch));
        inst->addDestReg(StinkyRegister("v", 0, 8));
        inst->addSrcReg(StinkyRegister("v", 8, 8));
        inst->addSrcReg(StinkyRegister("v", 8, 8));
        inst->addSrcReg(StinkyRegister("v", 0, 8));
        return inst;
    }

    void runPass(const std::vector<UnifiedOpcode>& opcodes) {
        auto pass = createRemoveInstructionPass(opcodes);
        ASSERT_NE(pass, nullptr);
        PassContext ctx;
        ctx.setGemmTileConfig(func->getGemmTileConfig());
        pass->run(*func, ctx, am);
    }

    void runPassMnemonics(const std::vector<std::string>& mnemonics) {
        auto pass = createRemoveInstructionPass(mnemonics);
        ASSERT_NE(pass, nullptr);
        PassContext ctx;
        ctx.setGemmTileConfig(func->getGemmTileConfig());
        pass->run(*func, ctx, am);
    }

    int countOpcode(GFX opcode) {
        int n = 0;
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            auto* inst = cast<StinkyInstruction>(&ir);
            if (inst->getUnifiedOpcode() == opcode) ++n;
        }
        return n;
    }

    std::unique_ptr<Function> func;
    BasicBlock* bb = nullptr;
    AnalysisManager am;
};

TEST_F(RemoveInstructionPassTest, RemovesTensorLoadOnly) {
    addTensorLoad();
    addVAdd();
    addDsLoad();
    addWmma();
    ASSERT_EQ(countOpcode(GFX::tensor_load_to_lds), 1);

    runPass({GFX::tensor_load_to_lds});

    EXPECT_EQ(countOpcode(GFX::tensor_load_to_lds), 0);
    EXPECT_EQ(countOpcode(GFX::v_add_f32), 1);
    EXPECT_EQ(countOpcode(GFX::ds_load_b128), 1);
    EXPECT_EQ(countOpcode(GFX::v_wmma_f32_16x16x16_bf16), 1);
}

TEST_F(RemoveInstructionPassTest, RemovesDsLoadOnly) {
    addTensorLoad();
    addVAdd();
    addDsLoad();
    addWmma();
    ASSERT_EQ(countOpcode(GFX::ds_load_b128), 1);

    runPass({GFX::ds_load_b128});

    EXPECT_EQ(countOpcode(GFX::tensor_load_to_lds), 1);
    EXPECT_EQ(countOpcode(GFX::v_add_f32), 1);
    EXPECT_EQ(countOpcode(GFX::ds_load_b128), 0);
    EXPECT_EQ(countOpcode(GFX::v_wmma_f32_16x16x16_bf16), 1);
}

TEST_F(RemoveInstructionPassTest, RemovesWmmaOnly) {
    addTensorLoad();
    addVAdd();
    addDsLoad();
    addWmma();
    ASSERT_EQ(countOpcode(GFX::v_wmma_f32_16x16x16_bf16), 1);

    runPass({GFX::v_wmma_f32_16x16x16_bf16});

    EXPECT_EQ(countOpcode(GFX::tensor_load_to_lds), 1);
    EXPECT_EQ(countOpcode(GFX::v_add_f32), 1);
    EXPECT_EQ(countOpcode(GFX::ds_load_b128), 1);
    EXPECT_EQ(countOpcode(GFX::v_wmma_f32_16x16x16_bf16), 0);
}

TEST_F(RemoveInstructionPassTest, RemovesMultipleOpcodeTypes) {
    addTensorLoad();
    addVAdd();
    addDsLoad();
    addWmma();

    runPass({GFX::tensor_load_to_lds, GFX::ds_load_b128, GFX::v_wmma_f32_16x16x16_bf16});

    EXPECT_EQ(countOpcode(GFX::tensor_load_to_lds), 0);
    EXPECT_EQ(countOpcode(GFX::ds_load_b128), 0);
    EXPECT_EQ(countOpcode(GFX::v_wmma_f32_16x16x16_bf16), 0);
    EXPECT_EQ(countOpcode(GFX::v_add_f32), 1);
}

TEST_F(RemoveInstructionPassTest, RemovesViaMnemonicList) {
    addTensorLoad();
    addVAdd();
    addDsLoad();
    addWmma();

    runPassMnemonics({"tensor_load_to_lds", "ds_load_b128", "v_wmma_f32_16x16x16_bf16"});

    EXPECT_EQ(countOpcode(GFX::tensor_load_to_lds), 0);
    EXPECT_EQ(countOpcode(GFX::ds_load_b128), 0);
    EXPECT_EQ(countOpcode(GFX::v_wmma_f32_16x16x16_bf16), 0);
    EXPECT_EQ(countOpcode(GFX::v_add_f32), 1);
}

TEST_F(RemoveInstructionPassTest, RemovesViaMnemonicCsv) {
    addTensorLoad();
    addDsLoad();
    addWmma();

    auto pass =
        createRemoveInstructionPass("tensor_load_to_lds,ds_load_b128,v_wmma_f32_16x16x16_bf16");
    ASSERT_NE(pass, nullptr);
    PassContext ctx;
    ctx.setGemmTileConfig(func->getGemmTileConfig());
    pass->run(*func, ctx, am);

    EXPECT_EQ(countOpcode(GFX::tensor_load_to_lds), 0);
    EXPECT_EQ(countOpcode(GFX::ds_load_b128), 0);
    EXPECT_EQ(countOpcode(GFX::v_wmma_f32_16x16x16_bf16), 0);
}

TEST_F(RemoveInstructionPassTest, EmptyFactoryReturnsNull) {
    EXPECT_EQ(createRemoveInstructionPass(), nullptr);
    EXPECT_EQ(createRemoveInstructionPass(std::vector<UnifiedOpcode>{}), nullptr);
    EXPECT_EQ(createRemoveInstructionPass(std::vector<std::string>{}), nullptr);
    EXPECT_EQ(createRemoveInstructionPass(""), nullptr);
}
