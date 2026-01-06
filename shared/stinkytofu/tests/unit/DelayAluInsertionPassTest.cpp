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
#include "ir/asm/DelayAluInsertionPass.hpp"
#include "ir/asm/DefUseChain.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include "isa/ArchHelper.hpp"
#include "stinkytofu.hpp"
#include "support/Casting.hpp"

#include <gtest/gtest.h>

using namespace stinkytofu;

// Test fixture for DelayAluInsertionPass
class DelayAluInsertionPassTest : public ::testing::Test
{
protected:
    GemmTileConfig            config;
    std::unique_ptr<Function> func;
    BasicBlock*               bb;
    std::unique_ptr<Pass>     pass;
    GfxArchID                 arch;

    void SetUp() override
    {
        // Create function
        func = std::make_unique<Function>("test_delay_alu");
        bb   = func->createBasicBlock("entry");
        func->setEntryBlock(bb);

        // Create pass
        pass = createDelayAluInsertionPass();

        arch           = getGfxArchID(12, 5, 0); // gfx1250
        config.arch[0] = 12;
        config.arch[1] = 5;
        config.arch[2] = 0;
    }

    void TearDown() override
    {
        func.reset();
        bb = nullptr;
        pass.reset();
    }

    // Helper to create v_mul_f32 instruction
    // Uses automatic use-def chain maintenance (LLVM-style: methods on instruction)
    StinkyInstruction* createMulF32(int destReg, int src1Reg, int src2Reg)
    {
        auto     builder   = StinkyInstIRBuilder(bb->getIR(), arch);
        uint16_t isaOpcode = getMnemonicToIsaOpcode("v_mul_f32", arch);
        auto*    inst      = builder.createStinkyInstBefore(
            bb->getIR().end(), getMCIDByIsaOp(static_cast<IsaOpcode>(isaOpcode), arch));

        // Use instruction methods to automatically maintain use-def chains (LLVM-style)
        inst->setDestRegs({StinkyRegister("v", destReg, 1)});
        inst->setSrcRegs({StinkyRegister("v", src1Reg, 1), StinkyRegister("v", src2Reg, 1)});
        return inst;
    }

    // Helper to create v_add_f32 instruction
    // Uses automatic use-def chain maintenance (LLVM-style: methods on instruction)
    StinkyInstruction* createAddF32(int destReg, int src1Reg, int src2Reg)
    {
        auto     builder   = StinkyInstIRBuilder(bb->getIR(), arch);
        uint16_t isaOpcode = getMnemonicToIsaOpcode("v_add_f32", arch);
        auto*    inst      = builder.createStinkyInstBefore(
            bb->getIR().end(), getMCIDByIsaOp(static_cast<IsaOpcode>(isaOpcode), arch));

        // Use instruction methods to automatically maintain use-def chains (LLVM-style)
        inst->setDestRegs({StinkyRegister("v", destReg, 1)});
        inst->setSrcRegs({StinkyRegister("v", src1Reg, 1), StinkyRegister("v", src2Reg, 1)});
        return inst;
    }

    // Helper to count s_delay_alu instructions
    int countDelayAluInstructions()
    {
        int count = 0;
        for(IRBase& irNode : bb->getIR())
        {
            if(irNode.getType() == IRBase::IRType::StinkyTofu)
            {
                auto* inst = cast<StinkyInstruction>(&irNode);
                if(inst->getHwInstDesc() && inst->getHwInstDesc()->mnemonic
                   && std::string(inst->getHwInstDesc()->mnemonic) == "s_delay_alu")
                {
                    count++;
                }
            }
        }
        return count;
    }

    // Helper to count total instructions
    int countInstructions()
    {
        int count = 0;
        for(IRBase& irNode : bb->getIR())
        {
            if(irNode.getType() == IRBase::IRType::StinkyTofu)
            {
                count++;
            }
        }
        return count;
    }
};

// Test 1: Pass should trigger UNREACHABLE on CDNA architectures (gfx942) without HasSchedMode
TEST_F(DelayAluInsertionPassTest, SkipsOnCDNA3)
{
    // Create simple VALU dependency: v0 = mul, v1 = add(v0, ...)
    createMulF32(0, 1, 2);
    createAddF32(3, 0, 2);

    // Run pass with CDNA3 arch (gfx942) - should trigger UNREACHABLE
    PassContext ctx;
    config.arch[0] = 9;
    config.arch[1] = 4;
    config.arch[2] = 2;
    arch           = getGfxArchID(9, 4, 2);
    ctx.setGemmTileConfig(config);

    // Expect death with the specific error message
    EXPECT_DEATH(pass->run(*func, ctx), "Expert Schedule mode not supported for this architecture");
}

// Test 2: Pass should insert delay on gfx1250 (CDNA5, modified from RDNA4) for VALU dependency
TEST_F(DelayAluInsertionPassTest, InsertsDelayOnGfx1250_VALU)
{
    // Create VALU dependency: v0 = mul, v1 = add(v0, ...)
    createMulF32(0, 1, 2);
    createAddF32(3, 0, 2);

    // Run pass with gfx1250 (CDNA5, has HasSchedMode like RDNA4)
    PassContext ctx;
    config.arch[0] = 12;
    config.arch[1] = 5;
    config.arch[2] = 0;
    arch           = getGfxArchID(12, 5, 0);
    ctx.setGemmTileConfig(config);

    pass->run(*func, ctx);

    // Should insert s_delay_alu between mul and add
    EXPECT_EQ(countDelayAluInstructions(), 1);
    EXPECT_EQ(countInstructions(), 3); // mul + s_delay_alu + add
}

// Test 3: No delay insertion when registers don't have dependency
TEST_F(DelayAluInsertionPassTest, NoDelayWhenNoDependency)
{
    // Create independent instructions: v0 = mul, v3 = add(v1, v2) - no dependency
    createMulF32(0, 1, 2);
    createAddF32(3, 1, 2); // Uses v1, v2, not v0

    // Run pass with gfx1250
    PassContext ctx;
    config.arch[0] = 12;
    config.arch[1] = 5;
    config.arch[2] = 0;
    arch           = getGfxArchID(12, 5, 0);
    ctx.setGemmTileConfig(config);

    pass->run(*func, ctx);

    // Should NOT insert s_delay_alu (no dependency)
    EXPECT_EQ(countDelayAluInstructions(), 0);
    EXPECT_EQ(countInstructions(), 2);
}

// Test 4: Multiple dependencies insert multiple delays
TEST_F(DelayAluInsertionPassTest, MultipleDelays)
{
    // v0 = mul(v1, v2)
    // v3 = add(v0, v2)  <- depends on v0
    // v4 = add(v3, v1)  <- depends on v3
    createMulF32(0, 1, 2);
    createAddF32(3, 0, 2);
    createAddF32(4, 3, 1);

    // Run pass with gfx1250
    PassContext ctx;
    config.arch[0] = 12;
    config.arch[1] = 5;
    config.arch[2] = 0;
    arch           = getGfxArchID(12, 5, 0);
    ctx.setGemmTileConfig(config);

    pass->run(*func, ctx);

    // Should insert 2 s_delay_alu instructions
    EXPECT_EQ(countDelayAluInstructions(), 2);
    EXPECT_EQ(countInstructions(), 5); // 3 insts + 2 delays
}

// Test 5: Empty function should not crash
TEST_F(DelayAluInsertionPassTest, EmptyFunction)
{
    PassContext ctx;
    config.arch[0] = 12;
    config.arch[1] = 5;
    config.arch[2] = 0;
    arch           = getGfxArchID(12, 5, 0);
    ctx.setGemmTileConfig(config);

    // Should not crash on empty function
    pass->run(*func, ctx);

    EXPECT_EQ(countInstructions(), 0);
}

// Test 6: Single instruction should not insert delay
TEST_F(DelayAluInsertionPassTest, SingleInstruction)
{
    createMulF32(0, 1, 2);

    PassContext ctx;
    config.arch[0] = 12;
    config.arch[1] = 5;
    config.arch[2] = 0;
    arch           = getGfxArchID(12, 5, 0);
    ctx.setGemmTileConfig(config);

    pass->run(*func, ctx);

    // No delays needed for single instruction
    EXPECT_EQ(countDelayAluInstructions(), 0);
    EXPECT_EQ(countInstructions(), 1);
}
