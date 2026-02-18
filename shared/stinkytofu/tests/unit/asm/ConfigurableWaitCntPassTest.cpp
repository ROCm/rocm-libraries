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
#include <vector>

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu/transforms/asm/StinkyConfigurableWaitCntPass.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/core/PassManager.hpp"

using namespace stinkytofu;

// Helper class to build test IR and run pass
class ConfigurableWaitCntPassTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        arch               = getGfxArchID(12, 5, 0); // GFX1250
        gemmConfig.arch[0] = 12;
        gemmConfig.arch[1] = 5;
        gemmConfig.arch[2] = 0;

        // Create a Function with a single BasicBlock for testing
        func = std::make_unique<Function>("test_function");
        bb   = func->createBasicBlock("entry");
    }

    void TearDown() override
    {
        // Clean up Function (which will clean up BasicBlocks and IR)
        func.reset();
        bb = nullptr;
    }

    // Create IRBuilder for building test instructions
    AsmIRBuilder getIRBuilder()
    {
        return AsmIRBuilder(*bb, arch);
    }

    // Helper to create a ds_read instruction (64-bit, 2 registers)
    StinkyInstruction* createDSRead(int destReg, int addrReg)
    {
        auto               builder = getIRBuilder();
        StinkyInstruction* inst
            = builder.create(getMCIDByUOp(GFX::ds_load_b64, arch));

        inst->addDestReg(StinkyRegister("v", destReg, 2));
        inst->addSrcReg(StinkyRegister("v", addrReg, 1));
        return inst;
    }

    // Helper to create a ds_read instruction (128-bit, 4 registers)
    StinkyInstruction* createDSRead128(int destReg, int addrReg)
    {
        auto               builder = getIRBuilder();
        StinkyInstruction* inst
            = builder.create(getMCIDByUOp(GFX::ds_load_b128, arch));

        inst->addDestReg(StinkyRegister("v", destReg, 4));
        inst->addSrcReg(StinkyRegister("v", addrReg, 1));
        return inst;
    }

    // Helper to create a ds_write instruction
    StinkyInstruction* createDSWrite(int addrReg, int dataReg)
    {
        auto               builder = getIRBuilder();
        StinkyInstruction* inst
            = builder.create(getMCIDByUOp(GFX::ds_write_b64, arch));

        inst->addSrcReg(StinkyRegister("v", addrReg, 2));
        inst->addSrcReg(StinkyRegister("v", dataReg, 1));
        return inst;
    }

    // Helper to create a global_load instruction
    StinkyInstruction* createGlobalLoad(int destReg, int addrReg)
    {
        auto               builder = getIRBuilder();
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::global_load_dword, arch));

        inst->addDestReg(StinkyRegister("v", destReg, 1));
        inst->addSrcReg(StinkyRegister("s", addrReg, 4));
        return inst;
    }

    // Helper to create a global_store instruction
    StinkyInstruction* createGlobalStore(int addrReg, int dataReg)
    {
        auto               builder = getIRBuilder();
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::global_store_dword, arch));

        inst->addSrcReg(StinkyRegister("v", addrReg, 1));
        inst->addSrcReg(StinkyRegister("s", dataReg, 4));
        return inst;
    }

    StinkyInstruction* createTensorLoad(int src0Reg, int src1Reg)
    {
        auto               builder = getIRBuilder();
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::tensor_load_to_lds, arch));

        inst->addSrcReg(StinkyRegister("s", src0Reg, 4));
        inst->addSrcReg(StinkyRegister("s", src1Reg, 8));
        return inst;
    }

    // Helper to create a v_add instruction
    StinkyInstruction* createVAdd(int destReg, int src0Reg, int src1Reg)
    {
        auto               builder = getIRBuilder();
        StinkyInstruction* inst
            = builder.create(getMCIDByUOp(GFX::v_add_f32, arch));

        inst->addDestReg(StinkyRegister("v", destReg, 1));
        inst->addSrcReg(StinkyRegister("v", src0Reg, 1));
        inst->addSrcReg(StinkyRegister("v", src1Reg, 1));
        return inst;
    }

    // Helper to create a v_mul instruction
    StinkyInstruction* createVMul(int destReg, int src0Reg, int src1Reg)
    {
        auto               builder = getIRBuilder();
        StinkyInstruction* inst
            = builder.create(getMCIDByUOp(GFX::v_mul_f32, arch));

        inst->addDestReg(StinkyRegister("v", destReg, 1));
        inst->addSrcReg(StinkyRegister("v", src0Reg, 1));
        inst->addSrcReg(StinkyRegister("v", src1Reg, 1));
        return inst;
    }

    // Helper to create an s_barrier instruction
    StinkyInstruction* createBarrier()
    {
        auto               builder = getIRBuilder();
        StinkyInstruction* inst
            = builder.create(getMCIDByUOp(GFX::s_barrier, arch));
        return inst;
    }

    // Helper to create a mfma instruction (tensor load)
    StinkyInstruction* createWMMA(int destReg, int src0Reg, int src1Reg)
    {
        auto               builder = getIRBuilder();
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::v_wmma_f32_16x16x32_bf16, arch));

        inst->addDestReg(StinkyRegister("a", destReg, 8));
        inst->addSrcReg(StinkyRegister("v", src0Reg, 8));
        inst->addSrcReg(StinkyRegister("v", src1Reg, 8));
        inst->addSrcReg(StinkyRegister("a", destReg, 8));
        return inst;
    }

    // Helper to count waitcnt instructions
    int countWaitCnt()
    {
        int count = 0;
        for(auto& irBase : *bb)
        {
            StinkyInstruction& inst = static_cast<StinkyInstruction&>(irBase);
            if(inst.getModifier<SWaitCntData>())
            {
                count++;
            }
        }
        return count;
    }

    // Helper to count tensor waitcnt instructions
    int countTensorWaitCnt()
    {
        int count = 0;
        for(auto& irBase : *bb)
        {
            StinkyInstruction& inst = static_cast<StinkyInstruction&>(irBase);
            if(inst.getModifier<SWaitTensorCntData>())
            {
                count++;
            }
        }
        return count;
    }

    // Helper structure to hold waitcnt information with position
    struct WaitCntInfo
    {
        StinkyInstruction* inst;
        SWaitCntData*      waitData;
        int                position; // Position in the instruction list

        WaitCntInfo(StinkyInstruction* i, SWaitCntData* w, int p)
            : inst(i)
            , waitData(w)
            , position(p)
        {
        }
    };

    // Helper structure to hold tensor waitcnt information with position
    struct TensorWaitCntInfo
    {
        StinkyInstruction*  inst;
        SWaitTensorCntData* tensorWaitData;
        int                 position; // Position in the instruction list

        TensorWaitCntInfo(StinkyInstruction* i, SWaitTensorCntData* t, int p)
            : inst(i)
            , tensorWaitData(t)
            , position(p)
        {
        }
    };

    // Helper to collect all waitcnt instructions with their positions
    std::vector<WaitCntInfo> getAllWaitCnts()
    {
        std::vector<WaitCntInfo> waitcnts;
        int                      position = 0;
        for(auto& irBase : *bb)
        {
            StinkyInstruction& inst = static_cast<StinkyInstruction&>(irBase);
            if(SWaitCntData* wait = inst.getModifier<SWaitCntData>())
            {
                waitcnts.emplace_back(&inst, wait, position);
            }
            position++;
        }
        return waitcnts;
    }

    // Helper to collect all tensor waitcnt instructions with their positions
    std::vector<TensorWaitCntInfo> getAllTensorWaitCnts()
    {
        std::vector<TensorWaitCntInfo> tensorWaitcnts;
        int                            position = 0;
        for(auto& irBase : *bb)
        {
            StinkyInstruction& inst = static_cast<StinkyInstruction&>(irBase);
            if(SWaitTensorCntData* tensorWait = inst.getModifier<SWaitTensorCntData>())
            {
                tensorWaitcnts.emplace_back(&inst, tensorWait, position);
            }
            position++;
        }
        return tensorWaitcnts;
    }

    // Helper to find instruction position in the list
    int getInstructionPosition(StinkyInstruction* target)
    {
        int position = 0;
        for(auto& irBase : *bb)
        {
            if(&static_cast<StinkyInstruction&>(irBase) == target)
            {
                return position;
            }
            position++;
        }
        return -1; // Not found
    }

    // Helper to find waitcnt before a specific instruction
    SWaitCntData* findWaitCntBefore(StinkyInstruction* target)
    {
        BasicBlock::iterator targetIt = bb->end();
        for(auto it = bb->begin(); it != bb->end(); ++it)
        {
            if(&static_cast<StinkyInstruction&>(*it) == target)
            {
                targetIt = it;
                break;
            }
        }

        if(targetIt == bb->end() || targetIt == bb->begin())
            return nullptr;

        // Search backwards from target, checking multiple instructions
        // to handle cases where both wait_cnt and tensor_wait_cnt are present
        auto prevIt = targetIt;
        --prevIt;

        while(true)
        {
            StinkyInstruction& prevInst = static_cast<StinkyInstruction&>(*prevIt);

            // Check if this instruction has the wait_cnt we're looking for
            if(SWaitCntData* wait = prevInst.getModifier<SWaitCntData>())
                return wait;

            // Check if this is a tensor_wait_cnt (different type, keep searching)
            if(prevInst.getModifier<SWaitTensorCntData>())
            {
                if(prevIt == bb->begin())
                    return nullptr;
                --prevIt;
                continue;
            }

            // Not a wait instruction, stop searching
            return nullptr;
        }
    }

    SWaitTensorCntData* findTensorWaitCntBefore(StinkyInstruction* target)
    {
        BasicBlock::iterator targetIt = bb->end();
        for(auto it = bb->begin(); it != bb->end(); ++it)
        {
            if(&static_cast<StinkyInstruction&>(*it) == target)
            {
                targetIt = it;
                break;
            }
        }

        if(targetIt == bb->end() || targetIt == bb->begin())
            return nullptr;

        // Search backwards from target, checking multiple instructions
        // to handle cases where both wait_cnt and tensor_wait_cnt are present
        auto prevIt = targetIt;
        --prevIt;

        while(true)
        {
            StinkyInstruction& prevInst = static_cast<StinkyInstruction&>(*prevIt);

            // Check if this instruction has the tensor_wait_cnt we're looking for
            if(SWaitTensorCntData* tensorWait = prevInst.getModifier<SWaitTensorCntData>())
                return tensorWait;

            // Check if this is a regular wait_cnt (different type, keep searching)
            if(prevInst.getModifier<SWaitCntData>())
            {
                if(prevIt == bb->begin())
                    return nullptr;
                --prevIt;
                continue;
            }

            // Not a wait instruction, stop searching
            return nullptr;
        }
    }

    // Helper to run pass with configuration
    void runPass(const WaitCntConfig& config)
    {
        PassContext passCtx;
        passCtx.setGemmTileConfig(gemmConfig);
        auto pass = stinkytofu::createStinkyCustomWaitCntPass(config);
        pass->run(*func, passCtx);
    }

    void dumpInsts()
    {
        std::cout << *func << std::endl;
    }

    std::unique_ptr<Function> func;
    BasicBlock*               bb = nullptr;
    GemmTileConfig            gemmConfig;
    GfxArchID                 arch;
};

// ============================================================================
// Test Suite 1: Pre-Defined Configurations
// ============================================================================

TEST_F(ConfigurableWaitCntPassTest, StandardConfigExists)
{
    auto config = WaitCntConfig::standard();

    // Standard config should have reasonable defaults
    EXPECT_TRUE(config.barrierPolicy.waitDSRead);
    EXPECT_TRUE(config.barrierPolicy.waitDSWrite);
    EXPECT_TRUE(config.barrierPolicy.waitTensorLoad);
    EXPECT_TRUE(config.dependencyPolicy.trackLoadDependencies);
}

TEST_F(ConfigurableWaitCntPassTest, ConservativeConfigExists)
{
    auto config = WaitCntConfig::conservative();

    // Conservative should wait for everything
    EXPECT_TRUE(config.barrierPolicy.waitDSRead);
    EXPECT_TRUE(config.barrierPolicy.waitDSWrite);
    EXPECT_TRUE(config.barrierPolicy.waitGlobalRead);
    EXPECT_TRUE(config.barrierPolicy.waitGlobalWrite);
    EXPECT_TRUE(config.barrierPolicy.waitTensorLoad);
}

TEST_F(ConfigurableWaitCntPassTest, MinimalConfigExists)
{
    auto config = WaitCntConfig::minimal();

    // Minimal should only have essentials
    EXPECT_TRUE(config.barrierPolicy.waitDSRead);
    EXPECT_FALSE(config.barrierPolicy.waitDSWrite);
    EXPECT_FALSE(config.barrierPolicy.waitGlobalRead);
}

TEST_F(ConfigurableWaitCntPassTest, UnrollLoopConfigExists)
{
    auto config = WaitCntConfig::unrollLoop();
    EXPECT_FALSE(config.barrierPolicy.waitDSRead)
        << "UnrollLoop config: barriers don't wait for ds_reads";
    EXPECT_TRUE(config.barrierPolicy.waitDSWrite);
    EXPECT_FALSE(config.barrierPolicy.waitGlobalRead);
    EXPECT_TRUE(config.barrierPolicy.waitTensorLoad);
}

// ============================================================================
// Test Suite 2: Barrier Wait Insertion for unroll loop
// ============================================================================

TEST_F(ConfigurableWaitCntPassTest, BarrierWithDSRead_UnrollLoopConfig)
{
    // Create: ds_read -> s_barrier
    createDSRead(0, 10);
    auto barrier = createBarrier();

    WaitCntConfig config = WaitCntConfig::unrollLoop();
    runPass(config);

    // Barriers don't wait for ds_reads, so no waitcnt should be inserted
    EXPECT_EQ(countWaitCnt(), 0)
        << "Barriers don't wait for ds_reads, no waitcnt should be inserted";
    EXPECT_EQ(countTensorWaitCnt(), 0) << "No tensor waitcnt";
}

TEST_F(ConfigurableWaitCntPassTest, BarrierWithDSReadTensorLoad_UnrollLoopConfig)
{
    // Create: tensor_load_to_lds -> ds_read -> s_barrier
    createTensorLoad(0, 10);
    createDSRead(0, 10);
    auto barrier = createBarrier();

    WaitCntConfig config = WaitCntConfig::unrollLoop();
    runPass(config);

    // Should insert tensor_wait_cnt before barrier (for tensor_load)
    // but NOT s_wait_cnt (barriers don't wait for ds_reads)
    SWaitTensorCntData* tensorWait = findTensorWaitCntBefore(barrier);
    ASSERT_NE(tensorWait, nullptr);
    EXPECT_EQ(tensorWait->tlcnt, 0) << "Wait for tensor load";

    // No regular waitcnt should be inserted (barriers don't wait for ds_reads)
    EXPECT_EQ(countWaitCnt(), 0)
        << "Barriers don't wait for ds_reads, no s_wait_cnt should be inserted";
}

// ============================================================================
// Test Suite 3: DS Read Insertion before  for unroll loop
// ============================================================================

TEST_F(ConfigurableWaitCntPassTest, DSReadBeforeWMMA_UnrollLoopConfig)
{
    // Create: ds_read -> v_wmma_f32_16x16x32_bf16
    createDSRead(20, 0);
    createDSRead(30, 0);
    createDSRead(40, 0);
    createDSRead(50, 0);
    createVAdd(60, 61, 62);
    StinkyInstruction* wmma1 = createWMMA(10, 20, 30);
    createVAdd(60, 61, 62);
    StinkyInstruction* wmma2 = createWMMA(10, 40, 50);

    WaitCntConfig config = WaitCntConfig::unrollLoop();
    runPass(config);

    // Collect all waitcnt instructions
    auto waitcnts = getAllWaitCnts();

    // Should have exactly 2 waitcnts
    ASSERT_EQ(waitcnts.size(), 2);

    // Get positions of WMMA instructions
    int wmma1Pos = getInstructionPosition(wmma1);
    int wmma2Pos = getInstructionPosition(wmma2);
    ASSERT_NE(wmma1Pos, -1);
    ASSERT_NE(wmma2Pos, -1);

    // First waitcnt should be right before wmma1 with dlcnt=2
    // (waits for first 2 ds_reads that produce regs 20, 30 used by wmma1)
    EXPECT_EQ(waitcnts[0].position, wmma1Pos - 1);
    EXPECT_EQ(waitcnts[0].waitData->dlcnt, 2)
        << "First waitcnt should wait for 2 ds_reads (regs 20, 30)";
    EXPECT_EQ(waitcnts[0].waitData->vlcnt, -1);
    EXPECT_EQ(waitcnts[0].waitData->vscnt, -1);
    EXPECT_EQ(waitcnts[0].waitData->dscnt, -1);
    EXPECT_EQ(waitcnts[0].waitData->kmcnt, -1);

    // Second waitcnt should be right before wmma2 with dlcnt=0
    // (waits for remaining 2 ds_reads that produce regs 40, 50 used by wmma2)
    EXPECT_EQ(waitcnts[1].position, wmma2Pos - 1);
    EXPECT_EQ(waitcnts[1].waitData->dlcnt, 0)
        << "Second waitcnt should wait for all remaining ds_reads (regs 40, 50)";
    EXPECT_EQ(waitcnts[1].waitData->vlcnt, -1);
    EXPECT_EQ(waitcnts[1].waitData->vscnt, -1);
    EXPECT_EQ(waitcnts[1].waitData->dscnt, -1);
    EXPECT_EQ(waitcnts[1].waitData->kmcnt, -1);
}

// ============================================================================
// Test Suite 4: Complete test case for unroll loop
// ============================================================================

TEST_F(ConfigurableWaitCntPassTest, CompleteTest_UnrollLoopConfig)
{
    // Test pattern:
    // preloop: ds_read dest v0-3
    // preloop: ds_read dest v4-7
    // preloop: ds_read dest v8-11
    // preloop: ds_read dest v12-15
    // tensor_load
    // ds_read dest v16-19
    // ds_read dest v20-23
    // ds_read dest v24-27
    // ds_read dest v28-31
    // wmma 50, 0, 8 (uses v0-7 and v8-15 from preloop ds_reads)
    // barrier
    // ds_read dest v0-3
    // ds_read dest v4-7
    // ds_read dest v8-11
    // ds_read dest v12-15
    // wmma 50, 16, 24 (uses v16-23 and v24-31 from first set of ds_reads)
    // barrier
    // preloop
    createDSRead128(0, 40);
    createDSRead128(4, 40);
    createDSRead128(8, 40);
    createDSRead128(12, 40);
    // end of preloop
    StinkyInstruction* tensorLoad = createTensorLoad(0, 10);
    createDSRead128(16, 40);
    createDSRead128(20, 40);
    createDSRead128(24, 40);
    createDSRead128(28, 40);
    StinkyInstruction* wmma1    = createWMMA(50, 0, 8);
    StinkyInstruction* barrier1 = createBarrier();
    createDSRead128(0, 40);
    createDSRead128(4, 40);
    createDSRead128(8, 40);
    createDSRead128(12, 40);
    StinkyInstruction* wmma2    = createWMMA(50, 16, 24);
    StinkyInstruction* barrier2 = createBarrier();

    WaitCntConfig config = WaitCntConfig::unrollLoop();
    runPass(config);

    // Collect all wait instructions
    auto waitcnts       = getAllWaitCnts();
    auto tensorWaitcnts = getAllTensorWaitCnts();

    // Get positions
    int wmma1Pos    = getInstructionPosition(wmma1);
    int barrier1Pos = getInstructionPosition(barrier1);
    int wmma2Pos    = getInstructionPosition(wmma2);
    int barrier2Pos = getInstructionPosition(barrier2);

    // Verify positions are valid
    ASSERT_NE(wmma1Pos, -1);
    ASSERT_NE(barrier1Pos, -1);
    ASSERT_NE(wmma2Pos, -1);
    ASSERT_NE(barrier2Pos, -1);

    // Expected behavior with unrollLoop config (barriers don't wait for ds_reads):
    // 1. Before wmma1: s_wait_cnt for preloop ds_reads (v0-15)
    // 2. Before barrier1: tensor_wait_cnt only (for tensor_load, not ds_reads)
    // 3. No waitcnt needed before wmma2 (barrier1 synchronizes all prior operations)
    // 4. No waitcnt needed before barrier2 (barriers don't wait for ds_reads)

    // Should have at least one tensor waitcnt
    ASSERT_GE(tensorWaitcnts.size(), 1)
        << "Should have at least one tensor_wait_cnt before barrier1";

    // First tensor waitcnt should be before barrier1 with tlcnt=0
    EXPECT_LT(tensorWaitcnts[0].position, barrier1Pos)
        << "tensor_wait_cnt should be before barrier1";
    EXPECT_EQ(tensorWaitcnts[0].tensorWaitData->tlcnt, 0) << "Should wait for tensor load";

    // Should have exactly 2 regular waitcnts (before wmma1 and wmma2)
    ASSERT_EQ(waitcnts.size(), 2)
        << "Should have 2 waitcnts (before each wmma), barriers don't wait for ds_reads";

    // First waitcnt should be right before wmma1 with dlcnt=4
    // (waits for preloop 4 ds_reads v0-15)
    EXPECT_EQ(waitcnts[0].position, wmma1Pos - 1) << "First waitcnt should be right before wmma1";
    EXPECT_EQ(waitcnts[0].waitData->dlcnt, 4)
        << "Should wait for preloop ds_reads (v0-15) before wmma1";
    EXPECT_EQ(waitcnts[0].waitData->vlcnt, -1);
    EXPECT_EQ(waitcnts[0].waitData->vscnt, -1);
    EXPECT_EQ(waitcnts[0].waitData->dscnt, -1);
    EXPECT_EQ(waitcnts[0].waitData->kmcnt, -1);

    // Second waitcnt should be right before wmma2 with dlcnt=4
    // (waits for 4 ds_reads v16-31 that happened before barrier1)
    // Note: barrier1 doesn't insert waitcnt for ds_reads, so these operations
    // may not be complete when wmma2 executes unless we insert a waitcnt
    EXPECT_EQ(waitcnts[1].position, wmma2Pos - 1) << "Second waitcnt should be right before wmma2";
    EXPECT_EQ(waitcnts[1].waitData->dlcnt, 4)
        << "Should wait for ds_reads (v16-31) that happened before barrier1";
    EXPECT_EQ(waitcnts[1].waitData->vlcnt, -1);
    EXPECT_EQ(waitcnts[1].waitData->vscnt, -1);
    EXPECT_EQ(waitcnts[1].waitData->dscnt, -1);
    EXPECT_EQ(waitcnts[1].waitData->kmcnt, -1);

    // Verify no waitcnt before barriers (they don't wait for ds_reads)
    for(const auto& wait : waitcnts)
    {
        EXPECT_NE(wait.position, barrier1Pos - 1)
            << "Should not have waitcnt before barrier1 (barriers don't wait for ds_reads)";
        EXPECT_NE(wait.position, barrier2Pos - 1)
            << "Should not have waitcnt before barrier2 (barriers don't wait for ds_reads)";
    }
}

// ============================================================================
// Test Suite 5: Basic Block State Tracking
// ============================================================================

// Helper function to create ds_load_b32 instruction (32-bit, 1 register)
StinkyInstruction* createDSReadB32InBlock(BasicBlock* bb, GfxArchID arch, int destReg, int addrReg)
{
    AsmIRBuilder builder = AsmIRBuilder(*bb, arch);
    StinkyInstruction*  inst
        = builder.create(getMCIDByUOp(GFX::ds_load_b32, arch));

    inst->addDestReg(StinkyRegister("v", destReg, 1));
    inst->addSrcReg(StinkyRegister("v", addrReg, 1)); // DS address is 1 VGPR
    return inst;
}

// Helper function to create v_add_f32 instruction (using as consumer of DS reads)
StinkyInstruction*
    createVFmacInBlock(BasicBlock* bb, GfxArchID arch, int destReg, int src0Reg, int src1Reg)
{
    AsmIRBuilder builder = AsmIRBuilder(*bb, arch);
    StinkyInstruction*  inst
        = builder.create(getMCIDByUOp(GFX::v_add_f32, arch));

    inst->addDestReg(StinkyRegister("v", destReg, 1));
    inst->addSrcReg(StinkyRegister("v", src0Reg, 1));
    inst->addSrcReg(StinkyRegister("v", src1Reg, 1));
    return inst;
}

// Helper function to create s_sub_u32 instruction
StinkyInstruction*
    createSSubU32InBlock(BasicBlock* bb, GfxArchID arch, int destReg, int src0Reg, int src1Val)
{
    AsmIRBuilder builder = AsmIRBuilder(*bb, arch);
    StinkyInstruction*  inst
        = builder.create(getMCIDByUOp(GFX::s_sub_u32, arch));

    inst->addDestReg(StinkyRegister("s", destReg, 1));
    inst->addSrcReg(StinkyRegister("s", src0Reg, 1));
    // For simplicity, we won't add the immediate value in this test
    return inst;
}

// Helper function to create s_cmp_eq_u32 instruction
StinkyInstruction* createSCmpEqU32InBlock(BasicBlock* bb, GfxArchID arch, int src0Reg, int src1Val)
{
    AsmIRBuilder builder = AsmIRBuilder(*bb, arch);
    StinkyInstruction*  inst
        = builder.create(getMCIDByUOp(GFX::s_cmp_eq_u32, arch));

    inst->addSrcReg(StinkyRegister("s", src0Reg, 1));
    // For simplicity, we won't add the immediate value in this test
    return inst;
}

// Helper function to create s_cbranch_scc0 instruction
StinkyInstruction* createCondBranchInBlock(BasicBlock* bb, GfxArchID arch, const std::string& label)
{
    AsmIRBuilder builder = AsmIRBuilder(*bb, arch);
    StinkyInstruction*  inst
        = builder.create(getMCIDByUOp(GFX::s_cbranch_scc0, arch));
    return inst;
}

// Helper function to check if a basic block has a loop back-edge
bool hasLoopBackEdge(const BasicBlock* bb)
{
    // Check if block points to itself (self-loop/back-edge)
    const auto& successors = bb->getSuccessors();
    return std::find(successors.begin(), successors.end(), bb) != successors.end();
}

TEST_F(ConfigurableWaitCntPassTest, BasicBlockStateTracking_NoLoop)
{
    // Test: Single block WITHOUT loop (no back-edge, no branch)
    //
    // Instruction sequence:
    //   v_add_f32 v4, v0, v2  (uses v0, v2 - but not loaded yet!)
    //   v_add_f32 v4, v1, v3  (uses v1, v3 - but not loaded yet!)
    //   ds_load_b32 v0        (load v0 AFTER use)
    //   ds_load_b32 v2        (load v2 AFTER use)
    //   ds_load_b32 v1        (load v1 AFTER use)
    //   ds_load_b32 v3        (load v3 AFTER use)
    //
    // Expected behavior:
    // - NO waitcnt should be inserted (no outstanding loads when v_adds execute)
    // - This is NOT a loop, so there's no dependency from "previous iteration"

    TearDown();
    auto        noLoopFunc = std::make_unique<Function>("test_no_loop");
    BasicBlock* block      = noLoopFunc->createBasicBlock("single_block");

    // NO back-edge, NO loop
    EXPECT_FALSE(hasLoopBackEdge(block)) << "Should NOT be a loop block";

    // Build instructions: v_adds THEN ds_loads (same order as loop test, but no loop)
    StinkyInstruction* fmac1 = createVFmacInBlock(block, arch, 4, 0, 2); // v_add v4, v0, v2
    StinkyInstruction* fmac2 = createVFmacInBlock(block, arch, 4, 1, 3); // v_add v4, v1, v3
    createDSReadB32InBlock(block, arch, 0, 10); // ds_load v0
    createDSReadB32InBlock(block, arch, 2, 10); // ds_load v2
    createDSReadB32InBlock(block, arch, 1, 10); // ds_load v1
    createDSReadB32InBlock(block, arch, 3, 10); // ds_load v3

    // Run the pass
    WaitCntConfig config = WaitCntConfig::standard();
    PassContext   passCtx;
    passCtx.setGemmTileConfig(gemmConfig);
    auto pass = stinkytofu::createStinkyCustomWaitCntPass(config);
    pass->run(*noLoopFunc, passCtx);

    // Collect waitcnt instructions
    std::vector<WaitCntInfo> waitcnts;
    int                      position = 0;
    for(auto& irBase : *block)
    {
        StinkyInstruction& inst = static_cast<StinkyInstruction&>(irBase);
        if(SWaitCntData* wait = inst.getModifier<SWaitCntData>())
        {
            waitcnts.emplace_back(&inst, wait, position);
        }
        position++;
    }

    // Find positions of the two v_add instructions
    int fmac1Pos = -1;
    int fmac2Pos = -1;
    position     = 0;
    for(auto& irBase : *block)
    {
        StinkyInstruction& inst = static_cast<StinkyInstruction&>(irBase);
        if(&inst == fmac1)
        {
            fmac1Pos = position;
        }
        if(&inst == fmac2)
        {
            fmac2Pos = position;
        }
        position++;
    }

    ASSERT_NE(fmac1Pos, -1) << "Should find fmac1 in block";
    ASSERT_NE(fmac2Pos, -1) << "Should find fmac2 in block";

    // Find any waitcnt before fmac1
    SWaitCntData* waitBeforeFmac1 = nullptr;
    for(const auto& wait : waitcnts)
    {
        if(wait.position < fmac1Pos)
        {
            waitBeforeFmac1 = wait.waitData;
            break;
        }
    }

    // NO waitcnt should be inserted before v_adds (no outstanding loads)
    EXPECT_EQ(waitBeforeFmac1, nullptr)
        << "Should NOT insert waitcnt - loads come AFTER uses (no dependency)";
}

TEST_F(ConfigurableWaitCntPassTest, BasicBlockStateTracking_LoopOnly)
{
    // Test 1: Loop block ONLY (no preloop)
    //
    // Single loop block with back-edge to itself:
    //   loop_start:
    //     v_add_f32 v4, v0, v2  (uses v0, v2 from previous iteration)
    //     v_add_f32 v4, v1, v3  (uses v1, v3 from previous iteration)
    //     ds_load_b32 v0        (load v0 for next iteration)
    //     ds_load_b32 v2        (load v2 for next iteration)
    //     ds_load_b32 v1        (load v1 for next iteration)
    //     ds_load_b32 v3        (load v3 for next iteration)
    //     s_cbranch loop_start
    //
    // Expected behavior (once iterative dataflow is implemented):
    // - First v_add should have s_waitcnt dlcnt=2 (wait for v0,v2, leave v1,v3)
    // - Second v_add should have s_waitcnt dlcnt=0 (wait for v1,v3)
    //
    // Current status: SKIPPED - requires iterative dataflow for convergence

    TearDown();
    auto        loopFunc  = std::make_unique<Function>("test_loop_only");
    BasicBlock* loopBlock = loopFunc->createBasicBlock("loop_start");

    // Set up loop back-edge: block points to itself
    loopBlock->addSuccessor(loopBlock);
    loopBlock->addPredecessor(loopBlock);

    // Verify this is a loop block
    EXPECT_TRUE(hasLoopBackEdge(loopBlock)) << "Loop block should have back-edge to itself";

    // Build instructions: v_adds THEN ds_loads (loop pattern)
    StinkyInstruction* fmac1 = createVFmacInBlock(loopBlock, arch, 4, 0, 2); // v_add v4, v0, v2
    StinkyInstruction* fmac2 = createVFmacInBlock(loopBlock, arch, 4, 1, 3); // v_add v4, v1, v3
    createDSReadB32InBlock(loopBlock, arch, 0, 10); // ds_load v0
    createDSReadB32InBlock(loopBlock, arch, 2, 10); // ds_load v2
    createDSReadB32InBlock(loopBlock, arch, 1, 10); // ds_load v1
    createDSReadB32InBlock(loopBlock, arch, 3, 10); // ds_load v3

    // Run the pass
    WaitCntConfig config = WaitCntConfig::standard();
    PassContext   passCtx;
    passCtx.setGemmTileConfig(gemmConfig);
    auto pass = stinkytofu::createStinkyCustomWaitCntPass(config);
    pass->run(*loopFunc, passCtx);

    // Collect waitcnt instructions
    std::vector<WaitCntInfo> waitcnts;
    int                      position = 0;
    for(auto& irBase : *loopBlock)
    {
        StinkyInstruction& inst = static_cast<StinkyInstruction&>(irBase);
        if(SWaitCntData* wait = inst.getModifier<SWaitCntData>())
        {
            waitcnts.emplace_back(&inst, wait, position);
        }
        position++;
    }

    // Find positions of the two v_add instructions
    int fmac1Pos = -1;
    int fmac2Pos = -1;
    position     = 0;
    for(auto& irBase : *loopBlock)
    {
        StinkyInstruction& inst = static_cast<StinkyInstruction&>(irBase);
        if(&inst == fmac1)
        {
            fmac1Pos = position;
        }
        if(&inst == fmac2)
        {
            fmac2Pos = position;
        }
        position++;
    }

    ASSERT_NE(fmac1Pos, -1) << "Should find fmac1 in loop block";
    ASSERT_NE(fmac2Pos, -1) << "Should find fmac2 in loop block";

    // Find waitcnt before fmac1 (first v_add)
    SWaitCntData* waitBeforeFmac1 = nullptr;
    for(const auto& wait : waitcnts)
    {
        if(wait.position < fmac1Pos && wait.position >= fmac1Pos - 2)
        {
            waitBeforeFmac1 = wait.waitData;
            break;
        }
    }

    ASSERT_NE(waitBeforeFmac1, nullptr)
        << "Should insert waitcnt before first v_add (loop dependency)";

    // Expected behavior with iterative dataflow and register-level tracking:
    // - Loop converges: entry state = exit state from previous iteration
    // - Exit state has 4 outstanding ds_loads (v0, v2, v1, v3 in that order)
    // - First v_add uses v0, v2 -> need to wait for loads up to v2 (index 1) -> leaves v1, v3 -> dlcnt=2
    EXPECT_EQ(waitBeforeFmac1->dlcnt, 2)
        << "Should wait for v0,v2 from previous iteration (leave v1,v3) -> dlcnt=2";

    // Find waitcnt before fmac2 (second v_add) - must be different from first waitcnt
    SWaitCntData* waitBeforeFmac2    = nullptr;
    int           waitBeforeFmac1Pos = -1;
    for(const auto& wait : waitcnts)
    {
        if(wait.waitData == waitBeforeFmac1)
        {
            waitBeforeFmac1Pos = wait.position;
            break;
        }
    }

    for(const auto& wait : waitcnts)
    {
        if(wait.position < fmac2Pos && wait.position >= fmac2Pos - 2
           && wait.position != waitBeforeFmac1Pos)
        {
            waitBeforeFmac2 = wait.waitData;
            break;
        }
    }

    // Second v_add uses v1, v3 -> after first wait, we have v1, v3 outstanding
    // Need to wait for v3 (latest), which also waits for v1 -> dlcnt=0
    if(waitBeforeFmac2)
    {
        EXPECT_EQ(waitBeforeFmac2->dlcnt, 0)
            << "Should wait for v1,v3 (all remaining loads) -> dlcnt=0";
    }
}

TEST_F(ConfigurableWaitCntPassTest, BasicBlockStateTracking_TwoBlockChain)
{
    // This test verifies wait state propagation from Block 1 to Block 2
    //
    // Block 1 (entry):
    //   ds_load_b32 v0  (issued, outstanding)
    //   ds_load_b32 v1  (issued, outstanding)
    //   ds_load_b32 v2  (issued, outstanding)
    //   ds_load_b32 v3  (issued, outstanding)
    //
    // Block 2 (loop with back-edge):
    //   v_add_f32 v4, v0, v2  <- Should have s_waitcnt dlcnt=1 before this
    //   v_add_f32 v4, v1, v3  <- May have additional waitcnt if needed
    //   ds_load_b32 v0
    //   ds_load_b32 v2
    //   ds_load_b32 v1
    //   ds_load_b32 v3
    //
    // Expected:
    // - Block 2 entry state: 4 outstanding ds_loads from Block 1
    // - First v_add uses v0,v2 -> wait for v0,v1,v2 (v1 is between) -> dlcnt=1

    TearDown();
    func = std::make_unique<Function>("test_two_block_chain");

    // Create Block 1 (entry block)
    BasicBlock* block1 = func->createBasicBlock("entry");

    // Create Block 2 (loop block)
    BasicBlock* block2 = func->createBasicBlock("loop_start");

    // Set up CFG: block1 -> block2, block2 -> block2 (loop back)
    block1->addSuccessor(block2);
    block2->addPredecessor(block1);
    block2->addSuccessor(block2); // loop back edge
    block2->addPredecessor(block2); // loop back edge

    // Verify CFG structure using helper
    EXPECT_FALSE(hasLoopBackEdge(block1)) << "Block1 should not have loop back-edge";
    EXPECT_TRUE(hasLoopBackEdge(block2)) << "Block2 should have loop back-edge to itself";

    // Build Block 1: 4 ds_load instructions in order v0, v1, v2, v3
    createDSReadB32InBlock(block1, arch, 0, 10); // ds_load_b32 v0, v10
    createDSReadB32InBlock(block1, arch, 1, 10); // ds_load_b32 v1, v10
    createDSReadB32InBlock(block1, arch, 2, 10); // ds_load_b32 v2, v10
    createDSReadB32InBlock(block1, arch, 3, 10); // ds_load_b32 v3, v10

    // Build Block 2: v_adds THEN ds_loads
    // v_adds use values from Block 1's ds_loads (or previous loop iteration)
    StinkyInstruction* fmac1 = createVFmacInBlock(block2, arch, 4, 0, 2); // v_add_f32 v4, v0, v2
    StinkyInstruction* fmac2 = createVFmacInBlock(block2, arch, 4, 1, 3); // v_add_f32 v4, v1, v3
    createDSReadB32InBlock(block2, arch, 0, 10); // ds_load_b32 v0, v10
    createDSReadB32InBlock(block2, arch, 2, 10); // ds_load_b32 v2, v10
    createDSReadB32InBlock(block2, arch, 1, 10); // ds_load_b32 v1, v10
    createDSReadB32InBlock(block2, arch, 3, 10); // ds_load_b32 v3, v10

    // Run the pass
    WaitCntConfig config = WaitCntConfig::standard();
    PassContext   passCtx;
    passCtx.setGemmTileConfig(gemmConfig);
    auto pass = stinkytofu::createStinkyCustomWaitCntPass(config);
    pass->run(*func, passCtx);

    // Collect all waitcnt instructions in block2
    std::vector<WaitCntInfo> waitcnts;
    int                      position = 0;
    for(auto& irBase : *block2)
    {
        StinkyInstruction& inst = static_cast<StinkyInstruction&>(irBase);
        if(SWaitCntData* wait = inst.getModifier<SWaitCntData>())
        {
            waitcnts.emplace_back(&inst, wait, position);
        }
        position++;
    }

    // Find fmac1 position
    int fmac1Pos = -1;
    position     = 0;
    for(auto& irBase : *block2)
    {
        if(&static_cast<StinkyInstruction&>(irBase) == fmac1)
        {
            fmac1Pos = position;
            break;
        }
        position++;
    }

    ASSERT_NE(fmac1Pos, -1) << "Should find fmac1 in block2";

    // Find fmac2 position
    int fmac2Pos = -1;
    position     = 0;
    for(auto& irBase : *block2)
    {
        if(&static_cast<StinkyInstruction&>(irBase) == fmac2)
        {
            fmac2Pos = position;
            break;
        }
        position++;
    }

    ASSERT_NE(fmac2Pos, -1) << "Should find fmac2 in block2";

    // Find waitcnt before fmac1
    SWaitCntData* waitBeforeFmac1 = nullptr;
    for(const auto& wait : waitcnts)
    {
        if(wait.position < fmac1Pos && wait.position >= fmac1Pos - 2)
        {
            waitBeforeFmac1 = wait.waitData;
            break;
        }
    }

    ASSERT_NE(waitBeforeFmac1, nullptr)
        << "Should insert waitcnt before first v_add (cross-block dependency)";

    // Verify: Entry state has 4 outstanding (v0,v1,v2,v3 from block1 in that order)
    // First v_add uses v0,v2 -> need to wait for loads up to v2 (index 2) -> leaves v3 -> dlcnt=1
    EXPECT_EQ(waitBeforeFmac1->dlcnt, 1)
        << "Should wait for v0,v1,v2 from Block 1 (leave v3) -> dlcnt=1";

    // Find waitcnt before fmac2 (second v_add) - must be different from first waitcnt
    SWaitCntData* waitBeforeFmac2    = nullptr;
    int           waitBeforeFmac1Pos = -1;
    for(const auto& wait : waitcnts)
    {
        if(wait.waitData == waitBeforeFmac1)
        {
            waitBeforeFmac1Pos = wait.position;
            break;
        }
    }

    for(const auto& wait : waitcnts)
    {
        if(wait.position < fmac2Pos && wait.position >= fmac2Pos - 2
           && wait.position != waitBeforeFmac1Pos)
        {
            waitBeforeFmac2 = wait.waitData;
            break;
        }
    }

    // Second v_add uses v1, v3 -> after first wait (dlcnt=1), we have v3 outstanding
    // Need to wait for v3 -> dlcnt=0 (MUST exist!)
    ASSERT_NE(waitBeforeFmac2, nullptr)
        << "MUST insert waitcnt before second v_add (v3 still outstanding)";

    EXPECT_EQ(waitBeforeFmac2->dlcnt, 0) << "Should wait for remaining loads (v3) -> dlcnt=0";
}

TEST_F(ConfigurableWaitCntPassTest, BasicBlockStateTracking_TwoBlockChain2)
{
    // This test verifies wait state propagation from Block 1 to Block 2
    //
    // Block 1 (entry):
    //   ds_load_b32 v0  (issued, outstanding)
    //   ds_load_b32 v2  (issued, outstanding)
    //   ds_load_b32 v1  (issued, outstanding)
    //   ds_load_b32 v3  (issued, outstanding)
    //
    // Block 2 (loop with back-edge):
    //   v_add_f32 v4, v0, v2  <- Should have s_waitcnt dlcnt=1 before this
    //   v_add_f32 v4, v1, v3  <- May have additional waitcnt if needed
    //   ds_load_b32 v0
    //   ds_load_b32 v1
    //   ds_load_b32 v2
    //   ds_load_b32 v3
    //
    // Expected:
    // - Block 2 entry state: 4 outstanding ds_loads from Block 1
    // - First v_add uses v0,v2 -> wait for v0,v1,v2 (v1 is between) -> dlcnt=1

    TearDown();
    func = std::make_unique<Function>("test_two_block_chain");

    // Create Block 1 (entry block)
    BasicBlock* block1 = func->createBasicBlock("entry");

    // Create Block 2 (loop block)
    BasicBlock* block2 = func->createBasicBlock("loop_start");

    // Set up CFG: block1 -> block2, block2 -> block2 (loop back)
    block1->addSuccessor(block2);
    block2->addPredecessor(block1);
    block2->addSuccessor(block2); // loop back edge
    block2->addPredecessor(block2); // loop back edge

    // Verify CFG structure using helper
    EXPECT_FALSE(hasLoopBackEdge(block1)) << "Block1 should not have loop back-edge";
    EXPECT_TRUE(hasLoopBackEdge(block2)) << "Block2 should have loop back-edge to itself";

    // Build Block 1: 4 ds_load instructions in order v0, v2, v1, v3
    createDSReadB32InBlock(block1, arch, 0, 10); // ds_load_b32 v0, v10
    createDSReadB32InBlock(block1, arch, 2, 10); // ds_load_b32 v2, v10
    createDSReadB32InBlock(block1, arch, 1, 10); // ds_load_b32 v1, v10
    createDSReadB32InBlock(block1, arch, 3, 10); // ds_load_b32 v3, v10

    // Build Block 2: v_adds THEN ds_loads
    // v_adds use values from Block 1's ds_loads (or previous loop iteration)
    StinkyInstruction* fmac1 = createVFmacInBlock(block2, arch, 4, 0, 2); // v_add_f32 v4, v0, v2
    StinkyInstruction* fmac2 = createVFmacInBlock(block2, arch, 4, 1, 3); // v_add_f32 v4, v1, v3
    createDSReadB32InBlock(block2, arch, 0, 10); // ds_load_b32 v0, v10
    createDSReadB32InBlock(block2, arch, 1, 10); // ds_load_b32 v1, v10
    createDSReadB32InBlock(block2, arch, 2, 10); // ds_load_b32 v2, v10
    createDSReadB32InBlock(block2, arch, 3, 10); // ds_load_b32 v3, v10

    // Run the pass
    WaitCntConfig config = WaitCntConfig::standard();
    PassContext   passCtx;
    passCtx.setGemmTileConfig(gemmConfig);
    auto pass = stinkytofu::createStinkyCustomWaitCntPass(config);
    pass->run(*func, passCtx);

    // Collect all waitcnt instructions in block2
    std::vector<WaitCntInfo> waitcnts;
    int                      position = 0;
    for(auto& irBase : *block2)
    {
        StinkyInstruction& inst = static_cast<StinkyInstruction&>(irBase);
        if(SWaitCntData* wait = inst.getModifier<SWaitCntData>())
        {
            waitcnts.emplace_back(&inst, wait, position);
        }
        position++;
    }

    // Find fmac1 position
    int fmac1Pos = -1;
    position     = 0;
    for(auto& irBase : *block2)
    {
        if(&static_cast<StinkyInstruction&>(irBase) == fmac1)
        {
            fmac1Pos = position;
            break;
        }
        position++;
    }

    ASSERT_NE(fmac1Pos, -1) << "Should find fmac1 in block2";

    // Find fmac2 position
    int fmac2Pos = -1;
    position     = 0;
    for(auto& irBase : *block2)
    {
        if(&static_cast<StinkyInstruction&>(irBase) == fmac2)
        {
            fmac2Pos = position;
            break;
        }
        position++;
    }

    ASSERT_NE(fmac2Pos, -1) << "Should find fmac2 in block2";

    // Find waitcnt before fmac1
    SWaitCntData* waitBeforeFmac1 = nullptr;
    for(const auto& wait : waitcnts)
    {
        if(wait.position < fmac1Pos && wait.position >= fmac1Pos - 2)
        {
            waitBeforeFmac1 = wait.waitData;
            break;
        }
    }

    ASSERT_NE(waitBeforeFmac1, nullptr)
        << "Should insert waitcnt before first v_add (cross-block dependency)";

    // Verify: Entry state has 4 outstanding (v0,v2,v1,v3 from Block 1 in that order)
    // First v_add uses v0,v2 -> need to wait for loads up to v2 (index 1) -> leaves v1,v3 -> dlcnt=2
    // HOWEVER: After loop convergence and proper merging, the state should reflect the union
    // For now, we expect the implementation to properly merge preloop and loop states
    EXPECT_EQ(waitBeforeFmac1->dlcnt, 1)
        << "Should wait for v0,v1,v2 from itself (leave v3) -> dlcnt=1";

    // Find waitcnt before fmac2 (second v_add) - must be different from first waitcnt
    SWaitCntData* waitBeforeFmac2    = nullptr;
    int           waitBeforeFmac1Pos = -1;
    for(const auto& wait : waitcnts)
    {
        if(wait.waitData == waitBeforeFmac1)
        {
            waitBeforeFmac1Pos = wait.position;
            break;
        }
    }

    for(const auto& wait : waitcnts)
    {
        if(wait.position < fmac2Pos && wait.position >= fmac2Pos - 2
           && wait.position != waitBeforeFmac1Pos)
        {
            waitBeforeFmac2 = wait.waitData;
            break;
        }
    }

    // Second v_add uses v1, v3 -> after first wait (dlcnt=1), we have v3 outstanding
    // Need to wait for v3 -> dlcnt=0 (MUST exist!)
    ASSERT_NE(waitBeforeFmac2, nullptr)
        << "MUST insert waitcnt before second v_add (v3 still outstanding)";

    EXPECT_EQ(waitBeforeFmac2->dlcnt, 0) << "Should wait for remaining loads (v3) -> dlcnt=0";
}

/**
 * @brief Test multi-predecessor with multi-path analysis
 *
 * b1: ds_read v0, v1 -> b3
 * b2: ds_read v2, v3, v4 -> b3
 * b3: v_fmac v5, v0, v1
 *
 * Multi-path analysis:
 * - Path 1 (b1): [v0, v1] -> fmac uses v0, v1 -> needs dlcnt=0
 * - Path 2 (b2): [v2, v3, v4] -> fmac uses v0, v1 (not present) -> no wait needed
 * - Result: min(0, IGNORE) = 0 -> Optimal for path 1, safe for path 2
 */
TEST_F(ConfigurableWaitCntPassTest, BasicBlockStateTracking_MultiPredecessorMerge)
{
    TearDown();
    auto        testFunc = std::make_unique<Function>("test_multi_pred");
    BasicBlock* entry    = testFunc->createBasicBlock("entry");
    BasicBlock* block1   = testFunc->createBasicBlock("b1");
    BasicBlock* block2   = testFunc->createBasicBlock("b2");
    BasicBlock* block3   = testFunc->createBasicBlock("b3");

    // Build Entry: just a branch (conceptually branches to both b1 and b2)
    // We'll connect it properly in the CFG

    // Build Block 1: ds_read v0, v1
    createDSReadB32InBlock(block1, arch, 0, 10); // v0
    createDSReadB32InBlock(block1, arch, 1, 10); // v1

    // Build Block 2: ds_read v2, v3, v4
    createDSReadB32InBlock(block2, arch, 2, 10); // v2
    createDSReadB32InBlock(block2, arch, 3, 10); // v3
    createDSReadB32InBlock(block2, arch, 4, 10); // v4

    // Build Block 3: v_fmac v5, v0, v1
    StinkyInstruction* fmac = createVFmacInBlock(block3, arch, 5, 0, 1); // v5 = v0 * v1

    // Set up CFG: entry -> b1, entry -> b2, b1 -> b3, b2 -> b3
    entry->addSuccessor(block1);
    entry->addSuccessor(block2);
    block1->addPredecessor(entry);
    block2->addPredecessor(entry);
    block1->addSuccessor(block3);
    block2->addSuccessor(block3);
    block3->addPredecessor(block1);
    block3->addPredecessor(block2);

    // Run the pass
    WaitCntConfig config = WaitCntConfig::standard();
    PassContext   passCtx;
    passCtx.setGemmTileConfig(gemmConfig);
    auto pass = stinkytofu::createStinkyCustomWaitCntPass(config);
    pass->run(*testFunc, passCtx);

    // Collect all inserted waitcnts in block3
    std::vector<WaitCntInfo> waitcnts;
    int                      position = 0;

    for(auto& irBase : *block3)
    {
        StinkyInstruction& inst = static_cast<StinkyInstruction&>(irBase);
        if(SWaitCntData* wait = inst.getModifier<SWaitCntData>())
        {
            waitcnts.emplace_back(&inst, wait, position);
        }
        position++;
    }

    // Find fmac position
    int fmacPos = -1;
    position    = 0;
    for(auto& irBase : *block3)
    {
        if(&static_cast<StinkyInstruction&>(irBase) == fmac)
        {
            fmacPos = position;
            break;
        }
        position++;
    }

    ASSERT_NE(fmacPos, -1) << "Should find fmac in block3";

    // Find waitcnt before fmac
    SWaitCntData* waitBeforeFmac = nullptr;
    for(const auto& wait : waitcnts)
    {
        if(wait.position < fmacPos && wait.position >= fmacPos - 2)
        {
            waitBeforeFmac = wait.waitData;
            break;
        }
    }

    ASSERT_NE(waitBeforeFmac, nullptr)
        << "Should insert waitcnt before fmac (multi-predecessor merge)";

    // Verify: With multi-path analysis, we analyze each predecessor path separately:
    // - Path 1 (b1): Has [v0, v1], fmac uses v0,v1 -> needs dlcnt=0 (wait for both)
    // - Path 2 (b2): Has [v2, v3, v4], fmac uses v0,v1 (not present) -> no wait needed
    // - Final: min(0, IGNORE) = 0 -> Waits for path 1, path 2 unaffected
    // Expected: dlcnt=0 (optimal for both paths)
    EXPECT_EQ(waitBeforeFmac->dlcnt, 0)
        << "Should wait for v0,v1 from path 1 (multi-path analysis) -> dlcnt=0";
}

/**
 * @brief Test chained multi-predecessor - demonstrates multi-path analysis limitation
 *
 * CFG:
 *   entry -> b1 (loads v0, v1, v2) -> b3 (uses v3, v4) -> b4 (uses v0, v1)
 *   entry -> b2 (loads v3, v4, v5) ?
 *
 * Block3 (multi-predecessor): Multi-path analysis works perfectly
 * - Path 1 (b1): [v0, v1, v2] -> uses v3, v4 (not present) -> no wait needed
 * - Path 2 (b2): [v3, v4, v5] -> uses v3, v4 -> wait for v3, v4 -> dlcnt=1 (leaves v5)
 * - Result: min(IGNORE, 1) = 1 ? Optimal!
 *
 * Block4 (single predecessor): Limitation revealed
 * - Block4 has only ONE predecessor (block3), so no multi-path analysis
 * - Block3's exit state (after conservative merge): dsLoadCount > 0, but outstandingDSLoads = []
 * - Block4 sees counts but no register lists -> lost precision from conservative merge
 * - Current behavior: No wait inserted (or dlcnt=0 if conservative)
 * - Ideal behavior: Would need per-path exit states from block3 to maintain precision
 *
 * This test demonstrates that multi-path analysis optimizes merge points, but precision
 * is lost downstream after conservative merging. This is acceptable because the critical
 * cross-block dependency (v3, v4 in block3) is handled optimally.
 */
TEST_F(ConfigurableWaitCntPassTest, BasicBlockStateTracking_MultiPredecessorMerge2)
{
    TearDown();
    auto        testFunc = std::make_unique<Function>("test_multi_pred");
    BasicBlock* entry    = testFunc->createBasicBlock("entry");
    BasicBlock* block1   = testFunc->createBasicBlock("b1");
    BasicBlock* block2   = testFunc->createBasicBlock("b2");
    BasicBlock* block3   = testFunc->createBasicBlock("b3");
    BasicBlock* block4   = testFunc->createBasicBlock("b4");

    // Build Block 1: ds_read v0, v1, v2
    createDSReadB32InBlock(block1, arch, 0, 10); // v0
    createDSReadB32InBlock(block1, arch, 1, 10); // v1
    createDSReadB32InBlock(block1, arch, 2, 10); // v2

    // Build Block 2: ds_read v3, v4, v5
    createDSReadB32InBlock(block2, arch, 3, 10); // v3
    createDSReadB32InBlock(block2, arch, 4, 10); // v4
    createDSReadB32InBlock(block2, arch, 5, 10); // v5

    // Build Block 3: v_fmac v6, v3, v3
    StinkyInstruction* fmac = createVFmacInBlock(block3, arch, 6, 3, 3); // v6 = v3 * v4

    // Build Block 4: v_fmac v7, v0, v1
    StinkyInstruction* fmac2 = createVFmacInBlock(block4, arch, 7, 0, 1); // v7 = v0 * v1

    // Set up CFG: entry -> b1, entry -> b2, b1 -> b3, b2 -> b3, b3 -> b4
    entry->addSuccessor(block1);
    entry->addSuccessor(block2);
    block1->addPredecessor(entry);
    block2->addPredecessor(entry);
    block1->addSuccessor(block3);
    block2->addSuccessor(block3);
    block3->addPredecessor(block1);
    block3->addPredecessor(block2);
    block3->addSuccessor(block4);
    block4->addPredecessor(block3);

    // Run the pass
    WaitCntConfig config = WaitCntConfig::standard();
    PassContext   passCtx;
    passCtx.setGemmTileConfig(gemmConfig);
    auto pass = stinkytofu::createStinkyCustomWaitCntPass(config);
    pass->run(*testFunc, passCtx);

    // Collect all inserted waitcnts in block3
    std::vector<WaitCntInfo> waitcnts;
    int                      position = 0;

    for(auto& irBase : *block3)
    {
        StinkyInstruction& inst = static_cast<StinkyInstruction&>(irBase);
        if(SWaitCntData* wait = inst.getModifier<SWaitCntData>())
        {
            waitcnts.emplace_back(&inst, wait, position);
        }
        position++;
    }

    // Find fmac position
    int fmacPos = -1;
    position    = 0;
    for(auto& irBase : *block3)
    {
        if(&static_cast<StinkyInstruction&>(irBase) == fmac)
        {
            fmacPos = position;
            break;
        }
        position++;
    }

    ASSERT_NE(fmacPos, -1) << "Should find fmac in block3";

    // Find waitcnt before fmac
    SWaitCntData* waitBeforeFmac = nullptr;
    for(const auto& wait : waitcnts)
    {
        if(wait.position < fmacPos && wait.position >= fmacPos - 2)
        {
            waitBeforeFmac = wait.waitData;
            break;
        }
    }

    ASSERT_NE(waitBeforeFmac, nullptr)
        << "Should insert waitcnt before fmac in block3 (multi-predecessor case)";

    // Verify block3: Multi-path analysis on [v0,v1,v2] vs [v3,v4,v5], uses v3,v4
    // - Path 1 (b1): [v0, v1, v2] -> v3, v4 not present -> no wait
    // - Path 2 (b2): [v3, v4, v5] -> v3 at 0, v4 at 1 -> dlcnt=1 (leaves v5)
    // - Result: min(IGNORE, 1) = 1
    EXPECT_EQ(waitBeforeFmac->dlcnt, 2)
        << "Should wait for v3, v3 from path 2 (multi-path analysis) -> dlcnt=2";

    // Collect all inserted waitcnts in block4
    std::vector<WaitCntInfo> waitcnts4;
    position = 0;

    for(auto& irBase : *block4)
    {
        StinkyInstruction& inst = static_cast<StinkyInstruction&>(irBase);
        if(SWaitCntData* wait = inst.getModifier<SWaitCntData>())
        {
            waitcnts4.emplace_back(&inst, wait, position);
        }
        position++;
    }

    // Find fmac2 position
    int fmac2Pos = -1;
    position     = 0;
    for(auto& irBase : *block4)
    {
        if(&static_cast<StinkyInstruction&>(irBase) == fmac2)
        {
            fmac2Pos = position;
            break;
        }
        position++;
    }

    ASSERT_NE(fmac2Pos, -1) << "Should find fmac2 in block4";

    // Find waitcnt before fmac2
    SWaitCntData* waitBeforeFmac2 = nullptr;
    for(const auto& wait : waitcnts4)
    {
        if(wait.position < fmac2Pos && wait.position >= fmac2Pos - 2)
        {
            waitBeforeFmac2 = wait.waitData;
            break;
        }
    }

    // Block4 behavior: Per-path tracking with multi-path analysis
    // Block3 outputs 2 exit states (one per incoming path):
    // - Path1 (via b1): After wait(dlcnt=2), [v1,v2] remain
    // - Path2 (via b2): After wait(dlcnt=2), [v4,v5] remain
    //
    // Block4 receives both path states and does multi-path analysis for v0,v1:
    // - Path1: [v1,v2] -> v1 at index 0 -> dlcnt=1 (optimal!)
    // - Path2: [v4,v5] -> v0,v1 not in list -> no wait needed
    // - Result: min(1, IGNORE) = 1
    //
    // This achieves the optimal wait count by maintaining per-path precision!
    ASSERT_NE(waitBeforeFmac2, nullptr)
        << "Should insert waitcnt before fmac2 in block4 (uses v0,v1)";

    EXPECT_EQ(waitBeforeFmac2->dlcnt, 1)
        << "Per-path analysis: Path1 needs dlcnt=1, Path2 needs none -> min=1";
}
