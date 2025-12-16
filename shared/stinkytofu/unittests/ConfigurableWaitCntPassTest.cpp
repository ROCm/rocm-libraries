/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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

#include "ir/asm/StinkyAsmIR.hpp"
#include "ir/asm/StinkyAsmPrinter.hpp"
#include "ir/asm/StinkyConfigurableWaitCntPass.hpp"
#include "stinkytofu.hpp"

using namespace stinkytofu;

// Helper class to build test IR and run pass
class ConfigurableWaitCntPassTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        arch               = getGfxArchID(12, 5, 0); // GFX1250
        kernelInfo.arch[0] = 12;
        kernelInfo.arch[1] = 5;
        kernelInfo.arch[2] = 0;
    }

    void TearDown() override
    {
        insts.clear();
    }

    // Create IRBuilder for building test instructions
    StinkyInstIRBuilder& getIRBuilder()
    {
        if(!irBuilder)
        {
            irBuilder = std::make_unique<StinkyInstIRBuilder>(insts, arch);
        }
        return *irBuilder;
    }

    // Helper to create a ds_read instruction (64-bit, 2 registers)
    StinkyInstruction* createDSRead(int destReg, int addrReg)
    {
        auto&              builder = getIRBuilder();
        StinkyInstruction* inst
            = builder.createStinkyInstBefore(insts.end(), getMCIDByUOp(GFX::ds_load_b64, arch));

        inst->destRegs.push_back(StinkyRegister("v", destReg, 2));
        inst->srcRegs.push_back(StinkyRegister("v", addrReg, 1));
        return inst;
    }

    // Helper to create a ds_read instruction (128-bit, 4 registers)
    StinkyInstruction* createDSRead128(int destReg, int addrReg)
    {
        auto&              builder = getIRBuilder();
        StinkyInstruction* inst
            = builder.createStinkyInstBefore(insts.end(), getMCIDByUOp(GFX::ds_load_b128, arch));

        inst->destRegs.push_back(StinkyRegister("v", destReg, 4));
        inst->srcRegs.push_back(StinkyRegister("v", addrReg, 1));
        return inst;
    }

    // Helper to create a ds_write instruction
    StinkyInstruction* createDSWrite(int addrReg, int dataReg)
    {
        auto&              builder = getIRBuilder();
        StinkyInstruction* inst
            = builder.createStinkyInstBefore(insts.end(), getMCIDByUOp(GFX::ds_write_b64, arch));

        inst->srcRegs.push_back(StinkyRegister("v", addrReg, 2));
        inst->srcRegs.push_back(StinkyRegister("v", dataReg, 1));
        return inst;
    }

    // Helper to create a global_load instruction
    StinkyInstruction* createGlobalLoad(int destReg, int addrReg)
    {
        auto&              builder = getIRBuilder();
        StinkyInstruction* inst    = builder.createStinkyInstBefore(
            insts.end(), getMCIDByUOp(GFX::global_load_dword, arch));

        inst->destRegs.push_back(StinkyRegister("v", destReg, 1));
        inst->srcRegs.push_back(StinkyRegister("s", addrReg, 4));
        return inst;
    }

    // Helper to create a global_store instruction
    StinkyInstruction* createGlobalStore(int addrReg, int dataReg)
    {
        auto&              builder = getIRBuilder();
        StinkyInstruction* inst    = builder.createStinkyInstBefore(
            insts.end(), getMCIDByUOp(GFX::global_store_dword, arch));

        inst->srcRegs.push_back(StinkyRegister("v", addrReg, 1));
        inst->srcRegs.push_back(StinkyRegister("s", dataReg, 4));
        return inst;
    }

    StinkyInstruction* createTensorLoad(int src0Reg, int src1Reg)
    {
        auto&              builder = getIRBuilder();
        StinkyInstruction* inst    = builder.createStinkyInstBefore(
            insts.end(), getMCIDByUOp(GFX::tensor_load_to_lds, arch));

        inst->srcRegs.push_back(StinkyRegister("s", src0Reg, 4));
        inst->srcRegs.push_back(StinkyRegister("s", src1Reg, 8));
        return inst;
    }

    // Helper to create a v_add instruction
    StinkyInstruction* createVAdd(int destReg, int src0Reg, int src1Reg)
    {
        auto&              builder = getIRBuilder();
        StinkyInstruction* inst
            = builder.createStinkyInstBefore(insts.end(), getMCIDByUOp(GFX::v_add_f32, arch));

        inst->destRegs.push_back(StinkyRegister("v", destReg, 1));
        inst->srcRegs.push_back(StinkyRegister("v", src0Reg, 1));
        inst->srcRegs.push_back(StinkyRegister("v", src1Reg, 1));
        return inst;
    }

    // Helper to create a v_mul instruction
    StinkyInstruction* createVMul(int destReg, int src0Reg, int src1Reg)
    {
        auto&              builder = getIRBuilder();
        StinkyInstruction* inst
            = builder.createStinkyInstBefore(insts.end(), getMCIDByUOp(GFX::v_mul_f32, arch));

        inst->destRegs.push_back(StinkyRegister("v", destReg, 1));
        inst->srcRegs.push_back(StinkyRegister("v", src0Reg, 1));
        inst->srcRegs.push_back(StinkyRegister("v", src1Reg, 1));
        return inst;
    }

    // Helper to create an s_barrier instruction
    StinkyInstruction* createBarrier()
    {
        auto&              builder = getIRBuilder();
        StinkyInstruction* inst
            = builder.createStinkyInstBefore(insts.end(), getMCIDByUOp(GFX::s_barrier, arch));
        return inst;
    }

    // Helper to create a mfma instruction (tensor load)
    StinkyInstruction* createWMMA(int destReg, int src0Reg, int src1Reg)
    {
        auto&              builder = getIRBuilder();
        StinkyInstruction* inst    = builder.createStinkyInstBefore(
            insts.end(), getMCIDByUOp(GFX::v_wmma_f32_16x16x32_bf16, arch));

        inst->destRegs.push_back(StinkyRegister("a", destReg, 8));
        inst->srcRegs.push_back(StinkyRegister("v", src0Reg, 8));
        inst->srcRegs.push_back(StinkyRegister("v", src1Reg, 8));
        inst->srcRegs.push_back(StinkyRegister("a", destReg, 8));
        return inst;
    }

    // Helper to count waitcnt instructions
    int countWaitCnt()
    {
        int count = 0;
        for(auto& irBase : insts)
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
        for(auto& irBase : insts)
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
        for(auto& irBase : insts)
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
        for(auto& irBase : insts)
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
        for(auto& irBase : insts)
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
        IRList::iterator targetIt = insts.end();
        for(auto it = insts.begin(); it != insts.end(); ++it)
        {
            if(&static_cast<StinkyInstruction&>(*it) == target)
            {
                targetIt = it;
                break;
            }
        }

        if(targetIt == insts.end() || targetIt == insts.begin())
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
                if(prevIt == insts.begin())
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
        IRList::iterator targetIt = insts.end();
        for(auto it = insts.begin(); it != insts.end(); ++it)
        {
            if(&static_cast<StinkyInstruction&>(*it) == target)
            {
                targetIt = it;
                break;
            }
        }

        if(targetIt == insts.end() || targetIt == insts.begin())
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
                if(prevIt == insts.begin())
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
        passCtx.addKernelInfo(kernelInfo);
        auto pass = stinkytofu::createStinkyCustomWaitCntPass(config);
        pass->run(insts, passCtx);
    }

    void dumpInsts()
    {
        std::cout << insts << std::endl;
    }

    IRList                               insts;
    StinkyKernelInfo                     kernelInfo;
    GfxArchID                            arch;
    std::unique_ptr<StinkyInstIRBuilder> irBuilder;
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
