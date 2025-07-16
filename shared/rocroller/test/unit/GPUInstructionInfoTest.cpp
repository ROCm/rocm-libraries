/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/CodeGen/WaitCount.hpp>
#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>

#include "SimpleFixture.hpp"

using namespace rocRoller;

class GPUInstructionInfoTest : public SimpleFixture
{
};

TEST_F(GPUInstructionInfoTest, BasicTest)
{
    GPUInstructionInfo Test(
        "test", 0, {GPUWaitQueueType::DSQueue, GPUWaitQueueType::LoadQueue}, 16, true, false, 8192);
    EXPECT_EQ(Test.getInstruction(), "test");
    EXPECT_EQ(Test.getWaitQueues().size(), 2);
    EXPECT_EQ(Test.getWaitQueues()[0], GPUWaitQueueType::DSQueue);
    EXPECT_EQ(Test.getWaitQueues()[1], GPUWaitQueueType::LoadQueue);
    EXPECT_EQ(Test.getWaitCount(), 0);
    EXPECT_EQ(Test.getLatency(), 16);
    EXPECT_EQ(Test.maxOffsetValue(), 8192);
}

TEST_F(GPUInstructionInfoTest, BasicTestLatency)
{
    GPUInstructionInfo Test("test", 0, {GPUWaitQueueType::DSQueue, GPUWaitQueueType::LoadQueue}, 8);
    EXPECT_EQ(Test.getInstruction(), "test");
    EXPECT_EQ(Test.getWaitQueues().size(), 2);
    EXPECT_EQ(Test.getWaitQueues()[0], GPUWaitQueueType::DSQueue);
    EXPECT_EQ(Test.getWaitQueues()[1], GPUWaitQueueType::LoadQueue);
    EXPECT_EQ(Test.getWaitCount(), 0);
    EXPECT_EQ(Test.getLatency(), 8);
}

#define EXPECT_CATEGORY_EQ(opcode, category, val)                                       \
    {                                                                                   \
        std::string str_(opcode);                                                       \
        auto        fromString_ = GPUInstructionInfo::category(str_);                   \
                                                                                        \
        Instruction inst_(str_, {}, {}, {}, "");                                        \
        auto        fromInstruction_ = GPUInstructionInfo::category(opcode);            \
                                                                                        \
        EXPECT_TRUE(fromString_ == fromInstruction_ && fromString_ == (val))            \
            << #category << "(" << opcode << ")\n"                                      \
            << ShowValue(fromString_) << ShowValue(fromInstruction_) << ShowValue(val); \
    }

TEST_F(GPUInstructionInfoTest, LDS)
{

    for(auto inst : {"ds_write_b128", "ds_write2_b64", "ds_write_b8"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isVCMPX, false);
        EXPECT_CATEGORY_EQ(inst, isVCMP, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, false);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSBarrier, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, false);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);

        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, true);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, true);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }

    for(auto inst : {"ds_read_b128", "ds_read2_b64", "ds_read_b8"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isVCMPX, false);
        EXPECT_CATEGORY_EQ(inst, isVCMP, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, false);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSBarrier, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, false);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);

        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, true);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, true);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }
}

TEST_F(GPUInstructionInfoTest, Scalar)
{
    for(auto inst : {"s_load_dword", "s_load_dwordx2", "s_load_dwordx16"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isVCMPX, false);
        EXPECT_CATEGORY_EQ(inst, isVCMP, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, true);
        EXPECT_CATEGORY_EQ(inst, isSMEM, true);
        EXPECT_CATEGORY_EQ(inst, isSBarrier, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, false);
        EXPECT_CATEGORY_EQ(inst, isVALU, false);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }

    for(auto inst : {"s_lshl_b64", "s_add_i32", "s_max_u32", "s_and_b64"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isVCMPX, false);
        EXPECT_CATEGORY_EQ(inst, isVCMP, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, true);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSBarrier, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, true);

        EXPECT_CATEGORY_EQ(inst, isVector, false);
        EXPECT_CATEGORY_EQ(inst, isVALU, false);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }

    for(auto inst : {"s_cbranch_vccz", "s_cbranch_vccnz"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isVCMPX, false);
        EXPECT_CATEGORY_EQ(inst, isVCMP, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, true);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSBarrier, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, true);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, false);
        EXPECT_CATEGORY_EQ(inst, isVALU, false);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }

    for(auto inst : {"s_barrier", "s_barrier_wait"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isVCMPX, false);
        EXPECT_CATEGORY_EQ(inst, isVCMP, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, true);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSBarrier, true);
        EXPECT_CATEGORY_EQ(inst, isSControl, true);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, false);
        EXPECT_CATEGORY_EQ(inst, isVALU, false);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }
}

TEST_F(GPUInstructionInfoTest, Vector)
{

    for(auto inst : {"v_mov_b32", "v_add_u32", "v_addc_co_u32", "v_or_b32"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isVCMPX, false);
        EXPECT_CATEGORY_EQ(inst, isVCMP, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, false);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSBarrier, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, true);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }

    for(auto inst : {"buffer_load_dword", "buffer_load_dwordx4", "buffer_load_short_d16"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isVCMPX, false);
        EXPECT_CATEGORY_EQ(inst, isVCMP, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, false);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSBarrier, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, false);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);

        EXPECT_CATEGORY_EQ(inst, isVMEM, true);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, true);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }

    for(auto inst : {"buffer_store_dword", "buffer_store_dwordx4", "buffer_store_short"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isVCMPX, false);
        EXPECT_CATEGORY_EQ(inst, isVCMP, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, false);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSBarrier, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, false);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);

        EXPECT_CATEGORY_EQ(inst, isVMEM, true);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, true);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }

    for(auto inst : {"v_dot2c_f32_f16", "v_dot4c_i32_i8"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, true);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isVCMPX, false);
        EXPECT_CATEGORY_EQ(inst, isVCMP, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, false);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSBarrier, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, false);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);

        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }

    for(auto inst : {"flat_load_dword", "flat_load_dwordx2"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isVCMPX, false);
        EXPECT_CATEGORY_EQ(inst, isVCMP, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, false);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSBarrier, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, false);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);

        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, true);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }

    for(auto inst : {"v_cmpx_ge_i32_e64", "v_cmpx_le_u64_e64"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isVCMPX, true);
        EXPECT_CATEGORY_EQ(inst, isVCMP, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, false);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSBarrier, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, true);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);

        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }
}

TEST_F(GPUInstructionInfoTest, AccMFMA)
{
    for(auto inst : {"v_accvgpr_read_b32"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isVCMPX, false);
        EXPECT_CATEGORY_EQ(inst, isVCMP, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, false);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSBarrier, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, true);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);

        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, true);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }

    for(auto inst : {"v_accvgpr_write_b32"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isVCMPX, false);
        EXPECT_CATEGORY_EQ(inst, isVCMP, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, false);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSBarrier, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, true);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);

        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, true);
    }

    for(auto inst : {"v_mfma_f32_16x16x16bf16", "v_mfma_f32_16x16x1f32", "v_mfma_f32_32x32x8f16"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, true);
        EXPECT_CATEGORY_EQ(inst, isVCMPX, false);
        EXPECT_CATEGORY_EQ(inst, isVCMP, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, false);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSBarrier, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, false);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);

        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }

    for(auto inst : {"v_mfma_f64_16x16x4f64", "v_mfma_f64_4x4x4f64"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, true);
        EXPECT_CATEGORY_EQ(inst, isVCMPX, false);
        EXPECT_CATEGORY_EQ(inst, isVCMP, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, false);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSBarrier, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, false);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, true);

        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }
}

TEST_F(GPUInstructionInfoTest, Signed)
{
    for(auto inst : {"v_add_u32", "v_add_u32_e32", "v_add_u32\n"})
    {
        EXPECT_CATEGORY_EQ(inst, isUIntInst, true);
        EXPECT_CATEGORY_EQ(inst, isIntInst, false);
    }

    for(auto inst : {"v_add_i32", "v_add_i32_e32"})
    {
        EXPECT_CATEGORY_EQ(inst, isUIntInst, false);
        EXPECT_CATEGORY_EQ(inst, isIntInst, true);
    }
}

TEST_F(GPUInstructionInfoTest, LaneInstructions)
{
    for(auto inst : {"v_readlane_b32"})
    {
        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, true);
        EXPECT_CATEGORY_EQ(inst, isVReadlane, true);
        EXPECT_CATEGORY_EQ(inst, isVWritelane, false);
        EXPECT_CATEGORY_EQ(inst, isVPermlane, false);
    }

    for(auto inst : {"v_writelane_b32"})
    {
        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, true);
        EXPECT_CATEGORY_EQ(inst, isVReadlane, false);
        EXPECT_CATEGORY_EQ(inst, isVWritelane, true);
        EXPECT_CATEGORY_EQ(inst, isVPermlane, false);
    }

    for(auto const& inst : {"v_permlane16_swap_b32", "v_permlane32_swap_b32"})
    {
        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, true);
        EXPECT_CATEGORY_EQ(inst, isVReadlane, false);
        EXPECT_CATEGORY_EQ(inst, isVWritelane, false);
        EXPECT_CATEGORY_EQ(inst, isVPermlane, true);
    }
}
