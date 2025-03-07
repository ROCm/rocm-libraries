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

#include <algorithm>

#include <vector>

#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>

#include "SimpleFixture.hpp"

using namespace rocRoller;

class GPUArchitectureTest : public SimpleFixture
{
};

TEST_F(GPUArchitectureTest, EmptyConstructor)
{
    GPUArchitecture Test;
    EXPECT_EQ(Test.HasCapability(GPUCapability::SupportedISA), false);
    Test.AddCapability(GPUCapability::SupportedISA, 0);
    EXPECT_EQ(Test.HasCapability(GPUCapability::SupportedISA), true);
    EXPECT_EQ(Test.HasCapability(GPUCapability::MaxVmcnt), false);
    Test.AddCapability(GPUCapability::MaxVmcnt, 15);
    EXPECT_EQ(Test.HasCapability(GPUCapability::MaxVmcnt), true);
    EXPECT_EQ(Test.GetCapability(GPUCapability::MaxVmcnt), 15);
}

TEST_F(GPUArchitectureTest, TargetConstructor)
{
    GPUArchitecture Test({GPUArchitectureGFX::GFX908});
    EXPECT_EQ(Test.HasCapability(GPUCapability::SupportedISA), false);
    Test.AddCapability(GPUCapability::SupportedISA, 0);
    EXPECT_EQ(Test.HasCapability(GPUCapability::SupportedISA), true);
    EXPECT_EQ(Test.HasCapability(GPUCapability::MaxVmcnt), false);
    Test.AddCapability(GPUCapability::MaxVmcnt, 15);
    EXPECT_EQ(Test.HasCapability(GPUCapability::MaxVmcnt), true);
    EXPECT_EQ(Test.GetCapability(GPUCapability::MaxVmcnt), 15);
}

TEST_F(GPUArchitectureTest, FullConstructor)
{
    std::map<GPUCapability, int> capabilities
        = {{GPUCapability::SupportedSource, 0}, {GPUCapability::MaxLgkmcnt, 63}};
    GPUArchitecture Test({GPUArchitectureGFX::GFX908}, capabilities, {});
    EXPECT_EQ(Test.HasCapability("SupportedSource"), true);
    EXPECT_EQ(Test.HasCapability("MaxLgkmcnt"), true);
    EXPECT_EQ(Test.GetCapability("MaxLgkmcnt"), 63);

    EXPECT_EQ(Test.HasCapability(GPUCapability::SupportedISA), false);
    Test.AddCapability(GPUCapability::SupportedISA, 0);
    EXPECT_EQ(Test.HasCapability(GPUCapability::SupportedISA), true);
    EXPECT_EQ(Test.HasCapability(GPUCapability::MaxVmcnt), false);
    Test.AddCapability(GPUCapability::MaxVmcnt, 15);
    EXPECT_EQ(Test.HasCapability(GPUCapability::MaxVmcnt), true);
    EXPECT_EQ(Test.GetCapability(GPUCapability::MaxVmcnt), 15);
}

TEST_F(GPUArchitectureTest, ValidateGeneratedDef)
{
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(
                  {GPUArchitectureGFX::GFX908, {.xnack = false}}, GPUCapability::HasExplicitNC),
              false);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(
                  {GPUArchitectureGFX::GFX908, {.xnack = false}}, GPUCapability::HasDirectToLds),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(
                  {GPUArchitectureGFX::GFX90A, {.xnack = false}}, GPUCapability::HasDirectToLds),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(
                  {GPUArchitectureGFX::GFX908, {.xnack = false}}, GPUCapability::HasAtomicAdd),
              true);

    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->GetCapability({GPUArchitectureGFX::GFX90A},
                                                                   GPUCapability::MaxVmcnt),
              63);

    EXPECT_EQ(GPUCapability("v_fma_f16"), GPUCapability::v_fma_f16);

    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(
                  {GPUArchitectureGFX::GFX908, {.xnack = false}}, GPUCapability::v_fma_f16),
              GPUArchitectureLibrary::getInstance()->HasCapability(
                  {GPUArchitectureGFX::GFX908, {.xnack = false}}, "v_fma_f16"));
}

TEST_F(GPUArchitectureTest, Xnack)
{
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability({GPUArchitectureGFX::GFX1030},
                                                                   GPUCapability::HasXnack),
              false);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(
                  {GPUArchitectureGFX::GFX1012, {.xnack = true}}, GPUCapability::HasXnack),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(
                  {GPUArchitectureGFX::GFX1012, {.xnack = false}}, GPUCapability::HasXnack),
              false);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(
                  {GPUArchitectureGFX::GFX1012, {.xnack = true}}, GPUCapability::HasXnack),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(
                  {GPUArchitectureGFX::GFX1012, {.xnack = false}}, GPUCapability::HasXnack),
              false);
}

TEST_F(GPUArchitectureTest, WaveFrontSize)
{
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability({GPUArchitectureGFX::GFX1030},
                                                                   GPUCapability::HasWave64),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(
                  {GPUArchitectureGFX::GFX1012, {.xnack = true}}, GPUCapability::HasWave64),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(
                  {GPUArchitectureGFX::GFX1012, {.xnack = false}}, GPUCapability::HasWave64),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(
                  {GPUArchitectureGFX::GFX908, {.xnack = true}}, GPUCapability::HasWave64),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(
                  {GPUArchitectureGFX::GFX908, {.xnack = false}}, GPUCapability::HasWave64),
              true);

    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(
                  {GPUArchitectureGFX::GFX908, {.xnack = true}}, GPUCapability::HasWave32),
              false);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(
                  {GPUArchitectureGFX::GFX908, {.xnack = false}}, GPUCapability::HasWave32),
              false);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability({GPUArchitectureGFX::GFX1030},
                                                                   GPUCapability::HasWave32),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(
                  {GPUArchitectureGFX::GFX1012, {.xnack = true}}, GPUCapability::HasWave32),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(
                  {GPUArchitectureGFX::GFX1012, {.xnack = false}}, GPUCapability::HasWave32),
              true);

    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->GetCapability(
                  {GPUArchitectureGFX::GFX90A}, GPUCapability::DefaultWavefrontSize),
              64);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->GetCapability(
                  {GPUArchitectureGFX::GFX1030}, GPUCapability::DefaultWavefrontSize),
              32);
}

TEST_F(GPUArchitectureTest, Validate90aInstructions)
{
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                  ->GetInstructionInfo({GPUArchitectureGFX::GFX90A}, "s_sendmsg")
                  .getWaitCount(),
              1);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                  ->GetInstructionInfo({GPUArchitectureGFX::GFX90A}, "s_sendmsg")
                  .getWaitQueues()[0],
              GPUWaitQueueType::SendMsgQueue);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                  ->GetInstructionInfo({GPUArchitectureGFX::GFX90A}, "s_sendmsg")
                  .getLatency(),
              0);

    EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                  ->GetInstructionInfo({GPUArchitectureGFX::GFX90A}, "v_mfma_f32_32x32x2f32")
                  .getLatency(),
              16);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                  ->GetInstructionInfo({GPUArchitectureGFX::GFX90A}, "v_accvgpr_read_b32")
                  .getLatency(),
              1);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                  ->GetInstructionInfo({GPUArchitectureGFX::GFX90A}, "v_accvgpr_write_b32")
                  .getLatency(),
              2);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                  ->GetInstructionInfo({GPUArchitectureGFX::GFX90A}, "v_accvgpr_write")
                  .getLatency(),
              2);

    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability({GPUArchitectureGFX::GFX90A},
                                                                   GPUCapability::v_mac_f32),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability({GPUArchitectureGFX::GFX90A},
                                                                   GPUCapability::v_fmac_f32),
              true);
}

TEST_F(GPUArchitectureTest, Validate908Instructions)
{

    EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                  ->GetInstructionInfo({GPUArchitectureGFX::GFX908}, "exp")
                  .getWaitCount(),
              1);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                  ->GetInstructionInfo({GPUArchitectureGFX::GFX908}, "exp")
                  .getWaitQueues()[0],
              GPUWaitQueueType::EXPQueue);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                  ->GetInstructionInfo({GPUArchitectureGFX::GFX908}, "exp")
                  .getLatency(),
              0);

    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability({GPUArchitectureGFX::GFX90A},
                                                                   GPUCapability::v_mac_f32),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability({GPUArchitectureGFX::GFX90A},
                                                                   GPUCapability::v_fmac_f32),
              true);
}

TEST_F(GPUArchitectureTest, Validate94xInstructions)
{
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability({GPUArchitectureGFX::GFX942},
                                                                   GPUCapability::v_mac_f32),
              false);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability({GPUArchitectureGFX::GFX942},
                                                                   GPUCapability::v_fmac_f32),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability({GPUArchitectureGFX::GFX942},
                                                                   GPUCapability::v_mov_b64),
              true);
}

TEST_F(GPUArchitectureTest, Validate95xInstructions)
{
    // Verify permlane instructions exist
    EXPECT_NO_THROW(GPUArchitectureLibrary::getInstance()->GetInstructionInfo(
        {GPUArchitectureGFX::GFX950}, "v_permlane16_swap_b32"));

    EXPECT_NO_THROW(GPUArchitectureLibrary::getInstance()->GetInstructionInfo(
        {GPUArchitectureGFX::GFX950}, "v_permlane32_swap_b32"));
}

TEST_F(GPUArchitectureTest, MFMA)
{
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(
                  {GPUArchitectureGFX::GFX942, {.sramecc = true}}, GPUCapability::HasMFMA_fp8),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability({GPUArchitectureGFX::GFX90A},
                                                                   GPUCapability::HasMFMA_fp8),
              false);
}

TEST_F(GPUArchitectureTest, TargetComparison)
{
    EXPECT_EQ(GPUArchitectureTarget{GPUArchitectureGFX::GFX908}
                  == GPUArchitectureTarget{GPUArchitectureGFX::GFX908},
              true);
    EXPECT_EQ(GPUArchitectureTarget{GPUArchitectureGFX::GFX90A}
                  != GPUArchitectureTarget{GPUArchitectureGFX::GFX908},
              true);
}

TEST_F(GPUArchitectureTest, ToLLVMString)
{
    {
        GPUArchitectureTarget target = {GPUArchitectureGFX::GFX908};
        EXPECT_EQ(target.features.toString(), "");
        EXPECT_EQ(toString(target.features), "");
        EXPECT_EQ(target.features.toLLVMString(), "");
    }

    {
        GPUArchitectureTarget target = {GPUArchitectureGFX::GFX908, {.sramecc = true}};
        EXPECT_EQ(target.features.toString(), "sramecc+");
        EXPECT_EQ(toString(target.features), "sramecc+");
        EXPECT_EQ(target.features.toLLVMString(), "+sramecc");
    }

    {
        GPUArchitectureTarget target = {GPUArchitectureGFX::GFX908, {.xnack = true}};
        EXPECT_EQ(target.features.toString(), "xnack+");
        EXPECT_EQ(toString(target.features), "xnack+");
        EXPECT_EQ(target.features.toLLVMString(), "+xnack");
    }

    {
        GPUArchitectureTarget target
            = {GPUArchitectureGFX::GFX908, {.sramecc = true, .xnack = true}};
        EXPECT_EQ(target.features.toString(), "sramecc+xnack+");
        EXPECT_EQ(toString(target.features), "sramecc+xnack+");
        EXPECT_EQ(target.features.toLLVMString(), "+xnack,+sramecc");
    }
}

TEST_F(GPUArchitectureTest, F8Mode)
{
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(
                  {GPUArchitectureTarget{GPUArchitectureGFX::GFX942}}, GPUCapability::HasNaNoo),
              true);
    auto allISAs = GPUArchitectureLibrary::getInstance()->getAllSupportedISAs();
    auto gfx942  = std::remove_if(allISAs.begin(), allISAs.end(), [](const auto& isa) {
        return isa.gfx == GPUArchitectureGFX::GFX942;
    });
    for(auto I = allISAs.begin(); I != gfx942; I++)
    {
        EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(*I, GPUCapability::HasNaNoo),
                  false);
    }
}
