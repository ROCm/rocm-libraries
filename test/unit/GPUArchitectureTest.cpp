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
              false);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(
                  {GPUArchitectureGFX::GFX908, {.xnack = false}}, GPUCapability::HasAtomicAdd),
              true);

    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->GetCapability({GPUArchitectureGFX::GFX803},
                                                                   GPUCapability::MaxVmcnt),
              15);
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
    //Check that some flat instructions belong to both LGKMDSQueue and VMQueue.
    std::vector<std::string> test_insts
        = {"flat_store_dword", "flat_store_dwordx2", "flat_store_dwordx3", "flat_store_dwordx4"};
    for(std::string& inst : test_insts)
    {
        auto queues = GPUArchitectureLibrary::getInstance()
                          ->GetInstructionInfo({GPUArchitectureGFX::GFX90A}, inst)
                          .getWaitQueues();
        EXPECT_NE(std::find(queues.begin(), queues.end(), GPUWaitQueueType::VMQueue), queues.end());
        EXPECT_NE(std::find(queues.begin(), queues.end(), GPUWaitQueueType::LGKMDSQueue),
                  queues.end());
        EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                      ->GetInstructionInfo({GPUArchitectureGFX::GFX90A}, inst)
                      .getWaitCount(),
                  0);
    }

    EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                  ->GetInstructionInfo({GPUArchitectureGFX::GFX90A}, "exp")
                  .getWaitCount(),
              1);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                  ->GetInstructionInfo({GPUArchitectureGFX::GFX90A}, "exp")
                  .getWaitQueues()[0],
              GPUWaitQueueType::EXPQueue);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                  ->GetInstructionInfo({GPUArchitectureGFX::GFX90A}, "exp")
                  .getLatency(),
              0);

    EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                  ->GetInstructionInfo({GPUArchitectureGFX::GFX90A}, "s_sendmsg")
                  .getWaitCount(),
              1);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                  ->GetInstructionInfo({GPUArchitectureGFX::GFX90A}, "s_sendmsg")
                  .getWaitQueues()[0],
              GPUWaitQueueType::LGKMSendMsgQueue);
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
