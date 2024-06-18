#include <algorithm>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>

#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>

using namespace rocRoller;

TEST(GPUArchitectureTest, EmptyConstructor)
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

TEST(GPUArchitectureTest, TargetConstructor)
{
    GPUArchitecture Test(GPUArchitectureTarget("gfx908"));
    EXPECT_EQ(Test.HasCapability(GPUCapability::SupportedISA), false);
    Test.AddCapability(GPUCapability::SupportedISA, 0);
    EXPECT_EQ(Test.HasCapability(GPUCapability::SupportedISA), true);
    EXPECT_EQ(Test.HasCapability(GPUCapability::MaxVmcnt), false);
    Test.AddCapability(GPUCapability::MaxVmcnt, 15);
    EXPECT_EQ(Test.HasCapability(GPUCapability::MaxVmcnt), true);
    EXPECT_EQ(Test.GetCapability(GPUCapability::MaxVmcnt), 15);
}

TEST(GPUArchitectureTest, FullConstructor)
{
    std::map<GPUCapability, int> capabilities
        = {{GPUCapability::SupportedSource, 0}, {GPUCapability::MaxLgkmcnt, 63}};
    GPUArchitecture Test(GPUArchitectureTarget("gfx908"), capabilities, {});
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

TEST(GPUArchitectureTest, ValidateGeneratedDef)
{
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(
                  GPUArchitectureTarget("gfx908:xnack-"), GPUCapability::HasExplicitNC),
              false);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(
                  GPUArchitectureTarget("gfx908:xnack-"), GPUCapability::HasDirectToLds),
              false);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(
                  GPUArchitectureTarget("gfx908:xnack-"), GPUCapability::HasAtomicAdd),
              true);

    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->GetCapability(GPUArchitectureTarget("gfx803"),
                                                                   GPUCapability::MaxVmcnt),
              15);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->GetCapability(GPUArchitectureTarget("gfx90a"),
                                                                   GPUCapability::MaxVmcnt),
              63);

    EXPECT_EQ(GPUCapability("v_fma_f16"), GPUCapability::v_fma_f16);

    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(
                  GPUArchitectureTarget("gfx908:xnack-"), GPUCapability::v_fma_f16),
              GPUArchitectureLibrary::getInstance()->HasCapability("gfx908:xnack-", "v_fma_f16"));
}

TEST(GPUArchitectureTest, Xnack)
{
    EXPECT_EQ(
        GPUArchitectureLibrary::getInstance()->HasCapability("gfx1030", GPUCapability::HasXnack),
        false);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability("gfx1012:xnack+",
                                                                   GPUCapability::HasXnack),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability("gfx1012:xnack-",
                                                                   GPUCapability::HasXnack),
              false);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability("gfx908:xnack+",
                                                                   GPUCapability::HasXnack),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability("gfx908:xnack-",
                                                                   GPUCapability::HasXnack),
              false);
}

TEST(GPUArchitectureTest, WaveFrontSize)
{
    EXPECT_EQ(
        GPUArchitectureLibrary::getInstance()->HasCapability("gfx1030", GPUCapability::HasWave64),
        true);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability("gfx1012:xnack+",
                                                                   GPUCapability::HasWave64),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability("gfx1012:xnack-",
                                                                   GPUCapability::HasWave64),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability("gfx908:xnack+",
                                                                   GPUCapability::HasWave64),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability("gfx908:xnack-",
                                                                   GPUCapability::HasWave64),
              true);

    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability("gfx908:xnack+",
                                                                   GPUCapability::HasWave32),
              false);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability("gfx908:xnack-",
                                                                   GPUCapability::HasWave32),
              false);
    EXPECT_EQ(
        GPUArchitectureLibrary::getInstance()->HasCapability("gfx1030", GPUCapability::HasWave32),
        true);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability("gfx1012:xnack+",
                                                                   GPUCapability::HasWave32),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability("gfx1012:xnack-",
                                                                   GPUCapability::HasWave32),
              true);

    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->GetCapability(
                  "gfx90a", GPUCapability::DefaultWavefrontSize),
              64);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->GetCapability(
                  "gfx1030", GPUCapability::DefaultWavefrontSize),
              32);
}

TEST(GPUArchitectureTest, Validate90aInstructions)
{
    //Check that some flat instructions belong to both LGKMDSQueue and VMQueue.
    std::vector<std::string> test_insts
        = {"flat_store_dword", "flat_store_dwordx2", "flat_store_dwordx3", "flat_store_dwordx4"};
    for(std::string& inst : test_insts)
    {
        auto queues = GPUArchitectureLibrary::getInstance()
                          ->GetInstructionInfo("gfx90a", inst)
                          .getWaitQueues();
        EXPECT_NE(std::find(queues.begin(), queues.end(), GPUWaitQueueType::VMQueue), queues.end());
        EXPECT_NE(std::find(queues.begin(), queues.end(), GPUWaitQueueType::LGKMDSQueue),
                  queues.end());
        EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                      ->GetInstructionInfo("gfx90a", inst)
                      .getWaitCount(),
                  0);
    }

    EXPECT_EQ(
        GPUArchitectureLibrary::getInstance()->GetInstructionInfo("gfx90a", "exp").getWaitCount(),
        1);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                  ->GetInstructionInfo("gfx90a", "exp")
                  .getWaitQueues()[0],
              GPUWaitQueueType::EXPQueue);
    EXPECT_EQ(
        GPUArchitectureLibrary::getInstance()->GetInstructionInfo("gfx90a", "exp").getLatency(), 0);

    EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                  ->GetInstructionInfo("gfx90a", "s_sendmsg")
                  .getWaitCount(),
              1);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                  ->GetInstructionInfo("gfx90a", "s_sendmsg")
                  .getWaitQueues()[0],
              GPUWaitQueueType::LGKMSendMsgQueue);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                  ->GetInstructionInfo("gfx90a", "s_sendmsg")
                  .getLatency(),
              0);

    EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                  ->GetInstructionInfo("gfx90a", "v_mfma_f32_32x32x2f32")
                  .getLatency(),
              16);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                  ->GetInstructionInfo("gfx90a", "v_accvgpr_read_b32")
                  .getLatency(),
              1);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                  ->GetInstructionInfo("gfx90a", "v_accvgpr_write_b32")
                  .getLatency(),
              2);
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()
                  ->GetInstructionInfo("gfx90a", "v_accvgpr_write")
                  .getLatency(),
              2);

    EXPECT_EQ(
        GPUArchitectureLibrary::getInstance()->HasCapability("gfx90a", GPUCapability::v_mac_f32),
        true);
    EXPECT_EQ(
        GPUArchitectureLibrary::getInstance()->HasCapability("gfx90a", GPUCapability::v_fmac_f32),
        true);
}

TEST(GPUArchitectureTest, Validate908Instructions)
{
    EXPECT_EQ(
        GPUArchitectureLibrary::getInstance()->HasCapability("gfx908", GPUCapability::v_mac_f32),
        true);
    EXPECT_EQ(
        GPUArchitectureLibrary::getInstance()->HasCapability("gfx908", GPUCapability::v_fmac_f32),
        true);
}

TEST(GPUArchitectureTest, Validate94xInstructions)
{
    EXPECT_EQ(
        GPUArchitectureLibrary::getInstance()->HasCapability("gfx942", GPUCapability::v_mac_f32),
        false);
    EXPECT_EQ(
        GPUArchitectureLibrary::getInstance()->HasCapability("gfx942", GPUCapability::v_fmac_f32),
        true);
}

TEST(GPUArchitectureTest, MFMA)
{
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability("gfx942:sramecc+",
                                                                   GPUCapability::HasMFMA_fp8),
              true);
    EXPECT_EQ(
        GPUArchitectureLibrary::getInstance()->HasCapability("gfx90a", GPUCapability::HasMFMA_fp8),
        false);
}

TEST(GPUArchitectureTest, CheckDefFile)
{
    EXPECT_EQ(GPUArchitectureLibrary::getInstance()->GetDevice("gfx90a").HasCapability(
                  GPUCapability::SupportedISA),
              true);
}

TEST(GPUArchitectureTest, TargetComparison)
{
    EXPECT_EQ(GPUArchitectureTarget("gfx908") == GPUArchitectureTarget("gfx908"), true);
    EXPECT_EQ(GPUArchitectureTarget("gfx90a") != GPUArchitectureTarget("gfx908"), true);
}

TEST(GPUArchitectureTest, F8Mode)
{
    EXPECT_EQ(
        GPUArchitectureLibrary::getInstance()->HasCapability("gfx942", GPUCapability::HasNaNoo),
        true);
    auto allISAs = GPUArchitectureLibrary::getInstance()->getAllSupportedISAs();
    auto gfx942  = std::remove_if(allISAs.begin(), allISAs.end(), [](const std::string& isa) {
        return isa.starts_with("gfx942");
    });
    for(auto I = allISAs.begin(); I != gfx942; I++)
    {
        EXPECT_EQ(GPUArchitectureLibrary::getInstance()->HasCapability(*I, GPUCapability::HasNaNoo),
                  false);
    }
}
