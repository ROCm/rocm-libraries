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
    EXPECT_EQ(GPUArchitectureLibrary::HasCapability(GPUArchitectureTarget("gfx908:xnack-"),
                                                    GPUCapability::HasExplicitNC),
              false);
    EXPECT_EQ(GPUArchitectureLibrary::HasCapability(GPUArchitectureTarget("gfx908:xnack-"),
                                                    GPUCapability::HasDirectToLds),
              false);
    EXPECT_EQ(GPUArchitectureLibrary::HasCapability(GPUArchitectureTarget("gfx908:xnack-"),
                                                    GPUCapability::HasAtomicAdd),
              true);

    EXPECT_EQ(GPUArchitectureLibrary::GetCapability(GPUArchitectureTarget("gfx803"),
                                                    GPUCapability::MaxVmcnt),
              15);
    EXPECT_EQ(GPUArchitectureLibrary::GetCapability(GPUArchitectureTarget("gfx90a"),
                                                    GPUCapability::MaxVmcnt),
              63);

    EXPECT_EQ(GPUCapability("v_fma_f16"), GPUCapability::v_fma_f16);

    EXPECT_EQ(GPUArchitectureLibrary::HasCapability(GPUArchitectureTarget("gfx908:xnack-"),
                                                    GPUCapability::v_fma_f16),
              GPUArchitectureLibrary::HasCapability("gfx908:xnack-", "v_fma_f16"));
}

TEST(GPUArchitectureTest, Xnack)
{
    EXPECT_EQ(GPUArchitectureLibrary::HasCapability("gfx1030", GPUCapability::HasXnack), false);
    EXPECT_EQ(GPUArchitectureLibrary::HasCapability("gfx1012:xnack+", GPUCapability::HasXnack),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::HasCapability("gfx1012:xnack-", GPUCapability::HasXnack),
              false);
    EXPECT_EQ(GPUArchitectureLibrary::HasCapability("gfx908:xnack+", GPUCapability::HasXnack),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::HasCapability("gfx908:xnack-", GPUCapability::HasXnack),
              false);
}

TEST(GPUArchitectureTest, WaveFrontSize)
{
    EXPECT_EQ(GPUArchitectureLibrary::HasCapability("gfx1030", GPUCapability::HasWave64), true);
    EXPECT_EQ(GPUArchitectureLibrary::HasCapability("gfx1012:xnack+", GPUCapability::HasWave64),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::HasCapability("gfx1012:xnack-", GPUCapability::HasWave64),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::HasCapability("gfx908:xnack+", GPUCapability::HasWave64),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::HasCapability("gfx908:xnack-", GPUCapability::HasWave64),
              true);

    EXPECT_EQ(GPUArchitectureLibrary::HasCapability("gfx908:xnack+", GPUCapability::HasWave32),
              false);
    EXPECT_EQ(GPUArchitectureLibrary::HasCapability("gfx908:xnack-", GPUCapability::HasWave32),
              false);
    EXPECT_EQ(GPUArchitectureLibrary::HasCapability("gfx1030", GPUCapability::HasWave32), true);
    EXPECT_EQ(GPUArchitectureLibrary::HasCapability("gfx1012:xnack+", GPUCapability::HasWave32),
              true);
    EXPECT_EQ(GPUArchitectureLibrary::HasCapability("gfx1012:xnack-", GPUCapability::HasWave32),
              true);

    EXPECT_EQ(GPUArchitectureLibrary::GetCapability("gfx90a", GPUCapability::DefaultWavefrontSize),
              64);
    EXPECT_EQ(GPUArchitectureLibrary::GetCapability("gfx1030", GPUCapability::DefaultWavefrontSize),
              32);
}

TEST(GPUArchitectureTest, Validate90aInstructions)
{
    //Check that some flat instructions belong to both LGKMDSQueue and VMQueue.
    std::vector<std::string> test_insts
        = {"flat_store_dword", "flat_store_dwordx2", "flat_store_dwordx3", "flat_store_dwordx4"};
    for(std::string& inst : test_insts)
    {
        auto queues = GPUArchitectureLibrary::GetInstructionInfo("gfx90a", inst).getWaitQueues();
        EXPECT_NE(std::find(queues.begin(), queues.end(), GPUWaitQueueType::VMQueue), queues.end());
        EXPECT_NE(std::find(queues.begin(), queues.end(), GPUWaitQueueType::LGKMDSQueue),
                  queues.end());
        EXPECT_EQ(GPUArchitectureLibrary::GetInstructionInfo("gfx90a", inst).getWaitCount(), 0);
    }

    EXPECT_EQ(GPUArchitectureLibrary::GetInstructionInfo("gfx90a", "exp").getWaitCount(), 1);
    EXPECT_EQ(GPUArchitectureLibrary::GetInstructionInfo("gfx90a", "exp").getWaitQueues()[0],
              GPUWaitQueueType::EXPQueue);
    EXPECT_EQ(GPUArchitectureLibrary::GetInstructionInfo("gfx90a", "exp").getLatency(), 0);

    EXPECT_EQ(GPUArchitectureLibrary::GetInstructionInfo("gfx90a", "s_sendmsg").getWaitCount(), 1);
    EXPECT_EQ(GPUArchitectureLibrary::GetInstructionInfo("gfx90a", "s_sendmsg").getWaitQueues()[0],
              GPUWaitQueueType::LGKMSendMsgQueue);
    EXPECT_EQ(GPUArchitectureLibrary::GetInstructionInfo("gfx90a", "s_sendmsg").getLatency(), 0);

    EXPECT_EQ(
        GPUArchitectureLibrary::GetInstructionInfo("gfx90a", "v_mfma_f32_32x32x2f32").getLatency(),
        16);
    EXPECT_EQ(
        GPUArchitectureLibrary::GetInstructionInfo("gfx90a", "v_accvgpr_read_b32").getLatency(), 1);
    EXPECT_EQ(
        GPUArchitectureLibrary::GetInstructionInfo("gfx90a", "v_accvgpr_write_b32").getLatency(),
        2);
    EXPECT_EQ(GPUArchitectureLibrary::GetInstructionInfo("gfx90a", "v_accvgpr_write").getLatency(),
              2);
}

TEST(GPUArchitectureTest, CheckDefFile)
{
    EXPECT_EQ(
        GPUArchitectureLibrary::GetDevice("gfx90a").HasCapability(GPUCapability::SupportedISA),
        true);
}

TEST(GPUArchitectureTest, TargetComparison)
{
    EXPECT_EQ(GPUArchitectureTarget("gfx908") == GPUArchitectureTarget("gfx908"), true);
    EXPECT_EQ(GPUArchitectureTarget("gfx90a") != GPUArchitectureTarget("gfx908"), true);
}
