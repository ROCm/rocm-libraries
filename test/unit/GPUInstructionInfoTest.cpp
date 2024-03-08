#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/CodeGen/WaitCount.hpp>
#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>

using namespace rocRoller;

TEST(GPUInstructionInfoTest, BasicTest)
{
    GPUInstructionInfo Test("test",
                            0,
                            {GPUWaitQueueType::LGKMDSQueue, GPUWaitQueueType::VMQueue},
                            16,
                            true,
                            false,
                            8192);
    EXPECT_EQ(Test.getInstruction(), "test");
    EXPECT_EQ(Test.getWaitQueues().size(), 2);
    EXPECT_EQ(Test.getWaitQueues()[0], GPUWaitQueueType::LGKMDSQueue);
    EXPECT_EQ(Test.getWaitQueues()[1], GPUWaitQueueType::VMQueue);
    EXPECT_EQ(Test.getWaitCount(), 0);
    EXPECT_EQ(Test.getLatency(), 16);
    EXPECT_EQ(Test.maxLiteralValue(), 8192);
}

TEST(GPUInstructionInfoTest, BasicTestLatency)
{
    GPUInstructionInfo Test(
        "test", 0, {GPUWaitQueueType::LGKMDSQueue, GPUWaitQueueType::VMQueue}, 8);
    EXPECT_EQ(Test.getInstruction(), "test");
    EXPECT_EQ(Test.getWaitQueues().size(), 2);
    EXPECT_EQ(Test.getWaitQueues()[0], GPUWaitQueueType::LGKMDSQueue);
    EXPECT_EQ(Test.getWaitQueues()[1], GPUWaitQueueType::VMQueue);
    EXPECT_EQ(Test.getWaitCount(), 0);
    EXPECT_EQ(Test.getLatency(), 8);
}
