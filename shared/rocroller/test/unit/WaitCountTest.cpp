// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CodeGen/WaitCount.hpp>
#include <rocRoller/Utilities/Logging.hpp>

#include "SimpleFixture.hpp"
#include <common/SourceMatcher.hpp>

using namespace rocRoller;

class WaitCountTest : public SimpleFixture
{
};

TEST_F(WaitCountTest, Basic)
{
    GPUArchitecture arch{GPUArchitectureTarget{GPUArchitectureGFX::GFX90A}};
    auto            wc = WaitCount::KMCnt(arch, 2);

    EXPECT_EQ(2, wc.kmcnt());
    EXPECT_EQ(-1, wc.dscnt());
    EXPECT_EQ(-1, wc.loadcnt());
    EXPECT_EQ(-1, wc.storecnt());
    EXPECT_EQ(-1, wc.vscnt());
    EXPECT_EQ(-1, wc.expcnt());

    EXPECT_EQ("s_waitcnt lgkmcnt(2)\n", wc.toString(LogLevel::Terse));
}

TEST_F(WaitCountTest, Combine)
{
    GPUArchitecture arch{GPUArchitectureTarget{GPUArchitectureGFX::GFX90A}};
    // Add capability so VSCnt can be generated
    arch.AddCapability(GPUCapability::SeparateVscnt, 0);
    arch.AddCapability(GPUCapability::HasExpcnt, 0);
    auto wc = WaitCount::KMCnt(arch, 2);

    wc.combine(WaitCount::LoadCnt(arch, 4));

    EXPECT_EQ(2, wc.kmcnt());
    EXPECT_EQ(4, wc.loadcnt());
    EXPECT_EQ(-1, wc.storecnt());
    EXPECT_EQ(-1, wc.dscnt());
    EXPECT_EQ(-1, wc.vscnt());
    EXPECT_EQ(-1, wc.expcnt());

    wc.combine(WaitCount::KMCnt(arch, 4));

    EXPECT_EQ(2, wc.kmcnt());
    EXPECT_EQ(4, wc.loadcnt());
    EXPECT_EQ(-1, wc.storecnt());
    EXPECT_EQ(-1, wc.dscnt());
    EXPECT_EQ(-1, wc.vscnt());
    EXPECT_EQ(-1, wc.expcnt());

    wc.combine(WaitCount::KMCnt(arch, 1, "Wait for LDS"));
    wc.combine(WaitCount::DSCnt(arch, 1, "Wait for LDS"));

    EXPECT_EQ(1, wc.kmcnt());
    EXPECT_EQ(1, wc.dscnt());
    EXPECT_EQ(4, wc.loadcnt());
    EXPECT_EQ(-1, wc.storecnt());
    EXPECT_EQ(-1, wc.vscnt());
    EXPECT_EQ(-1, wc.expcnt());

    // -1 is no wait, shouldn't affect the value.
    wc.combine(WaitCount::KMCnt(arch, -1));
    wc.combine(WaitCount::DSCnt(arch, -1));

    EXPECT_EQ(1, wc.kmcnt());
    EXPECT_EQ(1, wc.dscnt());
    EXPECT_EQ(4, wc.loadcnt());
    EXPECT_EQ(-1, wc.storecnt());
    EXPECT_EQ(-1, wc.vscnt());
    EXPECT_EQ(-1, wc.expcnt());

    wc.combine(WaitCount::VSCnt(arch, 2, "Wait for store"));

    EXPECT_EQ(1, wc.kmcnt());
    EXPECT_EQ(1, wc.dscnt());
    EXPECT_EQ(4, wc.loadcnt());
    EXPECT_EQ(2, wc.vscnt());
    EXPECT_EQ(-1, wc.storecnt());
    EXPECT_EQ(-1, wc.expcnt());

    wc.combine(WaitCount::EXPCnt(arch, 20));

    EXPECT_EQ(1, wc.kmcnt());
    EXPECT_EQ(1, wc.dscnt());
    EXPECT_EQ(4, wc.loadcnt());
    EXPECT_EQ(2, wc.vscnt());
    EXPECT_EQ(20, wc.expcnt());
    EXPECT_EQ(-1, wc.storecnt());

    auto stringValue = wc.toString(LogLevel::Debug);

    EXPECT_THAT(stringValue, testing::HasSubstr("// Wait for LDS"));
    EXPECT_THAT(stringValue, testing::HasSubstr("// Wait for store"));

    EXPECT_THAT(stringValue, testing::HasSubstr("s_waitcnt"));
    EXPECT_THAT(stringValue, testing::HasSubstr("vmcnt(4)"));
    EXPECT_THAT(stringValue, testing::HasSubstr("lgkmcnt(1)"));
    EXPECT_THAT(stringValue, testing::HasSubstr("expcnt(20)"));

    EXPECT_THAT(stringValue, testing::HasSubstr("s_waitcnt_vscnt 2"));

    stringValue = wc.toString(LogLevel::Terse);

    // No comments for terse version
    EXPECT_THAT(stringValue, testing::Not(testing::HasSubstr("// Wait for LDS")));
    EXPECT_THAT(stringValue, testing::Not(testing::HasSubstr("// Wait for store")));

    // Still need all the functional parts.
    EXPECT_THAT(stringValue, testing::HasSubstr("s_waitcnt"));
    EXPECT_THAT(stringValue, testing::HasSubstr("vmcnt(4)"));
    EXPECT_THAT(stringValue, testing::HasSubstr("lgkmcnt(1)"));
    EXPECT_THAT(stringValue, testing::HasSubstr("expcnt(20)"));

    EXPECT_THAT(stringValue, testing::HasSubstr("s_waitcnt_vscnt 2"));

    {
        WaitCount wc = WaitCount::SyncQueue(
            arch, GPUWaitQueueType::SMemQueue, "DEBUG: Wait for scalar queue");
        EXPECT_EQ(wc.queuesToSync().count(), 1);
        EXPECT_EQ(wc.queuesToSync()[GPUWaitQueueType::SMemQueue], true);
    }

    {
        WaitCount wc = WaitCount::SyncQueues(
            arch,
            EnumBitset<GPUWaitQueueType>{GPUWaitQueueType::SMemQueue, GPUWaitQueueType::DSQueue},
            "DEBUG: Wait for scalar and LDS queues");
        EXPECT_EQ(wc.queuesToSync().count(), 2);
        EXPECT_EQ(wc.queuesToSync()[GPUWaitQueueType::SMemQueue], true);
        EXPECT_EQ(wc.queuesToSync()[GPUWaitQueueType::DSQueue], true);
    }
}

/**
 * This test checks that if an architecture has the SeparateVscnt capability, then a waitcnt zero will
 * produce a wait zero for vscnt, and if it doesn't have that capability, then no vscnt wait is produced.
 **/
TEST_F(WaitCountTest, VSCnt)
{
    GPUArchitecture TestWithVSCnt{GPUArchitectureTarget{GPUArchitectureGFX::GFX90A}};
    TestWithVSCnt.AddCapability(GPUCapability::SupportedISA, 0);
    TestWithVSCnt.AddCapability(GPUCapability::SeparateVscnt, 0);
    TestWithVSCnt.AddCapability(GPUCapability::HasExpcnt, 0);
    EXPECT_EQ(TestWithVSCnt.HasCapability(GPUCapability::SupportedISA), true);
    EXPECT_EQ(TestWithVSCnt.HasCapability(GPUCapability::SeparateVscnt), true);

    auto wcWithVSCnt = WaitCount::Zero(TestWithVSCnt);

    EXPECT_EQ(0, wcWithVSCnt.kmcnt());
    EXPECT_EQ(0, wcWithVSCnt.dscnt());
    EXPECT_EQ(0, wcWithVSCnt.loadcnt());
    EXPECT_EQ(0, wcWithVSCnt.storecnt());
    EXPECT_EQ(0, wcWithVSCnt.vscnt());
    EXPECT_EQ(0, wcWithVSCnt.expcnt());

    std::string expectedWithVSCnt = R"(
                                        s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)
                                        s_waitcnt_vscnt 0
                                    )";

    EXPECT_EQ(NormalizedSource(wcWithVSCnt.toString(LogLevel::Debug)),
              NormalizedSource(expectedWithVSCnt));

    GPUArchitecture TestNoVSCnt{GPUArchitectureTarget{GPUArchitectureGFX::GFX90A}};
    TestNoVSCnt.AddCapability(GPUCapability::SupportedISA, 0);
    TestNoVSCnt.AddCapability(GPUCapability::HasExpcnt, 0);
    EXPECT_EQ(TestNoVSCnt.HasCapability(GPUCapability::SupportedISA), true);
    EXPECT_EQ(TestNoVSCnt.HasCapability(GPUCapability::SeparateVscnt), false);

    auto wcNoVSCnt = WaitCount::Zero(TestNoVSCnt);

    EXPECT_EQ(0, wcNoVSCnt.kmcnt());
    EXPECT_EQ(0, wcNoVSCnt.dscnt());
    EXPECT_EQ(0, wcNoVSCnt.loadcnt());
    EXPECT_EQ(0, wcNoVSCnt.storecnt());
    EXPECT_EQ(-1, wcNoVSCnt.vscnt());
    EXPECT_EQ(0, wcNoVSCnt.expcnt());

    std::string expectedNoVSCnt = R"(
                                        s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)
                                    )";

    EXPECT_EQ(NormalizedSource(wcNoVSCnt.toString(LogLevel::Debug)),
              NormalizedSource(expectedNoVSCnt));
}

TEST_F(WaitCountTest, TensorCnt)
{
    GPUArchitecture TestNoTensorCnt{GPUArchitectureTarget{GPUArchitectureGFX::GFX1250}};
    TestNoTensorCnt.AddCapability(GPUCapability::SupportedISA, 0);
    TestNoTensorCnt.AddCapability(GPUCapability::HasSplitWaitCounters, 0);
    EXPECT_EQ(TestNoTensorCnt.HasCapability(GPUCapability::SupportedISA), true);
    EXPECT_EQ(TestNoTensorCnt.HasCapability(GPUCapability::HasTensorcnt), false);

    auto wcNoTensorCnt = WaitCount::Zero(TestNoTensorCnt);

    EXPECT_EQ(0, wcNoTensorCnt.kmcnt());
    EXPECT_EQ(0, wcNoTensorCnt.dscnt());
    EXPECT_EQ(0, wcNoTensorCnt.loadcnt());
    EXPECT_EQ(0, wcNoTensorCnt.storecnt());
    EXPECT_EQ(-1, wcNoTensorCnt.vscnt());
    EXPECT_EQ(-1, wcNoTensorCnt.expcnt());
    EXPECT_EQ(-1, wcNoTensorCnt.tensorcnt());

    std::string expectedNoTensorCnt = R"(
                                        s_wait_loadcnt 0
                                        s_wait_storecnt 0
                                        s_wait_kmcnt 0
                                        s_wait_dscnt 0
                                    )";

    EXPECT_EQ(NormalizedSource(wcNoTensorCnt.toString(LogLevel::Debug)),
              NormalizedSource(expectedNoTensorCnt));

    GPUArchitecture TestWithTensorCnt{GPUArchitectureTarget{GPUArchitectureGFX::GFX1250}};
    TestWithTensorCnt.AddCapability(GPUCapability::SupportedISA, 0);
    TestWithTensorCnt.AddCapability(GPUCapability::HasSplitWaitCounters, 0);
    TestWithTensorCnt.AddCapability(GPUCapability::HasTensorcnt, 0);
    EXPECT_EQ(TestWithTensorCnt.HasCapability(GPUCapability::SupportedISA), true);
    EXPECT_EQ(TestWithTensorCnt.HasCapability(GPUCapability::HasTensorcnt), true);

    auto wcWithTensorCnt = WaitCount::Zero(TestWithTensorCnt);

    EXPECT_EQ(0, wcWithTensorCnt.kmcnt());
    EXPECT_EQ(0, wcWithTensorCnt.dscnt());
    EXPECT_EQ(0, wcWithTensorCnt.loadcnt());
    EXPECT_EQ(0, wcWithTensorCnt.storecnt());
    EXPECT_EQ(-1, wcWithTensorCnt.vscnt());
    EXPECT_EQ(-1, wcWithTensorCnt.expcnt());
    EXPECT_EQ(0, wcWithTensorCnt.tensorcnt());

    std::string expectedWithTensorCnt = R"(
                                        s_wait_loadcnt 0
                                        s_wait_storecnt 0
                                        s_wait_kmcnt 0
                                        s_wait_dscnt 0
                                        s_wait_tensorcnt 0
                                    )";

    EXPECT_EQ(NormalizedSource(wcWithTensorCnt.toString(LogLevel::Debug)),
              NormalizedSource(expectedWithTensorCnt));
}

TEST_F(WaitCountTest, SaturatedValues)
{
    GPUArchitecture testArch{GPUArchitectureTarget{GPUArchitectureGFX::GFX90A}};
    testArch.AddCapability(GPUCapability::MaxExpcnt, 7);
    testArch.AddCapability(GPUCapability::MaxLgkmcnt, 15);
    testArch.AddCapability(GPUCapability::MaxVmcnt, 63);
    testArch.AddCapability(GPUCapability::HasExpcnt, 0);
    EXPECT_EQ(testArch.HasCapability(GPUCapability::MaxExpcnt), true);
    EXPECT_EQ(testArch.HasCapability(GPUCapability::MaxLgkmcnt), true);
    EXPECT_EQ(testArch.HasCapability(GPUCapability::MaxVmcnt), true);

    auto wcZero = WaitCount::Zero(testArch).getAsSaturatedWaitCount(testArch);

    EXPECT_EQ(0, wcZero.kmcnt());
    EXPECT_EQ(0, wcZero.dscnt());
    EXPECT_EQ(0, wcZero.loadcnt());
    EXPECT_EQ(0, wcZero.storecnt());
    EXPECT_EQ(-1, wcZero.vscnt());
    EXPECT_EQ(0, wcZero.expcnt());

    auto wcExp = WaitCount::EXPCnt(testArch, 20);
    EXPECT_EQ(-1, wcExp.kmcnt());
    EXPECT_EQ(-1, wcExp.dscnt());
    EXPECT_EQ(-1, wcExp.loadcnt());
    EXPECT_EQ(-1, wcExp.storecnt());
    EXPECT_EQ(-1, wcExp.vscnt());
    EXPECT_EQ(20, wcExp.expcnt());

    auto wcSatExp = wcExp.getAsSaturatedWaitCount(testArch);
    EXPECT_EQ(-1, wcSatExp.kmcnt());
    EXPECT_EQ(-1, wcSatExp.dscnt());
    EXPECT_EQ(-1, wcSatExp.loadcnt());
    EXPECT_EQ(-1, wcSatExp.storecnt());
    EXPECT_EQ(-1, wcSatExp.vscnt());
    EXPECT_EQ(7, wcSatExp.expcnt());

    auto wcVm = WaitCount::LoadCnt(testArch, 100);
    EXPECT_EQ(100, wcVm.loadcnt());

    wcVm.combine(wcExp);
    auto wcComb = wcVm.getAsSaturatedWaitCount(testArch);
    EXPECT_EQ(-1, wcComb.kmcnt());
    EXPECT_EQ(-1, wcComb.dscnt());
    EXPECT_EQ(63, wcComb.loadcnt());
    EXPECT_EQ(-1, wcComb.storecnt());
    EXPECT_EQ(-1, wcComb.vscnt());
    EXPECT_EQ(7, wcComb.expcnt());
}

TEST_F(WaitCountTest, SaturatedValuesGFX1250)
{
    GPUArchitecture testArch{GPUArchitectureTarget{GPUArchitectureGFX::GFX1250}};
    testArch.AddCapability(GPUCapability::HasSplitWaitCounters, 0);
    testArch.AddCapability(GPUCapability::MaxExpcnt, 7);
    testArch.AddCapability(GPUCapability::MaxLgkmcnt, 15);
    testArch.AddCapability(GPUCapability::MaxVmcnt, 63);
    testArch.AddCapability(GPUCapability::MaxTensorcnt, 63);
    testArch.AddCapability(GPUCapability::HasExpcnt, 0);
    testArch.AddCapability(GPUCapability::HasTensorcnt, 0);
    EXPECT_EQ(testArch.HasCapability(GPUCapability::MaxExpcnt), true);
    EXPECT_EQ(testArch.HasCapability(GPUCapability::MaxLgkmcnt), true);
    EXPECT_EQ(testArch.HasCapability(GPUCapability::MaxVmcnt), true);
    EXPECT_EQ(testArch.HasCapability(GPUCapability::HasTensorcnt), true);

    auto wcZero = WaitCount::Zero(testArch).getAsSaturatedWaitCount(testArch);

    EXPECT_EQ(0, wcZero.kmcnt());
    EXPECT_EQ(0, wcZero.dscnt());
    EXPECT_EQ(0, wcZero.loadcnt());
    EXPECT_EQ(0, wcZero.storecnt());
    EXPECT_EQ(-1, wcZero.vscnt());
    EXPECT_EQ(0, wcZero.expcnt());
    EXPECT_EQ(0, wcZero.tensorcnt());

    auto wcExp = WaitCount::EXPCnt(testArch, 20);
    EXPECT_EQ(-1, wcExp.kmcnt());
    EXPECT_EQ(-1, wcExp.dscnt());
    EXPECT_EQ(-1, wcExp.loadcnt());
    EXPECT_EQ(-1, wcExp.storecnt());
    EXPECT_EQ(-1, wcExp.vscnt());
    EXPECT_EQ(20, wcExp.expcnt());
    EXPECT_EQ(-1, wcExp.tensorcnt());

    auto wcSatExp = wcExp.getAsSaturatedWaitCount(testArch);
    EXPECT_EQ(-1, wcSatExp.kmcnt());
    EXPECT_EQ(-1, wcSatExp.dscnt());
    EXPECT_EQ(-1, wcSatExp.loadcnt());
    EXPECT_EQ(-1, wcSatExp.storecnt());
    EXPECT_EQ(-1, wcSatExp.vscnt());
    EXPECT_EQ(7, wcSatExp.expcnt());
    EXPECT_EQ(-1, wcSatExp.tensorcnt());

    auto wcVm = WaitCount::LoadCnt(testArch, 100);
    EXPECT_EQ(100, wcVm.loadcnt());

    wcVm.combine(wcExp);
    auto wcComb = wcVm.getAsSaturatedWaitCount(testArch);
    EXPECT_EQ(-1, wcComb.kmcnt());
    EXPECT_EQ(-1, wcComb.dscnt());
    EXPECT_EQ(63, wcComb.loadcnt());
    EXPECT_EQ(-1, wcComb.storecnt());
    EXPECT_EQ(-1, wcComb.vscnt());
    EXPECT_EQ(7, wcComb.expcnt());
    EXPECT_EQ(-1, wcComb.tensorcnt());

    auto wcTensor = WaitCount::TensorCnt(testArch, 100);
    EXPECT_EQ(100, wcTensor.tensorcnt());

    wcTensor.combine(wcSatExp);
    wcComb = wcTensor.getAsSaturatedWaitCount(testArch);
    EXPECT_EQ(-1, wcComb.kmcnt());
    EXPECT_EQ(-1, wcComb.dscnt());
    EXPECT_EQ(-1, wcComb.loadcnt());
    EXPECT_EQ(-1, wcComb.storecnt());
    EXPECT_EQ(-1, wcComb.vscnt());
    EXPECT_EQ(7, wcComb.expcnt());
    EXPECT_EQ(63, wcComb.tensorcnt());
}
