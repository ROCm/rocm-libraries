#include <rocRoller/CodeGen/WaitCount.hpp>
#include <rocRoller/Utilities/Logging.hpp>

#include "SimpleFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;

class WaitCountTest : public SimpleFixture
{
};

TEST_F(WaitCountTest, Basic)
{
    auto wc = WaitCount::LGKMCnt(2);

    EXPECT_EQ(2, wc.lgkmcnt());
    EXPECT_EQ(-1, wc.vmcnt());
    EXPECT_EQ(-1, wc.vscnt());
    EXPECT_EQ(-1, wc.expcnt());

    EXPECT_EQ("s_waitcnt lgkmcnt(2)\n", wc.toString(LogLevel::Terse));
}

TEST_F(WaitCountTest, Combine)
{
    auto wc = WaitCount::LGKMCnt(2);

    wc.combine(WaitCount::VMCnt(4));

    EXPECT_EQ(2, wc.lgkmcnt());
    EXPECT_EQ(4, wc.vmcnt());
    EXPECT_EQ(-1, wc.vscnt());
    EXPECT_EQ(-1, wc.expcnt());

    wc.combine(WaitCount::LGKMCnt(4));

    EXPECT_EQ(2, wc.lgkmcnt());
    EXPECT_EQ(4, wc.vmcnt());
    EXPECT_EQ(-1, wc.vscnt());
    EXPECT_EQ(-1, wc.expcnt());

    wc.combine(WaitCount::LGKMCnt(1, "Wait for LDS"));

    EXPECT_EQ(1, wc.lgkmcnt());
    EXPECT_EQ(4, wc.vmcnt());
    EXPECT_EQ(-1, wc.vscnt());
    EXPECT_EQ(-1, wc.expcnt());

    // -1 is no wait, shouldn't affect the value.
    wc.combine(WaitCount::LGKMCnt(-1));

    EXPECT_EQ(1, wc.lgkmcnt());
    EXPECT_EQ(4, wc.vmcnt());
    EXPECT_EQ(-1, wc.vscnt());
    EXPECT_EQ(-1, wc.expcnt());

    wc.combine(WaitCount::VSCnt(2, "Wait for store"));

    EXPECT_EQ(1, wc.lgkmcnt());
    EXPECT_EQ(4, wc.vmcnt());
    EXPECT_EQ(2, wc.vscnt());
    EXPECT_EQ(-1, wc.expcnt());

    wc.combine(WaitCount::EXPCnt(20));

    EXPECT_EQ(1, wc.lgkmcnt());
    EXPECT_EQ(4, wc.vmcnt());
    EXPECT_EQ(2, wc.vscnt());
    EXPECT_EQ(20, wc.expcnt());

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
}

/**
 * This test checks that if an architecture has the SeparateVscnt capability, then a waitcnt zero will
 * produce a wait zero for vscnt, and if it doesn't have that capability, then no vscnt wait is produced.
 **/
TEST_F(WaitCountTest, VSCnt)
{
    GPUArchitecture TestWithVSCnt;
    TestWithVSCnt.AddCapability(GPUCapability::SupportedISA, 0);
    TestWithVSCnt.AddCapability(GPUCapability::SeparateVscnt, 0);
    EXPECT_EQ(TestWithVSCnt.HasCapability(GPUCapability::SupportedISA), true);
    EXPECT_EQ(TestWithVSCnt.HasCapability(GPUCapability::SeparateVscnt), true);

    auto wcWithVSCnt = WaitCount::Zero(TestWithVSCnt);

    EXPECT_EQ(0, wcWithVSCnt.lgkmcnt());
    EXPECT_EQ(0, wcWithVSCnt.vmcnt());
    EXPECT_EQ(0, wcWithVSCnt.vscnt());
    EXPECT_EQ(0, wcWithVSCnt.expcnt());

    std::string expectedWithVSCnt = R"(
                                        s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)
                                        s_waitcnt_vscnt 0
                                    )";

    EXPECT_EQ(NormalizedSource(wcWithVSCnt.toString(LogLevel::Debug)),
              NormalizedSource(expectedWithVSCnt));

    GPUArchitecture TestNoVSCnt;
    TestNoVSCnt.AddCapability(GPUCapability::SupportedISA, 0);
    EXPECT_EQ(TestNoVSCnt.HasCapability(GPUCapability::SupportedISA), true);
    EXPECT_EQ(TestNoVSCnt.HasCapability(GPUCapability::SeparateVscnt), false);

    auto wcNoVSCnt = WaitCount::Zero(TestNoVSCnt);

    EXPECT_EQ(0, wcNoVSCnt.lgkmcnt());
    EXPECT_EQ(0, wcNoVSCnt.vmcnt());
    EXPECT_EQ(-1, wcNoVSCnt.vscnt());
    EXPECT_EQ(0, wcNoVSCnt.expcnt());

    std::string expectedNoVSCnt = R"(
                                        s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)
                                    )";

    EXPECT_EQ(NormalizedSource(wcNoVSCnt.toString(LogLevel::Debug)),
              NormalizedSource(expectedNoVSCnt));
}

TEST_F(WaitCountTest, SaturatedValues)
{
    GPUArchitecture testArch;
    testArch.AddCapability(GPUCapability::MaxExpcnt, 7);
    testArch.AddCapability(GPUCapability::MaxLgkmcnt, 15);
    testArch.AddCapability(GPUCapability::MaxVmcnt, 63);
    EXPECT_EQ(testArch.HasCapability(GPUCapability::MaxExpcnt), true);
    EXPECT_EQ(testArch.HasCapability(GPUCapability::MaxLgkmcnt), true);
    EXPECT_EQ(testArch.HasCapability(GPUCapability::MaxVmcnt), true);

    auto wcZero = WaitCount::Zero(testArch).getAsSaturatedWaitCount(testArch);

    EXPECT_EQ(0, wcZero.lgkmcnt());
    EXPECT_EQ(0, wcZero.vmcnt());
    EXPECT_EQ(-1, wcZero.vscnt());
    EXPECT_EQ(0, wcZero.expcnt());

    auto wcExp = WaitCount::EXPCnt(20);
    EXPECT_EQ(-1, wcExp.lgkmcnt());
    EXPECT_EQ(-1, wcExp.vmcnt());
    EXPECT_EQ(-1, wcExp.vscnt());
    EXPECT_EQ(20, wcExp.expcnt());

    auto wcSatExp = wcExp.getAsSaturatedWaitCount(testArch);
    EXPECT_EQ(-1, wcSatExp.lgkmcnt());
    EXPECT_EQ(-1, wcSatExp.vmcnt());
    EXPECT_EQ(-1, wcSatExp.vscnt());
    EXPECT_EQ(7, wcSatExp.expcnt());

    auto wcVm = WaitCount::VMCnt(100);
    EXPECT_EQ(100, wcVm.vmcnt());

    wcVm.combine(wcExp);
    auto wcComb = wcVm.getAsSaturatedWaitCount(testArch);
    EXPECT_EQ(-1, wcComb.lgkmcnt());
    EXPECT_EQ(63, wcComb.vmcnt());
    EXPECT_EQ(-1, wcComb.vscnt());
    EXPECT_EQ(7, wcComb.expcnt());
}
