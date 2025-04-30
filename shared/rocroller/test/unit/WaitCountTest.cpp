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
