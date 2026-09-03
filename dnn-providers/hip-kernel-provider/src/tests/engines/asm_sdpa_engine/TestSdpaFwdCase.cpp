// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "AsmSdpaConfigHelpers.hpp"

using asm_sdpa_engine::SdpaFwdTestCase;

TEST(TestSdpaFwdCase, ConstructorDerivesKDimsFromQAndV)
{
    // Q: [B=2, H_q=4, S_q=256, D_qk=128]
    // V: [B=2, H_kv=1, S_kv=128, D_v=64]
    // Expected K: [B=2, H_kv=1, S_kv=128, D_qk=128]
    const SdpaFwdTestCase tc({2, 4, 256, 128}, {2, 1, 128, 64}, "gfx942");

    EXPECT_EQ(tc.kDims[0], 2); // B from Q
    EXPECT_EQ(tc.kDims[1], 1); // H_kv from V
    EXPECT_EQ(tc.kDims[2], 128); // S_kv from V
    EXPECT_EQ(tc.kDims[3], 128); // D_qk from Q
}

TEST(TestSdpaFwdCase, DefaultMaskIsNoMask)
{
    const SdpaFwdTestCase tc({1, 1, 128, 128}, {1, 1, 128, 128}, "gfx942");

    EXPECT_EQ(tc.leftBound, -1);
    EXPECT_EQ(tc.rightBound, -1);
    EXPECT_TRUE(tc.topLeftAlignment);
}

TEST(TestSdpaFwdCase, CausalMaskParameters)
{
    const SdpaFwdTestCase tc({1, 1, 128, 128}, {1, 1, 128, 128}, "gfx942", -1, 0, false);

    EXPECT_EQ(tc.leftBound, -1);
    EXPECT_EQ(tc.rightBound, 0);
    EXPECT_FALSE(tc.topLeftAlignment);
}

TEST(TestSdpaFwdCase, GetNameNoMask)
{
    const SdpaFwdTestCase tc({1, 4, 256, 128}, {1, 1, 256, 128}, "gfx942");
    const testing::TestParamInfo<SdpaFwdTestCase> info(tc, 0);

    auto name = SdpaFwdTestCase::getName(info);
    EXPECT_EQ(name, "gfx942_B1_Hq4_Hkv1_Sq256_Skv256_Dqk128_Dv128_NoMask");
}

TEST(TestSdpaFwdCase, GetNameBottomRightCausal)
{
    const SdpaFwdTestCase tc({2, 4, 256, 128}, {2, 4, 256, 128}, "gfx950", -1, 0, false);
    const testing::TestParamInfo<SdpaFwdTestCase> info(tc, 0);

    auto name = SdpaFwdTestCase::getName(info);
    EXPECT_EQ(name, "gfx950_B2_Hq4_Hkv4_Sq256_Skv256_Dqk128_Dv128_BottomRightCausal");
}

TEST(TestSdpaFwdCase, GetNameTopLeftCausal)
{
    const SdpaFwdTestCase tc({1, 2, 128, 192}, {1, 2, 128, 128}, "gfx942", 0, 0, true);
    const testing::TestParamInfo<SdpaFwdTestCase> info(tc, 0);

    auto name = SdpaFwdTestCase::getName(info);
    EXPECT_EQ(name, "gfx942_B1_Hq2_Hkv2_Sq128_Skv128_Dqk192_Dv128_TopLeftCausal");
}
