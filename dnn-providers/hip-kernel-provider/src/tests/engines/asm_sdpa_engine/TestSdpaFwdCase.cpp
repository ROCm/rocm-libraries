// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

#include "AsmSdpaConfigHelpers.hpp"
#include "hip_kernel_provider_common/SdpaConfigEnumerations.hpp"

using asm_sdpa_engine::GraphTestCase;
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

namespace
{

// Builds a GROUP-mode graph with the helper and exits 0 iff the frontend validates it.
[[noreturn]] void validateGroupModeGraph()
{
    const fmha_v3_fwdConfig config{"group_mode_kernel",
                                   "group_mode_object",
                                   "gfx942",
                                   "bf16",
                                   128,
                                   128,
                                   0,
                                   hip_kernel_provider_common::BatchMode::GROUP,
                                   0,
                                   256,
                                   32};
    const GraphTestCase tc(config, "GroupMode", "gfx942");
    const auto built = asm_sdpa_engine::buildSdpaFwdGraph(tc);
    const auto result = built.graph->validate();
    std::fprintf(stderr, "group mode validate: %s\n", result.get_message().c_str());
    std::exit(result.is_good() ? 0 : 1);
}

} // namespace

// GROUP mode adds seq_len_q / seq_len_kv, which the frontend requires to be INT32
// [B, 1, 1, 1] on every SDPA forward graph. Graph::validate() initializes frontend
// logging, which replaces this binary's log-recording callback for every later test,
// so it runs in a re-executed child (threadsafe death-test style). The suite name ends
// in Death because hipDNN's test-name validator reserves "Test" for the leading keyword.
TEST(TestSdpaFwdCaseDeath, GroupModeGraphPassesFrontendValidation)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    // EXPECT_EXIT expands to a switch over AssumeRole() with no default label.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wswitch-default"
    EXPECT_EXIT(validateGroupModeGraph(), testing::ExitedWithCode(0), "group mode validate");
#pragma clang diagnostic pop
}
