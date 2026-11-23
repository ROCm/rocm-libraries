// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionUnified3dGeometry.hpp"

/**
 * @file TestGfx942AttentionUnified3dSplitRule.cpp
 * @brief gfx942SplitRuleSegments() equals rocKE's `_num_segments(problem, "gfx942")` on
 *        every row of a table rocKE itself generated.
 *
 * The table (Gfx942AttentionUnified3dSplitRuleTable.inc) is produced by
 * gen_gfx942_attention_unified_3d_split_rule.py, which builds each problem through
 * `dispatch.attention.common._problem_with_num_cus` and asks
 * `kernels.common.attention_unified._num_segments`. The build generates it from the rocKE
 * source tree (kernel_ingestor_engine/CMakeLists.txt), so a change to the rule fails here.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

struct SplitRuleRow
{
    int64_t batch;
    int64_t hq;
    int64_t hkv;
    int64_t headSize;
    int64_t blockSize;
    int64_t seqlenK;
    int64_t numCus;
    int64_t numSegments;
};

const std::vector<SplitRuleRow> SPLIT_RULE_TABLE = {
#include "Gfx942AttentionUnified3dSplitRuleTable.inc"
};

} // namespace

TEST(TestGfx942AttentionUnified3dSplitRule, MatchesRockeOnEveryGeneratedRow)
{
    // 5 batches x 2 KV heads x 5 GQA x 8 seqlen_k x 3 D x 3 blocks at each of 304, 228, 76
    // and 38 CUs, + 9 at 80, + 10 pinned edges at 304. Only these sampled points are
    // checked.
    ASSERT_EQ(std::size(SPLIT_RULE_TABLE), 14419u);
    for(const auto& r : SPLIT_RULE_TABLE)
    {
        EXPECT_EQ(gfx942SplitRuleSegments(
                      r.batch, r.hq, r.hkv, r.headSize, r.blockSize, r.seqlenK, r.numCus),
                  r.numSegments)
            << "B=" << r.batch << " Hq=" << r.hq << " Hkv=" << r.hkv << " D=" << r.headSize
            << " block=" << r.blockSize << " Skv=" << r.seqlenK << " cus=" << r.numCus;
    }
}

// Hand-checked anchors, one per branch, so a failure above is easy to localise.
TEST(TestGfx942AttentionUnified3dSplitRule, BranchAnchors)
{
    // Decode rule, D128: ctas 4*8 = 32, 32*16 >= 304; min(4*304 // 32, 128) = 38.
    EXPECT_EQ(gfx942SplitRuleSegments(4, 32, 8, 128, 16, 8192, 304), 38);
    // Same shape at B=1: 8*16 < 304 at Skv 8192 leaves the decode rule; D128 Skv<32768
    // clamps select_3d's 128 to pre_bump: num_2d = (1 // 4 + 1) * 8 = 8, 480/8 -> 64.
    EXPECT_EQ(gfx942SplitRuleSegments(1, 32, 8, 128, 16, 8192, 304), 64);
    // Decode rule, D64 per_cu 6: 6*304 // (16*8) = 14.
    EXPECT_EQ(gfx942SplitRuleSegments(16, 32, 8, 64, 16, 8192, 304), 14);
    // Decode rule floor: 4*304 // (256*8) = 0 -> 2.
    EXPECT_EQ(gfx942SplitRuleSegments(256, 32, 8, 128, 16, 512, 304), 2);
    // Outside the decode rule (block 32): D128 Skv<=2048 clamp min(seg, 16).
    EXPECT_EQ(gfx942SplitRuleSegments(1, 32, 8, 128, 32, 2048, 304), 16);
    // D256 block 32 is in no decode rule; Skv<=2048 leaves select_3d's count:
    // num_2d = (1 // 4 + 1) * 8 = 8; next_pow2(ceil(1216 / 8)) = 256 -> 128.
    EXPECT_EQ(gfx942SplitRuleSegments(1, 32, 8, 256, 32, 2048, 304), 128);
    // Few CTAs at long KV (1*1*16 < 304, Skv >= 8192) leaves the decode rule; D128
    // Skv<32768 clamps to pre_bump: num_2d = 1 * 1 = 1, 480 -> 512 -> 128.
    EXPECT_EQ(gfx942SplitRuleSegments(1, 4, 1, 128, 16, 8192, 304), 128);
    // D256 block 16 rule: ctas 4*8 = 32; min(2*304 // 32, 128, 8192 // 32) = 19.
    EXPECT_EQ(gfx942SplitRuleSegments(4, 32, 8, 256, 16, 8192, 304), 19);
    // D256 block 16 floor: 32 // 32 = 1 -> 2.
    EXPECT_EQ(gfx942SplitRuleSegments(4, 32, 8, 256, 16, 32, 304), 2);
    // Decode-grid pre_bump: qpk 16 gives block_q 1. select_3d counts (1 + 1) * 8 = 16
    // q-blocks -> 128; pre_bump counts the decode grid, 1 * 8 = 8 -> 64 (the prefill
    // count would give 32).
    EXPECT_EQ(gfx942SplitRuleSegments(1, 128, 8, 128, 16, 8192, 304), 64);
    // Decode rule at few CTAs: Skv 4096 < 8192 keeps it, min(4*304 // 1, 64) = 64.
    EXPECT_EQ(gfx942SplitRuleSegments(1, 4, 1, 128, 16, 4096, 304), 64);
    // qpk 8 on select_3d: block_q 2, num_2d = (3 // 2 + 3) * 4 = 16 -> 128.
    EXPECT_EQ(gfx942SplitRuleSegments(3, 32, 4, 64, 32, 131072, 304), 128);
}

TEST(TestGfx942AttentionUnified3dSplitRule, RefusesInvalidInputs)
{
    EXPECT_THROW(gfx942SplitRuleSegments(1, 30, 8, 128, 16, 8192, 304),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(gfx942SplitRuleSegments(1, 32, 8, 128, 16, 8192, 0),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
