// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// CPU unit tests for L2 cache swizzle logic.
// Self-contained test: mirrors the L2 swizzle and GetTileIndex algorithms.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

namespace test_impl {

using index_t = int32_t;

// Mirror of ComputeL2Swizzle from fmha_fwd_kernel.hpp
index_t ComputeL2Swizzle(index_t seqlen_k,
                         index_t hdim_q,
                         index_t k_dtype_bytes,
                         index_t nhead_ratio_qk)
{
    constexpr index_t kL2PerXCD = 4 * 1024 * 1024; // 4MB
    const index_t kv_bytes_per_head = seqlen_k * hdim_q * k_dtype_bytes * 2;

    index_t raw = (kv_bytes_per_head > 0) ? (kL2PerXCD / kv_bytes_per_head) : 1;
    if(raw < 1)
        raw = 1;

    // Round down to power of 2
    index_t swizzle = 1;
    while(swizzle * 2 <= raw)
        swizzle *= 2;

    return swizzle * nhead_ratio_qk;
}

// Mirror of L2 swizzle head reordering in GetTileIndex
std::tuple<index_t, index_t> L2SwizzleRemap(index_t i_nhead,
                                            index_t i_tile_m_raw,
                                            index_t num_m_blocks,
                                            index_t l2_swizzle)
{
    // Linearize across heads and M-blocks
    const index_t linear = i_nhead * num_m_blocks + i_tile_m_raw;

    // Decompose into L2 sections
    const index_t section_size = l2_swizzle * num_m_blocks;
    const index_t section      = linear / section_size;
    const index_t remainder    = linear - section * section_size;

    // Within each section: iterate all M-blocks for head 0, then head 1, etc.
    const index_t i_nhead_new = section * l2_swizzle + remainder / num_m_blocks;
    const index_t i_tile_m    = remainder - (remainder / num_m_blocks) * num_m_blocks;

    return {i_tile_m, i_nhead_new};
}

// Mirror of XCD-interleave scheduling
index_t XCDRemap(index_t i_block, index_t grid_x)
{
    constexpr index_t kNumXCDs         = 8;
    constexpr index_t kMinTilesPerXCD  = 16;
    const index_t cus_per_xdim_per_xcd = grid_x / kNumXCDs;
    if(cus_per_xdim_per_xcd >= kMinTilesPerXCD)
    {
        const index_t cu_id  = i_block / kNumXCDs;
        const index_t xcd_id = i_block % kNumXCDs;
        if(cu_id < cus_per_xdim_per_xcd)
        {
            return xcd_id * cus_per_xdim_per_xcd + cu_id;
        }
    }
    return i_block;
}

} // namespace test_impl

// ===== L2 Swizzle Computation Tests =====

TEST(L2Swizzle, SmallKVFitsInL2)
{
    // Small seqlen_k=128, hdim=128, bf16 (2 bytes), nhead_ratio=1
    // KV bytes = 128 * 128 * 2 * 2 = 64KB
    // raw = 4MB / 64KB = 64, floor_pow2(64) = 64
    auto swizzle = test_impl::ComputeL2Swizzle(128, 128, 2, 1);
    EXPECT_EQ(swizzle, 64);
}

TEST(L2Swizzle, LargeKVExceedsL2)
{
    // Large seqlen_k=4096, hdim=128, bf16 (2 bytes), nhead_ratio=1
    // KV bytes = 4096 * 128 * 2 * 2 = 2MB
    // raw = 4MB / 2MB = 2, floor_pow2(2) = 2
    auto swizzle = test_impl::ComputeL2Swizzle(4096, 128, 2, 1);
    EXPECT_EQ(swizzle, 2);
}

TEST(L2Swizzle, GQARatioScaling)
{
    // With GQA ratio = 8 (8 Q-heads per KV head)
    auto swizzle = test_impl::ComputeL2Swizzle(128, 128, 2, 8);
    // 64 * 8 = 512
    EXPECT_EQ(swizzle, 512);
}

TEST(L2Swizzle, PowerOfTwoRounding)
{
    // seqlen_k=300, hdim=128, bf16
    // KV bytes = 300 * 128 * 2 * 2 = 150KB (approximately)
    // raw = 4MB / 150KB ~= 27, floor_pow2(27) = 16
    auto swizzle = test_impl::ComputeL2Swizzle(300, 128, 2, 1);
    EXPECT_EQ(swizzle, 16);
}

TEST(L2Swizzle, ZeroSeqlenHandled)
{
    // Edge case: seqlen_k=0 should not divide by zero
    auto swizzle = test_impl::ComputeL2Swizzle(0, 128, 2, 1);
    EXPECT_GE(swizzle, 1);
}

// ===== L2 Swizzle Remap Tests =====

TEST(L2SwizzleRemap, IdentityWhenSwizzleEqualsNhead)
{
    // When l2_swizzle == nhead, the remap should be identity for M-blocks
    const int nhead = 4, num_m = 8, l2_swizzle = 4;
    for(int h = 0; h < nhead; ++h)
    {
        for(int m = 0; m < num_m; ++m)
        {
            auto [m_new, h_new] = test_impl::L2SwizzleRemap(h, m, num_m, l2_swizzle);
            EXPECT_EQ(m_new, m) << "h=" << h << " m=" << m;
            EXPECT_EQ(h_new, h) << "h=" << h << " m=" << m;
        }
    }
}

TEST(L2SwizzleRemap, HeadGrouping)
{
    // With l2_swizzle=2, nhead=8, heads should be grouped in pairs
    const int num_m = 4, l2_swizzle = 2;
    // First 2 heads should stay in section 0
    auto [m0, h0] = test_impl::L2SwizzleRemap(0, 0, num_m, l2_swizzle);
    auto [m1, h1] = test_impl::L2SwizzleRemap(1, 0, num_m, l2_swizzle);
    EXPECT_EQ(h0 / l2_swizzle, h1 / l2_swizzle) << "heads 0,1 should be in same section";
}

TEST(L2SwizzleRemap, AllTilesCovered)
{
    // Every (head, m_tile) pair should be covered exactly once
    const int nhead = 4, num_m = 8, l2_swizzle = 2;
    std::vector<std::pair<int, int>> seen;
    for(int h = 0; h < nhead; ++h)
    {
        for(int m = 0; m < num_m; ++m)
        {
            auto [m_new, h_new] = test_impl::L2SwizzleRemap(h, m, num_m, l2_swizzle);
            ASSERT_GE(m_new, 0);
            ASSERT_LT(m_new, num_m);
            ASSERT_GE(h_new, 0);
            ASSERT_LT(h_new, nhead);
            seen.emplace_back(h_new, m_new);
        }
    }
    std::sort(seen.begin(), seen.end());
    auto it = std::unique(seen.begin(), seen.end());
    EXPECT_EQ(it - seen.begin(), nhead * num_m) << "all tiles should be unique";
}

// ===== XCD Remap Tests =====

TEST(XCDRemap, SmallGridNoRemap)
{
    // Grid too small for XCD remap (< 8 * 16 = 128 blocks)
    const int grid_x = 64;
    for(int i = 0; i < grid_x; ++i)
    {
        EXPECT_EQ(test_impl::XCDRemap(i, grid_x), i)
            << "small grid should not remap at i=" << i;
    }
}

TEST(XCDRemap, LargeGridRemaps)
{
    // Grid large enough: 256 blocks (= 8 XCDs * 32 tiles each)
    const int grid_x = 256;
    // Block 0 should stay at 0 (cu_id=0, xcd_id=0)
    EXPECT_EQ(test_impl::XCDRemap(0, grid_x), 0);
    // Block 1 should map to xcd_id=1 section: 1 * 32 + 0 = 32
    EXPECT_EQ(test_impl::XCDRemap(1, grid_x), 32);
    // Block 8 should map back: cu_id=1, xcd_id=0: 0 * 32 + 1 = 1
    EXPECT_EQ(test_impl::XCDRemap(8, grid_x), 1);
}

TEST(XCDRemap, AdjacentCUsOnSameXCD)
{
    // After remap, consecutive original block IDs that share an XCD
    // should map to adjacent positions
    const int grid_x = 256;

    // Blocks 0, 8, 16 all have xcd_id=0, should map to 0, 1, 2
    EXPECT_EQ(test_impl::XCDRemap(0, grid_x), 0);
    EXPECT_EQ(test_impl::XCDRemap(8, grid_x), 1);
    EXPECT_EQ(test_impl::XCDRemap(16, grid_x), 2);
}

TEST(XCDRemap, AllBlocksCovered)
{
    // Verify bijective mapping (all remapped values are unique)
    const int grid_x = 256;
    std::vector<int> remapped;
    for(int i = 0; i < grid_x; ++i)
    {
        remapped.push_back(test_impl::XCDRemap(i, grid_x));
    }
    std::sort(remapped.begin(), remapped.end());
    auto it = std::unique(remapped.begin(), remapped.end());
    EXPECT_EQ(it - remapped.begin(), grid_x) << "remap should be bijective";
}
