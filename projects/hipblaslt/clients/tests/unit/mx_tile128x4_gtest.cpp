// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only tests of the HIPBLASLT_ORDER_COL256_R128_EXT data tiling and the
// HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT scale shuffle the clients
// generate.

#include "hipblaslt_datatype2string.hpp"
#include "mx_tile128x4.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
    // The layouts as the kernels read them, written out per element rather than
    // through the helpers' tile arithmetic.
    size_t expectedLowOffset(size_t r, size_t k, size_t K)
    {
        size_t const o = 64 * ((r % 256) / 16) + 16 * ((k % 128) / 32) + r % 16;
        return ((r / 256) * (K / 128) + k / 128) * 24576 + 16 * o;
    }

    size_t expectedHighOffset(size_t r, size_t k, size_t K)
    {
        size_t const o = 64 * ((r % 256) / 16) + 16 * ((k % 128) / 32) + r % 16;
        return ((r / 256) * (K / 128) + k / 128) * 24576 + 16384 + 8 * o;
    }

    size_t expectedScaleOffset(size_t r, size_t kb, size_t R)
    {
        return (((kb / 4) * (R / 128) + r / 128) * 64 + 16 * (kb % 4) + r % 16) * 8
               + (r % 128) / 16;
    }
}

TEST(HostUnitMXTile128x4, smoke_DataGroupsLandWhereTheKernelReadsThem)
{
    for(size_t const R : {256, 512, 4096})
    {
        for(size_t const K : {128, 384, 65536, 131584})
        {
            for(size_t r = 0; r < R; r += (R > 512 ? 37 : 1))
            {
                for(size_t k = 0; k < K; k += (K > 512 ? 4096 + 32 : 32))
                {
                    EXPECT_EQ(mx_tile128x4::tiledGroupLowOffset(r, k / 32, K),
                              expectedLowOffset(r, k, K))
                        << "R=" << R << " K=" << K << " r=" << r << " k=" << k;
                    EXPECT_EQ(mx_tile128x4::tiledGroupHighOffset(r, k / 32, K),
                              expectedHighOffset(r, k, K))
                        << "R=" << R << " K=" << K << " r=" << r << " k=" << k;
                }
            }
        }
    }
}

// Every source byte lands on exactly one destination byte, in its group's planes.
TEST(HostUnitMXTile128x4, smoke_DataTilingIsAPermutationOfTheRows)
{
    size_t const         R = 512, K = 256, rowBytes = K * 6 / 8;
    std::vector<uint8_t> src(R * rowBytes), dst(src.size(), 0);
    std::vector<int>     hits(src.size(), 0);
    for(size_t i = 0; i < src.size(); ++i)
        src[i] = static_cast<uint8_t>(i * 131 + 7);
    mx_tile128x4::tileFp6Data(dst.data(), src.data(), R, K);

    for(size_t r = 0; r < R; ++r)
    {
        for(size_t g = 0; g < K / 32; ++g)
        {
            size_t const   low   = expectedLowOffset(r, g * 32, K);
            size_t const   high  = expectedHighOffset(r, g * 32, K);
            uint8_t const* group = src.data() + r * rowBytes + g * 24;
            for(size_t j = 0; j < 16; ++j)
            {
                ASSERT_EQ(dst[low + j], group[j]) << "r=" << r << " g=" << g << " j=" << j;
                ++hits[low + j];
            }
            for(size_t j = 0; j < 8; ++j)
            {
                ASSERT_EQ(dst[high + j], group[16 + j]) << "r=" << r << " g=" << g << " j=" << j;
                ++hits[high + j];
            }
        }
    }
    for(size_t i = 0; i < hits.size(); ++i)
        ASSERT_EQ(hits[i], 1) << "destination byte " << i;
}

TEST(HostUnitMXTile128x4, smoke_ScaleShuffleMatchesTheKernelLayout)
{
    for(size_t const R : {128, 4096})
    {
        for(size_t const kBlocks : {4, 2048, 4112})
        {
            std::vector<uint8_t> src(R * kBlocks);
            for(size_t i = 0; i < src.size(); ++i)
                src[i] = static_cast<uint8_t>(i * 37 + 11);
            auto const       dst = mx_tile128x4::shuffleScales(src, R, kBlocks);
            std::vector<int> hits(dst.size(), 0);
            for(size_t r = 0; r < R; ++r)
            {
                for(size_t kb = 0; kb < kBlocks; ++kb)
                {
                    size_t const off = expectedScaleOffset(r, kb, R);
                    ASSERT_EQ(mx_tile128x4::shuffledScaleOffset(r, kb, R), off);
                    ASSERT_EQ(dst[off], src[r * kBlocks + kb]) << "r=" << r << " kb=" << kb;
                    ++hits[off];
                }
            }
            for(size_t i = 0; i < hits.size(); ++i)
                ASSERT_EQ(hits[i], 1) << "R=" << R << " kBlocks=" << kBlocks << " offset " << i;
        }
    }
}

// A 256-row block holds K / 128 tiles of 24576 bytes, and a K tile of 4 scale blocks
// holds (M / 128) * 512 scale bytes: 0xC00000 at K = 65536, 0x1818000 at K = 131584.
TEST(HostUnitMXTile128x4, smoke_RowBlockPitch)
{
    for(auto const [K, pitch] :
        {std::pair<size_t, size_t>{65536, 0xC00000}, std::pair<size_t, size_t>{131584, 0x1818000}})
    {
        EXPECT_EQ(mx_tile128x4::tiledGroupLowOffset(256, 0, K), pitch) << "K=" << K;
        EXPECT_EQ(mx_tile128x4::tiledGroupLowOffset(4095, K / 32 - 1, K) / pitch, 15u) << "K=" << K;
    }
    EXPECT_EQ(mx_tile128x4::shuffledScaleOffset(0, 4, 4096), size_t(32) * 512);
}

TEST(HostUnitMXTile128x4, smoke_PartialTilesAreRejected)
{
    std::vector<uint8_t> buf(512 * 512);
    EXPECT_THROW(mx_tile128x4::tileFp6Data(buf.data(), buf.data(), 128, 128),
                 std::invalid_argument);
    EXPECT_THROW(mx_tile128x4::tileFp6Data(buf.data(), buf.data(), 256, 64), std::invalid_argument);
    EXPECT_THROW(mx_tile128x4::shuffleScales(std::vector<uint8_t>(64 * 4), 64, 4),
                 std::invalid_argument);
    EXPECT_THROW(mx_tile128x4::shuffleScales(std::vector<uint8_t>(128 * 2), 128, 2),
                 std::invalid_argument);
    EXPECT_THROW(mx_tile128x4::shuffleScales(std::vector<uint8_t>(128 * 4 - 1), 128, 4),
                 std::invalid_argument);
}

// One scale per 32 K elements, rounded up to whole 128-row by 4-block tiles.
TEST(HostUnitMXTile128x4, smoke_ScaleBufferCoversWholeTiles)
{
    auto const fmt = hipblaslt_scaling_format::Block_32_UE8M0_128_4_EXT;
    // K along the rows (transposed A, non-transposed B): dataRow = K, dataCol = M or N.
    EXPECT_EQ(blockScaleBufferSize(65536, 4096, fmt, true, false), size_t(4096) * 2048);
    EXPECT_EQ(blockScaleBufferSize(131584, 4096, fmt, true, false), size_t(4096) * 4112);
    EXPECT_EQ(blockScaleBufferSize(128, 128, fmt, true, false), size_t(128) * 4);
    EXPECT_EQ(blockScaleBufferSize(129, 129, fmt, true, false), size_t(256) * 8);
    EXPECT_EQ(blockScaleBufferSize(128, 128, fmt, true, true), size_t(128) * 4);
}
