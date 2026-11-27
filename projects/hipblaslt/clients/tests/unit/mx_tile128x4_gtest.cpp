// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only tests of the HIPBLASLT_ORDER_COL256_R128_* data tiling and the
// HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT and _128_4_PAGE24_EXT scale
// layouts the clients generate.

#include "hipblaslt_datatype2string.hpp"
#include "mx_tile128x4.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <set>
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

    using mx_tile128x4::TileOrder;

    // In-tile offset of byte b of the group of tile row c and K group kg.
    size_t expectedTileOffset(TileOrder order, size_t c, size_t kg, size_t b)
    {
        size_t const j = c / 32, p = (c / 16) % 2, l = 16 * kg + c % 16;
        if(order == TileOrder::P48)
            return 3072 * j + 48 * l + 24 * p + b;
        if(order == TileOrder::P16)
            return 3072 * j + 1024 * ((24 * p + b) / 16) + 16 * l + (24 * p + b) % 16;
        size_t const o = 64 * (c / 16) + 16 * kg + c % 16;
        return b < 16 ? 16 * o + b : 16384 + 8 * o + (b - 16);
    }

    // Placed offset of byte x of tile t at K tile s in half h, with aid_rowoff 1 and
    // phase 0: the 64 KiB block of block row r, then the 2 MiB page of (r, s / 24).
    size_t expectedPlacedOffset(size_t t, size_t s, size_t x, size_t h, bool split)
    {
        size_t const r = split ? (t & 7) : t;
        size_t const inBlock
            = (x / 6144) * 16384 + h * 8192 + ((x % 6144) / 3072) * 4096 + x % 3072;
        size_t const q    = 32 * r + s / 24;
        size_t const page = 2 * q + ((q >> 6) & 1);
        return page * (size_t(2) << 20) + (s % 24) * 65536 + (r & 7) * 65536 + inBlock;
    }

    // Placed offset of byte x of the 128x4 scales of 4096 rows in half h.
    size_t expectedPlacedScaleOffset(size_t x, size_t h)
    {
        size_t const s = x / 16384, g = (x % 16384) / 8192, z = x % 8192;
        return 2 * (s / 24) * (size_t(2) << 20) + (s % 24) * 32768 + g * 16384 + h * 8192 + z;
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

// Each tile order permutes the 24576 tile bytes, and P48 and P16 move bytes that
// HIPBLASLT_ORDER_COL256_R128_EXT keeps in place, and each other's.
TEST(HostUnitMXTile128x4, smoke_TileOrdersArePermutationsDistinctFromCol256R128)
{
    for(auto const order : {TileOrder::COL256_R128, TileOrder::P48, TileOrder::P16})
    {
        std::vector<int> hits(24576, 0);
        size_t           movedFromC0C1 = 0, movedFromP48 = 0;
        for(size_t c = 0; c < 256; ++c)
            for(size_t kg = 0; kg < 4; ++kg)
                for(size_t b = 0; b < 24; ++b)
                {
                    size_t const off = mx_tile128x4::tileByteOffset(order, c, kg, b);
                    ASSERT_EQ(off, expectedTileOffset(order, c, kg, b))
                        << "order=" << int(order) << " c=" << c << " kg=" << kg << " b=" << b;
                    ASSERT_LT(off, 24576u);
                    ++hits[off];
                    movedFromC0C1 += off != expectedTileOffset(TileOrder::COL256_R128, c, kg, b);
                    movedFromP48 += off != expectedTileOffset(TileOrder::P48, c, kg, b);
                }
        for(size_t i = 0; i < hits.size(); ++i)
            ASSERT_EQ(hits[i], 1) << "order=" << int(order) << " byte " << i;
        if(order != TileOrder::COL256_R128)
            EXPECT_GT(movedFromC0C1, 0u) << "order=" << int(order);
        if(order == TileOrder::P16)
            EXPECT_GT(movedFromP48, 0u);
    }
    // The COL256_R128 tile order is the one tiledGroupLowOffset and tiledGroupHighOffset give.
    EXPECT_EQ(mx_tile128x4::tileByteOffset(TileOrder::COL256_R128, 300, 5, 0),
              mx_tile128x4::tiledGroupLowOffset(300, 5, 256) % 24576);
    EXPECT_EQ(mx_tile128x4::tileByteOffset(TileOrder::COL256_R128, 300, 5, 16),
              mx_tile128x4::tiledGroupHighOffset(300, 5, 256) % 24576);
}

TEST(HostUnitMXTile128x4, smoke_TileOrderDataLandsAtItsOffsets)
{
    size_t const         R = 512, K = 256, rowBytes = K * 6 / 8;
    std::vector<uint8_t> src(R * rowBytes), plain(src.size());
    for(size_t i = 0; i < src.size(); ++i)
        src[i] = static_cast<uint8_t>(i * 131 + 7);
    mx_tile128x4::tileFp6Data(plain.data(), src.data(), R, K);
    for(auto const order : {TileOrder::COL256_R128, TileOrder::P48, TileOrder::P16})
    {
        std::vector<uint8_t> dst(src.size(), 0);
        mx_tile128x4::tileFp6Data(dst.data(), src.data(), R, K, order);
        if(order == TileOrder::COL256_R128)
            EXPECT_EQ(dst, plain);
        else
            EXPECT_NE(dst, plain) << "order=" << int(order);
        for(size_t r = 0; r < R; ++r)
            for(size_t g = 0; g < K / 32; ++g)
                for(size_t b = 0; b < 24; ++b)
                {
                    size_t const tile = (r / 256) * (K / 128) + g / 4;
                    ASSERT_EQ(dst[tile * 24576 + expectedTileOffset(order, r % 256, g % 4, b)],
                              src[r * rowBytes + g * 24 + b])
                        << "order=" << int(order) << " r=" << r << " g=" << g << " b=" << b;
                }
    }
}

// Every 3072-byte piece of every tile at every K tile lands at the page24 offset, at a
// distinct place inside the buffer; split tiles t and t + 8 share a block in two halves.
TEST(HostUnitMXTile128x4, smoke_PlacedPiecesFollowThePage24Layout)
{
    for(bool const split : {false, true})
    {
        size_t const     tiles = 16, kTiles = 768;
        size_t const     bytes = mx_tile128x4::placedFp6Bytes(tiles * 256, split);
        std::set<size_t> starts;
        for(size_t t = 0; t < tiles; ++t)
            for(size_t s = 0; s < kTiles; ++s)
                for(size_t h = 0; h < 2; ++h)
                {
                    if(split && h != t / 8)
                        continue;
                    for(size_t x = 0; x < 24576; x += 3072)
                    {
                        size_t const off = mx_tile128x4::placedFp6Offset(t, s, x, h, split);
                        ASSERT_EQ(off, expectedPlacedOffset(t, s, x, h, split))
                            << "split=" << split << " t=" << t << " s=" << s << " x=" << x;
                        ASSERT_EQ(mx_tile128x4::placedFp6Offset(t, s, x + 3071, h, split),
                                  off + 3071);
                        ASSERT_LE(off + 3072, bytes);
                        ASSERT_TRUE(starts.insert(off).second)
                            << "split=" << split << " t=" << t << " s=" << s << " x=" << x;
                    }
                }
        if(split)
            EXPECT_EQ(mx_tile128x4::placedFp6Offset(9, 5, 0, 1, true),
                      mx_tile128x4::placedFp6Offset(1, 5, 0, 0, true) + 8192);
    }
    EXPECT_EQ(mx_tile128x4::placedFp6Bytes(4096, true), size_t(514) << 21);
    EXPECT_EQ(mx_tile128x4::placedFp6Bytes(4096, false), size_t(1026) << 21);
}

TEST(HostUnitMXTile128x4, smoke_PlacedDataHoldsP48TilesInBothHalves)
{
    size_t const         R = 256, K = 128 * 26, rowBytes = K * 6 / 8;
    std::vector<uint8_t> src(R * rowBytes), p48(src.size());
    for(size_t i = 0; i < src.size(); ++i)
        src[i] = static_cast<uint8_t>(i * 131 + 7);
    mx_tile128x4::tileFp6Data(p48.data(), src.data(), R, K, TileOrder::P48);
    std::vector<uint8_t> dst(mx_tile128x4::placedFp6Bytes(R, false), 0);
    mx_tile128x4::placeFp6Data(dst.data(), src.data(), R, K, false);
    for(size_t s = 0; s < K / 128; ++s)
        for(size_t x = 0; x < 24576; ++x)
            for(size_t h = 0; h < 2; ++h)
                ASSERT_EQ(dst[expectedPlacedOffset(0, s, x, h, false)], p48[s * 24576 + x])
                    << "s=" << s << " x=" << x << " h=" << h;
}

TEST(HostUnitMXTile128x4, smoke_PlacedScalesFollowThePage24Layout)
{
    size_t const         R = 4096, kBlocks = 4 * 26;
    std::vector<uint8_t> src(R * kBlocks);
    for(size_t i = 0; i < src.size(); ++i)
        src[i] = static_cast<uint8_t>(i * 37 + 11);
    auto const dense  = mx_tile128x4::shuffleScales(src, R, kBlocks);
    auto const placed = mx_tile128x4::placeScales(src, kBlocks);
    ASSERT_EQ(placed.size(), mx_tile128x4::page24ScaleBytes(kBlocks));
    EXPECT_EQ(placed.size(), size_t(6) << 21);
    size_t moved = 0;
    for(size_t x = 0; x < dense.size(); ++x)
        for(size_t h = 0; h < 2; ++h)
        {
            size_t const off = expectedPlacedScaleOffset(x, h);
            ASSERT_EQ(mx_tile128x4::page24ScaleOffset(x, h), off) << "x=" << x << " h=" << h;
            ASSERT_EQ(placed[off], dense[x]) << "x=" << x << " h=" << h;
            moved += off != x;
        }
    EXPECT_GT(moved, 0u);
}

TEST(HostUnitMXTile128x4, smoke_PlacedShapesAreRejected)
{
    std::vector<uint8_t> buf(512 * 512);
    EXPECT_THROW(mx_tile128x4::placeFp6Data(buf.data(), buf.data(), 512, 128, true),
                 std::invalid_argument);
    EXPECT_THROW(mx_tile128x4::placeFp6Data(buf.data(), buf.data(), 256, 128 * 769, false),
                 std::invalid_argument);
    EXPECT_THROW(mx_tile128x4::placeScales(std::vector<uint8_t>(4096 * 4 * 769), 4 * 769),
                 std::invalid_argument);
    EXPECT_THROW(mx_tile128x4::placeScales(std::vector<uint8_t>(2048 * 4), 4),
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
