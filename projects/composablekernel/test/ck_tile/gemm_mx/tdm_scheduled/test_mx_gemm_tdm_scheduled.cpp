// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// comp_tdm_scheduled on gfx1250 against reference_mx_gemm_tdm_scheduled, bit for bit on the
// sparse check inputs: every type and output, both atoms and wave groups, each schedule variant,
// clusters, K paths (even and odd tile counts, K tails, K a multiple of 32 only), edges and FP8
// saturation. E goes through LDS and is stored by TDM. Needs a gfx1250 device.

#include "test_mx_gemm_tdm_scheduled_util.hpp"

namespace {

using namespace mx_tdm_scheduled_test;
using ck_tile::bf16_t;
using ck_tile::bf8_t;
using ck_tile::fp8_t;
using ck_tile::half_t;
using ck_tile::pk_fp4_t;
using ck_tile::pk_fp6x16_t;

template <typename O, typename Tuning, typename Traits>
using Fp4Big = Kernel<pk_fp4_t, pk_fp4_t, O, 256, 256, 512, 2, 2, 32, Tuning, Traits>;

// The tuned 4096 x 4096 FP4 instance: 256x256x512 tile, 4x4 cluster.
using Fp4C4x4      = Fp4Big<fp8_t, Tu<4, 4>, Shape<false, false, 0>>;
using Fp4C4x4Tail1 = Fp4Big<fp8_t, Tu<4, 4>, Shape<false, false, 1>>;
using Fp4C4x4Tail3 = Fp4Big<fp8_t, Tu<4, 4>, Shape<false, false, 3>>;

// FP4 128x128x1024 with a one-step K tail.
template <typename Tuning>
using Fp4K1024 =
    Kernel<pk_fp4_t, pk_fp4_t, fp8_t, 128, 128, 1024, 2, 2, 32, Tuning, Shape<false, false, 1>>;

TEST(MxGemmTdmScheduled, Fp4EvenTiles) { check<Fp4C4x4>(1024, 1024, 2048); }
TEST(MxGemmTdmScheduled, Fp4LongK) { check<Fp4C4x4>(1024, 1024, 65536); }
TEST(MxGemmTdmScheduled, Fp4OddTiles)
{
    check<Fp4C4x4>(1024, 1024, 512);
    check<Fp4C4x4>(1024, 1024, 1536);
    check<Fp4C4x4>(1024, 1024, 2560);
}
// One instance serves every whole-tile count: odd counts (1, 3) run its TailNumber::One kernel,
// even counts (2, 4) its Two kernel.
TEST(MxGemmTdmScheduled, Fp4TileCounts)
{
    for(index_t tiles = 1; tiles <= 4; ++tiles)
    {
        check<Fp4C4x4>(1024, 1024, tiles * 512);
        check<Fp4C4x4Tail1>(1024, 1024, tiles * 512 + 128);
    }
}
TEST(MxGemmTdmScheduled, Fp4KTail)
{
    check<Fp4C4x4Tail1>(1024, 1024, 2048 + 128);
    // A partial last K step: K a multiple of 32 only, its scales zero-padded.
    check<Fp4C4x4Tail1>(1024, 1024, 2048 + 32);
    check<Fp4C4x4Tail1>(1024, 1024, 2048 + 96);
    check<Fp4C4x4Tail3>(1024, 1024, 1536 + 384);
}

TEST(MxGemmTdmScheduled, Fp4OutputTypes)
{
    check<Fp4Big<half_t, Tu<4, 4>, Shape<false, false, 0>>>(1024, 1024, 2048);
    check<Fp4Big<bf16_t, Tu<4, 4>, Shape<false, false, 0>>>(1024, 1024, 2048);
}

TEST(MxGemmTdmScheduled, Fp4Clusters)
{
    check<Fp4Big<fp8_t, Tu<1, 1>, Shape<false, false, 0>>>(512, 768, 2048);
    check<Fp4Big<fp8_t, Tu<2, 2>, Shape<false, false, 0>>>(512, 1024, 2048);
    check<Fp4Big<fp8_t, Tu<4, 2>, Shape<false, false, 0>>>(512, 1024, 2048);
}

TEST(MxGemmTdmScheduled, Fp4ScheduleVariants)
{
    using S = Shape<false, false, 0>;
    check<Fp4Big<fp8_t, Tu<4, 4, 2>, S>>(1024, 1024, 4096);
    check<Fp4Big<fp8_t, Tu<4, 4, 4, true, false>, S>>(1024, 1024, 4096);
    check<Fp4Big<fp8_t, Tu<4, 4, 4, false, false>, S>>(1024, 1024, 4096);
    check<Fp4Big<fp8_t, Tu<4, 4, 4, true, true, false>, S>>(1024, 1024, 4096);
    check<Fp4Big<fp8_t, Tu<4, 4, 0>, S>>(1024, 1024, 4096);
    check<Fp4Big<fp8_t, Tu<4, 4, 8>, S>>(1024, 1024, 4096);
}

TEST(MxGemmTdmScheduled, Fp4TilesAndWaveGroups)
{
    check<Kernel<pk_fp4_t,
                 pk_fp4_t,
                 fp8_t,
                 256,
                 256,
                 256,
                 2,
                 2,
                 32,
                 Tu<4, 4>,
                 Shape<false, false, 0>>>(1024, 1024, 2048);
    check<Kernel<pk_fp4_t,
                 pk_fp4_t,
                 fp8_t,
                 128,
                 128,
                 1024,
                 2,
                 2,
                 32,
                 Tu<2, 2>,
                 Shape<false, false, 0>>>(512, 512, 4096);
    check<Kernel<pk_fp4_t,
                 pk_fp4_t,
                 fp8_t,
                 64,
                 256,
                 512,
                 1,
                 4,
                 32,
                 Tu<2, 2>,
                 Shape<false, false, 0>>>(512, 512, 2048);
    check<Kernel<pk_fp4_t,
                 pk_fp4_t,
                 fp8_t,
                 256,
                 64,
                 512,
                 4,
                 1,
                 32,
                 Tu<1, 1, 4, true, true, false>,
                 Shape<false, false, 0>>>(512, 512, 2048);
    // The 16x16 atom with FP4, and the 128x16x512 4x1-wave tile: tile_n 16, one N repeat.
    check<Kernel<pk_fp4_t,
                 pk_fp4_t,
                 fp8_t,
                 256,
                 256,
                 512,
                 2,
                 2,
                 16,
                 Tu<4, 4>,
                 Shape<false, false, 0>>>(1024, 1024, 2048);
    check<Kernel<pk_fp4_t,
                 pk_fp4_t,
                 fp8_t,
                 128,
                 16,
                 512,
                 4,
                 1,
                 16,
                 Tu<1, 4, 0, false, false>,
                 Shape<false, false, 0>>>(2048, 16, 4096);
    check<Kernel<pk_fp4_t,
                 pk_fp4_t,
                 fp8_t,
                 64,
                 16,
                 512,
                 4,
                 1,
                 16,
                 Tu<1, 1, 4, true, false>,
                 Shape<true, false, 0>>>(1000, 16, 2048);
}

TEST(MxGemmTdmScheduled, EightBitTypes)
{
    using S = Shape<false, false, 0>;
    check<Kernel<fp8_t, fp8_t, fp8_t, 128, 128, 256, 2, 2, 16, Tu<2, 2>, S>>(512, 512, 2048);
    check<Kernel<bf8_t, bf8_t, fp8_t, 128, 128, 256, 2, 2, 16, Tu<2, 2>, S>>(512, 512, 2048);
    check<Kernel<fp8_t, bf8_t, bf16_t, 128, 128, 256, 2, 2, 16, Tu<2, 2>, S>>(512, 512, 2048);
    check<Kernel<bf8_t, fp8_t, half_t, 128, 128, 128, 2, 2, 16, Tu<2, 2, 4, true, false>, S>>(
        512, 512, 2048);
    check<Kernel<fp8_t, fp8_t, fp8_t, 128, 128, 256, 2, 2, 16, Tu<2, 2, 4, false, false>, S>>(
        512, 512, 2048);
    // 8-bit edge and K tail: 1000 x 1000 x (2048 + 96) with bf8 B.
    check<Kernel<fp8_t, bf8_t, fp8_t, 128, 128, 256, 2, 2, 16, Tu<1, 1>, Shape<true, true, 1>>>(
        1000, 1000, 2048 + 96);
    check<Kernel<fp8_t, bf8_t, bf16_t, 128, 128, 256, 2, 2, 16, Tu<1, 1>, Shape<true, true, 1>>>(
        1000, 1001, 2048 + 96);
}

// FP6, bf16 out, with and without PrefetchLdsRead: the SM tuning (64x128x512, PrefetchLdsRead,
// GL2 distance 2, no SplitBLdsRead, regions not interleaved; its 384 B rows straddle GL2 lines)
// and the MAB tuning (256x16x512 4x1, no PrefetchLdsRead, no GL2 prefetch).
template <typename Tuning, typename Traits>
using Fp6Sm = Kernel<pk_fp6x16_t, pk_fp6x16_t, bf16_t, 64, 128, 512, 2, 2, 16, Tuning, Traits>;
template <typename Tuning, typename Traits>
using Fp6Mab   = Kernel<pk_fp6x16_t, pk_fp6x16_t, bf16_t, 256, 16, 512, 4, 1, 16, Tuning, Traits>;
using Fp6SmTu  = Tu<4, 4, 2, true, false, false>;
using Fp6MabTu = Tu<1, 2, 0, false, false>;

// Tile counts 1 to 4 (TailNumber::One and Two) with both tunings; 4 tiles of the SM tuning is
// 256 x 2048 x 2048, the problem it is tuned for.
TEST(MxGemmTdmScheduled, Fp6TileCounts)
{
    for(index_t tiles = 1; tiles <= 4; ++tiles)
    {
        check<Fp6Sm<Fp6SmTu, Shape<false, false, 0>>>(256, 2048, tiles * 512);
        check<Fp6Mab<Fp6MabTu, Shape<false, false, 0>>>(1024, 16, tiles * 512);
    }
}

// K tails of one K step (SM) and three (MAB), each after an even and an odd whole-tile count, the
// second with a partial last K step (K a multiple of 32 only); and a partial tile of every K
// step, which runs in the loop.
TEST(MxGemmTdmScheduled, Fp6KTail)
{
    using SmTail1  = Fp6Sm<Fp6SmTu, Shape<false, false, 1>>;
    using MabTail3 = Fp6Mab<Fp6MabTu, Shape<false, false, 3>>;
    check<SmTail1>(256, 2048, 2048 + 128);
    check<SmTail1>(256, 2048, 1536 + 32);
    check<MabTail3>(1024, 16, 1536 + 384);
    check<MabTail3>(1024, 16, 2048 + 352);
    check<Fp6Sm<Fp6SmTu, Shape<false, false, 0>>>(256, 2048, 1536 + 416);
    check<Fp6Mab<Fp6MabTu, Shape<false, false, 0>>>(1024, 16, 1536 + 480);
}

// Partial M and N tiles with both tunings, the SM one with a K tail. A and B load in iterate mode
// (SM: A 2 and B 4 rows per block; MAB: A 4 and B 2), two components of half the tile each: the
// last tile's rows end inside a block, fewer rows than one block included, and leave the second
// component of A or B empty.
TEST(MxGemmTdmScheduled, Fp6Edges)
{
    using SmEdge  = Fp6Sm<Tu<1, 1, 2, true, false, false>, Shape<true, true, 1>>;
    using MabEdge = Fp6Mab<Tu<1, 1, 0, false, false>, Shape<true, true, 0>>;
    // Last tile: A 32 + 9 rows, B 64 + 3; then A 17 + 0, B 3 + 0.
    check<SmEdge>(64 + 41, 128 + 67, 1024 + 128);
    check<SmEdge>(64 + 17, 128 + 3, 1024 + 128);
    // Last tile: A 128 + 105 rows, B 8 + 1; then A 17 + 0, B 8 + 0.
    check<MabEdge>(256 + 233, 16 + 9, 2048);
    check<MabEdge>(256 + 17, 8, 2048);
}

// tile_k 128 and 256 (96 B and 192 B rows: one GL2 line each, the one holding the tile's last
// byte), each with a partial last tile in the loop, whose line is clamped at the row's end;
// SplitBLdsRead with interleaved regions, with a cluster and without one (the refill's wait then
// counts the reads in flight, 3 per FP6 fragment); the 1x4 wave group; FP8 and FP16 out.
TEST(MxGemmTdmScheduled, Fp6TilesAndOutputs)
{
    using S = Shape<false, false, 0>;
    check<Kernel<pk_fp6x16_t, pk_fp6x16_t, fp8_t, 128, 128, 128, 2, 2, 16, Tu<2, 2>, S>>(
        512, 512, 1024 + 32);
    check<Kernel<pk_fp6x16_t, pk_fp6x16_t, half_t, 128, 128, 256, 2, 2, 16, Tu<1, 1>, S>>(
        512, 512, 2048 + 160);
    check<Kernel<pk_fp6x16_t, pk_fp6x16_t, bf16_t, 64, 256, 512, 1, 4, 16, Tu<2, 2>, S>>(
        512, 512, 2048);
}

// The tile after the K loop reads no next tile and refills no buffer. Each schedule variant with
// a K tail after an odd and an even whole-tile count, one whole tile included; SplitB without a
// cluster is the variant whose refill waited on the next tile's reads.
TEST(MxGemmTdmScheduled, KTailAfterLoop)
{
    using S = Shape<true, true, 1>;
    for(index_t tiles : {1, 7, 8})
    {
        const index_t K = tiles * 256 + 96;
        check<Kernel<fp8_t, bf8_t, fp8_t, 128, 128, 256, 2, 2, 16, Tu<1, 1>, S>>(1000, 1000, K);
        if(tiles % 2 != 0)
        {
            check_kernel<Kernel<fp8_t, bf8_t, fp8_t, 128, 128, 256, 2, 2, 16, Tu<1, 1>, S>>(
                1000, 1000, K, kSparseScaleShift);
            check_kernel<Kernel<fp8_t,
                                bf8_t,
                                fp8_t,
                                128,
                                128,
                                256,
                                2,
                                2,
                                16,
                                Tu<1, 1>,
                                Shape<true, true, 0>>>(1000, 1000, tiles * 256, kSparseScaleShift);
            check_kernel<
                Kernel<fp8_t, bf8_t, fp8_t, 128, 128, 256, 2, 2, 16, Tu<1, 1, 4, false, false>, S>>(
                1000, 1000, K, kSparseScaleShift);
            check_kernel<Kernel<fp8_t,
                                bf8_t,
                                fp8_t,
                                128,
                                128,
                                256,
                                2,
                                2,
                                16,
                                Tu<1, 1, 4, false, false>,
                                Shape<true, true, 0>>>(1000, 1000, tiles * 256, kSparseScaleShift);
        }
        check<Kernel<fp8_t, bf8_t, fp8_t, 128, 128, 256, 2, 2, 16, Tu<1, 1, 4, true, false>, S>>(
            1000, 1000, K);
        check<Kernel<fp8_t, bf8_t, fp8_t, 128, 128, 256, 2, 2, 16, Tu<1, 1, 4, false, false>, S>>(
            1000, 1000, K);
    }
    check<Fp4K1024<Tu<1, 1>>>(512, 512, 3 * 1024 + 128);
    check<Fp4K1024<Tu<1, 1>>>(512, 512, 4 * 1024 + 128);
}

// FP4 tile_k 1024: two GL2 lines per row. With a K tail of 512 or less, the tail's second line
// starts past the end of its row, and of the last row of A and B past the end of the buffer: the
// GL2 prefetch clamps it to the row's last byte.
TEST(MxGemmTdmScheduled, Gl2PrefetchKTail)
{
    check<Fp4K1024<Tu<1, 1>>>(512, 512, 2 * 1024 + 32);
    check<Fp4K1024<Tu<1, 1>>>(512, 512, 3 * 1024 + 32);
    check<Fp4K1024<Tu<2, 2>>>(512, 512, 4 * 1024 + 128);
    check<Fp4K1024<Tu<2, 2>>>(512, 512, 5 * 1024 + 32);
}

// A partial last tile of every K step (K mod tile_k above tile_k - 128) runs in the loop as one
// more tile on an instance without a K tail, zero-filled past K, its last K step partial (its
// scales zero-padded). Loop tile counts 3, 8 and 9 run the TailNumber::One and Two kernels.
TEST(MxGemmTdmScheduled, LongKTailInTheLoop)
{
    using S = Shape<false, false, 0>;
    for(index_t tiles : {2, 7, 8})
    {
        check<Kernel<fp8_t, bf8_t, fp8_t, 128, 128, 128, 2, 2, 16, Tu<1, 1>, S>>(
            1024, 1024, tiles * 128 + 32);
        check<Kernel<fp8_t, bf8_t, fp8_t, 128, 128, 256, 2, 2, 16, Tu<1, 1>, S>>(
            1024, 1024, tiles * 256 + 160);
        check<Kernel<fp8_t, bf8_t, fp8_t, 128, 128, 256, 2, 2, 16, Tu<1, 1, 4, false, false>, S>>(
            1024, 1024, tiles * 256 + 256 - 32);
        check<Fp4C4x4>(1024, 1024, tiles * 512 + 416);
    }
}

TEST(MxGemmTdmScheduled, Edges)
{
    using Edge = Fp4Big<fp8_t, Tu<1, 1>, Shape<true, true, 0>>;
    check<Edge>(1000, 1000, 2048);
    check<Edge>(1000, 1001, 2048);
    check<Edge>(1, 1000, 2048);
    check<Edge>(1000, 8, 2048);
    // An edge kernel runs whole tiles too.
    check<Edge>(1024, 1024, 2048);
    // N odd.
    check<Kernel<pk_fp4_t,
                 pk_fp4_t,
                 bf16_t,
                 128,
                 128,
                 512,
                 2,
                 2,
                 32,
                 Tu<1, 1>,
                 Shape<true, true, 0>>>(1024, 1001, 2048);
}

/// E past the output range: scales 2^3, 2^8 and 2^14 above the sparse check's on each operand
/// (alpha 1, 2^10 and 2^22) take E past +-448 while the f32 sums stay exact. FP8
/// saturates to +-448; FP16 rounds without clamping, to inf past 65504.
TEST(MxGemmTdmScheduled, Fp8Saturation)
{
    using S = Shape<false, false, 0>;
    for(int shift : {0, 5, 11})
    {
        check<Fp4Big<fp8_t, Tu<1, 1>, S>>(256, 256, 2048, shift);
        check<Fp4Big<half_t, Tu<1, 1>, S>>(256, 256, 2048, shift);
    }
}

} // namespace
