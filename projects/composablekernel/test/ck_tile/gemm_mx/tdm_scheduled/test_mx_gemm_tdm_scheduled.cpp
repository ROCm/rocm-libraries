// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// comp_tdm_scheduled on gfx1250 against reference_mx_gemm_tdm_scheduled, bit for bit on the
// sparse check inputs: every type and output, both atoms and wave groups, each schedule variant,
// clusters, K paths (even and odd tile counts, K tails, K a multiple of 32 only), edges, FP8
// saturation and the TdmEpilogue alternative. Needs a gfx1250 device.

#include "test_mx_gemm_tdm_scheduled_util.hpp"

namespace {

using namespace mx_tdm_scheduled_test;
using ck_tile::bf16_t;
using ck_tile::bf8_t;
using ck_tile::fp8_t;
using ck_tile::half_t;
using ck_tile::pk_fp4_t;

template <typename O, typename Tuning, typename Traits>
using Fp4Big = Kernel<pk_fp4_t, pk_fp4_t, O, 256, 256, 512, 2, 2, 32, Tuning, Traits>;
template <typename O, typename Tuning, typename Traits>
using Fp4BigTdm = TdmKernel<pk_fp4_t, pk_fp4_t, O, 256, 256, 512, 2, 2, 32, Tuning, Traits>;

// The tuned 4096 x 4096 FP4 instance: 256x256x512 tile, 4x4 cluster.
using Maf         = Fp4Big<fp8_t, Tu<4, 4>, Shape<false, false, 0, 8>>;
using MafTail     = Fp4Big<fp8_t, Tu<4, 4>, Shape<false, false, 1, 8>>;
using MafOddTail3 = Fp4Big<fp8_t, Tu<4, 4>, Shape<false, false, 3, 8>>;

TEST(MxGemmTdmScheduled, Fp4MafEvenTiles) { check<Maf>(1024, 1024, 2048); }
TEST(MxGemmTdmScheduled, Fp4MafLongK) { check<Maf>(1024, 1024, 65536); }
TEST(MxGemmTdmScheduled, Fp4MafOddTiles)
{
    check<Maf>(1024, 1024, 512);
    check<Maf>(1024, 1024, 1536);
    check<Maf>(1024, 1024, 2560);
}
// One instance serves every whole-tile count: 1 and 2 run without the loop, 3 and 4 with it,
// each with TailNumber One (odd) or Two (even).
TEST(MxGemmTdmScheduled, Fp4MafTileCounts)
{
    for(index_t tiles = 1; tiles <= 4; ++tiles)
    {
        check<Maf>(1024, 1024, tiles * 512);
        check<MafTail>(1024, 1024, tiles * 512 + 128);
    }
}
TEST(MxGemmTdmScheduled, Fp4KTail)
{
    check<MafTail>(1024, 1024, 2048 + 128);
    // A partial last K step: K a multiple of 32 only, its scales zero-padded.
    check<MafTail>(1024, 1024, 2048 + 32);
    check<MafTail>(1024, 1024, 2048 + 96);
    check<MafOddTail3>(1024, 1024, 1536 + 384);
}

TEST(MxGemmTdmScheduled, Fp4OutputTypes)
{
    check<Fp4Big<half_t, Tu<4, 4>, Shape<false, false, 0, 8>>>(1024, 1024, 2048);
    check<Fp4Big<bf16_t, Tu<4, 4>, Shape<false, false, 0, 8>>>(1024, 1024, 2048);
}

TEST(MxGemmTdmScheduled, Fp4Clusters)
{
    check<Fp4Big<fp8_t, Tu<1, 1>, Shape<false, false, 0, 8>>>(512, 768, 2048);
    check<Fp4Big<fp8_t, Tu<2, 2>, Shape<false, false, 0, 8>>>(512, 1024, 2048);
    check<Fp4Big<fp8_t, Tu<4, 2>, Shape<false, false, 0, 8>>>(512, 1024, 2048);
}

TEST(MxGemmTdmScheduled, Fp4ScheduleVariants)
{
    using S = Shape<false, false, 0, 8>;
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
                 Shape<false, false, 0, 8>>>(1024, 1024, 2048);
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
                 Shape<false, false, 0, 4>>>(512, 512, 4096);
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
                 Shape<false, false, 0, 4>>>(512, 512, 2048);
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
                 Shape<false, false, 0, 4>>>(512, 512, 2048);
    // The 16x16 atom with FP4, and the MAB shape: tile_n 16, one N repeat.
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
                 Shape<false, false, 0, 8>>>(1024, 1024, 2048);
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
                 Shape<false, false, 0, 1>>>(2048, 16, 4096);
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
                 Shape<true, false, 0, 1>>>(1000, 16, 2048);
}

TEST(MxGemmTdmScheduled, EightBitTypes)
{
    using S = Shape<false, false, 0, 4>;
    check<Kernel<fp8_t, fp8_t, fp8_t, 128, 128, 256, 2, 2, 16, Tu<2, 2>, S>>(512, 512, 2048);
    check<Kernel<bf8_t, bf8_t, fp8_t, 128, 128, 256, 2, 2, 16, Tu<2, 2>, S>>(512, 512, 2048);
    check<Kernel<fp8_t, bf8_t, bf16_t, 128, 128, 256, 2, 2, 16, Tu<2, 2>, S>>(512, 512, 2048);
    check<Kernel<bf8_t, fp8_t, half_t, 128, 128, 128, 2, 2, 16, Tu<2, 2, 4, true, false>, S>>(
        512, 512, 2048);
    check<Kernel<fp8_t, fp8_t, fp8_t, 128, 128, 256, 2, 2, 16, Tu<2, 2, 4, false, false>, S>>(
        512, 512, 2048);
    // 8-bit edge and K tail: 1000 x 1000 x (2048 + 96) with bf8 B.
    check<Kernel<fp8_t, bf8_t, fp8_t, 128, 128, 256, 2, 2, 16, Tu<1, 1>, Shape<true, true, 1, 4>>>(
        1000, 1000, 2048 + 96);
}

TEST(MxGemmTdmScheduled, Edges)
{
    using Edge = Fp4Big<fp8_t, Tu<1, 1>, Shape<true, true, 0, 8>>;
    check<Edge>(1000, 1000, 2048);
    check<Edge>(1, 1000, 2048);
    check<Edge>(1000, 8, 2048);
    // An edge kernel runs whole tiles too.
    check<Edge>(1024, 1024, 2048);
    // N odd: one-element stores.
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
                 Shape<true, true, 0, 1>>>(1024, 1001, 2048);
}

/// E past the output range: scales 2^3, 2^8 and 2^14 above the sparse check's on each operand
/// (alpha 1, 2^10 and 2^22) take E past +-448 while the f32 sums stay exact. FP8
/// saturates to +-448; FP16 rounds without clamping, to inf past 65504.
TEST(MxGemmTdmScheduled, Fp8Saturation)
{
    using S = Shape<false, false, 0, 8>;
    for(int shift : {0, 5, 11})
    {
        check<Fp4Big<fp8_t, Tu<1, 1>, S>>(256, 256, 2048, shift);
        check<Fp4Big<half_t, Tu<1, 1>, S>>(256, 256, 2048, shift);
        check<Fp4BigTdm<fp8_t, Tu<1, 1>, Shape<false, false, 0, 1>>>(256, 256, 2048, shift);
    }
}

/// TdmEpilogue (E through LDS, stored by TDM) instead of the direct store: the MAF and MAB shapes,
/// FP16 output and edges with N odd.
TEST(MxGemmTdmScheduled, TdmEpilogue)
{
    using S = Shape<false, false, 0, 1>;
    check<Fp4BigTdm<fp8_t, Tu<4, 4>, S>>(1024, 1024, 2048);
    check<Fp4BigTdm<half_t, Tu<4, 4>, S>>(1024, 1024, 2048);
    check<
        TdmKernel<pk_fp4_t, pk_fp4_t, fp8_t, 128, 16, 512, 4, 1, 16, Tu<1, 4, 0, false, false>, S>>(
        2048, 16, 4096);
    using Edge = Fp4BigTdm<fp8_t, Tu<1, 1>, Shape<true, true, 0, 1>>;
    check<Edge>(1000, 1001, 2048);
    check<Edge>(1, 1000, 2048);
    check<TdmKernel<fp8_t,
                    bf8_t,
                    bf16_t,
                    128,
                    128,
                    256,
                    2,
                    2,
                    16,
                    Tu<1, 1>,
                    Shape<true, true, 1, 1>>>(1000, 1001, 2048 + 96);
}

} // namespace
