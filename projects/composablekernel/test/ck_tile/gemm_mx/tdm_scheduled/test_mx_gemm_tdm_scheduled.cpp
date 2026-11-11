// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// comp_tdm_scheduled on gfx1250 against reference_mx_gemm_tdm_scheduled, bit for bit on rocKE's
// sparse check inputs: every type and output, both atoms and wave groups, each schedule variant,
// clusters, K paths (even and odd tile counts, K tails, K a multiple of 32 only), edges, alpha /
// beta and FP8 saturation. Needs a gfx1250 device.

#include "test_mx_gemm_tdm_scheduled_util.hpp"

namespace {

using namespace mx_tdm_scheduled_test;
using ck_tile::bf16_t;
using ck_tile::bf8_t;
using ck_tile::fp8_t;
using ck_tile::half_t;
using ck_tile::MxTdmScheduledHalfBOrder;
using ck_tile::MxTdmScheduledHalfPlr;
using ck_tile::pk_fp4_t;

template <typename O, typename Tuning, typename Traits>
using Fp4Big = Kernel<pk_fp4_t, pk_fp4_t, O, 256, 256, 512, 2, 2, 32, Tuning, Traits>;

// The rocKE MAF winner, mx_gemm_mxfp4_t256x256x512_c4x4.
using Maf         = Fp4Big<fp8_t, Tu<4, 4>, Shape<false, false, 0, 0, 8>>;
using MafOdd      = Fp4Big<fp8_t, Tu<4, 4>, Shape<false, false, 1, 0, 8>>;
using MafAny      = Fp4Big<fp8_t, Tu<4, 4>, Shape<false, false, -1, 0, 8>>;
using MafTail     = Fp4Big<fp8_t, Tu<4, 4>, Shape<false, false, 0, 1, 8>>;
using MafOddTail3 = Fp4Big<fp8_t, Tu<4, 4>, Shape<false, false, 1, 3, 8>>;

TEST(MxGemmTdmScheduled, Fp4MafEvenTiles) { check<Maf>(1024, 1024, 2048); }
TEST(MxGemmTdmScheduled, Fp4MafLongK) { check<Maf>(1024, 1024, 65536); }
TEST(MxGemmTdmScheduled, Fp4MafOddTilesPeel)
{
    check<MafOdd>(1024, 1024, 512);
    check<MafOdd>(1024, 1024, 1536);
}
TEST(MxGemmTdmScheduled, Fp4AnyParityRunsBoth)
{
    check<MafAny>(1024, 1024, 512);
    check<MafAny>(1024, 1024, 1024);
    check<MafAny>(1024, 1024, 2560);
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
    check<Fp4Big<half_t, Tu<4, 4>, Shape<false, false, 0, 0, 8>>>(1024, 1024, 2048);
    check<Fp4Big<bf16_t, Tu<4, 4>, Shape<false, false, 0, 0, 8>>>(1024, 1024, 2048);
}

TEST(MxGemmTdmScheduled, Fp4Clusters)
{
    check<Fp4Big<fp8_t, Tu<1, 1>, Shape<false, false, 0, 0, 8>>>(512, 768, 2048);
    check<Fp4Big<fp8_t, Tu<2, 2>, Shape<false, false, 0, 0, 8>>>(512, 1024, 2048);
    check<Fp4Big<fp8_t, Tu<4, 2>, Shape<false, false, 0, 0, 8>>>(512, 1024, 2048);
    check<Fp4Big<fp8_t, Tu<2, 2, 4, 3>, Shape<false, false, 0, 0, 8>>>(512, 1024, 2048);
}

TEST(MxGemmTdmScheduled, Fp4ScheduleVariants)
{
    using S = Shape<false, false, 0, 0, 8>;
    check<Fp4Big<fp8_t,
                 Tu<4, 4, 2, -1, 1, MxTdmScheduledHalfPlr::B, MxTdmScheduledHalfBOrder::Late>,
                 S>>(1024, 1024, 4096);
    check<Fp4Big<fp8_t, Tu<4, 4, 4, -1, 1, MxTdmScheduledHalfPlr::None>, S>>(1024, 1024, 4096);
    check<Fp4Big<fp8_t, Tu<4, 4, 4, -1, 0, MxTdmScheduledHalfPlr::None>, S>>(1024, 1024, 4096);
    check<
        Fp4Big<fp8_t,
               Tu<4, 4, 4, -1, 1, MxTdmScheduledHalfPlr::B, MxTdmScheduledHalfBOrder::Ahead, false>,
               S>>(1024, 1024, 4096);
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
                 Shape<false, false, 0, 0, 8>>>(1024, 1024, 2048);
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
                 Shape<false, false, 0, 0, 4>>>(512, 512, 4096);
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
                 Shape<false, false, 0, 0, 4>>>(512, 512, 2048);
    check<
        Kernel<pk_fp4_t,
               pk_fp4_t,
               fp8_t,
               256,
               64,
               512,
               4,
               1,
               32,
               Tu<1, 1, 4, -1, 1, MxTdmScheduledHalfPlr::B, MxTdmScheduledHalfBOrder::Late, false>,
               Shape<false, false, 0, 0, 4>>>(512, 512, 2048);
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
                 Shape<false, false, 0, 0, 8>>>(1024, 1024, 2048);
    check<Kernel<pk_fp4_t,
                 pk_fp4_t,
                 fp8_t,
                 128,
                 16,
                 512,
                 4,
                 1,
                 16,
                 Tu<1, 4, 0, -1, 0, MxTdmScheduledHalfPlr::None>,
                 Shape<false, false, 0, 0, 1>>>(2048, 16, 4096);
    check<Kernel<pk_fp4_t,
                 pk_fp4_t,
                 fp8_t,
                 64,
                 16,
                 512,
                 4,
                 1,
                 16,
                 Tu<1, 1, 4, -1, 1, MxTdmScheduledHalfPlr::None>,
                 Shape<true, false, 0, 0, 1>>>(1000, 16, 2048);
}

TEST(MxGemmTdmScheduled, EightBitTypes)
{
    using S = Shape<false, false, 0, 0, 4>;
    check<Kernel<fp8_t, fp8_t, fp8_t, 128, 128, 256, 2, 2, 16, Tu<2, 2>, S>>(512, 512, 2048);
    check<Kernel<bf8_t, bf8_t, fp8_t, 128, 128, 256, 2, 2, 16, Tu<2, 2>, S>>(512, 512, 2048);
    check<Kernel<fp8_t, bf8_t, bf16_t, 128, 128, 256, 2, 2, 16, Tu<2, 2>, S>>(512, 512, 2048);
    check<Kernel<bf8_t,
                 fp8_t,
                 half_t,
                 128,
                 128,
                 128,
                 2,
                 2,
                 16,
                 Tu<2, 2, 4, -1, 1, MxTdmScheduledHalfPlr::None>,
                 S>>(512, 512, 2048);
    check<Kernel<fp8_t,
                 fp8_t,
                 fp8_t,
                 128,
                 128,
                 256,
                 2,
                 2,
                 16,
                 Tu<2, 2, 4, -1, 1, MxTdmScheduledHalfPlr::B, MxTdmScheduledHalfBOrder::Late>,
                 S>>(512, 512, 2048);
    check<Kernel<fp8_t,
                 fp8_t,
                 fp8_t,
                 128,
                 128,
                 256,
                 2,
                 2,
                 16,
                 Tu<2, 2, 4, -1, 0, MxTdmScheduledHalfPlr::None>,
                 S>>(512, 512, 2048);
    // 8-bit edge and K tail (rocKE's 1000 x 1000 x (2048 + 96) bf8 check).
    check<
        Kernel<fp8_t, bf8_t, fp8_t, 128, 128, 256, 2, 2, 16, Tu<1, 1>, Shape<true, true, 0, 1, 4>>>(
        1000, 1000, 2048 + 96);
}

TEST(MxGemmTdmScheduled, Edges)
{
    using Edge = Fp4Big<fp8_t, Tu<1, 1>, Shape<true, true, 0, 0, 8>>;
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
                 Shape<true, true, 0, 0, 1>>>(1024, 1001, 2048);
}

/// alpha and beta beyond the sparse check: FP8 saturation (+-448, inf to NaN, NaN kept) and inf /
/// NaN in C.
TEST(MxGemmTdmScheduled, AlphaBetaAndSaturation)
{
    using K8        = Fp4Big<fp8_t, Tu<1, 1>, Shape<false, false, 0, 0, 8>>;
    using K16       = Fp4Big<half_t, Tu<1, 1>, Shape<false, false, 0, 0, 8>>;
    const index_t M = 256, N = 256, Kd = 2048;
    for(float alpha : {1.0f, -0.25f, 1024.0f, 65536.0f})
        for(float beta : {0.0f, 1.0f, -2.0f})
        {
            SCOPED_TRACE("alpha " + std::to_string(alpha) + " beta " + std::to_string(beta));
            {
                auto in = make_inputs(MxTdmScheduledDtype::Fp4,
                                      MxTdmScheduledDtype::Fp4,
                                      MxTdmScheduledOutDtype::Fp8E4M3,
                                      M,
                                      N,
                                      Kd,
                                      512);
                // fp8 C codes: +inf does not exist in e4m3; NaN (0x7F) and +-448.
                in.c[3]         = 0x7F;
                in.c[5]         = 0x7E;
                in.c[7]         = 0xFE;
                const auto got  = run_device<K8>(in, M, N, Kd, alpha, beta);
                const auto want = run_reference(MxTdmScheduledDtype::Fp4,
                                                MxTdmScheduledDtype::Fp4,
                                                MxTdmScheduledOutDtype::Fp8E4M3,
                                                in,
                                                M,
                                                N,
                                                Kd,
                                                alpha,
                                                beta);
                EXPECT_EQ(first_mismatch(got, want, MxTdmScheduledOutDtype::Fp8E4M3, N), "");
            }
            {
                auto in = make_inputs(MxTdmScheduledDtype::Fp4,
                                      MxTdmScheduledDtype::Fp4,
                                      MxTdmScheduledOutDtype::Fp16,
                                      M,
                                      N,
                                      Kd,
                                      512);
                sched::store_out(MxTdmScheduledOutDtype::Fp16,
                                 in.c.data(),
                                 3,
                                 std::numeric_limits<float>::infinity());
                sched::store_out(MxTdmScheduledOutDtype::Fp16,
                                 in.c.data(),
                                 5,
                                 std::numeric_limits<float>::quiet_NaN());
                const auto got  = run_device<K16>(in, M, N, Kd, alpha, beta);
                const auto want = run_reference(MxTdmScheduledDtype::Fp4,
                                                MxTdmScheduledDtype::Fp4,
                                                MxTdmScheduledOutDtype::Fp16,
                                                in,
                                                M,
                                                N,
                                                Kd,
                                                alpha,
                                                beta);
                EXPECT_EQ(first_mismatch(got, want, MxTdmScheduledOutDtype::Fp16, N), "");
            }
        }
}

TEST(MxGemmTdmScheduled, BetaZeroDoesNotNeedC)
{
    using K8        = Fp4Big<fp8_t, Tu<1, 1>, Shape<false, false, 0, 0, 8>>;
    const index_t M = 256, N = 256, Kd = 2048;
    auto in         = make_inputs(MxTdmScheduledDtype::Fp4,
                          MxTdmScheduledDtype::Fp4,
                          MxTdmScheduledOutDtype::Fp8E4M3,
                          M,
                          N,
                          Kd,
                          512);
    const auto got  = run_device<K8>(in, M, N, Kd, kSparseAlpha, 0.0f, /*pass_c=*/false);
    const auto want = run_reference(MxTdmScheduledDtype::Fp4,
                                    MxTdmScheduledDtype::Fp4,
                                    MxTdmScheduledOutDtype::Fp8E4M3,
                                    in,
                                    M,
                                    N,
                                    Kd,
                                    kSparseAlpha,
                                    0.0f);
    EXPECT_EQ(first_mismatch(got, want, MxTdmScheduledOutDtype::Fp8E4M3, N), "");
}

} // namespace
