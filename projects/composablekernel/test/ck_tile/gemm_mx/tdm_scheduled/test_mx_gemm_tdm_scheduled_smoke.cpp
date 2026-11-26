// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// comp_tdm_scheduled smoke test on gfx1250: a few small problems against
// reference_mx_gemm_tdm_scheduled, bit for bit on the sparse check inputs. FP4 whole tiles, an
// FP4 edge with a K tail, one 8-bit instance, FP6 with and without PrefetchLdsRead, and FP8
// saturation. test_mx_gemm_tdm_scheduled.cpp has the full coverage. Needs a gfx1250 device.

#include "test_mx_gemm_tdm_scheduled_util.hpp"

namespace {

using namespace mx_tdm_scheduled_test;
using ck_tile::bf16_t;
using ck_tile::fp8_t;
using ck_tile::pk_fp4_t;
using ck_tile::pk_fp6x16_t;

template <typename Traits>
using Fp4 = Kernel<pk_fp4_t, pk_fp4_t, fp8_t, 256, 256, 512, 2, 2, 32, Tu<1, 1>, Traits>;

TEST(MxGemmTdmScheduledSmoke, Fp4) { check<Fp4<Shape<false, false, 0>>>(256, 256, 1024); }

TEST(MxGemmTdmScheduledSmoke, Fp4EdgeKTail)
{
    check<Fp4<Shape<true, true, 1>>>(250, 250, 1024 + 128);
}

TEST(MxGemmTdmScheduledSmoke, EightBit)
{
    check<Kernel<fp8_t, fp8_t, fp8_t, 128, 128, 256, 2, 2, 16, Tu<1, 1>, Shape<false, false, 0>>>(
        256, 256, 1024);
}

// FP6 without a cluster, bf16 out, 2x2 tiles and a partial last tile of every K step: the SM
// tuning (PrefetchLdsRead, GL2 distance 2, its prefetch of the partial tile clamped at the row's
// end) and the MAB tuning (no PrefetchLdsRead, no GL2 prefetch).
TEST(MxGemmTdmScheduledSmoke, Fp6PrefetchLdsRead)
{
    check<Kernel<pk_fp6x16_t,
                 pk_fp6x16_t,
                 bf16_t,
                 64,
                 128,
                 512,
                 2,
                 2,
                 16,
                 Tu<1, 1, 2, true, false, false>,
                 Shape<false, false, 0>>>(128, 256, 1024 + 416);
}

TEST(MxGemmTdmScheduledSmoke, Fp6NoPrefetchLdsRead)
{
    check<Kernel<pk_fp6x16_t,
                 pk_fp6x16_t,
                 bf16_t,
                 256,
                 16,
                 512,
                 4,
                 1,
                 16,
                 Tu<1, 1, 0, false, false>,
                 Shape<false, false, 0>>>(512, 32, 1024 + 480);
}

/// Scales 2^14 above the sparse check's on each operand take E past +-448: FP8 saturates.
TEST(MxGemmTdmScheduledSmoke, Fp8Saturation)
{
    check<Fp4<Shape<false, false, 0>>>(256, 256, 1024, 11);
}

} // namespace
