// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only checks of MxGemmPipelineAgBgCrCompTDMScheduled's compile-time rules against rocKE: the
// LDS layout of the hipBLASLt solutions, the lane chunks of a K step, and the policy building and
// rejecting specs from a Problem and a tuning as rocKE does.

#include <algorithm>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/mx_gemm_pipeline_ag_bg_cr_comp_tdm_scheduled_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"

namespace {

using ck_tile::index_t;
using ck_tile::MxTdmScheduledReason;
using ck_tile::MxTdmScheduledSpec;

using D = ck_tile::MxTdmScheduledDtype;
using A = ck_tile::MxTdmScheduledMma;
using H = ck_tile::MxTdmScheduledHalfPlr;

// rocKE's test_lds_layout_matches_the_hipblaslt_solutions: LdsOffsetA / ldsBaseB with the
// component stride, LdsOffsetMXSA / MXSB, LdsOffsetA_Blk and LdsNumBytes (Tensile A = rocKE B).
TEST(MxGemmTdmScheduledHost, LayoutMatchesTheHipblasltSolutions)
{
    MxTdmScheduledSpec maf{4096, 4096, 65536};
    MxTdmScheduledSpec mab{65536, 16, 65536};
    mab.tile_n                 = 16;
    mab.warp_m                 = 4;
    mab.warp_n                 = 1;
    mab.cluster_x              = 1;
    mab.cluster_y              = 4;
    mab.half_plr               = H::None;
    mab.lds_segment_interleave = false;
    mab.mma                    = A::M16N16;
    MxTdmScheduledSpec mab8    = mab;
    mab8.a_dtype = mab8.b_dtype = D::Fp8;
    mab8.tile_k                 = 256;
    MxTdmScheduledSpec medium{2048, 2048, 65536};
    medium.a_dtype = medium.b_dtype = D::Fp8;
    medium.tile_m = medium.tile_n = 128;
    medium.tile_k                 = 256;
    medium.mma                    = A::M16N16;
    medium.half_plr               = H::None;
    struct Case
    {
        const char* label;
        MxTdmScheduledSpec spec;
        index_t want[8];
    };
    const Case cases[] = {
        {"maf fp4", maf, {0, 99584, 33024, 33280, 132608, 137216, 141312, 282624}},
        {"mab fp4", mab, {0, 2176, 9216, 33280, 4608, 5120, 131072, 206848}},
        {"mab fp8", mab8, {0, 2176, 6656, 33280, 4352, 4608, 131072, 204288}},
        {"medium fp8", medium, {0, 65536, 16640, 65536, 98816, 99840, 131072, 231936}},
    };
    for(const Case& c : cases)
    {
        SCOPED_TRACE(c.label);
        ASSERT_EQ(ck_tile::mx_tdm_scheduled_validate(c.spec), MxTdmScheduledReason::Ok);
        const ck_tile::MxTdmScheduledLdsLayout L = ck_tile::mx_tdm_scheduled_lds_layout(c.spec);
        const index_t got[8]                     = {L.b.base,
                                                    L.b.stride,
                                                    L.a.base,
                                                    L.a.stride,
                                                    L.b_scale.base,
                                                    L.a_scale.base,
                                                    L.buffer_stride,
                                                    L.lds_bytes};
        for(int i = 0; i < 8; ++i)
            EXPECT_EQ(got[i], c.want[i]) << "field " << i;
    }
}

// Lane group g reads chunk j of a row's K step at g * 16 + chunk_offset(j): (0, 32) for FP4 and
// (0, 32, 64, 96) for the 8-bit types, which tile the K step.
TEST(MxGemmTdmScheduledHost, LaneChunksTileTheKStep)
{
    for(D t : {D::Fp4, D::Fp8, D::Bf8})
    {
        MxTdmScheduledSpec s{};
        s.a_dtype = s.b_dtype = t;
        s.mma                 = t == D::Fp4 ? A::M32N16 : A::M16N16;
        std::vector<index_t> bytes;
        for(index_t g = 0; g < 2; ++g)
            for(index_t j = 0; j < s.chunks_per_lane(); ++j)
                for(index_t b = 0; b < s.chunk_bytes(); ++b)
                    bytes.push_back(g * s.chunk_bytes() + s.chunk_offset(j) + b);
        std::sort(bytes.begin(), bytes.end());
        ASSERT_EQ(static_cast<index_t>(bytes.size()), s.step_bytes());
        for(index_t i = 0; i < s.step_bytes(); ++i)
            EXPECT_EQ(bytes[i], i);
        EXPECT_EQ(s.chunk_offset(1), 32);
    }
}

// The cluster multicast masks hold exactly the workgroups sharing the coordinate, with bit
// x + y * cluster_x for workgroup (x, y), for every cluster of at most 16 workgroups.
TEST(MxGemmTdmScheduledHost, ClusterMasksHoldTheWorkgroupsSharingACoordinate)
{
    for(index_t cx = 1; cx <= 16; ++cx)
        for(index_t cy = 1; cx * cy <= 16; ++cy)
            for(index_t x = 0; x < cx; ++x)
                for(index_t y = 0; y < cy; ++y)
                {
                    uint32_t same_x = 0;
                    uint32_t same_y = 0;
                    for(index_t j = 0; j < cy; ++j)
                        for(index_t i = 0; i < cx; ++i)
                        {
                            same_x |= static_cast<uint32_t>(i == x) << (i + j * cx);
                            same_y |= static_cast<uint32_t>(j == y) << (i + j * cx);
                        }
                    EXPECT_EQ(ck_tile::cluster_mask_same_x(cx, cy, x), same_x);
                    EXPECT_EQ(ck_tile::cluster_mask_same_y(cx, y), same_y);
                }
}

// The policy builds the spec from a Problem and a tuning, both ways of writing a tuning agree,
// and the compile-time validator and name follow.
struct MafLateTuning : ck_tile::MxTdmScheduledDefaultTuning
{
    static constexpr index_t kGl2PrefetchDistance = 2;
    static constexpr ck_tile::MxTdmScheduledHalfBOrder kHalfBOrder =
        ck_tile::MxTdmScheduledHalfBOrder::Late;
};

using MafShape   = ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 512>,
                                          ck_tile::sequence<2, 2, 1>,
                                          ck_tile::sequence<32, 16, 128>>;
using MafProblem = ck_tile::
    MxGemmTdmScheduledProblem<ck_tile::pk_fp4_t, ck_tile::pk_fp4_t, ck_tile::fp8_t, MafShape>;
using MabShape   = ck_tile::TileGemmShape<ck_tile::sequence<128, 16, 512>,
                                          ck_tile::sequence<4, 1, 1>,
                                          ck_tile::sequence<16, 16, 128>>;
using MabProblem = ck_tile::
    MxGemmTdmScheduledProblem<ck_tile::pk_fp4_t, ck_tile::pk_fp4_t, ck_tile::half_t, MabShape>;
using MabTuning = ck_tile::MxTdmScheduledTuningT<1,
                                                 4,
                                                 0,
                                                 -1,
                                                 0,
                                                 ck_tile::MxTdmScheduledHalfPlr::None,
                                                 ck_tile::MxTdmScheduledHalfBOrder::Ahead,
                                                 true>;

TEST(MxGemmTdmScheduledHost, PolicyBuildsTheSpecFromProblemAndTuning)
{
    using Default = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<>;
    static_assert(Default::GetSmemSize<MafProblem>() == 282624);
    static_assert(Default::GetReason<MafProblem>() == MxTdmScheduledReason::Ok);
    static_assert(ck_tile::MxTdmScheduledValidate<MafProblem>() == nullptr);
    // 3 scale reads and 2 chunks of 4 x 2 A rows and 8 B fragments.
    static_assert(Default::GetDscntWait<MafProblem>(0) == 35);
    static_assert(Default::GetSignalWmmas<MafProblem>() == 6);
    const MxTdmScheduledSpec s = Default::GetSpec<MafProblem>(4096, 4096, 65536);
    EXPECT_EQ(ck_tile::mx_tdm_scheduled_kernel_name(s, "mxfp4_gemm"),
              "mxfp4_gemm_mxfp4_M4096N4096K65536_t256x256x512_c4x4");
    EXPECT_EQ(ck_tile::MxTdmScheduledTuningName<ck_tile::MxTdmScheduledDefaultTuning>(), "_c4x4");

    using Late = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<MafLateTuning>;
    static_assert(Late::GetSpec<MafProblem>().half_b_late());
    EXPECT_EQ(ck_tile::MxTdmScheduledTuningName<MafLateTuning>(), "_c4x4_pf2_hblate");
    using LateT = ck_tile::MxTdmScheduledTuningT<4,
                                                 4,
                                                 2,
                                                 -1,
                                                 1,
                                                 ck_tile::MxTdmScheduledHalfPlr::B,
                                                 ck_tile::MxTdmScheduledHalfBOrder::Late,
                                                 true>;
    EXPECT_EQ(ck_tile::MxTdmScheduledTuningName<LateT>(),
              ck_tile::MxTdmScheduledTuningName<MafLateTuning>());

    using Mab = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<MabTuning>;
    static_assert(Mab::GetReason<MabProblem>() == MxTdmScheduledReason::Ok);
    const MxTdmScheduledSpec m = Mab::GetSpec<MabProblem>(65536, 16, 65536);
    EXPECT_EQ(
        ck_tile::mx_tdm_scheduled_kernel_name(m, "mxfp4_gemm"),
        "mxfp4_gemm_mxfp4_M65536N16K65536_t128x16x512_c1x4_fp16_wg4x1_pf0_lp0_hplrnone_mma16");
    EXPECT_EQ(ck_tile::MxTdmScheduledTuningName<MabTuning>(), "_c1x4_pf0_lp0_hplrnone");
}

struct Cluster8x4 : ck_tile::MxTdmScheduledDefaultTuning
{
    static constexpr index_t kClusterDimX = 8;
};

TEST(MxGemmTdmScheduledHost, PolicyRejectsWhatRockeRejects)
{
    using Shape1x1 = ck_tile::TileGemmShape<ck_tile::sequence<16, 16, 128>,
                                            ck_tile::sequence<2, 2, 1>,
                                            ck_tile::sequence<32, 16, 128>>;
    using Small    = ck_tile::
        MxGemmTdmScheduledProblem<ck_tile::pk_fp4_t, ck_tile::pk_fp4_t, ck_tile::fp8_t, Shape1x1>;
    EXPECT_STREQ((ck_tile::MxTdmScheduledValidate<Small>()), "tile_atoms");
    using Fp8Atom32 = ck_tile::
        MxGemmTdmScheduledProblem<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::fp8_t, MafShape>;
    EXPECT_STREQ((ck_tile::MxTdmScheduledValidate<Fp8Atom32>()), "tile_k");
    using Mixed = ck_tile::
        MxGemmTdmScheduledProblem<ck_tile::pk_fp4_t, ck_tile::fp8_t, ck_tile::fp8_t, MafShape>;
    EXPECT_STREQ((ck_tile::MxTdmScheduledValidate<Mixed>()), "dtype_width");
    using F32Out =
        ck_tile::MxGemmTdmScheduledProblem<ck_tile::pk_fp4_t, ck_tile::pk_fp4_t, float, MafShape>;
    EXPECT_STREQ((ck_tile::MxTdmScheduledValidate<F32Out>()), "out_dtype");
    EXPECT_STREQ((ck_tile::MxTdmScheduledValidate<MafProblem, Cluster8x4>()), "cluster_size");
}

} // namespace
