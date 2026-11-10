// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only checks of GemmPipelineAgBgCrCompTDMV4's compile-time rules against rocKE: the LDS
// layout of the hipBLASLt solutions, the lane chunks of a K step, and the policy building and
// rejecting specs from a Problem and a tuning as rocKE does.

#include <algorithm>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_tdm_v4_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"

namespace {

using ck_tile::index_t;
using ck_tile::MxTdmV4Reason;
using ck_tile::MxTdmV4Spec;

using D = ck_tile::MxTdmV4Dtype;
using A = ck_tile::MxTdmV4Mma;
using H = ck_tile::MxTdmV4HalfPlr;

// rocKE's test_lds_layout_matches_the_hipblaslt_solutions: LdsOffsetA / ldsBaseB with the
// component stride, LdsOffsetMXSA / MXSB, LdsOffsetA_Blk and LdsNumBytes (Tensile A = rocKE B).
TEST(MxGemmTdmV4Host, LayoutMatchesTheHipblasltSolutions)
{
    MxTdmV4Spec maf{4096, 4096, 65536};
    MxTdmV4Spec mab{65536, 16, 65536};
    mab.tile_n                 = 16;
    mab.warp_m                 = 4;
    mab.warp_n                 = 1;
    mab.cluster_x              = 1;
    mab.cluster_y              = 4;
    mab.half_plr               = H::None;
    mab.lds_segment_interleave = false;
    mab.mma                    = A::M16N16;
    MxTdmV4Spec mab8           = mab;
    mab8.a_dtype = mab8.b_dtype = D::Fp8;
    mab8.tile_k                 = 256;
    MxTdmV4Spec medium{2048, 2048, 65536};
    medium.a_dtype = medium.b_dtype = D::Fp8;
    medium.tile_m = medium.tile_n = 128;
    medium.tile_k                 = 256;
    medium.mma                    = A::M16N16;
    medium.half_plr               = H::None;
    struct Case
    {
        const char* label;
        MxTdmV4Spec spec;
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
        ASSERT_EQ(ck_tile::mx_tdm_v4_validate(c.spec), MxTdmV4Reason::Ok);
        const ck_tile::MxTdmV4LdsLayout L = ck_tile::mx_tdm_v4_lds_layout(c.spec);
        const index_t got[8]              = {L.b.base,
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
TEST(MxGemmTdmV4Host, LaneChunksTileTheKStep)
{
    for(D t : {D::Fp4, D::Fp8, D::Bf8})
    {
        MxTdmV4Spec s{};
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

// The policy builds the spec from a Problem and a tuning, both ways of writing a tuning agree,
// and the compile-time validator and name follow.
struct MafLateTuning : ck_tile::MxTdmV4DefaultTuning
{
    static constexpr index_t kGl2PrefetchDistance           = 2;
    static constexpr ck_tile::MxTdmV4HalfBOrder kHalfBOrder = ck_tile::MxTdmV4HalfBOrder::Late;
};

using MafShape = ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 512>,
                                        ck_tile::sequence<2, 2, 1>,
                                        ck_tile::sequence<32, 16, 128>>;
using MafProblem =
    ck_tile::MxGemmTdmV4Problem<ck_tile::pk_fp4_t, ck_tile::pk_fp4_t, ck_tile::fp8_t, MafShape>;
using MabShape = ck_tile::TileGemmShape<ck_tile::sequence<128, 16, 512>,
                                        ck_tile::sequence<4, 1, 1>,
                                        ck_tile::sequence<16, 16, 128>>;
using MabProblem =
    ck_tile::MxGemmTdmV4Problem<ck_tile::pk_fp4_t, ck_tile::pk_fp4_t, ck_tile::half_t, MabShape>;
using MabTuning = ck_tile::MxTdmV4TuningT<1,
                                          4,
                                          0,
                                          -1,
                                          0,
                                          ck_tile::MxTdmV4HalfPlr::None,
                                          ck_tile::MxTdmV4HalfBOrder::Ahead,
                                          true>;

TEST(MxGemmTdmV4Host, PolicyBuildsTheSpecFromProblemAndTuning)
{
    using Default = ck_tile::GemmPipelineAgBgCrCompTDMV4Policy<>;
    static_assert(Default::GetSmemSize<MafProblem>() == 282624);
    static_assert(Default::GetReason<MafProblem>() == MxTdmV4Reason::Ok);
    static_assert(ck_tile::MxTdmV4Validate<MafProblem>() == nullptr);
    // 3 scale reads and 2 chunks of 4 x 2 A rows and 8 B fragments.
    static_assert(Default::GetDscntWait<MafProblem>(0) == 35);
    static_assert(Default::GetSignalWmmas<MafProblem>() == 6);
    const MxTdmV4Spec s = Default::GetSpec<MafProblem>(4096, 4096, 65536);
    EXPECT_EQ(ck_tile::mx_tdm_v4_kernel_name(s, "mxfp4_gemm"),
              "mxfp4_gemm_mxfp4_M4096N4096K65536_t256x256x512_c4x4");
    EXPECT_EQ(ck_tile::MxTdmV4TuningName<ck_tile::MxTdmV4DefaultTuning>(), "_c4x4");

    using Late = ck_tile::GemmPipelineAgBgCrCompTDMV4Policy<MafLateTuning>;
    static_assert(Late::GetSpec<MafProblem>().half_b_late());
    EXPECT_EQ(ck_tile::MxTdmV4TuningName<MafLateTuning>(), "_c4x4_pf2_hblate");
    using LateT = ck_tile::MxTdmV4TuningT<4,
                                          4,
                                          2,
                                          -1,
                                          1,
                                          ck_tile::MxTdmV4HalfPlr::B,
                                          ck_tile::MxTdmV4HalfBOrder::Late,
                                          true>;
    EXPECT_EQ(ck_tile::MxTdmV4TuningName<LateT>(), ck_tile::MxTdmV4TuningName<MafLateTuning>());

    using Mab = ck_tile::GemmPipelineAgBgCrCompTDMV4Policy<MabTuning>;
    static_assert(Mab::GetReason<MabProblem>() == MxTdmV4Reason::Ok);
    const MxTdmV4Spec m = Mab::GetSpec<MabProblem>(65536, 16, 65536);
    EXPECT_EQ(
        ck_tile::mx_tdm_v4_kernel_name(m, "mxfp4_gemm"),
        "mxfp4_gemm_mxfp4_M65536N16K65536_t128x16x512_c1x4_fp16_wg4x1_pf0_lp0_hplrnone_mma16");
    EXPECT_EQ(ck_tile::MxTdmV4TuningName<MabTuning>(), "_c1x4_pf0_lp0_hplrnone");
}

struct Cluster8x4 : ck_tile::MxTdmV4DefaultTuning
{
    static constexpr index_t kClusterDimX = 8;
};

TEST(MxGemmTdmV4Host, PolicyRejectsWhatRockeRejects)
{
    using Shape1x1 = ck_tile::TileGemmShape<ck_tile::sequence<16, 16, 128>,
                                            ck_tile::sequence<2, 2, 1>,
                                            ck_tile::sequence<32, 16, 128>>;
    using Small =
        ck_tile::MxGemmTdmV4Problem<ck_tile::pk_fp4_t, ck_tile::pk_fp4_t, ck_tile::fp8_t, Shape1x1>;
    EXPECT_STREQ((ck_tile::MxTdmV4Validate<Small>()), "tile_atoms");
    using Fp8Atom32 =
        ck_tile::MxGemmTdmV4Problem<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::fp8_t, MafShape>;
    EXPECT_STREQ((ck_tile::MxTdmV4Validate<Fp8Atom32>()), "tile_k");
    using Mixed =
        ck_tile::MxGemmTdmV4Problem<ck_tile::pk_fp4_t, ck_tile::fp8_t, ck_tile::fp8_t, MafShape>;
    EXPECT_STREQ((ck_tile::MxTdmV4Validate<Mixed>()), "dtype_width");
    using F32Out =
        ck_tile::MxGemmTdmV4Problem<ck_tile::pk_fp4_t, ck_tile::pk_fp4_t, float, MafShape>;
    EXPECT_STREQ((ck_tile::MxTdmV4Validate<F32Out>()), "out_dtype");
    EXPECT_STREQ((ck_tile::MxTdmV4Validate<MafProblem, Cluster8x4>()), "cluster_size");
}

} // namespace
