// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/block/variants.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_v128.hpp"
#include "fmha_fwd.hpp"

namespace tdm_v128_test {

template <int HeadDim,
          bool Group    = false,
          bool Mask     = false,
          bool Lse      = false,
          typename Data = ck_tile::bf16_t>
struct Model
{
    using Shape  = ck_tile::TileFmhaShape<ck_tile::sequence<128, 128, 32, 128, 32, HeadDim>,
                                          ck_tile::sequence<4, 1, 1>,
                                          ck_tile::sequence<16, 16, 32>,
                                          ck_tile::sequence<4, 1, 1>,
                                          ck_tile::sequence<16, 16, 32>,
                                          true>;
    using Traits = ck_tile::TileFmhaTraits<true,
                                           true,
                                           true,
                                           true,
                                           false,
                                           ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                           false,
                                           Lse,
                                           false,
                                           ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
                                           1,
                                           false,
                                           false>;
    using Problem =
        ck_tile::BlockFmhaPipelineProblem<Data,
                                          Data,
                                          Data,
                                          float,
                                          float,
                                          Data,
                                          std::uint8_t,
                                          float,
                                          Data,
                                          float,
                                          Data,
                                          Shape,
                                          Group,
                                          ck_tile::ComposedAttention<0, CK_TILE_FMHA_FWD_FAST_EXP2>,
                                          ck_tile::SimplifiedGenericAttentionMask<Mask>,
                                          false,
                                          Traits>;
    static_assert(Traits::kStoreLSE == Lse && !Traits::kHasBiasGrad);
    static_assert(Problem::kStoreLSE == Lse);
    using Geometry = ck_tile::FmhaTdmV128GeometryFor<Problem>;
};

// Test profiles are separate binaries, not a runtime mutation of a specialization.
template <typename Geometry, int Profile, bool Pack = false>
using Tuning =
    ck_tile::FmhaTdmV128Tuning<Profile == 1 || Profile >= 3 ? 1 : 0,
                               Profile >= 2 ? 1 : 0,
                               std::conditional_t<Profile == 4 || Profile == 6,
                                                  ck_tile::sequence<Geometry::kKSuLoadCount,
                                                                    Geometry::kKSuLoadCount,
                                                                    Geometry::kKSuLoadCount,
                                                                    Geometry::kVStageLoadCount>,
                                                  ck_tile::sequence<0, 0, 0, 0>>,
                               std::conditional_t<Profile == 5 || Profile == 6,
                                                  ck_tile::sequence<Geometry::kVStageLoadCount,
                                                                    Geometry::kVStageLoadCount,
                                                                    Geometry::kVStageLoadCount,
                                                                    Geometry::kKSuLoadCount>,
                                                  ck_tile::sequence<0, 0, 0, 0>>,
                               true,
                               Pack,
                               1,
                               Geometry>;

template <typename Problem, int Profile, bool Pack = false>
using Policy = ck_tile::BlockFmhaPipelineQRKSVSTdmV128Policy<
    ck_tile::FmhaTdmV128GeometryFor<Problem>,
    Tuning<ck_tile::FmhaTdmV128GeometryFor<Problem>, Profile, Pack>,
    ck_tile::FmhaTdmV128ScheduleFor<ck_tile::FmhaTdmV128GeometryFor<Problem>>>;

} // namespace tdm_v128_test
