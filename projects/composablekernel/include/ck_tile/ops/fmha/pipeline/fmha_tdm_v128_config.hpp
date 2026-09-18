// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/half.hpp"

namespace ck_tile {

namespace detail {

template <typename Tails, index_t I>
CK_TILE_HOST_DEVICE constexpr index_t fmha_tdm_v128_tail_at()
{
    if constexpr(Tails::size() > I)
    {
        return Tails::at(number<I>{});
    }
    else
    {
        return -1;
    }
}

} // namespace detail

// Homogeneous native 16-bit K32 geometry; FP8 requires separate layout validation.
template <typename Q,
          typename K,
          typename V,
          typename P,
          index_t HeadDimQK,
          index_t QkWmmaK,
          index_t PvWmmaK>
struct FmhaTdmV128Geometry
{
    using QDataType = Q;
    using KDataType = K;
    using VDataType = V;
    using PDataType = P;

    static_assert(QkWmmaK > 0 && PvWmmaK > 0, "WMMA reduction widths must be positive");
    static_assert(HeadDimQK % (QkWmmaK > 0 ? QkWmmaK : 1) == 0 &&
                      128 % (PvWmmaK > 0 ? PvWmmaK : 1) == 0,
                  "WMMA reduction widths must divide their logical dimensions");

    static constexpr index_t kM                    = 128;
    static constexpr index_t kN                    = 128;
    static constexpr index_t kDv                   = 128;
    static constexpr index_t kHeadDimQK            = HeadDimQK;
    static constexpr index_t kWaves                = 4;
    static constexpr index_t kWaveSize             = 32;
    static constexpr index_t kQueryRowsPerWave     = kM / kWaves;
    static constexpr index_t kQkSuColumns          = 32;
    static constexpr index_t kQkStages             = 4;
    static constexpr index_t kQkWmmaK              = QkWmmaK > 0 ? QkWmmaK : 1;
    static constexpr index_t kPvWmmaK              = PvWmmaK > 0 ? PvWmmaK : 1;
    static constexpr index_t kPvStages             = kN / kPvWmmaK;
    static constexpr index_t kQkWmmasPerStage      = 4 * (HeadDimQK / kQkWmmaK);
    static constexpr index_t kPvWmmasPerStage      = 16;
    static constexpr index_t kQkWmmasPerWaveKvTile = kQkStages * kQkWmmasPerStage;
    static constexpr index_t kPvWmmasPerWaveKvTile = kPvStages * kPvWmmasPerStage;
    static constexpr index_t kKSuLoadCount    = kQkSuColumns * HeadDimQK * sizeof(K) / (32 * 16);
    static constexpr index_t kVStageLoadCount = kPvWmmaK * kDv * sizeof(V) / (32 * 16);

    static_assert((std::is_same_v<QDataType, bf16_t> || std::is_same_v<QDataType, half_t>) &&
                      std::is_same_v<KDataType, QDataType> &&
                      std::is_same_v<VDataType, QDataType> && std::is_same_v<PDataType, QDataType>,
                  "V128 requires homogeneous native BF16 or FP16 Q/K/V/P");
    static_assert(HeadDimQK == 128 || HeadDimQK == 192, "V128 supports D128 or D192 QK geometry");
    static_assert(kQkWmmaK == 32 && kPvWmmaK == 32, "V128 supports only native K32 WMMA geometry");
    static_assert(kM % kWaves == 0 && kM / kWaves == kWaveSize,
                  "V128 geometry requires four wave32 waves");
    static_assert(kQkStages == kN / kQkSuColumns && kPvStages == 4,
                  "V128 stage geometry must remain four QK and four PV stages");
    static_assert(kQkWmmasPerStage == HeadDimQK / 8 && kPvWmmasPerStage == 16 &&
                      kQkWmmasPerWaveKvTile == HeadDimQK / 2 && kPvWmmasPerWaveKvTile == 64,
                  "native 16-bit instruction counts changed unexpectedly");
};

using LegacyD192Geometry  = FmhaTdmV128Geometry<bf16_t, bf16_t, bf16_t, bf16_t, 192, 32, 32>;
using FmhaTdmD128Geometry = FmhaTdmV128Geometry<bf16_t, bf16_t, bf16_t, bf16_t, 128, 32, 32>;

template <typename Problem>
using FmhaTdmV128GeometryFor =
    FmhaTdmV128Geometry<remove_cvref_t<typename Problem::QDataType>,
                        remove_cvref_t<typename Problem::KDataType>,
                        remove_cvref_t<typename Problem::VDataType>,
                        remove_cvref_t<typename Problem::PDataType>,
                        Problem::BlockFmhaShape::kQKHeaddim,
                        Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{}),
                        Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{})>;

template <typename Geometry>
struct FmhaTdmV128Layout
{
    static_assert(std::is_same_v<Geometry,
                                 FmhaTdmV128Geometry<typename Geometry::QDataType,
                                                     typename Geometry::KDataType,
                                                     typename Geometry::VDataType,
                                                     typename Geometry::PDataType,
                                                     Geometry::kHeadDimQK,
                                                     32,
                                                     32>>,
                  "Only native 16-bit D128/D192 LDS layouts are supported");
    static constexpr index_t kKPhysicalStride = Geometry::kHeadDimQK + 8;
    static constexpr index_t kVPhysicalStride = 144;
    static constexpr index_t kVPadAmount      = 7;
    static constexpr index_t kVPadInterval    = 5;
};

template <index_t KWait,
          index_t VWait,
          typename QkTails,
          typename PvTails,
          bool TailDrain,
          bool PackLds,
          index_t BlockPerCu,
          typename Geometry = LegacyD192Geometry>
struct FmhaTdmV128Tuning
{
    using QkTailDsCounts = QkTails;
    using PvTailDsCounts = PvTails;

    static constexpr index_t kKWait      = KWait;
    static constexpr index_t kVWait      = VWait;
    static constexpr bool kTailDrain     = TailDrain;
    static constexpr bool kPackLds       = PackLds;
    static constexpr index_t kBlockPerCu = BlockPerCu;
    static constexpr index_t kQkStages   = QkTails::size();
    static constexpr index_t kPvStages   = PvTails::size();

    static_assert(kKWait == 0 || kKWait == 1, "K wait count must be zero or one");
    static_assert(kVWait == 0 || kVWait == 1, "V wait count must be zero or one");
    static_assert(kQkStages == 4 && kPvStages == 4,
                  "QK and PV tail sequences must each contain four stages");

    static constexpr index_t kQkStage0TailDsCount = detail::fmha_tdm_v128_tail_at<QkTails, 0>();
    static constexpr index_t kQkStage1TailDsCount = detail::fmha_tdm_v128_tail_at<QkTails, 1>();
    static constexpr index_t kQkStage2TailDsCount = detail::fmha_tdm_v128_tail_at<QkTails, 2>();
    static constexpr index_t kQkStage3TailDsCount = detail::fmha_tdm_v128_tail_at<QkTails, 3>();
    static constexpr index_t kPvStage0TailDsCount = detail::fmha_tdm_v128_tail_at<PvTails, 0>();
    static constexpr index_t kPvStage1TailDsCount = detail::fmha_tdm_v128_tail_at<PvTails, 1>();
    static constexpr index_t kPvStage2TailDsCount = detail::fmha_tdm_v128_tail_at<PvTails, 2>();
    static constexpr index_t kPvStage3TailDsCount = detail::fmha_tdm_v128_tail_at<PvTails, 3>();
    static_assert(
        (kQkStage0TailDsCount == 0 || kQkStage0TailDsCount == Geometry::kKSuLoadCount) &&
            (kQkStage1TailDsCount == 0 || kQkStage1TailDsCount == Geometry::kKSuLoadCount) &&
            (kQkStage2TailDsCount == 0 || kQkStage2TailDsCount == Geometry::kKSuLoadCount) &&
            (kQkStage3TailDsCount == 0 || kQkStage3TailDsCount == Geometry::kVStageLoadCount),
        "QK tail waits must match the geometry producer budgets");
    static_assert(
        (kPvStage0TailDsCount == 0 || kPvStage0TailDsCount == Geometry::kVStageLoadCount) &&
            (kPvStage1TailDsCount == 0 || kPvStage1TailDsCount == Geometry::kVStageLoadCount) &&
            (kPvStage2TailDsCount == 0 || kPvStage2TailDsCount == Geometry::kVStageLoadCount) &&
            (kPvStage3TailDsCount == 0 || kPvStage3TailDsCount == Geometry::kKSuLoadCount),
        "PV tail waits must match the geometry producer budgets");
    static_assert(kBlockPerCu > 0, "BlockPerCu must be positive");
};

#ifndef CK_TILE_FMHA_GFX125_D192_K_PREFETCH_TENSORCNT
#define CK_TILE_FMHA_GFX125_D192_K_PREFETCH_TENSORCNT 0
#endif
#ifndef CK_TILE_FMHA_GFX125_D192_V_PREFETCH_TENSORCNT
#define CK_TILE_FMHA_GFX125_D192_V_PREFETCH_TENSORCNT 0
#endif
#ifndef CK_TILE_FMHA_GFX125_D192_PREFETCH_TAIL_DRAIN
#define CK_TILE_FMHA_GFX125_D192_PREFETCH_TAIL_DRAIN 1
#endif
#ifndef CK_TILE_FMHA_GFX125_D192_QK_STAGE0_TAIL_DSCNT
#define CK_TILE_FMHA_GFX125_D192_QK_STAGE0_TAIL_DSCNT 24
#endif
#ifndef CK_TILE_FMHA_GFX125_D192_QK_STAGE1_TAIL_DSCNT
#define CK_TILE_FMHA_GFX125_D192_QK_STAGE1_TAIL_DSCNT 24
#endif
#ifndef CK_TILE_FMHA_GFX125_D192_QK_STAGE2_TAIL_DSCNT
#define CK_TILE_FMHA_GFX125_D192_QK_STAGE2_TAIL_DSCNT 24
#endif
#ifndef CK_TILE_FMHA_GFX125_D192_QK_STAGE3_TAIL_DSCNT
#define CK_TILE_FMHA_GFX125_D192_QK_STAGE3_TAIL_DSCNT 16
#endif
#ifndef CK_TILE_FMHA_GFX125_D192_PV_STAGE0_TAIL_DSCNT
#define CK_TILE_FMHA_GFX125_D192_PV_STAGE0_TAIL_DSCNT 0
#endif
#ifndef CK_TILE_FMHA_GFX125_D192_PV_STAGE1_TAIL_DSCNT
#define CK_TILE_FMHA_GFX125_D192_PV_STAGE1_TAIL_DSCNT 0
#endif
#ifndef CK_TILE_FMHA_GFX125_D192_PV_STAGE2_TAIL_DSCNT
#define CK_TILE_FMHA_GFX125_D192_PV_STAGE2_TAIL_DSCNT 0
#endif
#ifndef CK_TILE_FMHA_GFX125_D192_PV_STAGE3_TAIL_DSCNT
#define CK_TILE_FMHA_GFX125_D192_PV_STAGE3_TAIL_DSCNT 0
#endif
#ifndef CK_TILE_FMHA_GFX125_D192_LDS_PACK
#define CK_TILE_FMHA_GFX125_D192_LDS_PACK 0
#endif
#ifndef CK_TILE_FMHA_GFX125_D192_BLOCK_PER_CU
#define CK_TILE_FMHA_GFX125_D192_BLOCK_PER_CU 1
#endif

using LegacyD192Tuning = FmhaTdmV128Tuning<CK_TILE_FMHA_GFX125_D192_K_PREFETCH_TENSORCNT,
                                           CK_TILE_FMHA_GFX125_D192_V_PREFETCH_TENSORCNT,
                                           sequence<CK_TILE_FMHA_GFX125_D192_QK_STAGE0_TAIL_DSCNT,
                                                    CK_TILE_FMHA_GFX125_D192_QK_STAGE1_TAIL_DSCNT,
                                                    CK_TILE_FMHA_GFX125_D192_QK_STAGE2_TAIL_DSCNT,
                                                    CK_TILE_FMHA_GFX125_D192_QK_STAGE3_TAIL_DSCNT>,
                                           sequence<CK_TILE_FMHA_GFX125_D192_PV_STAGE0_TAIL_DSCNT,
                                                    CK_TILE_FMHA_GFX125_D192_PV_STAGE1_TAIL_DSCNT,
                                                    CK_TILE_FMHA_GFX125_D192_PV_STAGE2_TAIL_DSCNT,
                                                    CK_TILE_FMHA_GFX125_D192_PV_STAGE3_TAIL_DSCNT>,
                                           CK_TILE_FMHA_GFX125_D192_PREFETCH_TAIL_DRAIN != 0,
                                           CK_TILE_FMHA_GFX125_D192_LDS_PACK != 0,
                                           CK_TILE_FMHA_GFX125_D192_BLOCK_PER_CU>;

} // namespace ck_tile
