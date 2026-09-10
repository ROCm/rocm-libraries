// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <utility>

#include <type_traits>

#if !defined(CK_TILE_FMHA_GFX125_D192_K_PREFETCH_TENSORCNT) && \
    !defined(CK_TILE_FMHA_GFX125_D192_V_PREFETCH_TENSORCNT) && \
    !defined(CK_TILE_FMHA_GFX125_D192_PREFETCH_TAIL_DRAIN) &&  \
    !defined(CK_TILE_FMHA_GFX125_D192_QK_STAGE0_TAIL_DSCNT) && \
    !defined(CK_TILE_FMHA_GFX125_D192_QK_STAGE1_TAIL_DSCNT) && \
    !defined(CK_TILE_FMHA_GFX125_D192_QK_STAGE2_TAIL_DSCNT) && \
    !defined(CK_TILE_FMHA_GFX125_D192_QK_STAGE3_TAIL_DSCNT) && \
    !defined(CK_TILE_FMHA_GFX125_D192_PV_STAGE0_TAIL_DSCNT) && \
    !defined(CK_TILE_FMHA_GFX125_D192_PV_STAGE1_TAIL_DSCNT) && \
    !defined(CK_TILE_FMHA_GFX125_D192_PV_STAGE2_TAIL_DSCNT) && \
    !defined(CK_TILE_FMHA_GFX125_D192_PV_STAGE3_TAIL_DSCNT) && \
    !defined(CK_TILE_FMHA_GFX125_D192_LDS_PACK) && !defined(CK_TILE_FMHA_GFX125_D192_BLOCK_PER_CU)
#define CK_TILE_FMHA_TDM_V128_TEST_NO_LEGACY_OVERRIDES 1
#endif

#include "ck_tile/ops/fmha/pipeline/fmha_tdm_v128_config.hpp"

#if defined(CK_TILE_FMHA_TDM_V128_TEST_INTEGRATION)
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_policy.hpp"
#endif

#if defined(CK_TILE_FMHA_TDM_V128_TEST_GENERATED_INSTANCE)
#include CK_TILE_FMHA_TDM_V128_TEST_GENERATED_INSTANCE
#endif

namespace {

using Geometry      = ck_tile::FmhaTdmV128Geometry<ck_tile::bf16_t,
                                                   ck_tile::bf16_t,
                                                   ck_tile::bf16_t,
                                                   ck_tile::bf16_t,
                                                   192,
                                                   32,
                                                   32>;
using NoTail        = ck_tile::sequence<0, 0, 0, 0>;
using QkTail        = ck_tile::sequence<24, 24, 24, 16>;
using PvTail        = ck_tile::sequence<16, 16, 16, 24>;
using MixedQkTail   = ck_tile::sequence<24, 0, 24, 16>;
using MixedPvTail   = ck_tile::sequence<16, 0, 16, 24>;
using SourceDefault = ck_tile::LegacyD192Tuning;
using K0V0NoTail    = ck_tile::FmhaTdmV128Tuning<0, 0, NoTail, NoTail, true, false, 1>;
using K1V0NoTail    = ck_tile::FmhaTdmV128Tuning<1, 0, NoTail, NoTail, true, false, 1>;
using K0V1NoTail    = ck_tile::FmhaTdmV128Tuning<0, 1, NoTail, NoTail, true, false, 1>;
using K1V1NoTail    = ck_tile::FmhaTdmV128Tuning<1, 1, NoTail, NoTail, true, false, 1>;
using K1V1AllTail   = ck_tile::FmhaTdmV128Tuning<1, 1, QkTail, PvTail, true, false, 1>;
using K1V1MixedTail = ck_tile::FmhaTdmV128Tuning<1, 1, MixedQkTail, MixedPvTail, false, false, 1>;
using BoolOccupancyVariant = ck_tile::FmhaTdmV128Tuning<1, 1, QkTail, PvTail, false, true, 3>;

using D128        = ck_tile::FmhaTdmD128Geometry;
using D128Tail    = ck_tile::sequence<16, 16, 16, 16>;
using D128NoTail  = ck_tile::FmhaTdmV128Tuning<0, 0, NoTail, NoTail, true, false, 1, D128>;
using D128AllTail = ck_tile::FmhaTdmV128Tuning<1, 1, D128Tail, D128Tail, true, false, 1, D128>;
static_assert(D128::kQkWmmasPerWaveKvTile == 64 && D128::kPvWmmasPerWaveKvTile == 64);
static_assert(D128::kKSuLoadCount == 16 && D128::kVStageLoadCount == 16);
static_assert(ck_tile::FmhaTdmV128Layout<D128>::kKPhysicalStride == 136);
static_assert(ck_tile::FmhaTdmV128Layout<D128>::kVPhysicalStride == 144);
static_assert(D128NoTail::kKWait == 0 && D128NoTail::kVWait == 0);
static_assert(D128AllTail::kQkStage0TailDsCount == 16 && D128AllTail::kPvStage3TailDsCount == 16);

template <int Dim>
constexpr bool CheckFp16Geometry()
{
    using Fp16   = ck_tile::FmhaTdmV128Geometry<ck_tile::half_t,
                                                ck_tile::half_t,
                                                ck_tile::half_t,
                                                ck_tile::half_t,
                                                Dim,
                                                32,
                                                32>;
    using Layout = ck_tile::FmhaTdmV128Layout<Fp16>;
    using Safe   = ck_tile::FmhaTdmV128Tuning<0, 0, NoTail, NoTail, true, false, 1, Fp16>;
    return Fp16::kKSuLoadCount == Dim / 8 && Fp16::kVStageLoadCount == 16 &&
           Fp16::kQkWmmasPerWaveKvTile == Dim / 2 && Fp16::kPvWmmasPerWaveKvTile == 64 &&
           Layout::kKPhysicalStride == Dim + 8 && Layout::kVPhysicalStride == 144 &&
           Safe::kKWait == 0 && Safe::kVWait == 0 && Safe::kTailDrain &&
           !std::is_same_v<Fp16, ck_tile::LegacyD192Geometry>;
}
static_assert(CheckFp16Geometry<128>() && CheckFp16Geometry<192>());

#if defined(CK_TILE_FMHA_TDM_V128_TEST_INTEGRATION)
using Policy = ck_tile::BlockFmhaPipelineQRKSVSTdmD192V128Policy;
static_assert(std::is_same_v<Policy::Tuning, SourceDefault>);
static_assert(Policy::kKPrefetchTensorCount == SourceDefault::kKWait);
static_assert(Policy::kVPrefetchTensorCount == SourceDefault::kVWait);
static_assert(Policy::kPrefetchTailDrain == SourceDefault::kTailDrain);
static_assert(Policy::kPackLds == SourceDefault::kPackLds);
static_assert(Policy::kQkStage0TailDsCount == SourceDefault::kQkStage0TailDsCount);
static_assert(Policy::kQkStage1TailDsCount == SourceDefault::kQkStage1TailDsCount);
static_assert(Policy::kQkStage2TailDsCount == SourceDefault::kQkStage2TailDsCount);
static_assert(Policy::kQkStage3TailDsCount == SourceDefault::kQkStage3TailDsCount);
static_assert(Policy::kPvStage0TailDsCount == SourceDefault::kPvStage0TailDsCount);
static_assert(Policy::kPvStage1TailDsCount == SourceDefault::kPvStage1TailDsCount);
static_assert(Policy::kPvStage2TailDsCount == SourceDefault::kPvStage2TailDsCount);
static_assert(Policy::kPvStage3TailDsCount == SourceDefault::kPvStage3TailDsCount);

using D128Policy = ck_tile::
    BlockFmhaPipelineQRKSVSTdmV128Policy<D128, D128NoTail, ck_tile::FmhaTdmV128ScheduleFor<D128>>;

template <typename Q, typename K, typename V, bool Group, bool Mask, bool Lse>
struct MaxPolicyProblem
{
    using QDataType                    = Q;
    using KDataType                    = K;
    using VDataType                    = V;
    static constexpr bool kIsGroupMode = Group;
    static constexpr bool kStoreLSE    = Lse;
    struct FmhaMask
    {
        static constexpr bool IsMasking = Mask;
    };
};

template <typename Q, typename K, typename V, std::size_t... States>
constexpr bool CheckMaxPolicyScope(std::index_sequence<States...>)
{
    constexpr bool bf16 = std::is_same_v<ck_tile::remove_cvref_t<Q>, ck_tile::bf16_t> &&
                          std::is_same_v<ck_tile::remove_cvref_t<K>, ck_tile::bf16_t> &&
                          std::is_same_v<ck_tile::remove_cvref_t<V>, ck_tile::bf16_t>;
    // State bits cover both modes, masks and LSE states using the production predicate.
    return ((Policy::UseCompilerMax<MaxPolicyProblem<Q,
                                                     K,
                                                     V,
                                                     (States & 4) != 0,
                                                     (States & 2) != 0,
                                                     (States & 1) != 0>>() ==
                 (bf16 && (States & 6) == 0) &&
             D128Policy::UseCompilerMax<MaxPolicyProblem<Q,
                                                         K,
                                                         V,
                                                         (States & 4) != 0,
                                                         (States & 2) != 0,
                                                         (States & 1) != 0>>() ==
                 ((States & 2) != 0)) &&
            ...);
}

using MaxPolicyStates = std::make_index_sequence<8>;
static_assert(
    CheckMaxPolicyScope<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t>(MaxPolicyStates{}));
static_assert(CheckMaxPolicyScope<const ck_tile::bf16_t&, ck_tile::bf16_t, ck_tile::bf16_t>(
    MaxPolicyStates{}));
static_assert(
    CheckMaxPolicyScope<ck_tile::half_t, ck_tile::half_t, ck_tile::half_t>(MaxPolicyStates{}));
static_assert(
    CheckMaxPolicyScope<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::fp8_t>(MaxPolicyStates{}));
static_assert(
    CheckMaxPolicyScope<ck_tile::half_t, ck_tile::bf16_t, ck_tile::bf16_t>(MaxPolicyStates{}));
static_assert(
    CheckMaxPolicyScope<ck_tile::bf16_t, ck_tile::half_t, ck_tile::bf16_t>(MaxPolicyStates{}));
static_assert(
    CheckMaxPolicyScope<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::half_t>(MaxPolicyStates{}));
#endif

#if defined(CK_TILE_FMHA_TDM_V128_TEST_GENERATED_INSTANCE)
// A legacy custom policy need not expose the new tuning adapter.
struct LegacyCustomPolicy : ck_tile::BlockFmhaPipelineQRKSVSTdmD192V128Policy
{
    private:
    using Tuning                     = void;
    static constexpr int kBlockPerCu = -1;
};

template <typename T, typename = void>
struct HasPublicTuning : std::false_type
{
};

template <typename T>
struct HasPublicTuning<T, std::void_t<typename T::Tuning>> : std::true_type
{
};

template <typename T, typename = void>
struct HasPublicBlockPerCu : std::false_type
{
};

template <typename T>
struct HasPublicBlockPerCu<T, std::void_t<decltype(T::kBlockPerCu)>> : std::true_type
{
};

using CustomPipeline =
    ck_tile::BlockFmhaPipelineQRKSVSTdmD192V128<fmha_pipeline_problem, LegacyCustomPolicy>;
using CommonPipeline = ck_tile::BlockFmhaPipelineQRKSVSTdmV128<fmha_pipeline_problem>;
static_assert(
    std::is_base_of_v<
        ck_tile::BlockFmhaPipelineQRKSVSTdmV128<fmha_pipeline_problem,
                                                ck_tile::BlockFmhaPipelineQRKSVSTdmD192V128Policy>,
        fmha_pipeline>);
static_assert(CommonPipeline::kBlockPerCu == SourceDefault::kBlockPerCu);
static_assert(CommonPipeline::GetSmemSize() == fmha_pipeline::GetSmemSize());
static_assert(std::is_same_v<typename CommonPipeline::Geometry, Geometry>);
static_assert(!HasPublicTuning<LegacyCustomPolicy>::value);
static_assert(!HasPublicBlockPerCu<LegacyCustomPolicy>::value);
static_assert(std::is_same_v<typename fmha_pipeline::Geometry, Geometry>);
static_assert(std::is_same_v<typename CustomPipeline::Geometry, Geometry>);
static_assert(fmha_pipeline::kBlockPerCu == SourceDefault::kBlockPerCu);
static_assert(CustomPipeline::kBlockPerCu == SourceDefault::kBlockPerCu);
static_assert(CustomPipeline::GetSmemSize() == fmha_pipeline::GetSmemSize());
#endif

static_assert(Geometry::kM == 128 && Geometry::kN == 128 && Geometry::kDv == 128);
static_assert(Geometry::kWaves == 4 && Geometry::kWaveSize == 32 &&
              Geometry::kQueryRowsPerWave == 32);
static_assert(Geometry::kQkSuColumns == 32 && Geometry::kQkStages == 4 && Geometry::kPvStages == 4);
static_assert(Geometry::kQkWmmasPerStage == 24 && Geometry::kPvWmmasPerStage == 16);
static_assert(Geometry::kQkWmmasPerWaveKvTile == 96 && Geometry::kPvWmmasPerWaveKvTile == 64);

static_assert(SourceDefault::kKWait == CK_TILE_FMHA_GFX125_D192_K_PREFETCH_TENSORCNT);
static_assert(SourceDefault::kVWait == CK_TILE_FMHA_GFX125_D192_V_PREFETCH_TENSORCNT);
static_assert(SourceDefault::kTailDrain == (CK_TILE_FMHA_GFX125_D192_PREFETCH_TAIL_DRAIN != 0));
static_assert(SourceDefault::kPackLds == (CK_TILE_FMHA_GFX125_D192_LDS_PACK != 0));
static_assert(SourceDefault::kBlockPerCu == CK_TILE_FMHA_GFX125_D192_BLOCK_PER_CU);
static_assert(SourceDefault::kQkStage0TailDsCount == CK_TILE_FMHA_GFX125_D192_QK_STAGE0_TAIL_DSCNT);
static_assert(SourceDefault::kQkStage1TailDsCount == CK_TILE_FMHA_GFX125_D192_QK_STAGE1_TAIL_DSCNT);
static_assert(SourceDefault::kQkStage2TailDsCount == CK_TILE_FMHA_GFX125_D192_QK_STAGE2_TAIL_DSCNT);
static_assert(SourceDefault::kQkStage3TailDsCount == CK_TILE_FMHA_GFX125_D192_QK_STAGE3_TAIL_DSCNT);
static_assert(SourceDefault::kPvStage0TailDsCount == CK_TILE_FMHA_GFX125_D192_PV_STAGE0_TAIL_DSCNT);
static_assert(SourceDefault::kPvStage1TailDsCount == CK_TILE_FMHA_GFX125_D192_PV_STAGE1_TAIL_DSCNT);
static_assert(SourceDefault::kPvStage2TailDsCount == CK_TILE_FMHA_GFX125_D192_PV_STAGE2_TAIL_DSCNT);
static_assert(SourceDefault::kPvStage3TailDsCount == CK_TILE_FMHA_GFX125_D192_PV_STAGE3_TAIL_DSCNT);
#if defined(CK_TILE_FMHA_TDM_V128_TEST_NO_LEGACY_OVERRIDES)
using ExpectedSourceDefault =
    ck_tile::FmhaTdmV128Tuning<0, 0, ck_tile::sequence<24, 24, 24, 16>, NoTail, true, false, 1>;
static_assert(std::is_same_v<SourceDefault, ExpectedSourceDefault>);
#endif

static_assert(K0V0NoTail::kKWait == 0 && K0V0NoTail::kVWait == 0 &&
              K0V0NoTail::kQkStage0TailDsCount == 0 && K0V0NoTail::kPvStage0TailDsCount == 0);
static_assert(K1V0NoTail::kKWait == 1 && K1V0NoTail::kVWait == 0);
static_assert(K0V1NoTail::kKWait == 0 && K0V1NoTail::kVWait == 1);
static_assert(K1V1NoTail::kKWait == 1 && K1V1NoTail::kVWait == 1 &&
              K1V1NoTail::kQkStage3TailDsCount == 0 && K1V1NoTail::kPvStage3TailDsCount == 0);
static_assert(K1V1AllTail::kKWait == 1 && K1V1AllTail::kVWait == 1 && K1V1AllTail::kTailDrain &&
              !K1V1AllTail::kPackLds && K1V1AllTail::kBlockPerCu == 1);
static_assert(K1V1AllTail::kQkStage0TailDsCount == 24 && K1V1AllTail::kQkStage1TailDsCount == 24 &&
              K1V1AllTail::kQkStage2TailDsCount == 24 && K1V1AllTail::kQkStage3TailDsCount == 16 &&
              K1V1AllTail::kPvStage0TailDsCount == 16 && K1V1AllTail::kPvStage1TailDsCount == 16 &&
              K1V1AllTail::kPvStage2TailDsCount == 16 && K1V1AllTail::kPvStage3TailDsCount == 24);
static_assert(
    !K1V1MixedTail::kTailDrain && !K1V1MixedTail::kPackLds &&
    K1V1MixedTail::kQkStage0TailDsCount == 24 && K1V1MixedTail::kQkStage1TailDsCount == 0 &&
    K1V1MixedTail::kQkStage2TailDsCount == 24 && K1V1MixedTail::kQkStage3TailDsCount == 16 &&
    K1V1MixedTail::kPvStage0TailDsCount == 16 && K1V1MixedTail::kPvStage1TailDsCount == 0 &&
    K1V1MixedTail::kPvStage2TailDsCount == 16 && K1V1MixedTail::kPvStage3TailDsCount == 24);
static_assert(!BoolOccupancyVariant::kTailDrain && BoolOccupancyVariant::kPackLds &&
              BoolOccupancyVariant::kBlockPerCu == 3);

#if defined(CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE)
#if CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE == 1
using NegativeCase = ck_tile::FmhaTdmV128Tuning<2, 0, NoTail, NoTail, true, false, 1>;
#elif CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE == 2
using NegativeCase = ck_tile::FmhaTdmV128Tuning<0, 2, NoTail, NoTail, true, false, 1>;
#elif CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE == 3
using NegativeCase =
    ck_tile::FmhaTdmV128Tuning<0, 0, ck_tile::sequence<24, 24, 24, 15>, NoTail, true, false, 1>;
#elif CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE == 4
using NegativeCase =
    ck_tile::FmhaTdmV128Tuning<0, 0, NoTail, ck_tile::sequence<16, 16, 16, 23>, true, false, 1>;
#elif CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE == 5
using NegativeCase =
    ck_tile::FmhaTdmV128Tuning<0, 0, ck_tile::sequence<0, 0, 0>, NoTail, true, false, 1>;
#elif CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE == 6
using NegativeCase =
    ck_tile::FmhaTdmV128Tuning<0, 0, NoTail, ck_tile::sequence<0, 0, 0>, true, false, 1>;
#elif CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE == 7
using NegativeCase =
    ck_tile::FmhaTdmV128Tuning<0, 0, NoTail, ck_tile::sequence<0, 0, 0, 0, 0>, true, false, 1>;
#elif CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE == 8
using NegativeCase = ck_tile::FmhaTdmV128Tuning<0, 0, NoTail, NoTail, true, false, 0>;
#elif CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE == 9
using NegativeCase = ck_tile::FmhaTdmV128Geometry<ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  192,
                                                  0,
                                                  32>;
#elif CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE == 10
using NegativeCase = ck_tile::FmhaTdmV128Geometry<ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  192,
                                                  32,
                                                  0>;
#elif CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE == 11
using NegativeCase = ck_tile::FmhaTdmV128Geometry<ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  192,
                                                  64,
                                                  32>;
#elif CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE == 12
using NegativeCase = ck_tile::FmhaTdmV128Geometry<ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  190,
                                                  32,
                                                  32>;
#elif CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE == 13
using NegativeCase = ck_tile::
    FmhaTdmV128Geometry<float, ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t, 192, 32, 32>;
#elif CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE == 14
using NegativeCase = ck_tile::FmhaTdmV128Geometry<ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  160,
                                                  32,
                                                  32>;
#elif CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE == 15
using NegativeCase =
    ck_tile::FmhaTdmV128Tuning<0, 0, ck_tile::sequence<0, 0, 0, 0, 0>, NoTail, true, false, 1>;
#elif CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE == 16
using NegativeCase = ck_tile::FmhaTdmV128Geometry<ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  192,
                                                  -32,
                                                  32>;
#elif CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE == 17
using NegativeCase = ck_tile::FmhaTdmV128Geometry<ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  192,
                                                  32,
                                                  -32>;
#elif CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE == 18
using NegativeCase = ck_tile::FmhaTdmV128Geometry<ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  192,
                                                  32,
                                                  64>;
#elif CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE == 19
using NegativeCase = ck_tile::FmhaTdmV128Tuning<0, 0, NoTail, NoTail, true, false, -1>;
#elif CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE == 20
using NegativeCase = ck_tile::FmhaTdmV128Geometry<ck_tile::bf16_t,
                                                  ck_tile::half_t,
                                                  ck_tile::half_t,
                                                  ck_tile::half_t,
                                                  192,
                                                  32,
                                                  32>;
#elif CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE == 21
using NegativeCase = ck_tile::FmhaTdmV128Geometry<ck_tile::half_t,
                                                  ck_tile::bf16_t,
                                                  ck_tile::half_t,
                                                  ck_tile::half_t,
                                                  192,
                                                  32,
                                                  32>;
#elif CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE == 22
using NegativeCase = ck_tile::FmhaTdmV128Geometry<ck_tile::half_t,
                                                  ck_tile::half_t,
                                                  ck_tile::bf16_t,
                                                  ck_tile::half_t,
                                                  192,
                                                  32,
                                                  32>;
#elif CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE == 23
using NegativeCase = ck_tile::FmhaTdmV128Geometry<ck_tile::half_t,
                                                  ck_tile::half_t,
                                                  ck_tile::half_t,
                                                  ck_tile::bf16_t,
                                                  192,
                                                  32,
                                                  32>;
#elif CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE == 24
using NegativeCase = ck_tile::FmhaTdmV128Geometry<float, float, float, float, 192, 32, 32>;
#else
#error "CK_TILE_FMHA_TDM_V128_NEGATIVE_CASE must be 1 through 24"
#endif
static_assert(sizeof(NegativeCase) > 0, "negative case must instantiate");
#endif

} // namespace

int main() { return 0; }
