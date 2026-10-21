// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "fmha_test_common.hpp"
#include "gtest/gtest.h"

#include <array>

namespace {

using namespace ck_tile;

template <typename T, int D, int N, int V>
struct Config
{
    using Problem  = typename test::Model<D, false, false, true, T, N, V>::Problem;
    using Geometry = FmhaTdmSchedGeometryFor<Problem>;
    using Policy   = FmhaTdmSchedPolicyFor<Problem>;
};

template <typename Policy>
struct WrongCountSchedule : Policy::Schedule
{
    static constexpr index_t kQkWmmasPerStage = Policy::Geometry::kQkWmmasPerStage - 1;
};
template <typename Policy>
struct WrongCountPolicy : Policy
{
    using Schedule = WrongCountSchedule<Policy>;
};

template <typename Problem, BlockAttentionBiasEnum Bias>
struct BiasProblem : Problem
{
    static constexpr auto BiasEnum = Bias;
};

// This fixture exercises only the max-policy predicate. Its mixed and FP8
// types do not imply that the corresponding full kernel problem is supported.
template <typename Q, typename K, typename V, bool Group, bool Mask, bool Lse>
struct CompilerMaxProblem
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

template <typename Policy, typename Q, typename K, typename V>
void ExpectCompilerMaxFlagMatrix(const std::array<bool, 8>& expected, const char* type_case)
{
    SCOPED_TRACE(type_case);
    static_for<0, 8, 1>{}([&](auto state) {
        constexpr int s = decltype(state)::value;
        using Problem   = CompilerMaxProblem<Q, K, V, (s & 4) != 0, (s & 2) != 0, (s & 1) != 0>;
        EXPECT_EQ((Policy::template UseCompilerMax<Problem>()), expected[s])
            << "group=" << Problem::kIsGroupMode << " mask=" << Problem::FmhaMask::IsMasking
            << " lse=" << Problem::kStoreLSE;
    });
}

template <typename Policy, bool Early>
struct OutputRescalePolicy : Policy
{
    static constexpr bool kEarlyOutputRescale = Early;
};

using Configs = ::testing::Types<Config<bf16_t, 64, 64, 64>,
                                 Config<half_t, 64, 64, 64>,
                                 Config<bf16_t, 128, 128, 128>,
                                 Config<half_t, 128, 128, 128>,
                                 Config<bf16_t, 192, 64, 128>,
                                 Config<half_t, 192, 64, 128>,
                                 Config<bf16_t, 192, 128, 128>,
                                 Config<half_t, 192, 128, 128>>;

template <typename T>
class QrTdmSchedConfig : public ::testing::Test
{
};
TYPED_TEST_SUITE(QrTdmSchedConfig, Configs);

TYPED_TEST(QrTdmSchedConfig, GeometryAndDerivedTails)
{
    using G = typename TypeParam::Geometry;
    using P = typename TypeParam::Policy;
    using T = typename P::Tuning;
    EXPECT_EQ(G::kWaves, 4);
    EXPECT_EQ(G::kWaveSize, 32);
    EXPECT_EQ(G::kQueryRowsPerWave, 32);
    EXPECT_EQ(G::kQkStages, G::kN / 32);
    EXPECT_EQ(G::kPvStages, G::kN / 32);
    EXPECT_EQ(G::kKSuLoadCount, G::kHeadDimQK / 8);
    EXPECT_EQ(G::kVStageLoadCount, G::kDv / 8);
    EXPECT_EQ(G::kQkWmmasPerWaveKvTile, G::kN * G::kHeadDimQK / 256);
    EXPECT_EQ(G::kPvWmmasPerWaveKvTile, G::kN * G::kDv / 256);
    EXPECT_TRUE(G::PRepackLayout::kInThread);
    EXPECT_EQ(T::kKWait, 1);
    EXPECT_EQ(T::kVWait, 1);
    EXPECT_TRUE(T::kTailDrain);
    EXPECT_EQ(T::kBlockPerCu, G::kN == 128 ? 1 : G::kHeadDimQK == 64 ? 3 : 2);
    static_for<0, G::kQkStages, 1>{}([&](auto stage) {
        constexpr int s = decltype(stage)::value;
        EXPECT_EQ(T::template GetQkTailDsCount<s>(),
                  s + 1 == G::kQkStages ? G::kVStageLoadCount : G::kKSuLoadCount);
    });
    static_for<0, G::kPvStages, 1>{}([&](auto stage) {
        constexpr int s = decltype(stage)::value;
        EXPECT_EQ(T::template GetPvTailDsCount<s>(),
                  s + 1 == G::kPvStages ? G::kKSuLoadCount : G::kVStageLoadCount);
    });
}

TYPED_TEST(QrTdmSchedConfig, PackedLdsAndKernelContract)
{
    using G        = typename TypeParam::Geometry;
    using P        = typename TypeParam::Policy;
    using Problem  = typename TypeParam::Problem;
    using Pipeline = BlockFmhaPipelineQRKSVSTdmSched<Problem>;
    EXPECT_EQ(P::kKPhysicalStride, G::kHeadDimQK + 8);
    EXPECT_EQ(P::kVPhysicalStride, G::kDv + 16);
    EXPECT_EQ(P::kLdsOffsetK0, 0);
    EXPECT_EQ(P::kLdsOffsetK1, P::kKFootprintBytes);
    EXPECT_EQ(P::kLdsOffsetV0, 2 * P::kKFootprintBytes);
    EXPECT_EQ(P::kLdsOffsetV1, P::kLdsOffsetV0 + P::kVFootprintBytes);
    EXPECT_EQ(P::kLdsArenaSize, 2 * (P::kKFootprintBytes + P::kVFootprintBytes));
    EXPECT_EQ(P::kLdsOffsetK1 % 16, 0);
    EXPECT_EQ(P::kLdsOffsetV0 % 16, 0);
    EXPECT_EQ(P::kLdsOffsetV1 % 16, 0);
    EXPECT_LE(G::kM * G::kHeadDimQK * sizeof(typename G::QDataType),
              static_cast<std::size_t>(2 * P::kKFootprintBytes));
    EXPECT_EQ(Pipeline::GetSmemSize(), P::kLdsArenaSize);
    // The host compiler target uses wave64; gfx125 device code uses wave32.
    // Check the target-dependent expression here and the physical device shape below.
    EXPECT_EQ(Pipeline::kBlockSize, G::kWaves * get_warp_size());
#if defined(__HIP_DEVICE_COMPILE__)
    static_assert(Pipeline::kBlockSize == G::kWaves * G::kWaveSize);
    static_assert(Pipeline::kBlockSize == 128);
#endif
    EXPECT_TRUE((P::template IsSupportedProblem<Problem>()));
    EXPECT_FALSE((P::template IsSupportedProblem<
                  BiasProblem<Problem, BlockAttentionBiasEnum::ELEMENTWISE_BIAS>>()));
    EXPECT_FALSE(
        (P::template IsSupportedProblem<BiasProblem<Problem, BlockAttentionBiasEnum::ALIBI>>()));
    EXPECT_TRUE(detail::uses_untransposed_v_kernel_path_v<Pipeline>);
    EXPECT_TRUE(detail::uses_tdm_affine_dram_path_v<Pipeline>);
    EXPECT_FALSE(detail::uses_qr_tdm_lds_arena_v<Pipeline>);
}

TYPED_TEST(QrTdmSchedConfig, SelectedScheduleAndInvalidCounts)
{
    using G       = typename TypeParam::Geometry;
    using P       = typename TypeParam::Policy;
    using Problem = typename TypeParam::Problem;
    using Mode    = FmhaTdmSchedMode<Problem, P>;
    EXPECT_TRUE(Mode::kIsLegal);
    EXPECT_EQ(Mode::kEarlyOutputRescale, G::kHeadDimQK == 128);
    EXPECT_EQ(Mode::kStreamOutputRescale, (std::is_same_v<G, FmhaTdmSchedBf16D192N128Geometry>));
    EXPECT_EQ(P::OutputFragments::kNumFragments, G::kPvMIter * G::kPvNIter);
    EXPECT_FALSE((FmhaTdmSchedMode<Problem, WrongCountPolicy<P>>::kIsLegal));
}

TYPED_TEST(QrTdmSchedConfig, CompilerMaxScopeAcrossFlagsAndTypes)
{
    using G = typename TypeParam::Geometry;
    using P = typename TypeParam::Policy;
    // State order: batch dense, batch masked, group dense, group masked;
    // each pair has LSE disabled then enabled. Pin the shipped scope directly.
    constexpr std::array<bool, 8> d64_d128_scope{true, true, true, true, true, true, true, true};
    constexpr std::array<bool, 8> d192_scope{true, true, false, false, false, false, false, false};
    constexpr auto expected = G::kHeadDimQK == 192 ? d192_scope : d64_d128_scope;
    ExpectCompilerMaxFlagMatrix<P, bf16_t, bf16_t, bf16_t>(expected, "BF16");
    ExpectCompilerMaxFlagMatrix<P, half_t, half_t, half_t>(expected, "FP16");
    ExpectCompilerMaxFlagMatrix<P, const bf16_t&, bf16_t, bf16_t>(expected, "BF16 Q cvref");
    ExpectCompilerMaxFlagMatrix<P, fp8_t, fp8_t, fp8_t>(expected, "FP8 predicate only");
    ExpectCompilerMaxFlagMatrix<P, half_t, bf16_t, bf16_t>(expected, "mixed Q predicate only");
    ExpectCompilerMaxFlagMatrix<P, bf16_t, half_t, bf16_t>(expected, "mixed K predicate only");
    ExpectCompilerMaxFlagMatrix<P, bf16_t, bf16_t, half_t>(expected, "mixed V predicate only");
}

TYPED_TEST(QrTdmSchedConfig, OutputRescaleModesRejectConflictingOwners)
{
    using P       = typename TypeParam::Policy;
    using Problem = typename TypeParam::Problem;
    using Late    = FmhaTdmSchedMode<Problem, OutputRescalePolicy<P, false>>;
    using Early   = FmhaTdmSchedMode<Problem, OutputRescalePolicy<P, true>>;
    // Only the native BF16 D192/N128 schedule owns streaming rescale. Other
    // schedules permit early rescale or the post-softmax fragment alternative.
    constexpr bool streaming =
        std::is_same_v<typename TypeParam::Geometry, FmhaTdmSchedBf16D192N128Geometry>;
    EXPECT_FALSE(Late::kEarlyOutputRescale);
    EXPECT_EQ(Late::kStreamOutputRescale, streaming);
    EXPECT_EQ(Late::kPostSoftmaxFragmentRescale, !streaming);
    EXPECT_TRUE(Late::kIsLegal);
    EXPECT_TRUE(Early::kEarlyOutputRescale);
    EXPECT_EQ(Early::kStreamOutputRescale, streaming);
    EXPECT_FALSE(Early::kPostSoftmaxFragmentRescale);
    EXPECT_EQ(Early::kIsLegal, !streaming);
}

// ---------------------------------------------------------------------------
// Split-kv copy
//
// BlockFmhaFwdSplitKVPipelineQRKSVSTdmSched is a copy of the prefill pipeline
// that shares the policy. The tests above instantiate the prefill pipeline only,
// so they stay green no matter what the copy does -- these cases are the ones
// that cover it. What they pin down is the contract between the copy and
// FmhaFwdSplitKVKernel: the kernel sizes the LDS allocation from
// FmhaPipeline::GetSmemSize() and the copy carves that block with the policy's
// own segment offsets, so the two have to agree exactly or the schedule writes
// outside its arena.
// ---------------------------------------------------------------------------

template <typename T, int D, int N, int V>
struct SplitConfig
{
    using Prefill  = typename test::Model<D, false, false, true, T, N, V>::Problem;
    using Model    = test::SplitKVModel<D, false, T, N, V>;
    using Problem  = typename Model::Problem;
    using Pipeline = typename Model::Pipeline;
    using Policy   = FmhaTdmSchedPolicyFor<Problem>;
    using Geometry = FmhaTdmSchedGeometryFor<Problem>;
};

using SplitConfigs = ::testing::Types<SplitConfig<bf16_t, 128, 128, 128>,
                                      SplitConfig<half_t, 128, 128, 128>>;

template <typename T>
class QrTdmSchedSplitKVConfig : public ::testing::Test
{
};
TYPED_TEST_SUITE(QrTdmSchedSplitKVConfig, SplitConfigs);

TYPED_TEST(QrTdmSchedSplitKVConfig, LdsArenaMatchesKernelAllocation)
{
    using P        = typename TypeParam::Policy;
    using Pipeline = typename TypeParam::Pipeline;
    // The kernel hands the pipeline one pointer; the pipeline splits it into
    // the four fixed segments. Both numbers come from the same policy only if
    // the copy keeps reporting the arena size.
    EXPECT_EQ(Pipeline::GetSmemSize(), P::kLdsArenaSize);
    EXPECT_EQ(P::kLdsOffsetK0, 0);
    EXPECT_EQ(P::kLdsOffsetK1, P::kKFootprintBytes);
    EXPECT_EQ(P::kLdsOffsetV0, 2 * P::kKFootprintBytes);
    EXPECT_EQ(P::kLdsOffsetV1, P::kLdsOffsetV0 + P::kVFootprintBytes);
    EXPECT_LE(P::kLdsOffsetV1 + P::kVFootprintBytes, Pipeline::GetSmemSize());
}

TYPED_TEST(QrTdmSchedSplitKVConfig, GeometryMatchesThePrefillPipeline)
{
    using G        = typename TypeParam::Geometry;
    using Pipeline = typename TypeParam::Pipeline;
    using Prefill  = BlockFmhaPipelineQRKSVSTdmSched<typename TypeParam::Prefill>;
    // The copy must keep the tile shape the hand-tuned schedule was written
    // for, otherwise the shared policy is describing a different kernel.
    EXPECT_EQ(Pipeline::kM0, Prefill::kM0);
    EXPECT_EQ(Pipeline::kN0, Prefill::kN0);
    EXPECT_EQ(Pipeline::kK0, Prefill::kK0);
    EXPECT_EQ(Pipeline::kN1, Prefill::kN1);
    EXPECT_EQ(Pipeline::kK1, Prefill::kK1);
    EXPECT_EQ(Pipeline::kSubQKHeaddim, Prefill::kSubQKHeaddim);
    EXPECT_EQ(Pipeline::kQLoadOnce, Prefill::kQLoadOnce);
    EXPECT_EQ(G::kM, Pipeline::kM0);
    EXPECT_EQ(G::kN, Pipeline::kN0);
    EXPECT_EQ(G::kDv, Pipeline::kN1);
}

TYPED_TEST(QrTdmSchedSplitKVConfig, SplitProblemCarriesThePrefillPolicyInputs)
{
    using Problem = typename TypeParam::Problem;
    // The split-kv problem has to answer the members the shared policy and the
    // copied body read; split-kv supports none of these features, so the values
    // are the prefill defaults for qr_tdm_sched. A new policy read landing here
    // is what breaks first when rebasing onto a newer develop.
    EXPECT_FALSE(Problem::kHasDropout);
    EXPECT_EQ(Problem::QScaleEnum, BlockAttentionQuantScaleEnum::NO_SCALE);
    EXPECT_FALSE(Problem::kUseDoubleKVLdsBuffer);
    EXPECT_FALSE(Problem::kProgressiveDsLoadK);
    EXPECT_TRUE(Problem::kStoreLSE);
    EXPECT_FALSE(Problem::kIsPagedKV);
    EXPECT_EQ(Problem::BiasEnum, BlockAttentionBiasEnum::NO_BIAS);
    static_assert(std::is_same_v<typename Problem::RandValOutputDataType, std::uint8_t>);
}

TYPED_TEST(QrTdmSchedSplitKVConfig, SplitTailIsAlwaysPredicated)
{
    using Pipeline = typename TypeParam::Pipeline;
    // A split bound is ceil(seqlen_k / num_splits) and is not kN0 aligned, so
    // the last tile of a split can reach past it. The copy hardwires the
    // predicate on instead of taking it from the traits -- if that ever becomes
    // conditional again, the tail of an uneven split silently joins the softmax.
    EXPECT_TRUE(Pipeline::kHasUnevenSplits);
}

// ---------------------------------------------------------------------------
// qr_tdm split-kv copy
//
// BlockFmhaFwdSplitKVPipelineQRKSVSTdm carries only the single-V / double-K
// path of BlockFmhaPipelineQRKSVSTdm. Unlike the qr_tdm_sched copy it does not
// go through FmhaTdmSchedSupportedGeometry, so nothing else pins its tile down;
// these cases do. They also pin the three flags the copy overrides locally
// instead of reading from the problem -- each one was a bug at some point
// during the port.
// ---------------------------------------------------------------------------

template <typename T, int D, int N, int V>
struct QrTdmSplitConfig
{
    // bm0 64 is the tile codegen pairs with this pipeline (occupancy 2).
    using Model    = test::SplitKVModel<D, false, T, N, V, true, 64>;
    using Problem  = typename Model::Problem;
    using Pipeline = typename Model::TdmPipeline;
    using Policy   = BlockFmhaPipelineQRKSVSTdmDefaultPolicy;
    using Layout   = typename Policy::template LdsArenaLayout<Problem>;
};

using QrTdmSplitConfigs = ::testing::Types<QrTdmSplitConfig<bf16_t, 128, 64, 128>,
                                           QrTdmSplitConfig<half_t, 128, 64, 128>>;

template <typename T>
class QrTdmSplitKVConfig : public ::testing::Test
{
};
TYPED_TEST_SUITE(QrTdmSplitKVConfig, QrTdmSplitConfigs);

TYPED_TEST(QrTdmSplitKVConfig, UsesTheSingleVArenaLayout)
{
    using Layout   = typename TypeParam::Layout;
    using Pipeline = typename TypeParam::Pipeline;
    // The whole point of this copy: the shared policy must resolve to the
    // single-V layout, which it does because split-kv problems hardwire
    // kUseDoubleKVLdsBuffer to false. V1 collapsing onto V0 is what proves it,
    // and the S region only exists on this path.
    EXPECT_EQ(Layout::kV1Offset, Layout::kV0Offset);
    EXPECT_GT(Layout::kSOffset, 0);
    EXPECT_EQ(Pipeline::GetSmemSize(), Layout::kArenaBytes);
}

TYPED_TEST(QrTdmSplitKVConfig, KernelPathFlagsAreDeclared)
{
    using Pipeline = typename TypeParam::Pipeline;
    // fmha_fwd_kernel.hpp and fmha_fwd_splitkv_kernel.hpp do NOT default to the
    // same K/V views, which is why the prefill pipeline declares neither flag
    // and still works there. This copy must declare both: it steps V with a
    // seqlen stride on dim 0 (needs the untransposed view) and loads K/V with
    // bare load_tile_tdm (needs the unpadded affine view).
    EXPECT_TRUE(Pipeline::kUsesUntransposedVKernelPath);
    EXPECT_TRUE(Pipeline::kUsesTdmAffineDramPath);
}

TYPED_TEST(QrTdmSplitKVConfig, ProgressiveKReloadStaysOn)
{
    using Pipeline = typename TypeParam::Pipeline;
    // Progressive K reload is the single biggest contributor to this pipeline's
    // speed: the bn0 32 tile probe, which has to fall back to the plain
    // traversal, measured 4-5x slower. The split-kv problem hardwires
    // kProgressiveDsLoadK to false for qr_tdm_sched's benefit, so this copy
    // overrides it locally, gated on the bn0 == 64 the reload's own
    // static_asserts require.
    EXPECT_TRUE(Pipeline::kProgressiveDsLoadK);
    EXPECT_EQ(Pipeline::kN0, 64);
    EXPECT_FALSE(Pipeline::kSkipMinSeqlenQ);
}

TYPED_TEST(QrTdmSplitKVConfig, SplitTailIsAlwaysPredicated)
{
    using Pipeline = typename TypeParam::Pipeline;
    EXPECT_TRUE(Pipeline::kHasUnevenSplits);
    EXPECT_EQ(Pipeline::kM0, 64);
}

} // namespace
