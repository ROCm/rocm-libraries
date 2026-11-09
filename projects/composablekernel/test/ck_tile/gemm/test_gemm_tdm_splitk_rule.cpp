// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Which TDM pipelines UniversalGemmKernel lets run split-K (k_batch > 1).
//
// The rule (universal_gemm_kernel.hpp, IsSupportedArgument): a TDM pipeline runs split-K only if
// it declares kSupportsTdmSplitK, and only with an epilogue that adds the splits into C (CShuffle).
// The TDM epilogue overwrites C, so it rejects split-K whatever the pipeline. A pipeline declares
// kSupportsTdmSplitK once its TDM loads are clipped to each split's K range and its own gtests run
// split-K numerically; comp_tdm_v3 does.
//
// These tests check the rule against each pipeline's declaration rather than a list of
// pipelines, so declaring kSupportsTdmSplitK on another pipeline needs no change here, while
// loosening the kernel's guard fails every pipeline that does not declare it. They do not check
// that a declaring pipeline computes split-K correctly; its numeric gtests do.
//
// Argument validation only: nothing is allocated or launched. Without a gfx125 device
// IsSupportedArgument rejects every TDM argument, so the rejections are checked on any host and
// the acceptances only on gfx125.

#include "gtest/gtest.h"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/host/device_prop.hpp"

namespace {

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

// fp32 C, so CShuffle can add the splits with scalar atomics at any vector size.
using ADataType   = ck_tile::half_t;
using BDataType   = ck_tile::half_t;
using AccDataType = float;
using CDataType   = float;

constexpr ck_tile::index_t kTileM = 128;
constexpr ck_tile::index_t kTileN = 128;
constexpr ck_tile::index_t kTileK = 64;

using Shape       = ck_tile::TileGemmShape<ck_tile::sequence<kTileM, kTileN, kTileK>,
                                           ck_tile::sequence<2, 2, 1>,
                                           ck_tile::sequence<16, 16, 32>>;
using Partitioner = ck_tile::GemmTile1DPartitioner<Shape>;
using Traits      = ck_tile::TileGemmUniversalTraits<false, // PadM
                                                     false, // PadN
                                                     false, // PadK
                                                     true,  // DoubleSmemBuffer
                                                     Row,
                                                     Col,
                                                     Row>;
using Problem     = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                          BDataType,
                                                          AccDataType,
                                                          Shape,
                                                          Traits,
                                                          ck_tile::GemmPipelineScheduler::Intrawave>;
using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<ADataType,
                                                         BDataType,
                                                         ck_tile::tuple<>,
                                                         AccDataType,
                                                         CDataType,
                                                         ck_tile::tuple<>,
                                                         Row,
                                                         ck_tile::element_wise::PassThrough,
                                                         kTileM,
                                                         kTileN,
                                                         2,
                                                         2,
                                                         16,
                                                         16,
                                                         32,
                                                         false, // isCTransposed
                                                         1,     // kNumWaveGroups
                                                         false, // FixedVectorSize
                                                         1,     // VectorSizeC
                                                         1,     // BlockedXDLN_PerWarp
                                                         true>; // DoubleSmemBuffer

template <template <typename...> class Pipeline, template <typename...> class Epilogue>
using KernelOf = ck_tile::GemmKernel<Partitioner, Pipeline<Problem>, Epilogue<EpilogueProblem>>;

template <typename T>
using has_tdm_split_k = decltype(T::kSupportsTdmSplitK);

// The pipeline's own declaration, read here rather than through the kernel, so a kernel that
// misreads it is caught too.
template <typename Pipeline>
constexpr bool declares_tdm_split_k()
{
    if constexpr(ck_tile::is_detected<has_tdm_split_k, Pipeline>{})
        return static_cast<bool>(Pipeline::kSupportsTdmSplitK);
    else
        return false;
}

template <template <typename...> class Pipeline, template <typename...> class Epilogue>
bool supports_split_k(ck_tile::index_t k_batch)
{
    using Kernel = KernelOf<Pipeline, Epilogue>;
    // K = 1024 leaves every split of k_batch 2 several K tiles, so only the split-K rule decides.
    constexpr ck_tile::index_t M = kTileM;
    constexpr ck_tile::index_t N = kTileN;
    constexpr ck_tile::index_t K = 1024;
    const ck_tile::GemmHostArgs args{nullptr, nullptr, nullptr, k_batch, M, N, K, K, K, N};
    return Kernel::IsSupportedArgument(Kernel::MakeKernelArgs(args));
}

template <template <typename...> class Pipeline_>
struct TdmPipeline
{
    template <typename... Ts>
    using type = Pipeline_<Ts...>;
};

template <typename P>
class TdmSplitKRule : public ::testing::Test
{
};

// Every pipeline UniversalGemmKernel treats as TDM (kIsTdmPipeline: it exposes
// skipCheckValidLaunchParams). A new TDM pipeline belongs in this list.
using TdmPipelines = ::testing::Types<TdmPipeline<ck_tile::GemmPipelineAgBgCrCompTDMV1>,
                                      TdmPipeline<ck_tile::GemmPipelineAgBgCrCompTDMV2>,
                                      TdmPipeline<ck_tile::GemmPipelineAgBgCrCompTDMV3>,
                                      TdmPipeline<ck_tile::GemmPipelineAgBgCrCompTDMProducerConsumer>>;
TYPED_TEST_SUITE(TdmSplitKRule, TdmPipelines);

TYPED_TEST(TdmSplitKRule, IsATdmPipeline)
{
    using Kernel = KernelOf<TypeParam::template type, ck_tile::TdmEpilogue>;
    EXPECT_TRUE(Kernel::UniversalGemmKernel::kIsTdmPipeline);
}

// The TDM epilogue overwrites C, so no pipeline may split K with it.
TYPED_TEST(TdmSplitKRule, TdmEpilogueRejectsSplitK)
{
    EXPECT_FALSE((supports_split_k<TypeParam::template type, ck_tile::TdmEpilogue>(2)));
    if(!ck_tile::is_gfx125_supported())
    {
        GTEST_SKIP() << "accepting a TDM argument requires a gfx125 device";
    }
    // The rejection above comes from the split-K rule, not from the shape.
    EXPECT_TRUE((supports_split_k<TypeParam::template type, ck_tile::TdmEpilogue>(1)));
}

// With CShuffle, which adds the splits into C, split-K is accepted exactly when the pipeline
// declares kSupportsTdmSplitK.
TYPED_TEST(TdmSplitKRule, CShuffleSplitKFollowsTheDeclaration)
{
    constexpr bool declared =
        declares_tdm_split_k<typename TypeParam::template type<Problem>>();
    if constexpr(!declared)
    {
        EXPECT_FALSE((supports_split_k<TypeParam::template type, ck_tile::CShuffleEpilogue>(2)));
    }
    if(!ck_tile::is_gfx125_supported())
    {
        GTEST_SKIP() << "accepting a TDM argument requires a gfx125 device";
    }
    EXPECT_TRUE((supports_split_k<TypeParam::template type, ck_tile::CShuffleEpilogue>(1)));
    EXPECT_EQ((supports_split_k<TypeParam::template type, ck_tile::CShuffleEpilogue>(2)),
              declared);
}

// Keeps the suite from passing vacuously: if no pipeline declared the trait, the accepting side
// of the rule would never be checked.
TEST(TdmSplitKRuleCoverage, CompTdmV3DeclaresSplitK)
{
    EXPECT_TRUE(declares_tdm_split_k<ck_tile::GemmPipelineAgBgCrCompTDMV3<Problem>>());
}

} // namespace
