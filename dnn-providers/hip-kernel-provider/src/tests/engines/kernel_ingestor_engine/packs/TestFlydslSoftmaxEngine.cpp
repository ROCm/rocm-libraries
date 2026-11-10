// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslSoftmaxCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSoftmaxTestGraphs.hpp"

/**
 * @file TestFlydslSoftmaxEngine.cpp
 * @brief hipkernel:flydsl_softmax's matchers, scorer and problem resolution, host-only:
 *        what the pack accepts, what it declines, the [rows, columns] problem a launch
 *        gets, and which object the shipped tiers' priorities pick. The launches
 *        themselves are TestFlydslSoftmaxObjects.cpp.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using data_objects::DataType;

bool matches(const SoftmaxGraphSpec& spec)
{
    const GraphFixture fixture(buildFlydslSoftmaxGraph(spec));
    return matchesGraph(FLYDSL_SOFTMAX, fixture.context()).has_value();
}

/// The problem the pack's shared reading resolves for @p spec's graph.
std::optional<flydsl_softmax::SoftmaxProblem> problemFor(const SoftmaxGraphSpec& spec)
{
    const GraphFixture fixture(buildFlydslSoftmaxGraph(spec));
    const auto context = fixture.context();
    const auto& node = context.graph.getNodeWrapper(0);
    const auto& attributes = node.attributesAs<data_objects::SoftmaxAttributes>();
    const auto& tensors = context.graph.getTensorMap();
    const auto stats = tensors.find(FLYDSL_SOFTMAX_STATS_UID);
    return flydsl_softmax::resolveSoftmaxProblem(attributes,
                                                 *tensors.at(FLYDSL_SOFTMAX_P_UID),
                                                 *tensors.at(FLYDSL_SOFTMAX_S_UID),
                                                 stats == tensors.end() ? nullptr : stats->second);
}

/// A softmax object's descriptor as _instances.py writes it.
struct SoftmaxKernelSpec
{
    std::string dtype = "bf16";
    std::string outDtype = "bf16";
    int64_t n = 0;
    int64_t nMax = 0;
    int64_t nAlign = 8;
    int64_t blockThreads = 256;
    int64_t threadsPerRow = 256;
    int64_t priority = 10;
};

KernelDefinition makeSoftmaxKernel(const SoftmaxKernelSpec& spec)
{
    KernelDefinition kernel;
    kernel.name = "softmax_" + spec.dtype + "_out" + spec.outDtype + "_n" + std::to_string(spec.n)
                  + "_nmax" + std::to_string(spec.nMax) + "_align" + std::to_string(spec.nAlign)
                  + "_bt" + std::to_string(spec.blockThreads) + "_tpr"
                  + std::to_string(spec.threadsPerRow);
    kernel.priority = spec.priority;
    kernel.metadata = {{"dtype", spec.dtype},
                       {"out_dtype", spec.outDtype},
                       {"N", spec.n},
                       {"n_max", spec.nMax},
                       {"n_align", spec.nAlign},
                       {"block_threads", spec.blockThreads},
                       {"threads_per_row", spec.threadsPerRow},
                       {"unroll", int64_t{1}}};
    return kernel;
}

bool kernelMatches(const SoftmaxGraphSpec& graph, const SoftmaxKernelSpec& kernel)
{
    const GraphFixture fixture(buildFlydslSoftmaxGraph(graph));
    const auto bound = matchesGraph(FLYDSL_SOFTMAX, fixture.context());
    return bound.has_value()
           && matchesKernel(FLYDSL_SOFTMAX, fixture.context(), makeSoftmaxKernel(kernel), *bound);
}

// ---------------------------------------------------------------------------
// graph_match: accepted
// ---------------------------------------------------------------------------

struct SpecCase
{
    std::string name;
    SoftmaxGraphSpec spec;
};

std::string specCaseName(const ::testing::TestParamInfo<SpecCase>& info)
{
    return info.param.name;
}

SoftmaxGraphSpec with(void (*edit)(SoftmaxGraphSpec&))
{
    SoftmaxGraphSpec spec;
    edit(spec);
    return spec;
}

class TestFlydslSoftmaxGraphMatcherAcceptance : public ::testing::TestWithParam<SpecCase>
{
};

TEST_P(TestFlydslSoftmaxGraphMatcherAcceptance, Accepts)
{
    EXPECT_TRUE(matches(GetParam().spec));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestFlydslSoftmaxGraphMatcherAcceptance,
    ::testing::Values(
        // Qwen3.8's router: 512 experts.
        SpecCase{"QwenRouter", SoftmaxGraphSpec{}},
        // Qwen3.8's sampling softmax over the vocabulary.
        SpecCase{"QwenVocab", with([](SoftmaxGraphSpec& s) {
                     s.pDataType = DataType::FLOAT;
                     s.pDims = {1, 4, 248320};
                 })},
        SpecCase{"Stats", with([](SoftmaxGraphSpec& s) { s.stats = true; })},
        SpecCase{"StatsLog2", with([](SoftmaxGraphSpec& s) {
                     s.stats = true;
                     s.statsLog2 = true;
                 })},
        SpecCase{"Bf16LogitsF32Probabilities",
                 with([](SoftmaxGraphSpec& s) { s.sDataType = DataType::FLOAT; })},
        SpecCase{"F32LogitsBf16Probabilities", with([](SoftmaxGraphSpec& s) {
                     s.pDataType = DataType::FLOAT;
                     s.sDataType = DataType::BFLOAT16;
                 })},
        SpecCase{"Half", with([](SoftmaxGraphSpec& s) { s.pDataType = DataType::HALF; })},
        SpecCase{"Rank1", with([](SoftmaxGraphSpec& s) { s.pDims = {1000}; })},
        SpecCase{"Rank4AttentionScores", with([](SoftmaxGraphSpec& s) {
                     s.pDims = {2, 8, 64, 333};
                     s.stats = true;
                 })},
        SpecCase{"OddWidth", with([](SoftmaxGraphSpec& s) { s.pDims = {5, 1001}; })},
        SpecCase{"PaddedRows", with([](SoftmaxGraphSpec& s) {
                     s.pDims = {3, 4, 500};
                     s.pStrides = std::vector<int64_t>{2048, 512, 1};
                 })},
        SpecCase{"SInItsOwnPadding", with([](SoftmaxGraphSpec& s) {
                     s.pDims = {4, 500};
                     s.sStrides = std::vector<int64_t>{640, 1};
                 })},
        SpecCase{"ArbitraryStridesOnUnitAxes", with([](SoftmaxGraphSpec& s) {
                     s.pDims = {1, 7, 1, 512};
                     s.pStrides = std::vector<int64_t>{0, 512, 12345, 1};
                     s.stats = true;
                     s.statsStrides = std::vector<int64_t>{99, 1, 7, 3};
                 })},
        SpecCase{"WidthOne", with([](SoftmaxGraphSpec& s) { s.pDims = {9, 1}; })},
        SpecCase{"ElementAlignedOnly", with([](SoftmaxGraphSpec& s) {
                     s.pAlignment = 2;
                     s.sAlignment = 2;
                     s.stats = true;
                     s.statsAlignment = 4;
                 })},
        SpecCase{"ComputeTypeFloat",
                 with([](SoftmaxGraphSpec& s) { s.computeDataType = DataType::FLOAT; })},
        SpecCase{"ComputeTypeUnset",
                 with([](SoftmaxGraphSpec& s) { s.computeDataType = DataType::UNSET; })},
        // More bytes than one launch addresses: the dispatch splits it into row slices.
        SpecCase{"PastOneLaunch", with([](SoftmaxGraphSpec& s) {
                     s.pDataType = DataType::FLOAT;
                     s.pDims = {4096, 248320};
                 })}),
    specCaseName);

// ---------------------------------------------------------------------------
// graph_match: declined
// ---------------------------------------------------------------------------

class TestFlydslSoftmaxGraphMatcherRefusal : public ::testing::TestWithParam<SpecCase>
{
};

TEST_P(TestFlydslSoftmaxGraphMatcherRefusal, Refuses)
{
    EXPECT_FALSE(matches(GetParam().spec));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestFlydslSoftmaxGraphMatcherRefusal,
    ::testing::Values(
        SpecCase{"UnsupportedPDtype", with([](SoftmaxGraphSpec& s) {
                     s.pDataType = DataType::DOUBLE;
                     s.sDataType = DataType::FLOAT;
                     s.computeDataType = DataType::FLOAT;
                 })},
        SpecCase{"UnsupportedSDtype",
                 with([](SoftmaxGraphSpec& s) { s.sDataType = DataType::INT32; })},
        SpecCase{"ComputeTypeDouble",
                 with([](SoftmaxGraphSpec& s) { s.computeDataType = DataType::DOUBLE; })},
        SpecCase{"SOfAnotherShape",
                 with([](SoftmaxGraphSpec& s) { s.sDims = std::vector<int64_t>{1, 7, 511}; })},
        SpecCase{"StridedLastAxis",
                 with([](SoftmaxGraphSpec& s) { s.pStrides = std::vector<int64_t>{1024, 1, 7}; })},
        SpecCase{"OverlappingRows", with([](SoftmaxGraphSpec& s) {
                     s.pStrides = std::vector<int64_t>{3584, 256, 1};
                 })},
        SpecCase{"BroadcastRows", with([](SoftmaxGraphSpec& s) {
                     s.pDims = {2, 7, 512};
                     s.pStrides = std::vector<int64_t>{0, 512, 1};
                 })},
        // [B, S, N] with S outer in memory: no one row stride steps both leading axes.
        SpecCase{"LeadingAxesNotCollapsible", with([](SoftmaxGraphSpec& s) {
                     s.pDims = {2, 7, 512};
                     s.pStrides = std::vector<int64_t>{512, 1024, 1};
                 })},
        SpecCase{"ZeroExtent", with([](SoftmaxGraphSpec& s) { s.pDims = {0, 512}; })},
        SpecCase{"StridesOfAnotherRank",
                 with([](SoftmaxGraphSpec& s) { s.pStrides = std::vector<int64_t>{512, 1}; })},
        SpecCase{"PMisalignedForItsElements", with([](SoftmaxGraphSpec& s) { s.pAlignment = 1; })},
        SpecCase{"SMisalignedForItsElements", with([](SoftmaxGraphSpec& s) {
                     s.sDataType = DataType::FLOAT;
                     s.sAlignment = 2;
                 })},
        SpecCase{"StatsNotFloat", with([](SoftmaxGraphSpec& s) {
                     s.stats = true;
                     s.statsDataType = DataType::BFLOAT16;
                 })},
        SpecCase{"StatsOfPsShape", with([](SoftmaxGraphSpec& s) {
                     s.stats = true;
                     s.statsDims = std::vector<int64_t>{1, 7, 512};
                 })},
        SpecCase{"StatsMissingAnAxis", with([](SoftmaxGraphSpec& s) {
                     s.stats = true;
                     s.statsDims = std::vector<int64_t>{7, 1};
                 })},
        SpecCase{"StatsNotPacked", with([](SoftmaxGraphSpec& s) {
                     s.pDims = {2, 7, 512};
                     s.stats = true;
                     s.statsStrides = std::vector<int64_t>{1, 2, 1};
                 })},
        SpecCase{"StatsMisaligned", with([](SoftmaxGraphSpec& s) {
                     s.stats = true;
                     s.statsAlignment = 2;
                 })},
        // One f32 row past a launch's 32-bit byte offsets.
        SpecCase{"RowPastOneLaunch", with([](SoftmaxGraphSpec& s) {
                     s.pDataType = DataType::FLOAT;
                     s.pDims = {1, int64_t{1} << 29};
                 })},
        SpecCase{"VirtualP",
                 with([](SoftmaxGraphSpec& s) { s.virtualTensors = {FLYDSL_SOFTMAX_P_UID}; })},
        SpecCase{"VirtualS",
                 with([](SoftmaxGraphSpec& s) { s.virtualTensors = {FLYDSL_SOFTMAX_S_UID}; })},
        SpecCase{"VirtualStats", with([](SoftmaxGraphSpec& s) {
                     s.stats = true;
                     s.virtualTensors = {FLYDSL_SOFTMAX_STATS_UID};
                 })},
        SpecCase{"PassByValueP", with([](SoftmaxGraphSpec& s) {
                     s.pDims = {1};
                     s.passByValueTensors = {FLYDSL_SOFTMAX_P_UID};
                 })},
        SpecCase{"TwoNodes", with([](SoftmaxGraphSpec& s) { s.secondNode = true; })},
        SpecCase{"OverrideShape",
                 with([](SoftmaxGraphSpec& s) { s.overrideShapeEnabled = true; })}),
    specCaseName);

// ---------------------------------------------------------------------------
// the problem a launch gets
// ---------------------------------------------------------------------------

TEST(TestFlydslSoftmaxProblem, CollapsesTheLeadingAxesIntoRows)
{
    SoftmaxGraphSpec spec;
    spec.pDims = {2, 8, 64, 333};
    spec.stats = true;
    spec.statsLog2 = true;
    const auto problem = problemFor(spec);
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->rows, 2 * 8 * 64);
    EXPECT_EQ(problem->columns, 333);
    EXPECT_EQ(problem->pRowStride, 333);
    EXPECT_EQ(problem->sRowStride, 333);
    EXPECT_TRUE(problem->storeStats);
    EXPECT_TRUE(problem->statsLog2);
}

TEST(TestFlydslSoftmaxProblem, ReadsEachOperandsOwnRowPitch)
{
    SoftmaxGraphSpec spec;
    spec.pDims = {3, 4, 500};
    spec.pStrides = std::vector<int64_t>{2048, 512, 1};
    spec.sStrides = flydslSoftmaxPackedStrides(spec.pDims);
    const auto problem = problemFor(spec);
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->rows, 12);
    EXPECT_EQ(problem->pRowStride, 512);
    EXPECT_EQ(problem->sRowStride, 500);
    EXPECT_FALSE(problem->storeStats);
}

TEST(TestFlydslSoftmaxProblem, ReadsRankOneAsOneRow)
{
    SoftmaxGraphSpec spec;
    spec.pDims = {1000};
    const auto problem = problemFor(spec);
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->rows, 1);
    EXPECT_EQ(problem->columns, 1000);
}

TEST(TestFlydslSoftmaxProblem, SplitsWhatOneLaunchCannotAddressIntoWholeRowSlices)
{
    SoftmaxGraphSpec spec;
    spec.pDataType = DataType::FLOAT;
    spec.sDataType = DataType::BFLOAT16;
    spec.pDims = {4096, 248320};
    const auto problem = problemFor(spec);
    ASSERT_TRUE(problem.has_value());
    // (2^31 - 1) / 4 bytes of f32 p hold 2162 rows of 248320.
    EXPECT_EQ(flydsl_softmax::rowsPerLaunch(*problem, 1), 2162);
    // A multiple of the rows a block serves.
    EXPECT_EQ(flydsl_softmax::rowsPerLaunch(*problem, 4), 2160);
    // Within one launch: every row, whatever the rows per block.
    spec.pDims = {7, 512};
    const auto small = problemFor(spec);
    ASSERT_TRUE(small.has_value());
    EXPECT_EQ(flydsl_softmax::rowsPerLaunch(*small, 4), 7);
}

// ---------------------------------------------------------------------------
// kernel_match, score, binding
// ---------------------------------------------------------------------------

TEST(TestFlydslSoftmaxKernelMatcher, PairsBothDtypesWithTheObjectBuiltForThem)
{
    const std::vector<std::pair<DataType, std::string>> types{
        {DataType::BFLOAT16, "bf16"}, {DataType::HALF, "f16"}, {DataType::FLOAT, "f32"}};
    for(const auto& [pType, pName] : types)
    {
        for(const auto& [sType, sName] : types)
        {
            SoftmaxGraphSpec graph;
            graph.pDataType = pType;
            graph.sDataType = sType;
            for(const auto& candidateP : types)
            {
                for(const auto& candidateS : types)
                {
                    SoftmaxKernelSpec kernel;
                    kernel.dtype = candidateP.second;
                    kernel.outDtype = candidateS.second;
                    kernel.nAlign = 1;
                    EXPECT_EQ(kernelMatches(graph, kernel),
                              candidateP.second == pName && candidateS.second == sName)
                        << pName << "->" << sName << " graph against a " << candidateP.second
                        << "->" << candidateS.second << " object";
                }
            }
        }
    }
}

TEST(TestFlydslSoftmaxKernelMatcher, ServesABakedWidthExactly)
{
    SoftmaxKernelSpec baked;
    baked.n = 512;
    baked.nMax = 512;
    baked.threadsPerRow = 64;
    EXPECT_TRUE(kernelMatches(SoftmaxGraphSpec{}, baked));
    EXPECT_FALSE(kernelMatches(with([](SoftmaxGraphSpec& s) { s.pDims = {7, 256}; }), baked));
    EXPECT_FALSE(kernelMatches(with([](SoftmaxGraphSpec& s) { s.pDims = {7, 1024}; }), baked));
}

TEST(TestFlydslSoftmaxKernelMatcher, ServesARuntimeWidthUpToItsBound)
{
    SoftmaxKernelSpec bounded;
    bounded.nMax = 2048;
    bounded.threadsPerRow = 64;
    EXPECT_TRUE(kernelMatches(with([](SoftmaxGraphSpec& s) { s.pDims = {7, 2048}; }), bounded));
    EXPECT_FALSE(kernelMatches(with([](SoftmaxGraphSpec& s) { s.pDims = {7, 2056}; }), bounded));
    SoftmaxKernelSpec unbounded;
    unbounded.nMax = 0;
    EXPECT_TRUE(kernelMatches(with([](SoftmaxGraphSpec& s) { s.pDims = {1, 248320}; }), unbounded));
}

TEST(TestFlydslSoftmaxKernelMatcher, ServesAVectorObjectOnlyWhereItsCopiesFit)
{
    const SoftmaxKernelSpec vector;
    SoftmaxKernelSpec element;
    element.nAlign = 1;
    // The width a multiple of 8, packed, 16-byte bases: both serve.
    EXPECT_TRUE(kernelMatches(SoftmaxGraphSpec{}, vector));
    EXPECT_TRUE(kernelMatches(SoftmaxGraphSpec{}, element));
    // An odd width, a row pitch or a base off the vector grid: the element object only.
    const auto oddWidth = with([](SoftmaxGraphSpec& s) { s.pDims = {7, 1001}; });
    const auto oddPPitch = with([](SoftmaxGraphSpec& s) {
        s.pDims = {7, 512};
        s.pStrides = std::vector<int64_t>{516, 1};
        s.sStrides = std::vector<int64_t>{512, 1};
    });
    const auto oddSPitch = with([](SoftmaxGraphSpec& s) {
        s.pDims = {7, 512};
        s.sStrides = std::vector<int64_t>{516, 1};
    });
    const auto smallBase = with([](SoftmaxGraphSpec& s) { s.sAlignment = 8; });
    for(const auto& spec : {oddWidth, oddPPitch, oddSPitch, smallBase})
    {
        EXPECT_FALSE(kernelMatches(spec, vector));
        EXPECT_TRUE(kernelMatches(spec, element));
    }
}

/// The objects _instances.py ships per dtype pair (bf16 here), as (n, n_max, n_align,
/// block_threads, threads_per_row, priority).
std::vector<SoftmaxKernelSpec> shippedTiers()
{
    std::vector<SoftmaxKernelSpec> tiers;
    const auto add = [&tiers](int64_t n,
                              int64_t nMax,
                              int64_t nAlign,
                              int64_t blockThreads,
                              int64_t threadsPerRow,
                              int64_t priority) {
        SoftmaxKernelSpec spec;
        spec.n = n;
        spec.nMax = nMax;
        spec.nAlign = nAlign;
        spec.blockThreads = blockThreads;
        spec.threadsPerRow = threadsPerRow;
        spec.priority = priority;
        tiers.push_back(spec);
    };
    add(0, 256, 8, 256, 16, 40);
    add(0, 2048, 8, 256, 64, 30);
    add(0, 16384, 8, 256, 256, 20);
    add(0, 0, 8, 1024, 1024, 15);
    add(0, 2048, 1, 256, 64, 12);
    add(0, 0, 1, 256, 256, 10);
    add(512, 512, 8, 256, 64, 100);
    return tiers;
}

/// The object the shipped priorities pick for a [7, @p columns] bf16 graph: the highest
/// priority among those kernel_match admits, as the ingestor ranks them.
std::optional<SoftmaxKernelSpec> pickFor(int64_t columns)
{
    SoftmaxGraphSpec graph;
    graph.pDims = {7, columns};
    std::optional<SoftmaxKernelSpec> best;
    for(const auto& tier : shippedTiers())
    {
        if(kernelMatches(graph, tier) && (!best.has_value() || tier.priority > best->priority))
        {
            best = tier;
        }
    }
    return best;
}

TEST(TestFlydslSoftmaxScore, ThePrioritiesPickTheTierForEachWidth)
{
    struct Expected
    {
        int64_t columns;
        int64_t threadsPerRow;
        int64_t nAlign;
        int64_t n;
    };
    for(const auto& expected : {Expected{8, 16, 8, 0},
                                Expected{256, 16, 8, 0},
                                Expected{512, 64, 8, 512},
                                Expected{384, 64, 8, 0},
                                Expected{2048, 64, 8, 0},
                                Expected{4096, 256, 8, 0},
                                Expected{16384, 256, 8, 0},
                                Expected{16392, 1024, 8, 0},
                                Expected{248320, 1024, 8, 0},
                                Expected{1001, 64, 1, 0},
                                Expected{248321, 256, 1, 0}})
    {
        const auto picked = pickFor(expected.columns);
        ASSERT_TRUE(picked.has_value()) << expected.columns;
        EXPECT_EQ(picked->threadsPerRow, expected.threadsPerRow) << expected.columns;
        EXPECT_EQ(picked->nAlign, expected.nAlign) << expected.columns;
        EXPECT_EQ(picked->n, expected.n) << expected.columns;
    }
}

TEST(TestFlydslSoftmaxScore, IsTheDescriptorPriority)
{
    const GraphFixture fixture(buildFlydslSoftmaxGraph());
    const auto bound = matchesGraph(FLYDSL_SOFTMAX, fixture.context());
    ASSERT_TRUE(bound.has_value());
    SoftmaxKernelSpec kernel;
    kernel.priority = 37;
    EXPECT_EQ(scoreKernel(FLYDSL_SOFTMAX, fixture.context(), makeSoftmaxKernel(kernel), *bound),
              37.0);
}

TEST(TestFlydslSoftmaxBinding, BindsTheOperandsAndTheRowGeometry)
{
    SoftmaxGraphSpec spec;
    spec.stats = true;
    spec.pDims = {3, 4, 500};
    spec.pStrides = std::vector<int64_t>{2048, 512, 1};
    spec.sStrides = flydslSoftmaxPackedStrides(spec.pDims);
    const GraphFixture fixture(buildFlydslSoftmaxGraph(spec));
    const auto bound = matchesGraph(FLYDSL_SOFTMAX, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_SOFTMAX.inputAToken), FLYDSL_SOFTMAX_P_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_SOFTMAX.inputBToken), FLYDSL_SOFTMAX_STATS_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_SOFTMAX.outputToken), FLYDSL_SOFTMAX_S_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_SOFTMAX_COLUMNS_TOKEN), 500);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_SOFTMAX_P_ROW_STRIDE_TOKEN), 512);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_SOFTMAX_S_ROW_STRIDE_TOKEN), 500);
}

TEST(TestFlydslSoftmaxBinding, BindsNoStatsWhenTheGraphHasNone)
{
    const GraphFixture fixture(buildFlydslSoftmaxGraph());
    const auto bound = matchesGraph(FLYDSL_SOFTMAX, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_FALSE(tryGetBoundInt(*bound, FLYDSL_SOFTMAX.inputBToken).has_value());
}

TEST(TestFlydslSoftmaxDispatch, NeedsNoWorkspace)
{
    const GraphFixture fixture(buildFlydslSoftmaxGraph());
    const auto bound = matchesGraph(FLYDSL_SOFTMAX, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(dispatchHandler(FLYDSL_SOFTMAX)
                  .workspaceBytes(fixture.context(), *bound, makeSoftmaxKernel({})),
              0U);
}

TEST(TestFlydslSoftmaxDispatch, RefusesToPrepareWithoutTheMatchersBindings)
{
    const GraphFixture fixture(buildFlydslSoftmaxGraph());
    EXPECT_THROW(
        static_cast<void>(dispatchHandler(FLYDSL_SOFTMAX)
                              .prepare(fixture.context(), BoundTokens{}, makeSoftmaxKernel({}))),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
