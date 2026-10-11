// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Adapted from dnn-providers/hip-kernel-provider/src/tests/engines/kernel_ingestor_engine/packs/TestFlydslRmsNormEngine.cpp at commit 03911535bd0 (FlyDSL provider)

// Redundant with the CMake gate on purpose: if this file lands on an ungated source list
// it compiles to an empty translation unit instead of a suite reading an unstaged shard.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_TRITON)

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/packs/TritonRmsNormTestGraphs.hpp"

/**
 * @file TestTritonRmsNormMatchers.cpp
 * @brief Device-free tests of the Triton RMS-norm pack's graph match, kernel match and
 *        dispatch handler.
 *
 * Kernels are built by hand with the four KMD metadata fields. Device cases are in
 * TestTritonRmsNormDispatch.cpp, the shard census in TestTritonRmsNormPacks.cpp.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

using data_objects::DataType;
using data_objects::NormFwdPhase;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

bool matches(const MatchContext& context)
{
    return matchesGraph(TRITON_RMSNORM, context).has_value();
}

/// The bindings a real plan build would hand the dispatch, from running the graph match.
BoundTokens bindingsFor(const MatchContext& context)
{
    auto bound = matchesGraph(TRITON_RMSNORM, context);
    if(!bound.has_value())
    {
        throw std::logic_error("test graph does not match the pack it is dispatched against");
    }
    return std::move(*bound);
}

/// A kernel carrying the four metadata fields this pack's KMD declares; source is left
/// default-constructed because these cases read metadata only.
KernelDefinition makeTritonKernel(const std::string& dtype,
                                  int64_t blockN,
                                  int64_t priority,
                                  int64_t blockThreads = 256,
                                  int64_t sharedMemBytes = 2048)
{
    KernelDefinition kernel;
    kernel.name = "rmsnorm_" + dtype + "_bn" + std::to_string(blockN);
    kernel.priority = priority;
    kernel.metadata = {{std::string(TRITON_DTYPE_FIELD), dtype},
                       {std::string(TRITON_BLOCK_N_FIELD), blockN},
                       {std::string(TRITON_BLOCK_THREADS_FIELD), blockThreads},
                       {std::string(TRITON_SHARED_MEM_BYTES_FIELD), sharedMemBytes}};
    return kernel;
}

/// The four shipped instances per dtype, with the descriptors' priorities.
std::vector<KernelDefinition> shippedLadder(const std::string& dtype)
{
    return {makeTritonKernel(dtype, 1024, 400),
            makeTritonKernel(dtype, 2048, 300),
            makeTritonKernel(dtype, 4096, 200, 512),
            makeTritonKernel(dtype, 8192, 100, 512)};
}

// ---------------------------------------------------------------------------
// graph_match: what this pack accepts
// ---------------------------------------------------------------------------

/// Plain function pointer rather than std::function: FlatBufferBuilder is move-only.
struct GraphCase
{
    std::string name;
    flatbuffers::FlatBufferBuilder (*buildGraph)();
};

std::string graphCaseName(const ::testing::TestParamInfo<GraphCase>& info)
{
    return info.param.name;
}

/// A default graph with @p mutate applied: one field changed per case.
template <typename Mutate>
flatbuffers::FlatBufferBuilder graphWith(Mutate&& mutate)
{
    TritonRmsNormGraphOptions options;
    mutate(options);
    return buildTritonRmsNormGraph(options);
}

class TestTritonRmsNormGraphMatcherAcceptance : public ::testing::TestWithParam<GraphCase>
{
};

TEST_P(TestTritonRmsNormGraphMatcherAcceptance, Accepts)
{
    const GraphFixture fixture(GetParam().buildGraph());
    EXPECT_TRUE(matches(fixture.context()));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestTritonRmsNormGraphMatcherAcceptance,
    ::testing::Values(
        // The default graph carries the frontend's default alignment of 16 on every operand.
        GraphCase{"DefaultGraphAtAlignment16", [] { return buildTritonRmsNormGraph(); }},
        GraphCase{"Half", [] { return graphWith([](auto& o) { o.dataType = DataType::HALF; }); }},
        // Rank above two collapses: the leading axes multiply into the row count.
        GraphCase{"Rank4CollapsesToRows",
                  [] { return graphWith([](auto& o) { o.xDims = {2, 3, 4, 4096}; }); }},
        // Any width matches the graph; whether a kernel covers it is the kernel matcher's call.
        GraphCase{"AWidthNoKernelCovers",
                  [] { return graphWith([](auto& o) { o.xDims = {8, 8193}; }); }},
        // Epsilon is a kernarg: no value is refused.
        GraphCase{"ABakedEpsilonOtherThanOneEMinusFive",
                  [] { return graphWith([](auto& o) { o.epsilon = 1e-2F; }); }},
        // Not inspectable at match time; launch() reads it.
        GraphCase{"ARuntimeEpsilon",
                  [] { return graphWith([](auto& o) { o.runtimeEpsilon = true; }); }},
        GraphCase{
            "InferencePhase",
            [] { return graphWith([](auto& o) { o.forwardPhase = NormFwdPhase::INFERENCE; }); }},
        // Any multiple of 16 is accepted.
        GraphCase{"AlignmentAboveSixteen",
                  [] {
                      return graphWith([](auto& o) {
                          o.xAlignment = 32;
                          o.scaleAlignment = 64;
                          o.yAlignment = 256;
                      });
                  }}),
    graphCaseName);

TEST(TestTritonRmsNormBinding, BindsEveryTokenTheDispatchReads)
{
    const GraphFixture fixture(graphWith([](auto& o) { o.xDims = {2, 3, 4096}; }));
    const auto bound = matchesGraph(TRITON_RMSNORM, fixture.context());
    ASSERT_TRUE(bound.has_value());

    EXPECT_EQ(tryGetBoundInt(*bound, TRITON_RMSNORM.inputAToken), TRITON_X_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, TRITON_RMSNORM.inputBToken), TRITON_SCALE_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, TRITON_EPSILON_TOKEN), TRITON_EPSILON_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, TRITON_RMSNORM.outputToken), TRITON_Y_UID);

    // The grid and kernarg values. Rows is the product of every axis but the innermost.
    EXPECT_EQ(tryGetBoundInt(*bound, TRITON_ROWS_TOKEN), 6);
    EXPECT_EQ(tryGetBoundInt(*bound, TRITON_COLUMNS_TOKEN), 4096);
}

TEST(TestTritonRmsNormBinding, CollapsesRank4UnderALeadingUnitScale)
{
    // A scale of (1,1,1,W) is accepted like the rank-1 {W}: the matcher requires only one
    // row and an innermost extent equal to the width. x's leading axes collapse to 2x8x4.
    const GraphFixture fixture(graphWith([](auto& o) {
        o.xDims = {2, 8, 4, 4096};
        o.scaleDims = std::vector<int64_t>{1, 1, 1, 4096};
    }));

    const auto bound = matchesGraph(TRITON_RMSNORM, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, TRITON_ROWS_TOKEN), 64);
    EXPECT_EQ(tryGetBoundInt(*bound, TRITON_COLUMNS_TOKEN), 4096);
}

// ---------------------------------------------------------------------------
// graph_match: what this pack declines
// ---------------------------------------------------------------------------

class TestTritonRmsNormGraphMatcherRefusal : public ::testing::TestWithParam<GraphCase>
{
};

TEST_P(TestTritonRmsNormGraphMatcherRefusal, Refuses)
{
    const GraphFixture fixture(GetParam().buildGraph());
    EXPECT_FALSE(matches(fixture.context()));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestTritonRmsNormGraphMatcherRefusal,
    ::testing::Values(
        // The kernels compute y = x * rsqrt(mean(x^2) + eps) * scale only.
        GraphCase{"WithABias", [] { return graphWith([](auto& o) { o.withBias = true; }); }},
        GraphCase{"SavingTheInverseRms",
                  [] { return graphWith([](auto& o) { o.withInvRms = true; }); }},
        // Training needs the saved statistic even if the graph names no tensor for it.
        GraphCase{
            "TrainingPhase",
            [] { return graphWith([](auto& o) { o.forwardPhase = NormFwdPhase::TRAINING; }); }},
        // Only bf16 and f16 are built.
        GraphCase{"AnUnsupportedDtype",
                  [] { return graphWith([](auto& o) { o.dataType = DataType::FLOAT; }); }},
        // One dtype for all three operands: scale is read at x's type.
        GraphCase{"ScaleAtADifferentDtype",
                  [] { return graphWith([](auto& o) { o.scaleDataType = DataType::HALF; }); }},
        // Rank one has no row to reduce over.
        GraphCase{"Rank1Input", [] { return graphWith([](auto& o) { o.xDims = {4096}; }); }},
        // The kernel addresses row r at r*N with unit inner stride: transposed or padded
        // operands cannot be described.
        GraphCase{
            "NonPackedStrides",
            [] { return graphWith([](auto& o) { o.xStrides = std::vector<int64_t>{1, 8}; }); }},
        // Channels-last rank 4: dims() is in logical order and only the strides express the
        // layout, so a matcher reading dims() alone would normalise over W while memory has C
        // innermost. The packed check expects stride 1 on the innermost axis; W's stride is C.
        GraphCase{"ChannelsLastStrides",
                  [] {
                      return graphWith([](auto& o) {
                          o.xDims = {2, 8, 4, 4096};
                          o.scaleDims = std::vector<int64_t>{1, 1, 1, 4096};
                          o.xStrides = hipdnn_data_sdk::utilities::generateStrides(
                              o.xDims, hipdnn_data_sdk::utilities::strideOrderNhwc(o.xDims.size()));
                      });
                  }},
        // Scale is indexed along the normalised axis with no broadcast: it must be exactly
        // that wide...
        GraphCase{"ScaleNarrowerThanTheAxis",
                  [] { return graphWith([](auto& o) { o.scaleDims = std::vector<int64_t>{1}; }); }},
        // ...and one vector, not one per row (a broadcast-shaped scale).
        GraphCase{
            "ScaleWithARowPerInputRow",
            [] { return graphWith([](auto& o) { o.scaleDims = std::vector<int64_t>{8, 4096}; }); }},
        // The kernels need 16-byte aligned base pointers, so declaring less on any one of
        // the three is declined.
        GraphCase{"XAtAlignment8", [] { return graphWith([](auto& o) { o.xAlignment = 8; }); }},
        GraphCase{"ScaleAtAlignment8",
                  [] { return graphWith([](auto& o) { o.scaleAlignment = 8; }); }},
        GraphCase{"YAtAlignment8", [] { return graphWith([](auto& o) { o.yAlignment = 8; }); }},
        // The backend checks only `ptr % alignment == 0`, so 24 admits an 8-mod-16 pointer:
        // the declaration must be a multiple of 16. Zero means undeclared.
        GraphCase{"XAtAlignment24", [] { return graphWith([](auto& o) { o.xAlignment = 24; }); }},
        GraphCase{"ScaleAtAlignment24",
                  [] { return graphWith([](auto& o) { o.scaleAlignment = 24; }); }},
        GraphCase{"YAtAlignment24", [] { return graphWith([](auto& o) { o.yAlignment = 24; }); }},
        GraphCase{"XAtAlignment0", [] { return graphWith([](auto& o) { o.xAlignment = 0; }); }},
        GraphCase{"ScaleAtAlignment0",
                  [] { return graphWith([](auto& o) { o.scaleAlignment = 0; }); }},
        GraphCase{"YAtAlignment0", [] { return graphWith([](auto& o) { o.yAlignment = 0; }); }},
        // The frontend rejects these, but the matcher sees unvalidated graphs and must
        // decline them.
        GraphCase{"ZeroRows", [] { return graphWith([](auto& o) { o.xDims = {0, 4096}; }); }},
        GraphCase{"NegativeRows", [] { return graphWith([](auto& o) { o.xDims = {-8, 4096}; }); }},
        GraphCase{"ZeroColumns",
                  [] {
                      return graphWith([](auto& o) {
                          o.xDims = {8, 0};
                          o.xStrides = std::vector<int64_t>{0, 1};
                      });
                  }},
        GraphCase{"NegativeColumns",
                  [] {
                      return graphWith([](auto& o) {
                          o.xDims = {8, -4096};
                          o.xStrides = std::vector<int64_t>{-4096, 1};
                      });
                  }},
        // Leading extents whose product (2^64) overflows int64.
        GraphCase{"RowCountOverflowingInt64",
                  [] {
                      return graphWith(
                          [](auto& o) { o.xDims = {int64_t{1} << 32, int64_t{1} << 32, 8}; });
                  }},
        // Fit int64 but not 32 bits.
        GraphCase{"RowCountAboveInt32",
                  [] { return graphWith([](auto& o) { o.xDims = {int64_t{1} << 31, 4096}; }); }},
        GraphCase{"ColumnsAboveInt32",
                  [] { return graphWith([](auto& o) { o.xDims = {2, int64_t{1} << 31}; }); }},
        // The kernel addresses row r at r*N; a ragged offset tensor on any operand cannot
        // be honoured.
        GraphCase{"RaggedX", [] { return graphWith([](auto& o) { o.xRagged = true; }); }},
        GraphCase{"RaggedScale", [] { return graphWith([](auto& o) { o.scaleRagged = true; }); }},
        GraphCase{"RaggedY", [] { return graphWith([](auto& o) { o.yRagged = true; }); }},
        // Each kernel serves one complete single-node graph.
        GraphCase{"TwoNodes", [] { return graphWith([](auto& o) { o.secondNode = true; }); }},
        // Layer norm is a sibling union arm, not a mode of RMS norm.
        GraphCase{"ALayernormNode",
                  [] { return graphWith([](auto& o) { o.asLayernorm = true; }); }},
        // The node names a y uid the graph does not carry.
        GraphCase{"NoYTensor", [] { return graphWith([](auto& o) { o.omitY = true; }); }},
        // Every operand is read or written through a device pointer.
        GraphCase{"VirtualX", [] { return graphWith([](auto& o) { o.xVirtual = true; }); }},
        GraphCase{"VirtualScale", [] { return graphWith([](auto& o) { o.scaleVirtual = true; }); }},
        GraphCase{"VirtualY", [] { return graphWith([](auto& o) { o.yVirtual = true; }); }},
        GraphCase{"PassByValueX", [] { return graphWith([](auto& o) { o.xPassByValue = true; }); }},
        // y is written at x's element type.
        GraphCase{"YAtADifferentDtype",
                  [] { return graphWith([](auto& o) { o.yDataType = DataType::HALF; }); }},
        // Unvalidated graphs may carry strides that do not describe the dims. The leading two
        // strides are x's packed ones, so only the rank check can refuse this graph.
        GraphCase{"XStridesOfADifferentRankThanDims",
                  [] {
                      return graphWith(
                          [](auto& o) { o.xStrides = std::vector<int64_t>{4096, 1, 1}; });
                  }},
        // y is written over x's index space, densely.
        GraphCase{
            "YShapeDiffersFromX",
            [] { return graphWith([](auto& o) { o.yDims = std::vector<int64_t>{8, 2048}; }); }},
        GraphCase{
            "NonPackedY",
            [] { return graphWith([](auto& o) { o.yStrides = std::vector<int64_t>{1, 8}; }); }},
        // The default rank-1 scale {4096} at stride 2: a real stride, not one on an
        // extent-1 axis, which would still be dense.
        GraphCase{
            "NonPackedScale",
            [] { return graphWith([](auto& o) { o.scaleStrides = std::vector<int64_t>{2}; }); }},
        // An epsilon that cannot be read as a scalar operand.
        GraphCase{"UnclassifiableEpsilon",
                  [] { return graphWith([](auto& o) { o.epsilonDataType = DataType::UNSET; }); }}),
    graphCaseName);

// ---------------------------------------------------------------------------
// kernel_match: dtype, block width and grid size
// ---------------------------------------------------------------------------

TEST(TestTritonRmsNormKernelMatcher, AdmitsEveryBlockAtLeastAsWideAsTheAxis)
{
    const GraphFixture fixture(graphWith([](auto& o) { o.xDims = {8, 1000}; }));
    const auto bound = bindingsFor(fixture.context());

    for(const auto& kernel : shippedLadder("bf16"))
    {
        EXPECT_TRUE(matchesKernel(TRITON_RMSNORM, fixture.context(), kernel, bound)) << kernel.name;
    }
}

TEST(TestTritonRmsNormKernelMatcher, AdmitsABlockExactlyAsWideAsTheAxisAndNotNarrower)
{
    // N == block_n is admitted (the tile covers the row exactly); N == block_n + 1 is not.
    const GraphFixture atBoundary(graphWith([](auto& o) { o.xDims = {8, 2048}; }));
    const GraphFixture pastBoundary(graphWith([](auto& o) { o.xDims = {8, 2049}; }));
    const auto atBound = bindingsFor(atBoundary.context());
    const auto pastBound = bindingsFor(pastBoundary.context());

    const auto bn1024 = makeTritonKernel("bf16", 1024, 400);
    const auto bn2048 = makeTritonKernel("bf16", 2048, 300);

    EXPECT_FALSE(matchesKernel(TRITON_RMSNORM, atBoundary.context(), bn1024, atBound));
    EXPECT_TRUE(matchesKernel(TRITON_RMSNORM, atBoundary.context(), bn2048, atBound));
    EXPECT_FALSE(matchesKernel(TRITON_RMSNORM, pastBoundary.context(), bn2048, pastBound));
}

TEST(TestTritonRmsNormKernelMatcher, RefusesEveryBlockForAWidthAboveTheLargest)
{
    // N = 8193 matches the graph but no kernel, so the engine declines: a tile narrower
    // than the row would skip its tail.
    const GraphFixture fixture(graphWith([](auto& o) { o.xDims = {8, 8193}; }));
    const auto bound = bindingsFor(fixture.context());

    for(const auto& dtype : {std::string("bf16"), std::string("f16")})
    {
        for(const auto& kernel : shippedLadder(dtype))
        {
            EXPECT_FALSE(matchesKernel(TRITON_RMSNORM, fixture.context(), kernel, bound))
                << kernel.name;
        }
    }
}

TEST(TestTritonRmsNormKernelMatcher, RefusesAKernelOfTheWrongDtype)
{
    const GraphFixture fixture(graphWith([](auto& o) { o.xDims = {8, 1000}; }));
    const auto bound = bindingsFor(fixture.context());

    for(const auto& kernel : shippedLadder("f16"))
    {
        EXPECT_FALSE(matchesKernel(TRITON_RMSNORM, fixture.context(), kernel, bound))
            << kernel.name;
    }
}

TEST(TestTritonRmsNormKernelMatcher, ReadsTheDescriptorDtypeVocabularyNotTheFlatbufferEnum)
{
    const GraphFixture fixture(graphWith([](auto& o) { o.xDims = {8, 1000}; }));
    const auto bound = bindingsFor(fixture.context());

    // The flatbuffer spelling "BFLOAT16" is not accepted; descriptors say "bf16".
    EXPECT_FALSE(matchesKernel(
        TRITON_RMSNORM, fixture.context(), makeTritonKernel("BFLOAT16", 4096, 200), bound));
}

TEST(TestTritonRmsNormKernelMatcher, ServesHalfGraphsFromHalfKernels)
{
    const GraphFixture fixture(graphWith([](auto& o) {
        o.dataType = DataType::HALF;
        o.xDims = {8, 1000};
    }));
    const auto bound = bindingsFor(fixture.context());

    EXPECT_TRUE(matchesKernel(
        TRITON_RMSNORM, fixture.context(), makeTritonKernel("f16", 1024, 400), bound));
    EXPECT_FALSE(matchesKernel(
        TRITON_RMSNORM, fixture.context(), makeTritonKernel("bf16", 1024, 400), bound));
}

TEST(TestTritonRmsNormKernelMatcher, AdmitsTheLargestGridA256ThreadBlockCanLaunch)
{
    // HIP caps a grid dimension at UINT32_MAX threads: 16777215 rows of 256 threads fit,
    // 16777216 rows (2^32 threads) do not.
    const GraphFixture largest(graphWith([](auto& o) { o.xDims = {16777215, 1024}; }));
    const GraphFixture tooLarge(graphWith([](auto& o) { o.xDims = {16777216, 1024}; }));
    const auto largestBound = bindingsFor(largest.context());
    const auto tooLargeBound = bindingsFor(tooLarge.context());
    const auto bn1024 = makeTritonKernel("bf16", 1024, 400, 256);

    EXPECT_TRUE(matchesKernel(TRITON_RMSNORM, largest.context(), bn1024, largestBound));
    EXPECT_FALSE(matchesKernel(TRITON_RMSNORM, tooLarge.context(), bn1024, tooLargeBound));
}

TEST(TestTritonRmsNormKernelMatcher, AdmitsTheLargestGridA512ThreadBlockCanLaunch)
{
    // The bound is per instance: at 512 threads it halves, to 8388607 rows.
    const GraphFixture largest(graphWith([](auto& o) { o.xDims = {8388607, 4096}; }));
    const GraphFixture tooLarge(graphWith([](auto& o) { o.xDims = {8388608, 4096}; }));
    const auto largestBound = bindingsFor(largest.context());
    const auto tooLargeBound = bindingsFor(tooLarge.context());
    const auto bn4096 = makeTritonKernel("bf16", 4096, 200, 512);

    EXPECT_TRUE(matchesKernel(TRITON_RMSNORM, largest.context(), bn4096, largestBound));
    EXPECT_FALSE(matchesKernel(TRITON_RMSNORM, tooLarge.context(), bn4096, tooLargeBound));
}

// ---------------------------------------------------------------------------
// dispatch: the seams that need no device
// ---------------------------------------------------------------------------

TEST(TestTritonRmsNormDispatch, NeedsNoWorkspace)
{
    const GraphFixture fixture(graphWith([](auto& o) { o.xDims = {8, 1000}; }));
    const auto& handler = dispatchHandler(TRITON_RMSNORM);
    const auto bound = bindingsFor(fixture.context());

    // The reduction is entirely in-block (one program per row, LDS only).
    for(const auto& kernel : shippedLadder("bf16"))
    {
        EXPECT_EQ(handler.workspaceBytes(fixture.context(), bound, kernel), 0U) << kernel.name;
    }
}

TEST(TestTritonRmsNormDispatch, RefusesToPrepareWithoutTheMatcherSBindings)
{
    const GraphFixture fixture(graphWith([](auto& o) { o.xDims = {8, 1000}; }));
    const auto& handler = dispatchHandler(TRITON_RMSNORM);

    // The binding lookup throws before any archive is opened.
    EXPECT_THROW(
        handler.prepare(fixture.context(), BoundTokens{}, makeTritonKernel("bf16", 1024, 400)),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_TRITON
