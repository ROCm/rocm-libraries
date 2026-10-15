// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Adapted from dnn-providers/hip-kernel-provider/src/tests/engines/kernel_ingestor_engine/packs/TestFlydslRmsNormEngine.cpp at commit 03911535bd0 (FlyDSL provider)

// Redundant with the CMake gate on purpose: if this file lands on an ungated source list
// it compiles to an empty translation unit instead of a suite reading an unstaged shard.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_TRITON)

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
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
 * Kernels are built by hand with the KMD metadata fields. The kernel matcher's per-instance
 * boundaries (block width, grid, dtype, alignment) are generated from the staged kernels in
 * TestTritonRmsNormPacks.cpp, and the engine's own gate is in
 * TestTritonRmsNormEngineTier.cpp. Device cases are in TestTritonRmsNormDispatch.cpp.
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

/// A kernel carrying the metadata fields this pack's KMD declares; `alignment` is omitted
/// when @p alignment is nullopt. Source is left default-constructed because these cases
/// read metadata only.
KernelDefinition makeTritonKernel(const std::string& dtype,
                                  int64_t blockN = 8192,
                                  int64_t blockThreads = 256,
                                  std::optional<int64_t> alignment = 16)
{
    KernelDefinition kernel;
    kernel.name = "rmsnorm_" + dtype + "_bn" + std::to_string(blockN);
    kernel.metadata = {{std::string(TRITON_DTYPE_FIELD), dtype},
                       {std::string(TRITON_BLOCK_N_FIELD), blockN},
                       {std::string(TRITON_BLOCK_THREADS_FIELD), blockThreads},
                       {std::string(TRITON_SHARED_MEM_BYTES_FIELD), int64_t{2048}}};
    if(alignment.has_value())
    {
        kernel.metadata.emplace(std::string(TRITON_ALIGNMENT_FIELD), *alignment);
    }
    return kernel;
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
        // Alignment is the kernel matcher's call (against each kernel's own); whether the
        // engine serves a non-default alignment is the engine tier's.
        GraphCase{"GraphMatcherAdmitsAlignmentAboveSixteen",
                  [] {
                      return graphWith([](auto& o) {
                          o.xAlignment = 32;
                          o.scaleAlignment = 64;
                          o.yAlignment = 256;
                      });
                  }},
        // The int32 bound on rows and columns admits the bound itself.
        GraphCase{"RowsAtInt32Max",
                  [] {
                      return graphWith(
                          [](auto& o) { o.xDims = {std::numeric_limits<int32_t>::max(), 1}; });
                  }},
        GraphCase{"ColumnsAtInt32Max",
                  [] {
                      return graphWith(
                          [](auto& o) { o.xDims = {1, std::numeric_limits<int32_t>::max()}; });
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
        // The grid and n_cols come from the graph; launch takes no execute-time shapes.
        GraphCase{"OverrideShapesEnabled",
                  [] { return graphWith([](auto& o) { o.overrideShapeEnabled = true; }); }},
        // Each kernel serves one complete single-node graph, which an empty one is not.
        GraphCase{"NoNode", [] { return graphWith([](auto& o) { o.noNode = true; }); }},
        // The kernels compute y = x * rsqrt(mean(x^2) + eps) * scale only.
        GraphCase{"WithABias", [] { return graphWith([](auto& o) { o.withBias = true; }); }},
        GraphCase{"SavingTheInverseRms",
                  [] { return graphWith([](auto& o) { o.withInvRms = true; }); }},
        // Training needs the saved statistic even if the graph names no tensor for it.
        GraphCase{
            "TrainingPhase",
            [] { return graphWith([](auto& o) { o.forwardPhase = NormFwdPhase::TRAINING; }); }},
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
        // Unit inner stride but padded rows: only the outermost stride is wrong.
        GraphCase{
            "XWithPaddedRows",
            [] { return graphWith([](auto& o) { o.xStrides = std::vector<int64_t>{8192, 1}; }); }},
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
        // The node names a uid the graph does not carry.
        GraphCase{"NoXTensor", [] { return graphWith([](auto& o) { o.omitX = true; }); }},
        GraphCase{"NoScaleTensor", [] { return graphWith([](auto& o) { o.omitScale = true; }); }},
        GraphCase{"NoYTensor", [] { return graphWith([](auto& o) { o.omitY = true; }); }},
        // Unvalidated graphs may omit a tensor's dims or strides.
        GraphCase{"XWithoutDims", [] { return graphWith([](auto& o) { o.xWithoutDims = true; }); }},
        GraphCase{"XWithoutStrides",
                  [] { return graphWith([](auto& o) { o.xWithoutStrides = true; }); }},
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
        // The same for y and scale: their dims describe a dense tensor that only the rank
        // check tells apart from what the strides describe.
        GraphCase{"YStridesOfADifferentRankThanDims",
                  [] {
                      return graphWith(
                          [](auto& o) { o.yStrides = std::vector<int64_t>{4096, 1, 1}; });
                  }},
        GraphCase{
            "ScaleStridesOfADifferentRankThanDims",
            [] { return graphWith([](auto& o) { o.scaleStrides = std::vector<int64_t>{1, 1}; }); }},
        // y is written over x's index space, densely.
        GraphCase{
            "YShapeDiffersFromX",
            [] { return graphWith([](auto& o) { o.yDims = std::vector<int64_t>{8, 2048}; }); }},
        // A trailing unit axis keeps every extent x has, so only the rank comparison refuses.
        GraphCase{
            "YOfAHigherRankThanX",
            [] { return graphWith([](auto& o) { o.yDims = std::vector<int64_t>{8, 4096, 1}; }); }},
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
// kernel_match: dtype, alignment and the tokens it reads
// ---------------------------------------------------------------------------

/// A graph read by the kernel matcher, and the kernel it is asked about. The bound tokens
/// are always the default graph's, so a case changes only what it names.
struct KernelCase
{
    std::string name;
    flatbuffers::FlatBufferBuilder (*buildGraph)();
    KernelDefinition (*buildKernel)();
};

std::string kernelCaseName(const ::testing::TestParamInfo<KernelCase>& info)
{
    return info.param.name;
}

/// The default graph's bindings: 8 rows of 4096.
BoundTokens defaultBindings()
{
    const GraphFixture fixture(buildTritonRmsNormGraph());
    return bindingsFor(TRITON_RMSNORM, fixture.context());
}

KernelDefinition bf16Kernel()
{
    return makeTritonKernel("bf16");
}

class TestTritonRmsNormKernelMatcherAcceptance : public ::testing::TestWithParam<KernelCase>
{
};

TEST_P(TestTritonRmsNormKernelMatcherAcceptance, Admits)
{
    const GraphFixture fixture(GetParam().buildGraph());
    EXPECT_TRUE(matchesKernel(
        TRITON_RMSNORM, fixture.context(), GetParam().buildKernel(), defaultBindings()));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestTritonRmsNormKernelMatcherAcceptance,
    ::testing::Values(
        // The frontend's default alignment, 16, is what a 16-aligned kernel needs.
        KernelCase{
            "DefaultGraphAtAlignment16", [] { return buildTritonRmsNormGraph(); }, bf16Kernel},
        // Any multiple of the kernel's alignment guarantees it.
        KernelCase{"XAtAlignment32",
                   [] { return graphWith([](auto& o) { o.xAlignment = 32; }); },
                   bf16Kernel},
        KernelCase{"ScaleAtAlignment32",
                   [] { return graphWith([](auto& o) { o.scaleAlignment = 32; }); },
                   bf16Kernel},
        KernelCase{"YAtAlignment32",
                   [] { return graphWith([](auto& o) { o.yAlignment = 32; }); },
                   bf16Kernel},
        KernelCase{"HalfGraphOnAHalfKernel",
                   [] { return graphWith([](auto& o) { o.dataType = DataType::HALF; }); },
                   [] { return makeTritonKernel("f16"); }}),
    kernelCaseName);

class TestTritonRmsNormKernelMatcherRefusal : public ::testing::TestWithParam<KernelCase>
{
};

TEST_P(TestTritonRmsNormKernelMatcherRefusal, Refuses)
{
    const GraphFixture fixture(GetParam().buildGraph());
    EXPECT_FALSE(matchesKernel(
        TRITON_RMSNORM, fixture.context(), GetParam().buildKernel(), defaultBindings()));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestTritonRmsNormKernelMatcherRefusal,
    ::testing::Values(
        // The kernel matcher re-reads the node and its operands, and must be total itself.
        KernelCase{
            "NoNode", [] { return graphWith([](auto& o) { o.noNode = true; }); }, bf16Kernel},
        KernelCase{
            "NoXTensor", [] { return graphWith([](auto& o) { o.omitX = true; }); }, bf16Kernel},
        KernelCase{"NoScaleTensor",
                   [] { return graphWith([](auto& o) { o.omitScale = true; }); },
                   bf16Kernel},
        KernelCase{
            "NoYTensor", [] { return graphWith([](auto& o) { o.omitY = true; }); }, bf16Kernel},
        // The kernel's base pointers carry its alignment as a divisibility hint, so declaring
        // less on any one of the three is declined.
        KernelCase{"XAtAlignment8",
                   [] { return graphWith([](auto& o) { o.xAlignment = 8; }); },
                   bf16Kernel},
        KernelCase{"ScaleAtAlignment8",
                   [] { return graphWith([](auto& o) { o.scaleAlignment = 8; }); },
                   bf16Kernel},
        KernelCase{"YAtAlignment8",
                   [] { return graphWith([](auto& o) { o.yAlignment = 8; }); },
                   bf16Kernel},
        // The backend checks only `ptr % alignment == 0`, so 24 admits an 8-mod-16 pointer:
        // the declaration must be a multiple of the kernel's. Zero means undeclared, and a
        // negative multiple guarantees nothing.
        KernelCase{"XAtAlignment24",
                   [] { return graphWith([](auto& o) { o.xAlignment = 24; }); },
                   bf16Kernel},
        KernelCase{"ScaleAtAlignment24",
                   [] { return graphWith([](auto& o) { o.scaleAlignment = 24; }); },
                   bf16Kernel},
        KernelCase{"YAtAlignment24",
                   [] { return graphWith([](auto& o) { o.yAlignment = 24; }); },
                   bf16Kernel},
        KernelCase{"XAtAlignment0",
                   [] { return graphWith([](auto& o) { o.xAlignment = 0; }); },
                   bf16Kernel},
        KernelCase{"ScaleAtAlignment0",
                   [] { return graphWith([](auto& o) { o.scaleAlignment = 0; }); },
                   bf16Kernel},
        KernelCase{"YAtAlignment0",
                   [] { return graphWith([](auto& o) { o.yAlignment = 0; }); },
                   bf16Kernel},
        KernelCase{"XAtNegativeAlignment",
                   [] { return graphWith([](auto& o) { o.xAlignment = -16; }); },
                   bf16Kernel},
        // The requirement is the kernel's own alignment, not a fixed 16.
        KernelCase{"OperandsAt16OnAKernelAlignedTo32",
                   [] { return buildTritonRmsNormGraph(); },
                   [] { return makeTritonKernel("bf16", 8192, 256, 32); }},
        // A kernel that states no usable alignment admits nothing.
        KernelCase{"AKernelWithoutAlignment",
                   [] { return buildTritonRmsNormGraph(); },
                   [] { return makeTritonKernel("bf16", 8192, 256, std::nullopt); }},
        KernelCase{"AKernelWithANonIntegerAlignment",
                   [] { return buildTritonRmsNormGraph(); },
                   [] {
                       auto kernel = makeTritonKernel("bf16", 8192, 256, std::nullopt);
                       kernel.metadata.emplace(std::string(TRITON_ALIGNMENT_FIELD),
                                               std::string("16"));
                       return kernel;
                   }},
        KernelCase{"AKernelAtAlignment0",
                   [] { return buildTritonRmsNormGraph(); },
                   [] { return makeTritonKernel("bf16", 8192, 256, 0); }},
        KernelCase{"AKernelAtNegativeAlignment",
                   [] { return buildTritonRmsNormGraph(); },
                   [] { return makeTritonKernel("bf16", 8192, 256, -16); }},
        // A kernel with no threads cannot be launched (and would divide by zero).
        KernelCase{"AKernelWithZeroBlockThreads",
                   [] { return buildTritonRmsNormGraph(); },
                   [] { return makeTritonKernel("bf16", 8192, 0); }}),
    kernelCaseName);

TEST(TestTritonRmsNormKernelMatcher, RefusesAFloatGraphOnEveryKernelDtype)
{
    // The graph matcher leaves dtypes to the kernels; FLOAT has no kernel spelling.
    const GraphFixture fixture(graphWith([](auto& o) { o.dataType = DataType::FLOAT; }));
    const auto bound = bindingsFor(TRITON_RMSNORM, fixture.context());

    for(const auto& dtype : {std::string("bf16"), std::string("f16")})
    {
        EXPECT_FALSE(
            matchesKernel(TRITON_RMSNORM, fixture.context(), makeTritonKernel(dtype), bound))
            << dtype;
    }
}

TEST(TestTritonRmsNormKernelMatcher, ReadsTheDescriptorDtypeVocabularyNotTheFlatbufferEnum)
{
    const GraphFixture fixture(buildTritonRmsNormGraph());

    // The flatbuffer spelling "BFLOAT16" is not accepted; descriptors say "bf16".
    EXPECT_FALSE(matchesKernel(
        TRITON_RMSNORM, fixture.context(), makeTritonKernel("BFLOAT16"), defaultBindings()));
}

TEST(TestTritonRmsNormKernelMatcher, RefusesWithoutTheColumnsToken)
{
    const GraphFixture fixture(buildTritonRmsNormGraph());
    auto bound = defaultBindings();
    bound.erase(TRITON_COLUMNS_TOKEN);

    EXPECT_FALSE(matchesKernel(TRITON_RMSNORM, fixture.context(), bf16Kernel(), bound));
}

TEST(TestTritonRmsNormKernelMatcher, RefusesWithoutTheRowsToken)
{
    const GraphFixture fixture(buildTritonRmsNormGraph());
    auto bound = defaultBindings();
    bound.erase(TRITON_ROWS_TOKEN);

    EXPECT_FALSE(matchesKernel(TRITON_RMSNORM, fixture.context(), bf16Kernel(), bound));
}

// ---------------------------------------------------------------------------
// dispatch: the seams that need no device
// ---------------------------------------------------------------------------

TEST(TestTritonRmsNormDispatch, NeedsNoWorkspace)
{
    const GraphFixture fixture(buildTritonRmsNormGraph());
    const auto& handler = dispatchHandler(TRITON_RMSNORM);

    // The reduction is entirely in-block (one program per row, LDS only).
    EXPECT_EQ(handler.workspaceBytes(fixture.context(), defaultBindings(), bf16Kernel()), 0U);
}

TEST(TestTritonRmsNormDispatch, RefusesToPrepareWithoutTheMatcherSBindings)
{
    const GraphFixture fixture(graphWith([](auto& o) { o.xDims = {8, 1000}; }));
    const auto& handler = dispatchHandler(TRITON_RMSNORM);

    // The binding lookup throws before any archive is opened.
    EXPECT_THROW(handler.prepare(fixture.context(), BoundTokens{}, makeTritonKernel("bf16", 1024)),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_TRITON
