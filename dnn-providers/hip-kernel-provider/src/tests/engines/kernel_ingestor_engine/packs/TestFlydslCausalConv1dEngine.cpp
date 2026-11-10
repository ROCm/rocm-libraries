// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <array>
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

#include "engines/kernel_ingestor_engine/packs/FlydslCausalConv1dCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslCausalConv1dTestGraphs.hpp"

/**
 * @file TestFlydslCausalConv1dEngine.cpp
 * @brief hipkernel:flydsl_causal_conv1d's matchers, scorer and problem resolution, host-only:
 *        what the pack accepts, what it declines, and the problem a launch would get. The
 *        launches themselves are TestFlydslCausalConv1dObjects.cpp.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using data_objects::CausalConv1dActivation;
using data_objects::DataType;

bool matches(const CausalConv1dGraphSpec& spec)
{
    const GraphFixture fixture(buildFlydslCausalConv1dGraph(spec));
    return matchesGraph(FLYDSL_CAUSAL_CONV1D, fixture.context()).has_value();
}

std::optional<flydsl_causal_conv1d::CausalConv1dProblem>
    problemFor(const CausalConv1dGraphSpec& spec)
{
    const GraphFixture fixture(buildFlydslCausalConv1dGraph(spec));
    const auto context = fixture.context();
    const auto& node = context.graph.getNodeWrapper(0);
    return flydsl_causal_conv1d::resolveCausalConv1dProblem(
        context, node.attributesAs<data_objects::CausalConv1dAttributes>());
}

KernelDefinition makeConvKernel(const std::string& dtype,
                                int64_t width = 4,
                                const std::string& activation = "silu",
                                int64_t lanes = 4,
                                int64_t priority = 100)
{
    KernelDefinition kernel;
    kernel.name = "causal_conv1d_" + dtype;
    kernel.priority = priority;
    kernel.metadata = {{"dtype", dtype},
                       {"width", width},
                       {"activation", activation},
                       {"lanes", lanes},
                       {"block_threads", int64_t{64}},
                       {"tile_tokens", int64_t{16}}};
    return kernel;
}

struct SpecCase
{
    std::string name;
    CausalConv1dGraphSpec spec;
};

std::string specCaseName(const ::testing::TestParamInfo<SpecCase>& info)
{
    return info.param.name;
}

CausalConv1dGraphSpec with(void (*edit)(CausalConv1dGraphSpec&))
{
    CausalConv1dGraphSpec spec;
    edit(spec);
    return spec;
}

// ---------------------------------------------------------------------------
// graph_match: accepted
// ---------------------------------------------------------------------------

class TestFlydslCausalConv1dGraphMatcherAcceptance : public ::testing::TestWithParam<SpecCase>
{
};

TEST_P(TestFlydslCausalConv1dGraphMatcherAcceptance, Accepts)
{
    EXPECT_TRUE(matches(GetParam().spec));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestFlydslCausalConv1dGraphMatcherAcceptance,
    ::testing::Values(
        SpecCase{"QwenDecodeStep", CausalConv1dGraphSpec{}},
        SpecCase{"QwenPrefill", with([](CausalConv1dGraphSpec& s) {
                     s.tokens = 8192;
                     s.sequences = 1;
                     s.channels = 10240;
                 })},
        SpecCase{"Identity", with([](CausalConv1dGraphSpec& s) {
                     s.activation = CausalConv1dActivation::IDENTITY;
                 })},
        SpecCase{"Bias", with([](CausalConv1dGraphSpec& s) { s.bias = true; })},
        SpecCase{"NoStates", with([](CausalConv1dGraphSpec& s) {
                     s.initialState = false;
                     s.finalState = false;
                 })},
        SpecCase{"FinalStateFromZeros",
                 with([](CausalConv1dGraphSpec& s) { s.initialState = false; })},
        SpecCase{"InitialStateOnly", with([](CausalConv1dGraphSpec& s) { s.finalState = false; })},
        SpecCase{"Pool", with([](CausalConv1dGraphSpec& s) { s.poolRows = 64; })},
        SpecCase{"PoolWithoutFinalState", with([](CausalConv1dGraphSpec& s) {
                     s.poolRows = 64;
                     s.finalState = false;
                 })},
        SpecCase{"Int64Offsets",
                 with([](CausalConv1dGraphSpec& s) { s.cuSeqlensDataType = DataType::INT64; })},
        SpecCase{"Int64Indices", with([](CausalConv1dGraphSpec& s) {
                     s.poolRows = 9;
                     s.stateIndicesDataType = DataType::INT64;
                 })},
        SpecCase{"Half", with([](CausalConv1dGraphSpec& s) { s.dataType = DataType::HALF; })},
        SpecCase{"Float", with([](CausalConv1dGraphSpec& s) { s.dataType = DataType::FLOAT; })},
        SpecCase{"ComputeTypeFloat",
                 with([](CausalConv1dGraphSpec& s) { s.computeDataType = DataType::FLOAT; })},
        SpecCase{"WidthTwo", with([](CausalConv1dGraphSpec& s) { s.width = 2; })},
        SpecCase{"OddChannels", with([](CausalConv1dGraphSpec& s) { s.channels = 37; })},
        SpecCase{"PaddedTokenRows", with([](CausalConv1dGraphSpec& s) {
                     s.xStrides = std::vector<int64_t>{int64_t{3} * 2048, 1};
                     s.yStrides = std::vector<int64_t>{int64_t{2} * 2048, 1};
                 })},
        SpecCase{"TransposedWeight", with([](CausalConv1dGraphSpec& s) {
                     s.weightStrides = std::vector<int64_t>{1, 2048};
                 })},
        SpecCase{"ChannelsInnermostStates", with([](CausalConv1dGraphSpec& s) {
                     s.initialStrides = std::vector<int64_t>{int64_t{3} * 2048, 1, 2048};
                     s.finalStrides = std::vector<int64_t>{int64_t{3} * 2048, 1, 2048};
                 })}),
    specCaseName);

// ---------------------------------------------------------------------------
// graph_match: declined
// ---------------------------------------------------------------------------

class TestFlydslCausalConv1dGraphMatcherRefusal : public ::testing::TestWithParam<SpecCase>
{
};

TEST_P(TestFlydslCausalConv1dGraphMatcherRefusal, Refuses)
{
    EXPECT_FALSE(matches(GetParam().spec));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestFlydslCausalConv1dGraphMatcherRefusal,
    ::testing::Values(
        SpecCase{"UnsupportedDtype",
                 with([](CausalConv1dGraphSpec& s) { s.dataType = DataType::DOUBLE; })},
        SpecCase{"WeightOfAnotherDtype",
                 with([](CausalConv1dGraphSpec& s) { s.weightDataType = DataType::FLOAT; })},
        SpecCase{"StateOfAnotherDtype",
                 with([](CausalConv1dGraphSpec& s) { s.stateDataType = DataType::FLOAT; })},
        SpecCase{"YOfAnotherDtype",
                 with([](CausalConv1dGraphSpec& s) { s.yDataType = DataType::HALF; })},
        SpecCase{"YOfAnotherShape",
                 with([](CausalConv1dGraphSpec& s) { s.yDims = std::vector<int64_t>{4, 1024}; })},
        SpecCase{"OffsetsNotIntegers",
                 with([](CausalConv1dGraphSpec& s) { s.cuSeqlensDataType = DataType::FLOAT; })},
        SpecCase{"IndicesNotIntegers", with([](CausalConv1dGraphSpec& s) {
                     s.poolRows = 9;
                     s.stateIndicesDataType = DataType::HALF;
                 })},
        SpecCase{"IndicesWithoutInitialState", with([](CausalConv1dGraphSpec& s) {
                     s.poolRows = 9;
                     s.initialState = false;
                 })},
        SpecCase{"FinalPoolOfAnotherSize", with([](CausalConv1dGraphSpec& s) {
                     s.poolRows = 9;
                     s.finalDims = std::vector<int64_t>{10, 2048, 3};
                 })},
        SpecCase{"StateOfAnotherWidth", with([](CausalConv1dGraphSpec& s) {
                     s.finalDims = std::vector<int64_t>{4, 2048, 2};
                 })},
        SpecCase{"StridedChannels", with([](CausalConv1dGraphSpec& s) {
                     s.xStrides = std::vector<int64_t>{4096, 2};
                 })},
        SpecCase{"OverlappingOutputRows", with([](CausalConv1dGraphSpec& s) {
                     s.yStrides = std::vector<int64_t>{1024, 1};
                 })},
        SpecCase{"BroadcastFinalState", with([](CausalConv1dGraphSpec& s) {
                     s.finalStrides = std::vector<int64_t>{0, 3, 1};
                 })},
        SpecCase{"StridedOffsets", with([](CausalConv1dGraphSpec& s) {
                     s.cuSeqlensStrides = std::vector<int64_t>{2};
                 })},
        SpecCase{"UnalignedOffsets",
                 with([](CausalConv1dGraphSpec& s) { s.cuSeqlensAlignment = 2; })},
        SpecCase{"NegativeStride", with([](CausalConv1dGraphSpec& s) {
                     s.weightStrides = std::vector<int64_t>{-4, 1};
                 })},
        SpecCase{"TooManyTokens", with([](CausalConv1dGraphSpec& s) {
                     s.tokens = int64_t{1} << 31;
                     s.sequences = 1;
                     s.channels = 4;
                     s.initialState = false;
                     s.finalState = false;
                 })},
        SpecCase{"ComputeTypeDouble",
                 with([](CausalConv1dGraphSpec& s) { s.computeDataType = DataType::DOUBLE; })},
        SpecCase{"VirtualX", with([](CausalConv1dGraphSpec& s) {
                     s.virtualTensors = {FLYDSL_CAUSAL_CONV1D_X_UID};
                 })},
        SpecCase{"VirtualFinalState", with([](CausalConv1dGraphSpec& s) {
                     s.virtualTensors = {FLYDSL_CAUSAL_CONV1D_FINAL_STATE_UID};
                 })},
        SpecCase{"TwoNodes", with([](CausalConv1dGraphSpec& s) { s.secondNode = true; })},
        SpecCase{"OverrideShape",
                 with([](CausalConv1dGraphSpec& s) { s.overrideShapeEnabled = true; })}),
    specCaseName);

// ---------------------------------------------------------------------------
// the problem a launch gets
// ---------------------------------------------------------------------------

TEST(TestFlydslCausalConv1dProblem, ReadsAPackedDecodeStep)
{
    const auto problem = problemFor(CausalConv1dGraphSpec{});
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->tokens, 4);
    EXPECT_EQ(problem->channels, 2048);
    EXPECT_EQ(problem->width, 4);
    EXPECT_EQ(problem->sequences, 4);
    EXPECT_EQ(problem->stateRows, 4);
    EXPECT_TRUE(problem->silu);
    EXPECT_EQ(problem->xStride, 2048);
    EXPECT_EQ(problem->yStride, 2048);
    EXPECT_EQ(problem->weightStrideC, 4);
    EXPECT_EQ(problem->weightStrideK, 1);
    EXPECT_EQ(problem->initialStrides, (std::array<int64_t, 3>{int64_t{2048} * 3, 3, 1}));
    EXPECT_EQ(problem->finalStrides, problem->initialStrides);
    EXPECT_EQ(problem->cuSeqlensStep, 1);
    EXPECT_EQ(problem->flags,
              flydsl_causal_conv1d::FLAG_INITIAL_STATE | flydsl_causal_conv1d::FLAG_FINAL_STATE);
}

TEST(TestFlydslCausalConv1dProblem, ReadsAPoolWithInt64IndicesAndOffsets)
{
    CausalConv1dGraphSpec spec;
    spec.poolRows = 9;
    spec.bias = true;
    spec.cuSeqlensDataType = DataType::INT64;
    spec.stateIndicesDataType = DataType::INT64;
    const auto problem = problemFor(spec);
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->stateRows, 9);
    EXPECT_EQ(problem->cuSeqlensStep, 2);
    EXPECT_EQ(problem->stateIndicesStep, 2);
    EXPECT_EQ(problem->flags,
              flydsl_causal_conv1d::FLAG_BIAS | flydsl_causal_conv1d::FLAG_INITIAL_STATE
                  | flydsl_causal_conv1d::FLAG_FINAL_STATE
                  | flydsl_causal_conv1d::FLAG_STATE_INDICES);
}

TEST(TestFlydslCausalConv1dProblem, NormalisesTheStrideOfAUnitAxis)
{
    // One token: x's token stride is never stepped, whatever the graph says.
    CausalConv1dGraphSpec spec;
    spec.tokens = 1;
    spec.sequences = 1;
    spec.xStrides = std::vector<int64_t>{12345, 1};
    const auto problem = problemFor(spec);
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->xStride, 2048);
}

TEST(TestFlydslCausalConv1dProblem, SizesTheGridForSequencesAndTokenTiles)
{
    CausalConv1dGraphSpec spec;
    spec.tokens = 100;
    spec.sequences = 3;
    spec.channels = 10240;
    const auto problem = problemFor(spec);
    ASSERT_TRUE(problem.has_value());
    const auto grid = flydsl_causal_conv1d::launchGrid(*problem, 64, 4, 16);
    ASSERT_TRUE(grid.has_value());
    EXPECT_EQ(grid->first, 3 + 7);
    EXPECT_EQ(grid->second, 40);
}

// ---------------------------------------------------------------------------
// kernel_match, score, binding
// ---------------------------------------------------------------------------

TEST(TestFlydslCausalConv1dKernelMatcher, PairsTheGraphDtypeWithTheObjectBuiltForIt)
{
    for(const auto& [type, name] : {std::pair{DataType::BFLOAT16, "bf16"},
                                    std::pair{DataType::HALF, "f16"},
                                    std::pair{DataType::FLOAT, "f32"}})
    {
        CausalConv1dGraphSpec spec;
        spec.dataType = type;
        const GraphFixture fixture(buildFlydslCausalConv1dGraph(spec));
        const auto bound = matchesGraph(FLYDSL_CAUSAL_CONV1D, fixture.context());
        ASSERT_TRUE(bound.has_value()) << name;
        for(const auto* candidate : {"bf16", "f16", "f32"})
        {
            EXPECT_EQ(
                matchesKernel(
                    FLYDSL_CAUSAL_CONV1D, fixture.context(), makeConvKernel(candidate), *bound),
                std::string(candidate) == name)
                << name << " graph against a " << candidate << " object";
        }
    }
}

TEST(TestFlydslCausalConv1dKernelMatcher, NeedsTheGraphsWidthAndActivation)
{
    const GraphFixture fixture(buildFlydslCausalConv1dGraph());
    const auto bound = matchesGraph(FLYDSL_CAUSAL_CONV1D, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_TRUE(
        matchesKernel(FLYDSL_CAUSAL_CONV1D, fixture.context(), makeConvKernel("bf16"), *bound));
    EXPECT_FALSE(
        matchesKernel(FLYDSL_CAUSAL_CONV1D, fixture.context(), makeConvKernel("bf16", 3), *bound));
    EXPECT_FALSE(matchesKernel(
        FLYDSL_CAUSAL_CONV1D, fixture.context(), makeConvKernel("bf16", 4, "identity"), *bound));
}

TEST(TestFlydslCausalConv1dKernelMatcher, NeedsChannelsAMultipleOfTheLanes)
{
    CausalConv1dGraphSpec spec;
    spec.channels = 37;
    const GraphFixture fixture(buildFlydslCausalConv1dGraph(spec));
    const auto bound = matchesGraph(FLYDSL_CAUSAL_CONV1D, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_FALSE(matchesKernel(
        FLYDSL_CAUSAL_CONV1D, fixture.context(), makeConvKernel("bf16", 4, "silu", 4), *bound));
    EXPECT_TRUE(matchesKernel(
        FLYDSL_CAUSAL_CONV1D, fixture.context(), makeConvKernel("bf16", 4, "silu", 1), *bound));
}

TEST(TestFlydslCausalConv1dKernelMatcher, DeclinesAGridPastTheDevicesLimits)
{
    // 65536 channel blocks of 64 one-lane threads.
    CausalConv1dGraphSpec spec;
    spec.channels = int64_t{65536} * 64;
    spec.initialState = false;
    spec.finalState = false;
    const GraphFixture fixture(buildFlydslCausalConv1dGraph(spec));
    const auto bound = matchesGraph(FLYDSL_CAUSAL_CONV1D, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_FALSE(matchesKernel(
        FLYDSL_CAUSAL_CONV1D, fixture.context(), makeConvKernel("bf16", 4, "silu", 1), *bound));
}

TEST(TestFlydslCausalConv1dScore, IsTheDescriptorPriority)
{
    const GraphFixture fixture(buildFlydslCausalConv1dGraph());
    const auto bound = matchesGraph(FLYDSL_CAUSAL_CONV1D, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(scoreKernel(FLYDSL_CAUSAL_CONV1D,
                          fixture.context(),
                          makeConvKernel("bf16", 4, "silu", 4, 37),
                          *bound),
              37.0);
}

TEST(TestFlydslCausalConv1dBinding, BindsEveryOperandItWasGiven)
{
    CausalConv1dGraphSpec spec;
    spec.bias = true;
    spec.poolRows = 9;
    const GraphFixture fixture(buildFlydslCausalConv1dGraph(spec));
    const auto bound = matchesGraph(FLYDSL_CAUSAL_CONV1D, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_CAUSAL_CONV1D.inputAToken), FLYDSL_CAUSAL_CONV1D_X_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_CAUSAL_CONV1D.inputBToken),
              FLYDSL_CAUSAL_CONV1D_WEIGHT_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_CAUSAL_CONV1D.outputToken), FLYDSL_CAUSAL_CONV1D_Y_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_CAUSAL_CONV1D_CU_SEQLENS_TOKEN),
              FLYDSL_CAUSAL_CONV1D_CU_SEQLENS_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_CAUSAL_CONV1D_BIAS_TOKEN),
              FLYDSL_CAUSAL_CONV1D_BIAS_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_CAUSAL_CONV1D_INITIAL_STATE_TOKEN),
              FLYDSL_CAUSAL_CONV1D_INITIAL_STATE_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_CAUSAL_CONV1D_STATE_INDICES_TOKEN),
              FLYDSL_CAUSAL_CONV1D_STATE_INDICES_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_CAUSAL_CONV1D_FINAL_STATE_TOKEN),
              FLYDSL_CAUSAL_CONV1D_FINAL_STATE_UID);
}

TEST(TestFlydslCausalConv1dBinding, LeavesUnboundOperandsUnbound)
{
    CausalConv1dGraphSpec spec;
    spec.initialState = false;
    spec.finalState = false;
    const GraphFixture fixture(buildFlydslCausalConv1dGraph(spec));
    const auto bound = matchesGraph(FLYDSL_CAUSAL_CONV1D, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_FALSE(tryGetBoundInt(*bound, FLYDSL_CAUSAL_CONV1D_BIAS_TOKEN).has_value());
    EXPECT_FALSE(tryGetBoundInt(*bound, FLYDSL_CAUSAL_CONV1D_INITIAL_STATE_TOKEN).has_value());
    EXPECT_FALSE(tryGetBoundInt(*bound, FLYDSL_CAUSAL_CONV1D_STATE_INDICES_TOKEN).has_value());
    EXPECT_FALSE(tryGetBoundInt(*bound, FLYDSL_CAUSAL_CONV1D_FINAL_STATE_TOKEN).has_value());
}

TEST(TestFlydslCausalConv1dDispatch, NeedsNoWorkspace)
{
    const GraphFixture fixture(buildFlydslCausalConv1dGraph());
    const auto bound = matchesGraph(FLYDSL_CAUSAL_CONV1D, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(dispatchHandler(FLYDSL_CAUSAL_CONV1D)
                  .workspaceBytes(fixture.context(), *bound, makeConvKernel("bf16")),
              0U);
}

TEST(TestFlydslCausalConv1dDispatch, RefusesToPrepareWithoutTheMatchersBindings)
{
    const GraphFixture fixture(buildFlydslCausalConv1dGraph());
    EXPECT_THROW(
        static_cast<void>(dispatchHandler(FLYDSL_CAUSAL_CONV1D)
                              .prepare(fixture.context(), BoundTokens{}, makeConvKernel("bf16"))),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
