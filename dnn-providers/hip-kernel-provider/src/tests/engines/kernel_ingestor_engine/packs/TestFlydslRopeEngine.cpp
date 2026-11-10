// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslRopeCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslRopeTestGraphs.hpp"

/**
 * @file TestFlydslRopeEngine.cpp
 * @brief hipkernel:flydsl_rope's matchers, scorer and problem resolution, host-only: what
 *        the pack accepts, what it declines, and the rank-4 problem a launch would get.
 *        The launches themselves are TestFlydslRopeObjects.cpp.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using data_objects::DataType;

bool matches(const RopeGraphSpec& spec)
{
    const GraphFixture fixture(buildFlydslRopeGraph(spec));
    return matchesGraph(FLYDSL_ROPE, fixture.context()).has_value();
}

/// The problem the pack's shared reading resolves for @p spec's graph.
std::optional<flydsl_rope::RopeProblem> problemFor(const RopeGraphSpec& spec)
{
    const GraphFixture fixture(buildFlydslRopeGraph(spec));
    const auto context = fixture.context();
    const auto& node = context.graph.getNodeWrapper(0);
    const auto& attributes = node.attributesAs<data_objects::RopeAttributes>();
    const auto& tensors = context.graph.getTensorMap();
    return flydsl_rope::resolveRopeProblem(attributes,
                                           *tensors.at(FLYDSL_ROPE_X_UID),
                                           *tensors.at(FLYDSL_ROPE_FREQS_UID),
                                           *tensors.at(FLYDSL_ROPE_Y_UID));
}

KernelDefinition makeRopeKernel(const std::string& dtype, int64_t priority = 10)
{
    KernelDefinition kernel;
    kernel.name = "rope_" + dtype + "_bt64";
    kernel.priority = priority;
    kernel.metadata = {{"dtype", dtype}, {"block_threads", int64_t{64}}};
    return kernel;
}

// ---------------------------------------------------------------------------
// graph_match: accepted
// ---------------------------------------------------------------------------

struct SpecCase
{
    std::string name;
    RopeGraphSpec spec;
};

std::string specCaseName(const ::testing::TestParamInfo<SpecCase>& info)
{
    return info.param.name;
}

RopeGraphSpec with(void (*edit)(RopeGraphSpec&))
{
    RopeGraphSpec spec;
    edit(spec);
    return spec;
}

class TestFlydslRopeGraphMatcherAcceptance : public ::testing::TestWithParam<SpecCase>
{
};

TEST_P(TestFlydslRopeGraphMatcherAcceptance, Accepts)
{
    EXPECT_TRUE(matches(GetParam().spec));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestFlydslRopeGraphMatcherAcceptance,
    ::testing::Values(
        // Qwen3's attention RoPE: BSHD, D = 256, the leading 64 rotated.
        SpecCase{"QwenPartialRotaryBshd", RopeGraphSpec{}},
        SpecCase{"QwenPerBatchFreqs", with([](RopeGraphSpec& s) {
                     s.xDims = {2, 16, 7, 256};
                     s.freqsDims = {2, 1, 7, 64};
                 })},
        SpecCase{"CudnnTrailingDefaultOffset",
                 with([](RopeGraphSpec& s) { s.ropeOffset = std::nullopt; })},
        SpecCase{"FullRotationRopeDimZero", with([](RopeGraphSpec& s) {
                     s.xDims = {1, 4, 7, 128};
                     s.freqsDims = {1, 1, 7, 128};
                     s.ropeDim = 0;
                 })},
        SpecCase{"MiddleSegment", with([](RopeGraphSpec& s) {
                     s.ropeDim = 32;
                     s.ropeOffset = 100;
                     s.freqsDims = {1, 1, 7, 32};
                 })},
        SpecCase{"Half", with([](RopeGraphSpec& s) { s.dataType = DataType::HALF; })},
        SpecCase{"Float", with([](RopeGraphSpec& s) { s.dataType = DataType::FLOAT; })},
        SpecCase{"ComputeTypeFloat",
                 with([](RopeGraphSpec& s) { s.computeDataType = DataType::FLOAT; })},
        SpecCase{"ComputeTypeUnset",
                 with([](RopeGraphSpec& s) { s.computeDataType = DataType::UNSET; })},
        SpecCase{"PackedBhsd",
                 with([](RopeGraphSpec& s) { s.xStrides = flydslRopePackedStrides(s.xDims); })},
        SpecCase{"YInADifferentLayout",
                 with([](RopeGraphSpec& s) { s.yStrides = flydslRopePackedStrides(s.xDims); })},
        SpecCase{"Rank3", with([](RopeGraphSpec& s) {
                     s.xDims = {16, 7, 128};
                     s.freqsDims = {1, 7, 64};
                 })},
        SpecCase{"Rank2", with([](RopeGraphSpec& s) {
                     s.xDims = {7, 128};
                     s.freqsDims = {7, 64};
                 })},
        SpecCase{"BroadcastFreqsWithArbitraryUnitStrides", with([](RopeGraphSpec& s) {
                     s.freqsStrides = std::vector<int64_t>{12345, 0, 64, 1};
                 })},
        SpecCase{"StridedHeadDim", with([](RopeGraphSpec& s) {
                     s.xDims = {1, 2, 3, 8};
                     s.freqsDims = {1, 1, 3, 4};
                     s.ropeDim = 4;
                     s.xStrides = std::vector<int64_t>{48, 1, 2, 6};
                     s.yStrides = std::vector<int64_t>{48, 1, 2, 6};
                 })}),
    specCaseName);

// ---------------------------------------------------------------------------
// graph_match: declined
// ---------------------------------------------------------------------------

class TestFlydslRopeGraphMatcherRefusal : public ::testing::TestWithParam<SpecCase>
{
};

TEST_P(TestFlydslRopeGraphMatcherRefusal, Refuses)
{
    EXPECT_FALSE(matches(GetParam().spec));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestFlydslRopeGraphMatcherRefusal,
    ::testing::Values(
        SpecCase{"OddRopeDim", with([](RopeGraphSpec& s) {
                     s.ropeDim = 63;
                     s.freqsDims = {1, 1, 7, 63};
                 })},
        SpecCase{"NegativeRopeDim", with([](RopeGraphSpec& s) { s.ropeDim = -64; })},
        SpecCase{"RopeDimPastHeadDim", with([](RopeGraphSpec& s) {
                     s.ropeDim = 258;
                     s.freqsDims = {1, 1, 7, 258};
                 })},
        SpecCase{"OddHeadDimFullRotation", with([](RopeGraphSpec& s) {
                     s.xDims = {1, 2, 7, 129};
                     s.freqsDims = {1, 1, 7, 129};
                     s.ropeDim = 0;
                 })},
        SpecCase{"NegativeOffset", with([](RopeGraphSpec& s) { s.ropeOffset = -1; })},
        SpecCase{"SegmentPastHeadDim", with([](RopeGraphSpec& s) { s.ropeOffset = 193; })},
        SpecCase{"FreqsLastDimNotRopeDim",
                 with([](RopeGraphSpec& s) { s.freqsDims = {1, 1, 7, 32}; })},
        SpecCase{"FreqsNotBroadcastable",
                 with([](RopeGraphSpec& s) { s.freqsDims = {1, 2, 7, 64}; })},
        SpecCase{"FreqsOfAnotherRank", with([](RopeGraphSpec& s) { s.freqsDims = {7, 64}; })},
        SpecCase{"FreqsNotFloat",
                 with([](RopeGraphSpec& s) { s.freqsDataType = DataType::BFLOAT16; })},
        SpecCase{"YOfAnotherDtype", with([](RopeGraphSpec& s) { s.yDataType = DataType::FLOAT; })},
        SpecCase{"YOfAnotherShape",
                 with([](RopeGraphSpec& s) { s.yDims = std::vector<int64_t>{1, 16, 8, 256}; })},
        SpecCase{"UnsupportedDtype", with([](RopeGraphSpec& s) { s.dataType = DataType::DOUBLE; })},
        SpecCase{"ComputeTypeDouble",
                 with([](RopeGraphSpec& s) { s.computeDataType = DataType::DOUBLE; })},
        SpecCase{"Rank5", with([](RopeGraphSpec& s) {
                     s.xDims = {1, 1, 16, 7, 256};
                     s.freqsDims = {1, 1, 1, 7, 64};
                 })},
        SpecCase{"Rank1", with([](RopeGraphSpec& s) {
                     s.xDims = {256};
                     s.freqsDims = {64};
                 })},
        SpecCase{"NegativeStride", with([](RopeGraphSpec& s) {
                     s.xStrides = std::vector<int64_t>{28672, 256, 4096, -1};
                 })},
        SpecCase{"YRowsWrittenToOnePlace", with([](RopeGraphSpec& s) {
                     s.yStrides = std::vector<int64_t>{28672, 256, 0, 1};
                 })},
        SpecCase{"TooManyHeadsForTheGrid", with([](RopeGraphSpec& s) {
                     s.xDims = {1, 65536, 1, 64};
                     s.freqsDims = {1, 1, 1, 64};
                     s.xStrides = flydslRopePackedStrides(s.xDims);
                 })},
        SpecCase{"VirtualX",
                 with([](RopeGraphSpec& s) { s.virtualTensors = {FLYDSL_ROPE_X_UID}; })},
        SpecCase{"VirtualY",
                 with([](RopeGraphSpec& s) { s.virtualTensors = {FLYDSL_ROPE_Y_UID}; })},
        SpecCase{"PassByValueFreqs",
                 with([](RopeGraphSpec& s) { s.passByValueTensors = {FLYDSL_ROPE_FREQS_UID}; })},
        SpecCase{"TwoNodes", with([](RopeGraphSpec& s) { s.secondNode = true; })},
        SpecCase{"OverrideShape", with([](RopeGraphSpec& s) { s.overrideShapeEnabled = true; })}),
    specCaseName);

// ---------------------------------------------------------------------------
// the problem a launch gets
// ---------------------------------------------------------------------------

TEST(TestFlydslRopeProblem, ReadsQwenBshdStridesAndZeroesBroadcastFreqsAxes)
{
    const auto problem = problemFor(RopeGraphSpec{});
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->dims, (std::array<int64_t, 4>{1, 16, 7, 256}));
    // B has extent 1, so its stride is never stepped and is passed as 0.
    EXPECT_EQ(problem->xStrides, (std::array<int64_t, 4>{0, 256, 4096, 1}));
    EXPECT_EQ(problem->yStrides, problem->xStrides);
    EXPECT_EQ(problem->freqsStrides, (std::array<int64_t, 4>{0, 0, 64, 1}));
    EXPECT_EQ(problem->headDim, 256);
    EXPECT_EQ(problem->ropeDim, 64);
    EXPECT_EQ(problem->ropeOffset, 0);
}

TEST(TestFlydslRopeProblem, DefaultsTheOffsetToTheTrailingSegmentAsCudnnDoes)
{
    RopeGraphSpec spec;
    spec.ropeOffset = std::nullopt;
    const auto problem = problemFor(spec);
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->ropeOffset, 256 - 64);
}

TEST(TestFlydslRopeProblem, ReadsRopeDimZeroAsTheWholeHeadDim)
{
    RopeGraphSpec spec;
    spec.xDims = {1, 4, 7, 128};
    spec.freqsDims = {1, 1, 7, 128};
    spec.ropeDim = 0;
    spec.ropeOffset = std::nullopt;
    const auto problem = problemFor(spec);
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->ropeDim, 128);
    EXPECT_EQ(problem->ropeOffset, 0);
}

TEST(TestFlydslRopeProblem, PadsLowerRanksWithLeadingUnitAxes)
{
    RopeGraphSpec spec;
    spec.xDims = {7, 128};
    spec.freqsDims = {7, 64};
    const auto problem = problemFor(spec);
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->dims, (std::array<int64_t, 4>{1, 1, 7, 128}));
    EXPECT_EQ(problem->xStrides, (std::array<int64_t, 4>{0, 0, 128, 1}));
    EXPECT_EQ(problem->freqsStrides, (std::array<int64_t, 4>{0, 0, 64, 1}));
}

// ---------------------------------------------------------------------------
// kernel_match, score, binding
// ---------------------------------------------------------------------------

TEST(TestFlydslRopeKernelMatcher, PairsTheGraphDtypeWithTheObjectBuiltForIt)
{
    for(const auto& [type, name] : {std::pair{DataType::BFLOAT16, "bf16"},
                                    std::pair{DataType::HALF, "f16"},
                                    std::pair{DataType::FLOAT, "f32"}})
    {
        RopeGraphSpec spec;
        spec.dataType = type;
        const GraphFixture fixture(buildFlydslRopeGraph(spec));
        const auto bound = matchesGraph(FLYDSL_ROPE, fixture.context());
        ASSERT_TRUE(bound.has_value()) << name;
        for(const auto* candidate : {"bf16", "f16", "f32"})
        {
            EXPECT_EQ(
                matchesKernel(FLYDSL_ROPE, fixture.context(), makeRopeKernel(candidate), *bound),
                std::string(candidate) == name)
                << name << " graph against a " << candidate << " object";
        }
    }
}

TEST(TestFlydslRopeScore, IsTheDescriptorPriority)
{
    const GraphFixture fixture(buildFlydslRopeGraph());
    const auto bound = matchesGraph(FLYDSL_ROPE, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(scoreKernel(FLYDSL_ROPE, fixture.context(), makeRopeKernel("bf16", 37), *bound),
              37.0);
}

TEST(TestFlydslRopeBinding, BindsTheThreeOperands)
{
    const GraphFixture fixture(buildFlydslRopeGraph());
    const auto bound = matchesGraph(FLYDSL_ROPE, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_ROPE.inputAToken), FLYDSL_ROPE_X_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_ROPE.inputBToken), FLYDSL_ROPE_FREQS_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_ROPE.outputToken), FLYDSL_ROPE_Y_UID);
}

TEST(TestFlydslRopeDispatch, NeedsNoWorkspace)
{
    const GraphFixture fixture(buildFlydslRopeGraph());
    const auto bound = matchesGraph(FLYDSL_ROPE, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(dispatchHandler(FLYDSL_ROPE)
                  .workspaceBytes(fixture.context(), *bound, makeRopeKernel("bf16")),
              0U);
}

TEST(TestFlydslRopeDispatch, RefusesToPrepareWithoutTheMatchersBindings)
{
    const GraphFixture fixture(buildFlydslRopeGraph());
    EXPECT_THROW(
        static_cast<void>(dispatchHandler(FLYDSL_ROPE)
                              .prepare(fixture.context(), BoundTokens{}, makeRopeKernel("bf16"))),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
