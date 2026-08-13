// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/**
 * @file TestPointwiseAddMatchers.cpp
 * @brief The pack's two matcher shapes: what each accepts, and what each refuses.
 *
 * An under-specified refusal accepts a graph the kernel cannot serve, which is silently
 * wrong rather than merely a missed optimization.
 *
 * The matchers are reached through the registry rather than called directly: they are
 * internal to PointwiseAddNative.cpp, and the registry is the only door the descriptors
 * reach them by, so testing through it also proves the pack registered what its
 * descriptors name.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::MatchContext;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// Runs the graph matcher, discarding what it binds; binding is asserted separately below.
bool matches(const MatchContext& context)
{
    BoundTokens bound;
    return matchesGraph(POINTWISE_ADD, context, bound);
}

// ---------------------------------------------------------------------------
// Graph-scoped matcher: acceptances
// ---------------------------------------------------------------------------

TEST(TestPointwiseAddGraphMatcher, AcceptsASingleElementFloatAdd)
{
    const GraphFixture fixture(buildPointwiseGraph());

    EXPECT_TRUE(matches(fixture.context()));
}

TEST(TestPointwiseAddGraphMatcher, AcceptsAHalfPrecisionAdd)
{
    // The graph-level gate is dtype-agnostic; pinning the kernel's baked dtype is the
    // kernel-scoped matcher's job.
    const GraphFixture fixture(
        buildPointwiseGraph(data_objects::PointwiseMode::ADD, data_objects::DataType::HALF));

    EXPECT_TRUE(matches(fixture.context()));
}

TEST(TestPointwiseAddGraphMatcher, AcceptsTheUpperSupportedRank)
{
    const GraphFixture fixture(buildPointwiseGraph(
        data_objects::PointwiseMode::ADD, data_objects::DataType::FLOAT, {1, 1, 1, 1, 1}));

    EXPECT_TRUE(matches(fixture.context()));
}

// ---------------------------------------------------------------------------
// Graph-scoped matcher: refusals
// ---------------------------------------------------------------------------

/// One graph the matcher must refuse, plus a readable name for a failing run. The builder
/// is a plain function pointer, not std::function, since FlatBufferBuilder is move-only.
struct GraphMatcherRefusalCase
{
    std::string name;
    flatbuffers::FlatBufferBuilder (*buildGraph)();
};

class TestPointwiseAddGraphMatcherRefusal : public ::testing::TestWithParam<GraphMatcherRefusalCase>
{
};

TEST_P(TestPointwiseAddGraphMatcherRefusal, Refuses)
{
    const GraphFixture fixture(GetParam().buildGraph());

    EXPECT_FALSE(matches(fixture.context()));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestPointwiseAddGraphMatcherRefusal,
    ::testing::ValuesIn(std::vector<GraphMatcherRefusalCase>{
        {"AnotherPointwiseOperation",
         []() { return buildPointwiseGraph(data_objects::PointwiseMode::MUL); }},
        {"MultiElementTensors",
         // The kernel writes only element 0; a larger tensor leaves the rest unwritten.
         []() {
             return buildPointwiseGraph(
                 data_objects::PointwiseMode::ADD, data_objects::DataType::FLOAT, {1, 1, 2, 2});
         }},
        {"ARankTheDispatchPathCannotServe",
         // The provider derives layout from tensor rank and rejects anything below rank 4.
         []() {
             return buildPointwiseGraph(
                 data_objects::PointwiseMode::ADD, data_objects::DataType::FLOAT, {1});
         }},
        {"AStrideOrderTheDispatchPathCannotClassify",
         // The provider derives layout from tensor strides and only classifies NCHW or
         // NHWC; other orders reach plan build and fail there (RFC 0017 §8.6).
         []() {
             return buildPointwiseGraph(data_objects::PointwiseMode::ADD,
                                        data_objects::DataType::FLOAT,
                                        {1, 1, 1, 1},
                                        std::nullopt,
                                        /*binary=*/true,
                                        /*explicitStrides=*/std::vector<int64_t>{8, 2, 4, 1});
         }},
        {"AUnaryPointwise",
         []() {
             return buildPointwiseGraph(data_objects::PointwiseMode::ADD,
                                        data_objects::DataType::FLOAT,
                                        {1, 1, 1, 1},
                                        std::nullopt,
                                        /*binary=*/false);
         }},
        {"AMultiNodeGraph",
         // A prebuilt kernel serves one complete graph, not a graph containing it.
         []() { return buildTwoNodePointwiseGraph(); }},
        {"CrossOperandDtypeMismatch",
         // This pack's kernel reads one dtype for every operand; mismatched inputs are
         // unreadable, not merely unoptimized.
         []() {
             return buildPointwiseGraph(data_objects::PointwiseMode::ADD,
                                        data_objects::DataType::FLOAT,
                                        {1, 1, 1, 1},
                                        std::nullopt,
                                        /*binary=*/true,
                                        /*explicitStrides=*/std::nullopt,
                                        /*inputBDataType=*/data_objects::DataType::HALF);
         }},
        {"AThirdOperand",
         // A third operand is a different operation from this pack's binary add, not a
         // larger instance of the same one.
         []() {
             return buildPointwiseGraph(data_objects::PointwiseMode::ADD,
                                        data_objects::DataType::FLOAT,
                                        {1, 1, 1, 1},
                                        std::nullopt,
                                        /*binary=*/true,
                                        /*explicitStrides=*/std::nullopt,
                                        /*inputBDataType=*/std::nullopt,
                                        /*includeThirdOperand=*/true);
         }},
        {"ADanglingTensorUid",
         // The uid names a tensor absent from the graph's tensor map, distinct from one
         // the matcher reads and declines.
         []() {
             return buildPointwiseGraph(data_objects::PointwiseMode::ADD,
                                        data_objects::DataType::FLOAT,
                                        {1, 1, 1, 1},
                                        std::nullopt,
                                        /*binary=*/true,
                                        /*explicitStrides=*/std::nullopt,
                                        /*inputBDataType=*/std::nullopt,
                                        /*includeThirdOperand=*/false,
                                        /*danglingInputBUid=*/DEFAULT_DANGLING_UID);
         }},
        {"AVirtualOperand",
         // A virtual tensor never appears in the launch's variant pack.
         []() {
             return buildPointwiseGraph(data_objects::PointwiseMode::ADD,
                                        data_objects::DataType::FLOAT,
                                        {1, 1, 1, 1},
                                        std::nullopt,
                                        /*binary=*/true,
                                        /*explicitStrides=*/std::nullopt,
                                        /*inputBDataType=*/std::nullopt,
                                        /*includeThirdOperand=*/false,
                                        /*danglingInputBUid=*/std::nullopt,
                                        /*inputAVirtual=*/true);
         }},
        {"ARuntimePassByValueOperand",
         // This pack's supported shape is indistinguishable from a pass-by-value
         // scalar, whose variant-pack slot is a host pointer, not a device one.
         []() {
             return buildPointwiseGraph(data_objects::PointwiseMode::ADD,
                                        data_objects::DataType::FLOAT,
                                        {1, 1, 1, 1},
                                        std::nullopt,
                                        /*binary=*/true,
                                        /*explicitStrides=*/std::nullopt,
                                        /*inputBDataType=*/std::nullopt,
                                        /*includeThirdOperand=*/false,
                                        /*danglingInputBUid=*/std::nullopt,
                                        /*inputAVirtual=*/false,
                                        /*inputAIsRuntimePassByValue=*/true);
         }},
    }),
    [](const ::testing::TestParamInfo<GraphMatcherRefusalCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// Kernel-scoped matcher
// ---------------------------------------------------------------------------

TEST(TestPointwiseAddKernelMatcher, AcceptsAKernelWhoseDtypeMatchesTheGraph)
{
    const GraphFixture fixture(buildPointwiseGraph());

    EXPECT_TRUE(matchesKernel(POINTWISE_ADD, fixture.context(), makeKernel(64, "FLOAT")));
}

TEST(TestPointwiseAddKernelMatcher, RefusesAKernelBakedForAnotherDtype)
{
    // An f16 kernel handed f32 operands does not fail; it returns wrong numbers.
    const GraphFixture fixture(buildPointwiseGraph());

    EXPECT_FALSE(matchesKernel(POINTWISE_ADD, fixture.context(), makeKernel(64, "HALF")));
}

TEST(TestPointwiseAddKernelMatcher, AcceptsAHalfKernelForAHalfGraph)
{
    const GraphFixture fixture(
        buildPointwiseGraph(data_objects::PointwiseMode::ADD, data_objects::DataType::HALF));

    EXPECT_TRUE(matchesKernel(POINTWISE_ADD, fixture.context(), makeKernel(64, "HALF")));
}

TEST(TestPointwiseAddKernelMatcher, IgnoresBlockSizeWhichTheGraphDoesNotConstrain)
{
    // block_size ranks kernels but never gates applicability.
    const GraphFixture fixture(buildPointwiseGraph());

    EXPECT_TRUE(matchesKernel(POINTWISE_ADD, fixture.context(), makeKernel(64, "FLOAT")));
    EXPECT_TRUE(matchesKernel(POINTWISE_ADD, fixture.context(), makeKernel(256, "FLOAT")));
}

// ---------------------------------------------------------------------------
// Score and binding
// ---------------------------------------------------------------------------

TEST(TestPointwiseAddScore, PrefersTheLargerBlockSize)
{
    const GraphFixture fixture(buildPointwiseGraph());

    EXPECT_GT(scoreKernel(POINTWISE_ADD, makeKernel(256, "FLOAT"), fixture.context()),
              scoreKernel(POINTWISE_ADD, makeKernel(64, "FLOAT"), fixture.context()));
}

TEST(TestPointwiseAddBinding, TheMatcherBindsTheOperandUidsItResolved)
{
    const GraphFixture fixture(buildPointwiseGraph());

    BoundTokens bound;
    ASSERT_TRUE(matchesGraph(POINTWISE_ADD, fixture.context(), bound));

    // Asserted by token name rather than through a binding struct: the names are the
    // contract a descriptor's dispatch formulas will reference (RFC 0017 §5), and they
    // are what survives this pack's C++ becoming data.
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, POINTWISE_ADD.inputAToken),
              INPUT_A_UID);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, POINTWISE_ADD.inputBToken),
              INPUT_B_UID);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, POINTWISE_ADD.outputToken),
              OUTPUT_UID);
}

TEST(TestPointwiseAddBinding, ARejectedGraphBindsNothingToDispatchFrom)
{
    const GraphFixture fixture(buildTwoNodePointwiseGraph());

    BoundTokens bound;
    ASSERT_FALSE(matchesGraph(POINTWISE_ADD, fixture.context(), bound));

    // Dispatch reads these tokens back; a refused graph must leave nothing for it to
    // read, or a later pack's dispatch could prepare against another pack's operands.
    EXPECT_TRUE(bound.empty());
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
