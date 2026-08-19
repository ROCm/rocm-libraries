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
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::MatchContext;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

bool matches(const MatchContext& context)
{
    BoundTokens bound;
    return matchesGraph(POINTWISE_ADD, context, bound);
}

// Graph-scoped matcher: acceptances

TEST(TestPointwiseAddGraphMatcher, AcceptsASingleElementFloatAdd)
{
    const GraphFixture fixture(buildPointwiseGraph());

    EXPECT_TRUE(matches(fixture.context()));
}

TEST(TestPointwiseAddGraphMatcher, AcceptsAHalfPrecisionAdd)
{
    // Graph-level gate is dtype-agnostic; the kernel-scoped matcher pins dtype.
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

// Graph-scoped matcher: refusals

/// Builder is a plain function pointer, not std::function: FlatBufferBuilder is move-only.
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
        {"MultiElementTensors",
         // The kernel writes only element 0; a larger tensor leaves the rest unwritten.
         []() {
             return buildPointwiseGraph(
                 data_objects::PointwiseMode::ADD, data_objects::DataType::FLOAT, {1, 1, 2, 2});
         }},
        {"ATensorWithNoStrides",
         // The layout classifier dereferences strides(); applicability runs before validation.
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
                                        /*inputAIsRuntimePassByValue=*/false,
                                        /*outputVirtual=*/false,
                                        /*omitStrides=*/true);
         }},
        {"DimsWhoseProductIsOneButAreNotAllOne",
         // {-1,-1,1,1} multiplies to 1; the kernel indexes element 0 only.
         []() {
             return buildPointwiseGraph(
                 data_objects::PointwiseMode::ADD, data_objects::DataType::FLOAT, {-1, -1, 1, 1});
         }},
        {"ARankTheDispatchPathCannotServe",
         []() {
             return buildPointwiseGraph(
                 data_objects::PointwiseMode::ADD, data_objects::DataType::FLOAT, {1});
         }},
        {"AStrideOrderTheDispatchPathCannotClassify",
         // Layout derives from stride order; only NCHW/NHWC classify.
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
        {"AMultiNodeGraph", []() { return buildTwoNodePointwiseGraph(); }},
        {"CrossOperandDtypeMismatch",
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
        {"AVirtualOutput",
         // Checks every operand, not just input A: a virtual output has no buffer to
         // resolve at launch.
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
                                        /*inputAIsRuntimePassByValue=*/false,
                                        /*outputVirtual=*/true);
         }},
    }),
    [](const ::testing::TestParamInfo<GraphMatcherRefusalCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// Graph-scoped operation matchers: the one fact separating this engine's packs
// ---------------------------------------------------------------------------

/// The shared matcher deliberately admits any operation; these are what stop a
/// multiplication reaching an add kernel. Both directions are asserted because a
/// matcher that always returned true would satisfy only one of the two claims.
TEST(TestPointwiseOperationMatchers, EachPackAdmitsOnlyItsOwnOperation)
{
    const GraphFixture add(buildPointwiseGraph(data_objects::PointwiseMode::ADD));
    const GraphFixture mul(buildPointwiseGraph(data_objects::PointwiseMode::MUL));

    BoundTokens bound;
    EXPECT_TRUE(matchesOperation(POINTWISE_ADD, add.context(), bound));
    EXPECT_FALSE(matchesOperation(POINTWISE_ADD, mul.context(), bound));

    EXPECT_TRUE(matchesOperation(POINTWISE_MUL, mul.context(), bound));
    EXPECT_FALSE(matchesOperation(POINTWISE_MUL, add.context(), bound));
}

/// The shared matcher has already bound every token the dispatch reads; a second writer
/// of the same token would be a silent last-one-wins.
TEST(TestPointwiseOperationMatchers, BindNothingOfTheirOwn)
{
    const GraphFixture add(buildPointwiseGraph(data_objects::PointwiseMode::ADD));

    BoundTokens bound;
    ASSERT_TRUE(matchesOperation(POINTWISE_ADD, add.context(), bound));
    EXPECT_TRUE(bound.empty());
}

/// The shared half of the split, stated as its own claim: the expensive checks do not
/// re-run per pack, so they must not encode an operation.
TEST(TestPointwiseGraphMatcher, AdmitsEveryOperationItsPacksBetweenThemServe)
{
    const GraphFixture add(buildPointwiseGraph(data_objects::PointwiseMode::ADD));
    const GraphFixture mul(buildPointwiseGraph(data_objects::PointwiseMode::MUL));

    BoundTokens bound;
    EXPECT_TRUE(matchesGraph(POINTWISE_ADD, add.context(), bound));
    EXPECT_TRUE(matchesGraph(POINTWISE_ADD, mul.context(), bound));
}

// ---------------------------------------------------------------------------
// Kernel-scoped matcher

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

// Score and binding

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

    // Asserted by token name: the contract a descriptor's dispatch formulas reference.
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

    // A refused graph must leave nothing bound, or a later pack could read stale operands.
    EXPECT_TRUE(bound.empty());
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
