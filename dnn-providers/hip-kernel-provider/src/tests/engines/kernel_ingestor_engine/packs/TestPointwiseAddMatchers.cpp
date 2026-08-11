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

#include "engines/kernel_ingestor_engine/packs/PointwiseAddMatchers.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseAddSymbols.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseAddTestGraphs.hpp"

/**
 * @file TestPointwiseAddMatchers.cpp
 * @brief The pack's two matcher shapes: what each accepts, and what each refuses.
 *
 * The refusals matter more than the acceptances here. An under-specified decline accepts
 * a graph the kernel cannot serve, which is a wrong answer rather than a missed
 * optimization — and for a prebuilt kernel it is a silent one.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::MatchContext;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// Runs the graph matcher where only its verdict is under test, discarding what it
/// bound. What it binds on success is asserted separately, below.
bool matches(const MatchContext& context)
{
    BoundTokens bound;
    return pointwiseAddGraphMatches(context, bound);
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
    // The graph-level gate is dtype-agnostic within the pack's declared set; pinning the
    // kernel's baked dtype is the kernel-scoped matcher's job.
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
//
// Collapsed into one TEST_P: each case differs only in which graph it hands the
// matcher, and every one asserts the identical verdict. Nine cases in total -- the six
// original shape refusals, plus the three this pack's matcher previously left
// unreachable from the builder (cross-operand dtype mismatch, a third operand, and a
// dangling tensor uid).
// ---------------------------------------------------------------------------

/// One graph the matcher must refuse, plus a readable name for a failing run. The
/// builder is a captureless-lambda-convertible function pointer rather than a
/// std::function, because flatbuffers::FlatBufferBuilder is move-only and every case
/// here needs only a zero-argument factory.
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
         // The kernel writes element 0 and nothing else, so anything larger would
         // silently leave most of the output untouched.
         []() {
             return buildPointwiseGraph(
                 data_objects::PointwiseMode::ADD, data_objects::DataType::FLOAT, {1, 1, 2, 2});
         }},
        {"ARankTheDispatchPathCannotServe",
         // A 1-element 1-D tensor suits the kernel, which indexes element 0, but the
         // provider's compile options derive layout from the tensor and reject anything
         // below rank 4. Accepting it would trade a free decline at applicability for a
         // failed plan build, which the caller pays for.
         []() {
             return buildPointwiseGraph(
                 data_objects::PointwiseMode::ADD, data_objects::DataType::FLOAT, {1});
         }},
        {"AStrideOrderTheDispatchPathCannotClassify",
         // Same class as the rank refusal above, and the same root cause: the
         // provider's compile options derive a layout from the tensor's strides and
         // throw on any 4D order that is neither NCHW nor NHWC. A 1-element tensor
         // viewing into a larger buffer can carry such an order, so without this the
         // matcher accepts a graph the plan build then fails on, which RFC 0017 section
         // 8.6 forbids.
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
         // A prebuilt kernel serves one complete graph, so a larger graph is a
         // different problem even though it contains this one.
         []() { return buildTwoNodePointwiseGraph(); }},
        {"CrossOperandDtypeMismatch",
         // PointwiseAddMatchers.cpp:165-168. The highest-value refusal here: this pack's
         // kernel reads one dtype for every operand, so a binary add whose inputs
         // disagree is not merely unoptimized, it is unreadable. Unreachable from the
         // pre-extension builder, which forced every tensor to share one dtype.
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
         // PointwiseAddMatchers.cpp:144, the in_2_tensor_uid().has_value() half. A third
         // operand is a different operation from this pack's binary add, not a larger
         // instance of the same one.
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
         // PointwiseAddMatchers.cpp:152-155. in_1_tensor_uid names a uid absent from
         // the graph's tensor map -- a node the matcher cannot even read, as distinct
         // from one it reads and declines.
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
    }),
    [](const ::testing::TestParamInfo<GraphMatcherRefusalCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// Kernel-scoped matcher
// ---------------------------------------------------------------------------

TEST(TestPointwiseAddKernelMatcher, AcceptsAKernelWhoseDtypeMatchesTheGraph)
{
    const GraphFixture fixture(buildPointwiseGraph());

    EXPECT_TRUE(pointwiseAddKernelMatches(fixture.context(), makeKernel(64, "FLOAT")));
}

TEST(TestPointwiseAddKernelMatcher, RefusesAKernelBakedForAnotherDtype)
{
    // The failure this prevents is silent: an f16 binary handed f32 operands does not
    // fail, it returns wrong numbers.
    const GraphFixture fixture(buildPointwiseGraph());

    EXPECT_FALSE(pointwiseAddKernelMatches(fixture.context(), makeKernel(64, "HALF")));
}

TEST(TestPointwiseAddKernelMatcher, AcceptsAHalfKernelForAHalfGraph)
{
    const GraphFixture fixture(
        buildPointwiseGraph(data_objects::PointwiseMode::ADD, data_objects::DataType::HALF));

    EXPECT_TRUE(pointwiseAddKernelMatches(fixture.context(), makeKernel(64, "HALF")));
}

TEST(TestPointwiseAddKernelMatcher, IgnoresBlockSizeWhichTheGraphDoesNotConstrain)
{
    // block_size is a ranking and launch axis, not an applicability one: every block
    // size serves every graph this pack accepts.
    const GraphFixture fixture(buildPointwiseGraph());

    EXPECT_TRUE(pointwiseAddKernelMatches(fixture.context(), makeKernel(64, "FLOAT")));
    EXPECT_TRUE(pointwiseAddKernelMatches(fixture.context(), makeKernel(256, "FLOAT")));
}

// ---------------------------------------------------------------------------
// Score and binding
// ---------------------------------------------------------------------------

TEST(TestPointwiseAddScore, PrefersTheLargerBlockSize)
{
    const GraphFixture fixture(buildPointwiseGraph());

    EXPECT_GT(pointwiseAddScore(makeKernel(256, "FLOAT"), fixture.context()),
              pointwiseAddScore(makeKernel(64, "FLOAT"), fixture.context()));
}

TEST(TestPointwiseAddBinding, TheMatcherBindsTheOperandUidsItResolved)
{
    // Matching does double duty: deciding the kernel applies also binds the fields the
    // launch will use, so dispatch reads these rather than re-deriving the graph shape
    // with a second notion of what a pointwise add looks like.
    const GraphFixture fixture(buildPointwiseGraph());

    BoundTokens bound;
    ASSERT_TRUE(pointwiseAddGraphMatches(fixture.context(), bound));

    const auto binding = pointwiseAddBinding(bound);
    EXPECT_EQ(binding.inputA, INPUT_A_UID);
    EXPECT_EQ(binding.inputB, INPUT_B_UID);
    EXPECT_EQ(binding.output, OUTPUT_UID);
}

TEST(TestPointwiseAddBinding, ARejectedGraphBindsNothingToDispatchFrom)
{
    // A matcher that declines must not leave partial bindings behind: its pack is pruned,
    // so anything it wrote would be state no surviving kernel could correctly read.
    const GraphFixture fixture(buildTwoNodePointwiseGraph());

    BoundTokens bound;
    ASSERT_FALSE(pointwiseAddGraphMatches(fixture.context(), bound));

    EXPECT_TRUE(bound.empty());
    EXPECT_THROW(pointwiseAddBinding(bound), hipdnn_plugin_sdk::HipdnnPluginException);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
