// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/**
 * @file TestConvFwdPack.cpp
 * @brief The conv-forward pack's matcher shapes: what it accepts, what it refuses, and
 *        that the engine split by graph node type actually holds.
 *
 * Modelled on TestPointwiseAddMatchers.cpp. This engine has one pack and no operation
 * matcher, so there is no operation-matcher section to mirror; in its place is the
 * claim the second engine exists to make -- a conv graph and a pointwise graph each
 * reach only their own engine's matcher.
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
    return matchesGraph(CONV_FWD, context, bound);
}

// ---------------------------------------------------------------------------
// Graph-scoped matcher: the supported case
// ---------------------------------------------------------------------------

TEST(TestConvFwdGraphMatcher, AcceptsUnitStrideNoPaddingCrossCorrelation)
{
    const GraphFixture fixture(buildConvFwdGraph());

    EXPECT_TRUE(matches(fixture.context()));
}

TEST(TestConvFwdGraphMatcher, AcceptsAHalfPrecisionConv)
{
    const GraphFixture fixture(buildConvFwdGraph(data_objects::DataType::HALF));

    EXPECT_TRUE(matches(fixture.context()));
}

TEST(TestConvFwdBinding, BindsAllThreeOperandUids)
{
    const GraphFixture fixture(buildConvFwdGraph());

    BoundTokens bound;
    ASSERT_TRUE(matchesGraph(CONV_FWD, fixture.context(), bound));

    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, CONV_FWD.inputAToken), CONV_X_UID);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, CONV_FWD.inputBToken), CONV_W_UID);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, CONV_FWD.outputToken), CONV_Y_UID);
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

class TestConvFwdGraphMatcherRefusal : public ::testing::TestWithParam<GraphMatcherRefusalCase>
{
};

TEST_P(TestConvFwdGraphMatcherRefusal, Refuses)
{
    const GraphFixture fixture(GetParam().buildGraph());

    EXPECT_FALSE(matches(fixture.context()));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestConvFwdGraphMatcherRefusal,
    ::testing::ValuesIn(std::vector<GraphMatcherRefusalCase>{
        {"StrideTwo",
         // The in-kernel p = h - r + 1 formula is only correct for unit stride.
         []() {
             return buildConvFwdGraph(data_objects::DataType::FLOAT,
                                      data_objects::ConvMode::CROSS_CORRELATION,
                                      /*stride=*/std::vector<int64_t>{2, 2});
         }},
        {"DilationTwo",
         []() {
             return buildConvFwdGraph(data_objects::DataType::FLOAT,
                                      data_objects::ConvMode::CROSS_CORRELATION,
                                      /*stride=*/std::vector<int64_t>{1, 1},
                                      /*dilation=*/std::vector<int64_t>{2, 2});
         }},
        {"Padded",
         // The kernel's flat index arithmetic never adds padding.
         []() {
             return buildConvFwdGraph(data_objects::DataType::FLOAT,
                                      data_objects::ConvMode::CROSS_CORRELATION,
                                      /*stride=*/std::vector<int64_t>{1, 1},
                                      /*dilation=*/std::vector<int64_t>{1, 1},
                                      /*prePadding=*/std::vector<int64_t>{1, 1});
         }},
        {"ConvolutionMode",
         // Only CROSS_CORRELATION is supported; true CONVOLUTION flips the kernel,
         // which this reference implementation does not.
         []() {
             return buildConvFwdGraph(data_objects::DataType::FLOAT,
                                      data_objects::ConvMode::CONVOLUTION);
         }},
        {"CrossOperandDtypeMismatch",
         []() {
             return buildConvFwdGraph(data_objects::DataType::FLOAT,
                                      data_objects::ConvMode::CROSS_CORRELATION,
                                      /*stride=*/std::vector<int64_t>{1, 1},
                                      /*dilation=*/std::vector<int64_t>{1, 1},
                                      /*prePadding=*/std::vector<int64_t>{0, 0},
                                      /*postPadding=*/std::vector<int64_t>{0, 0},
                                      /*xDims=*/std::vector<int64_t>{1, 1, 3, 3},
                                      /*wDims=*/std::nullopt,
                                      /*yDims=*/std::nullopt,
                                      /*wDataType=*/data_objects::DataType::HALF);
         }},
        {"APointwiseGraph",
         // This engine's matcher only ever admits a ConvolutionFwdAttributes node.
         []() { return buildPointwiseGraph(); }},
        {"FilterChannelsDisagreeWithInput",
         // w's channel count (4) disagrees with x's (1) -- also the group-count
         // refusal, since this pack has no notion of groups.
         []() {
             return buildConvFwdGraph(data_objects::DataType::FLOAT,
                                      data_objects::ConvMode::CROSS_CORRELATION,
                                      /*stride=*/std::vector<int64_t>{1, 1},
                                      /*dilation=*/std::vector<int64_t>{1, 1},
                                      /*prePadding=*/std::vector<int64_t>{0, 0},
                                      /*postPadding=*/std::vector<int64_t>{0, 0},
                                      /*xDims=*/std::vector<int64_t>{1, 1, 3, 3},
                                      /*wDims=*/std::vector<int64_t>{1, 4, 2, 2});
         }},
        {"OutputDimsInconsistentWithInputAndFilter",
         // y's shape disagrees with n/k/p/q, which is entirely what the kernel
         // actually computes it from -- a smaller y is an out-of-bounds write.
         []() {
             return buildConvFwdGraph(data_objects::DataType::FLOAT,
                                      data_objects::ConvMode::CROSS_CORRELATION,
                                      /*stride=*/std::vector<int64_t>{1, 1},
                                      /*dilation=*/std::vector<int64_t>{1, 1},
                                      /*prePadding=*/std::vector<int64_t>{0, 0},
                                      /*postPadding=*/std::vector<int64_t>{0, 0},
                                      /*xDims=*/std::vector<int64_t>{1, 1, 3, 3},
                                      /*wDims=*/std::nullopt,
                                      /*yDims=*/std::vector<int64_t>{1, 3, 9, 9});
         }},
        {"PostPaddingOnly",
         // Padding on one side is still padding; the flat index arithmetic never
         // adds any.
         []() {
             return buildConvFwdGraph(data_objects::DataType::FLOAT,
                                      data_objects::ConvMode::CROSS_CORRELATION,
                                      /*stride=*/std::vector<int64_t>{1, 1},
                                      /*dilation=*/std::vector<int64_t>{1, 1},
                                      /*prePadding=*/std::vector<int64_t>{0, 0},
                                      /*postPadding=*/std::vector<int64_t>{1, 1});
         }},
        {"NonPackedStrides",
         // Valid strides, but not packed row-major; the kernel takes no strides of
         // its own and assumes contiguous NCHW.
         []() {
             return buildConvFwdGraph(data_objects::DataType::FLOAT,
                                      data_objects::ConvMode::CROSS_CORRELATION,
                                      /*stride=*/std::vector<int64_t>{1, 1},
                                      /*dilation=*/std::vector<int64_t>{1, 1},
                                      /*prePadding=*/std::vector<int64_t>{0, 0},
                                      /*postPadding=*/std::vector<int64_t>{0, 0},
                                      /*xDims=*/std::vector<int64_t>{1, 1, 3, 3},
                                      /*wDims=*/std::nullopt,
                                      /*yDims=*/std::nullopt,
                                      /*wDataType=*/std::nullopt,
                                      /*xStridesOverride=*/std::vector<int64_t>{9, 9, 1, 3});
         }},
        {"Rank3Tensors",
         // Rank 4 is required; a rank-3 x is refused before any cross-operand
         // comparison runs, so w/y here only need to be constructible.
         []() {
             return buildConvFwdGraph(data_objects::DataType::FLOAT,
                                      data_objects::ConvMode::CROSS_CORRELATION,
                                      /*stride=*/std::vector<int64_t>{1, 1},
                                      /*dilation=*/std::vector<int64_t>{1, 1},
                                      /*prePadding=*/std::vector<int64_t>{0, 0},
                                      /*postPadding=*/std::vector<int64_t>{0, 0},
                                      /*xDims=*/std::vector<int64_t>{1, 1, 3},
                                      /*wDims=*/std::vector<int64_t>{1, 1, 2, 2},
                                      /*yDims=*/std::vector<int64_t>{1, 1, 2, 2});
         }},
        {"UnsupportedDtype",
         // Only FLOAT and HALF are supported; the reference kernel has no other
         // instantiation.
         []() { return buildConvFwdGraph(data_objects::DataType::INT32); }},
    }),
    [](const ::testing::TestParamInfo<GraphMatcherRefusalCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// The engine split: each graph type reaches only its own matcher
// ---------------------------------------------------------------------------

/// The claim the split by graph node type exists to make: a conv graph never satisfies
/// the pointwise matcher, and a pointwise graph never satisfies the conv matcher.
TEST(TestConvFwdGraphMatcher, DoesNotOverlapWithThePointwiseEngine)
{
    const GraphFixture convFixture(buildConvFwdGraph());
    const GraphFixture pointwiseFixture(buildPointwiseGraph());

    BoundTokens bound;
    EXPECT_TRUE(matchesGraph(CONV_FWD, convFixture.context(), bound));
    EXPECT_FALSE(matchesGraph(CONV_FWD, pointwiseFixture.context(), bound));

    EXPECT_TRUE(matchesGraph(POINTWISE_ADD, pointwiseFixture.context(), bound));
    EXPECT_FALSE(matchesGraph(POINTWISE_ADD, convFixture.context(), bound));
}

// ---------------------------------------------------------------------------
// Kernel-scoped matcher
// ---------------------------------------------------------------------------

TEST(TestConvFwdKernelMatcher, AcceptsAKernelWhoseDtypeMatchesTheGraph)
{
    const GraphFixture fixture(buildConvFwdGraph());

    EXPECT_TRUE(matchesKernel(CONV_FWD, fixture.context(), makeKernel(64, "FLOAT", "ConvFwd")));
}

TEST(TestConvFwdKernelMatcher, RefusesAKernelBakedForAnotherDtype)
{
    const GraphFixture fixture(buildConvFwdGraph());

    EXPECT_FALSE(matchesKernel(CONV_FWD, fixture.context(), makeKernel(64, "HALF", "ConvFwd")));
}

TEST(TestConvFwdKernelMatcher, AcceptsAHalfKernelForAHalfGraph)
{
    const GraphFixture fixture(buildConvFwdGraph(data_objects::DataType::HALF));

    EXPECT_TRUE(matchesKernel(CONV_FWD, fixture.context(), makeKernel(64, "HALF", "ConvFwd")));
}

// ---------------------------------------------------------------------------
// Score
// ---------------------------------------------------------------------------

TEST(TestConvFwdScore, PrefersTheLargerBlockSize)
{
    const GraphFixture fixture(buildConvFwdGraph());

    EXPECT_GT(scoreKernel(CONV_FWD, makeKernel(256, "FLOAT", "ConvFwd"), fixture.context()),
              scoreKernel(CONV_FWD, makeKernel(64, "FLOAT", "ConvFwd"), fixture.context()));
}

// ---------------------------------------------------------------------------
// Shipped descriptor set
// ---------------------------------------------------------------------------
//
// Every test above resolves symbols compiled into this binary and hand-builds
// KernelDefinitions via makeKernel() -- none of it loads conv_fwd/*.json. Without this
// section, a broken shipped descriptor (wrong entry_point, a missing kernel, a knob
// naming no KMD field) passes every unit test and only shows up in the slow GPU suite.

TEST(TestConvFwdPack, ShipsThreeKernelsCoveringTwoBlockSizesAndTwoDataTypes)
{
    const auto& set = loadedSet("hipkernel:ConvFwd");

    ASSERT_EQ(set.packs.size(), 1U);
    const auto& kernels = set.packs.front().kernels;
    ASSERT_EQ(kernels.size(), 3U);

    const auto describes = [&kernels](int64_t blockSize, const std::string& dtype) {
        return std::any_of(kernels.begin(), kernels.end(), [&](const auto& kernel) {
            return std::get<int64_t>(kernel.metadata.at(std::string(BLOCK_SIZE_FIELD))) == blockSize
                   && std::get<std::string>(kernel.metadata.at(std::string(DTYPE_FIELD))) == dtype
                   && kernel.source.entryPoint == "ConvFwd";
        });
    };

    EXPECT_TRUE(describes(64, "FLOAT"));
    EXPECT_TRUE(describes(256, "FLOAT"));
    EXPECT_TRUE(describes(64, "HALF"));
}

TEST(TestConvFwdPack, ExposesBlockSizeAsTheOneKnob)
{
    const auto& set = loadedSet("hipkernel:ConvFwd");

    ASSERT_EQ(set.engine.knobs.size(), 1U);
    EXPECT_EQ(set.engine.knobs.front(), std::string(BLOCK_SIZE_FIELD));
}

TEST(TestConvFwdPack, HasOneGraphMatcherAndOneKernelMatcher)
{
    const auto& set = loadedSet("hipkernel:ConvFwd");

    EXPECT_EQ(std::count_if(set.matchers.begin(),
                            set.matchers.end(),
                            [](const auto& matcher) {
                                return matcher.scope
                                       == hipdnn_plugin_sdk::ingestor::MatchScope::GRAPH;
                            }),
              1);
    EXPECT_EQ(std::count_if(set.matchers.begin(),
                            set.matchers.end(),
                            [](const auto& matcher) {
                                return matcher.scope
                                       == hipdnn_plugin_sdk::ingestor::MatchScope::KERNEL;
                            }),
              1);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
