// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

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

    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, CONV_FWD.inputAToken),
              CONV_X_UID);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, CONV_FWD.inputBToken),
              CONV_W_UID);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, CONV_FWD.outputToken),
              CONV_Y_UID);
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

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
