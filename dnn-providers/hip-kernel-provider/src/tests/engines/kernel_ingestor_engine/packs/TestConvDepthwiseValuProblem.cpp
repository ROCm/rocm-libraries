// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/ConvDepthwiseValuProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestConvDepthwiseValuProblem.cpp
 * @brief Host-only checks of the VALU depthwise engines' acceptance rules: what each direction
 *        accepts and the forward convolution it runs (layout, rank, mirrored padding, filter
 *        reflection), one refusal per rule beyond the shared naive parse, and the launch
 *        sizes. No device is used.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_naive_test;
using conv_depthwise_valu::DepthwiseLayout;
using conv_depthwise_valu::DepthwiseTile;
using conv_depthwise_valu::DepthwiseVariant;
using conv_depthwise_valu::launchFor;
using conv_depthwise_valu::parseDepthwiseValuProblem;
using conv_naive::ConvDirection;
using hipdnn_flatbuffers_sdk::data_objects::ConvMode;

std::optional<conv_depthwise_valu::DepthwiseValuProblem> parse(const ConvGraphSpec& spec)
{
    const GraphFixture fixture(buildGraph(spec));
    return parseDepthwiseValuProblem(fixture.context(), spec.direction);
}

std::optional<conv_depthwise_valu::DepthwiseValuProblem> parse(const ConvShape& shape,
                                                               ConvDirection direction)
{
    return parse(specFor(shape, direction));
}

/// A depthwise 3x3, padding 1, over @p channels channels.
ConvShape depthwise(int64_t channels = 5, bool channelsLast = true)
{
    ConvShape shape;
    shape.c = shape.k = shape.groups = channels;
    shape.input = {8, 9};
    shape.filter = {3, 3};
    shape.prePadding = shape.postPadding = {1, 1};
    shape.channelsLast = channelsLast;
    return shape;
}

TEST(TestConvDepthwiseValuProblem, ForwardReadsXIntoY)
{
    for(const bool last : {true, false})
    {
        const auto problem = parse(depthwise(5, last), ConvDirection::FORWARD);
        ASSERT_TRUE(problem.has_value()) << "channels last " << last;
        EXPECT_EQ(problem->layout,
                  last ? DepthwiseLayout::CHANNELS_LAST : DepthwiseLayout::CHANNELS_FIRST);
        EXPECT_EQ(problem->ndims, 2);
        EXPECT_EQ(problem->inUid, X_UID);
        EXPECT_EQ(problem->outUid, Y_UID);
        EXPECT_EQ(problem->args.hi, 8);
        EXPECT_EQ(problem->args.ho, 8);
        EXPECT_EQ(problem->args.ph, 1);
        EXPECT_FALSE(problem->flipFilter);
        EXPECT_TRUE(problem->unitStrideAndDilation);
        // The depth axis of a 2-D problem is degenerate.
        EXPECT_EQ(problem->args.di, 1);
        EXPECT_EQ(problem->args.kd, 1);
    }
}

TEST(TestConvDepthwiseValuProblem, BackwardDataRunsDyIntoDxWithMirroredPaddingAndAReflectedFilter)
{
    auto shape = depthwise();
    shape.prePadding = {2, 0};
    shape.postPadding = {0, 2};
    shape.dilation = {1, 2};
    const auto problem = parse(shape, ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->inUid, Y_UID);
    EXPECT_EQ(problem->outUid, X_UID);
    const auto output = shape.output();
    EXPECT_EQ(problem->args.hi, output[0]);
    EXPECT_EQ(problem->args.wi, output[1]);
    EXPECT_EQ(problem->args.ho, 8);
    EXPECT_EQ(problem->args.wo, 9);
    // pre' = dilation * (filter - 1) - pre
    EXPECT_EQ(problem->args.ph, 1 * 2 - 2);
    EXPECT_EQ(problem->args.pw, 2 * 2 - 0);
    EXPECT_TRUE(problem->flipFilter);
    EXPECT_FALSE(problem->unitStrideAndDilation);
}

TEST(TestConvDepthwiseValuProblem, ConvolutionModeReflectsTheFilterOnceMore)
{
    auto shape = depthwise();
    shape.mode = ConvMode::CONVOLUTION;
    const auto forward = parse(shape, ConvDirection::FORWARD);
    const auto backward = parse(shape, ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(forward.has_value());
    ASSERT_TRUE(backward.has_value());
    EXPECT_TRUE(forward->flipFilter);
    EXPECT_FALSE(backward->flipFilter);
}

TEST(TestConvDepthwiseValuProblem, OneDRunsAsTwoDAndThreeDAsThreeD)
{
    auto oneD = depthwise();
    oneD.input = {11};
    oneD.filter = {3};
    oneD.prePadding = oneD.postPadding = {1};
    oneD.stride = oneD.dilation = {1};
    const auto one = parse(oneD, ConvDirection::FORWARD);
    ASSERT_TRUE(one.has_value());
    EXPECT_EQ(one->ndims, 2);
    EXPECT_EQ(one->args.hi, 1);
    EXPECT_EQ(one->args.kh, 1);
    EXPECT_EQ(one->args.wi, 11);

    auto threeD = depthwise();
    threeD.input = {4, 5, 6};
    threeD.filter = {3, 3, 3};
    threeD.prePadding = threeD.postPadding = {1, 1, 1};
    threeD.stride = threeD.dilation = {1, 1, 1};
    for(const bool last : {true, false})
    {
        threeD.channelsLast = last;
        const auto three = parse(threeD, ConvDirection::FORWARD);
        ASSERT_TRUE(three.has_value());
        EXPECT_EQ(three->ndims, 3);
        EXPECT_EQ(three->args.di, 4);
        EXPECT_EQ(three->args.kd, 3);
    }
}

TEST(TestConvDepthwiseValuProblem, DeclinesWhatIsNotDepthwise)
{
    auto multiplier = depthwise();
    multiplier.k = 10; // two outputs per channel
    auto dense = depthwise();
    dense.groups = 1;
    dense.k = 5;
    auto grouped = depthwise(6);
    grouped.groups = 3;
    for(const auto& shape : {multiplier, dense, grouped})
    {
        EXPECT_FALSE(parse(shape, ConvDirection::FORWARD).has_value());
    }
}

TEST(TestConvDepthwiseValuProblem, DeclinesBackwardWeights)
{
    EXPECT_FALSE(parse(depthwise(), ConvDirection::BACKWARD_WEIGHTS).has_value());
}

TEST(TestConvDepthwiseValuProblem, StridedBackwardDataIsGatherOnly)
{
    auto shape = depthwise();
    shape.stride = {2, 1};
    const auto forward = parse(shape, ConvDirection::FORWARD);
    ASSERT_TRUE(forward.has_value());
    EXPECT_FALSE(forward->gatherOnly);
    const auto backward = parse(shape, ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(backward.has_value());
    EXPECT_TRUE(backward->gatherOnly);
    // Gather terms: dy in, dx out, the forward convolution's padding and stride.
    const auto& g = backward->gatherArgs;
    EXPECT_EQ(g.hi, static_cast<int>(backward->base.args.outSpatial[1]));
    EXPECT_EQ(g.ho, static_cast<int>(backward->base.args.inSpatial[1]));
    EXPECT_EQ(g.ph, static_cast<int>(backward->base.args.prePadding[1]));
    EXPECT_EQ(g.sh, 2);

    shape.stride = {1, 1};
    const auto unit = parse(shape, ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(unit.has_value());
    EXPECT_FALSE(unit->gatherOnly);
}

TEST(TestConvDepthwiseValuProblem, DeclinesMixedLayouts)
{
    auto spec = specFor(depthwise(5, true), ConvDirection::FORWARD);
    spec.y.strides = packedStrides(spec.y.dims, false);
    EXPECT_FALSE(parse(spec).has_value());
}

TEST(TestConvDepthwiseValuProblem, DeclinesUnpackedTensors)
{
    for(const char* operand : {"x", "w", "y"})
    {
        auto spec = specFor(depthwise(5, true), ConvDirection::FORWARD);
        auto& tensor = std::string(operand) == "x"   ? spec.x
                       : std::string(operand) == "w" ? spec.w
                                                     : spec.y;
        tensor.strides[0] += 3; // a gap after every image (filter: every channel)
        EXPECT_FALSE(parse(spec).has_value()) << operand;
    }
}

TEST(TestConvDepthwiseValuProblem, AcceptsAnyStrideOnExtentOneAxes)
{
    auto shape = depthwise(1, true);
    shape.n = 1;
    auto spec = specFor(shape, ConvDirection::FORWARD);
    spec.x.strides[0] = 12345; // n and c are extent 1: never read through
    spec.x.strides[1] = 7;
    EXPECT_TRUE(parse(spec).has_value());
}

TEST(TestConvDepthwiseValuProblem, LaunchCoversTheProblemPerVariant)
{
    const auto problem = parse(depthwise(40, true), ConvDirection::FORWARD);
    ASSERT_TRUE(problem.has_value());

    DepthwiseTile floor;
    floor.wstrip = 4;
    const auto floorLaunch = launchFor(*problem, floor);
    ASSERT_TRUE(floorLaunch.has_value());
    EXPECT_EQ(floorLaunch->block, 256);
    // n * ho * ceil(wo / 4) * c threads: 2 * 8 * 3 * 40 = 1920 -> 8 blocks.
    EXPECT_EQ(floorLaunch->grid, 8);

    DepthwiseTile fused;
    fused.variant = DepthwiseVariant::FUSED;
    fused.bh = fused.bw = 8;
    fused.bk = 32;
    fused.rh = fused.rw = 2;
    const auto fusedLaunch = launchFor(*problem, fused);
    ASSERT_TRUE(fusedLaunch.has_value());
    EXPECT_EQ(fusedLaunch->block, 4 * 4 * 32);
    // n * ceil(ho / 8) * ceil(wo / 8) * ceil(c / 32) = 2 * 1 * 2 * 2.
    EXPECT_EQ(fusedLaunch->grid, 8);

    const auto nchw = parse(depthwise(40, false), ConvDirection::FORWARD);
    ASSERT_TRUE(nchw.has_value());
    const auto nchwLaunch = launchFor(*nchw, DepthwiseTile{});
    ASSERT_TRUE(nchwLaunch.has_value());
    // One thread per output element: 2 * 40 * 8 * 9 = 5760 -> 23 blocks.
    EXPECT_EQ(nchwLaunch->grid, 23);
}

TEST(TestConvDepthwiseValuProblem, LaunchDeclinesABlockOverTheLimit)
{
    const auto problem = parse(depthwise(), ConvDirection::FORWARD);
    ASSERT_TRUE(problem.has_value());
    DepthwiseTile huge;
    huge.variant = DepthwiseVariant::LDS;
    huge.bh = huge.bw = 32;
    huge.bk = 32;
    huge.rh = huge.rw = 1;
    EXPECT_FALSE(launchFor(*problem, huge).has_value());
}

TEST(TestConvDepthwiseValuComputeType, ServesCudnnsComputeTypesPerElementType)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        expectCudnnComputeTypes(
            depthwise(),
            {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16},
            [direction](const ConvShape& s) { return parse(s, direction).has_value(); });
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_CONV
