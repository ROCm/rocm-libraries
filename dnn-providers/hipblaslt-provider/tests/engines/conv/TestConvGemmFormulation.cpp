// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "common/ConvGemmFormulation.hpp"
#include "engines/conv/ConvGemmTestGraphs.hpp"

/**
 * @file TestConvGemmFormulation.cpp
 * @brief Which catalog row a convolution resolves to, how many GEMM calls it takes, and the
 *        shapes and layouts that are declined because no direct GEMM reads them.
 */
namespace
{

using namespace hipblaslt_plugin;
using namespace hipblaslt_plugin::test;

std::optional<ConvGemmPlanCalls> resolve(const ConvGemmShape& shape, ConvDirection direction)
{
    const ConvGemmGraph graph(shape, direction);
    const auto problem = parseConvProblem(graph.graph(), direction);
    if(!problem.has_value())
    {
        ADD_FAILURE() << "the test graph is not a well-formed convolution";
        return std::nullopt;
    }
    return directGemmCalls(*problem);
}

constexpr ConvDirection ALL_DIRECTIONS[]
    = {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA, ConvDirection::BACKWARD_WEIGHTS};

ConvGemmShape channelsLast(ConvGemmShape shape)
{
    shape.xChannelsLast = true;
    shape.wChannelsLast = true;
    shape.yChannelsLast = true;
    return shape;
}

TEST(TestConvGemmFormulation, OneByOneChannelsLastIsOneCallBatchedOverGroups)
{
    ConvGemmShape shape = channelsLast({});
    shape.c = 6;
    shape.k = 9;
    shape.groups = 3;
    for(const auto direction : ALL_DIRECTIONS)
    {
        const auto calls = resolve(shape, direction);
        ASSERT_TRUE(calls.has_value());
        EXPECT_EQ(calls->row, ConvGemmRow::ONE_BY_ONE);
        ASSERT_EQ(calls->stages[0].calls.size(), 1u);
        EXPECT_EQ(calls->stages[0].calls[0].batchCount, 3);
    }
}

TEST(TestConvGemmFormulation, OneByOneChannelsFirstBatchesImagesWithoutGroups)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        const auto calls = resolve(ConvGemmShape{}, direction);
        ASSERT_TRUE(calls.has_value());
        ASSERT_EQ(calls->stages[0].calls.size(), 1u);
        EXPECT_EQ(calls->stages[0].calls[0].batchCount, 2); // n
    }
}

TEST(TestConvGemmFormulation, OneByOneChannelsFirstWithGroupsIsOneCallPerImage)
{
    ConvGemmShape shape;
    shape.n = 3;
    shape.groups = 2;
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        const auto calls = resolve(shape, direction);
        ASSERT_TRUE(calls.has_value());
        ASSERT_EQ(calls->stages[0].calls.size(), 3u);
        for(size_t image = 0; image < calls->stages[0].calls.size(); ++image)
        {
            EXPECT_EQ(calls->stages[0].calls[image].batchCount, 2); // groups
        }
    }
}

TEST(TestConvGemmFormulation, OneByOneChannelsFirstWeightGradientNeedsOneImage)
{
    EXPECT_FALSE(resolve(ConvGemmShape{}, ConvDirection::BACKWARD_WEIGHTS).has_value());
    ConvGemmShape single;
    single.n = 1;
    EXPECT_TRUE(resolve(single, ConvDirection::BACKWARD_WEIGHTS).has_value());
}

TEST(TestConvGemmFormulation, OneByOneIgnoresConvolutionMode)
{
    ConvGemmShape shape = channelsLast({});
    shape.mode = conv_data_objects::ConvMode::CONVOLUTION;
    EXPECT_TRUE(resolve(shape, ConvDirection::FORWARD).has_value());
}

TEST(TestConvGemmFormulation, PointOutputInEveryLayoutWhoseColumnsAgree)
{
    ConvGemmShape shape;
    shape.input = {3, 4};
    shape.filter = {3, 4};
    for(const bool last : {false, true})
    {
        ConvGemmShape layout = last ? channelsLast(shape) : shape;
        for(const auto direction : ALL_DIRECTIONS)
        {
            const auto calls = resolve(layout, direction);
            ASSERT_TRUE(calls.has_value()) << "channels last " << last;
            EXPECT_EQ(calls->row, ConvGemmRow::POINT_OUTPUT);
            EXPECT_EQ(calls->stages[0].calls.size(), 1u);
        }
    }
}

TEST(TestConvGemmFormulation, PointOutputWithGroupsNeedsChannelsFirst)
{
    ConvGemmShape shape;
    shape.input = {3, 4};
    shape.filter = {3, 4};
    shape.groups = 2;
    EXPECT_TRUE(resolve(shape, ConvDirection::FORWARD).has_value());
    EXPECT_FALSE(resolve(channelsLast(shape), ConvDirection::FORWARD).has_value());
}

TEST(TestConvGemmFormulation, PointOutputDeclinesWhenXAndWOrderColumnsDifferently)
{
    ConvGemmShape shape;
    shape.input = {3, 4};
    shape.filter = {3, 4};
    shape.wChannelsLast = true; // NCHW x, KRSC w
    EXPECT_FALSE(resolve(shape, ConvDirection::FORWARD).has_value());
}

TEST(TestConvGemmFormulation, PointOutputDeclinesTrueConvolution)
{
    ConvGemmShape shape;
    shape.input = {3, 4};
    shape.filter = {3, 4};
    shape.mode = conv_data_objects::ConvMode::CONVOLUTION;
    EXPECT_FALSE(resolve(shape, ConvDirection::FORWARD).has_value());
}

TEST(TestConvGemmFormulation, DeclinesWhatNeedsARepack)
{
    ConvGemmShape padded3x3;
    padded3x3.filter = {3, 3};
    padded3x3.prePadding = {1, 1};
    padded3x3.postPadding = {1, 1};
    ConvGemmShape strided1x1;
    strided1x1.stride = {2, 2};
    ConvGemmShape padded1x1;
    padded1x1.prePadding = {1, 0};
    for(const auto& shape : {padded3x3, strided1x1, padded1x1})
    {
        for(const auto direction : ALL_DIRECTIONS)
        {
            EXPECT_FALSE(resolve(shape, direction).has_value());
        }
    }
}

TEST(TestConvGemmFormulation, OneByOneThreeDChannelsLast)
{
    ConvGemmShape shape = channelsLast({});
    shape.input = {3, 4, 5};
    shape.filter = {1, 1, 1};
    shape.prePadding = {0, 0, 0};
    shape.postPadding = {0, 0, 0};
    shape.stride = {1, 1, 1};
    shape.dilation = {1, 1, 1};
    for(const auto direction : ALL_DIRECTIONS)
    {
        EXPECT_TRUE(resolve(shape, direction).has_value());
    }
}

TEST(TestConvGemmFormulation, ServesCudnnsComputeTypesPerElementType)
{
    using conv_data_objects::DataType;
    // cuDNN's convolution compute types (cudnn-frontend docs/developer/graph-api.mdx): FLOAT
    // for every element type, HALF also for HALF; UNSET reads as FLOAT.
    for(const auto element : {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16})
    {
        for(const auto compute : {DataType::UNSET,
                                  DataType::FLOAT,
                                  DataType::HALF,
                                  DataType::BFLOAT16,
                                  DataType::DOUBLE,
                                  DataType::INT32,
                                  DataType::INT8})
        {
            ConvGemmShape shape;
            shape.dataType = element;
            shape.computeType = compute;
            const bool served = compute == DataType::UNSET || compute == DataType::FLOAT
                                || (compute == DataType::HALF && element == DataType::HALF);
            for(const auto direction : ALL_DIRECTIONS)
            {
                const ConvGemmGraph graph(shape, direction);
                EXPECT_EQ(parseConvProblem(graph.graph(), direction).has_value(), served)
                    << conv_data_objects::EnumNameDataType(element) << " elements, "
                    << conv_data_objects::EnumNameDataType(compute) << " compute";
            }
        }
    }
}

} // namespace
