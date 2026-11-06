// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <optional>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "common/ConvRepackFormulation.hpp"
#include "engines/conv/ConvGemmTestGraphs.hpp"

/**
 * @file TestConvRepackFormulation.cpp
 * @brief The IM2COL plans: which steps run around the GEMM in each direction and layout, the
 *        workspace they take, the column order and geometry handed to the kernels, and the
 *        shapes that are declined.
 */
namespace
{

using namespace hipblaslt_plugin;
using namespace hipblaslt_plugin::test;

constexpr ConvDirection ALL_DIRECTIONS[]
    = {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA, ConvDirection::BACKWARD_WEIGHTS};

ConvProblem parse(const ConvGemmShape& shape, ConvDirection direction)
{
    const ConvGemmGraph graph(shape, direction);
    const auto problem = parseConvProblem(graph.graph(), direction);
    EXPECT_TRUE(problem.has_value()) << "the test graph is not a well-formed convolution";
    return problem.value_or(ConvProblem{});
}

std::optional<ConvGemmPlanCalls> plan(const ConvGemmShape& shape, ConvDirection direction)
{
    return convGemmCalls(parse(shape, direction));
}

ConvGemmShape channelsLast(ConvGemmShape shape)
{
    shape.xChannelsLast = shape.wChannelsLast = shape.yChannelsLast = true;
    return shape;
}

ConvGemmShape padded3x3()
{
    ConvGemmShape shape;
    shape.filter = {3, 3};
    shape.prePadding = shape.postPadding = {1, 1};
    return shape;
}

template <typename Args>
const Args* stepArgs(const RepackStep& step)
{
    return std::get_if<Args>(&step.args);
}

TEST(TestConvRepackFormulation, DirectRowsArePreferred)
{
    for(const auto direction : ALL_DIRECTIONS)
    {
        const auto calls = plan(channelsLast({}), direction);
        ASSERT_TRUE(calls.has_value());
        EXPECT_EQ(calls->row, ConvGemmRow::ONE_BY_ONE);
        EXPECT_FALSE(calls->hasRepack());
        EXPECT_TRUE(calls->workspace.empty());
        EXPECT_EQ(calls->workspaceBytes, 0u);
    }
}

TEST(TestConvRepackFormulation, ForwardIsIm2colThenGemm)
{
    for(const bool last : {false, true})
    {
        const auto shape = last ? channelsLast(padded3x3()) : padded3x3();
        const auto calls = plan(shape, ConvDirection::FORWARD);
        ASSERT_TRUE(calls.has_value()) << "channels last " << last;
        EXPECT_EQ(calls->row, ConvGemmRow::IM2COL);
        ASSERT_EQ(calls->before.size(), 1u);
        EXPECT_TRUE(calls->after.empty());
        const auto* args = stepArgs<Im2ColArgs>(calls->before[0]);
        ASSERT_NE(args, nullptr);
        EXPECT_EQ(calls->before[0].srcUid, CONV_X_UID);
        EXPECT_EQ(calls->before[0].dstUid, calls->workspace[0].uid);
        // rows (n * output positions) x columns (c * filter taps)
        EXPECT_EQ(args->total, 2 * 5 * 7 * 4 * 9);
        EXPECT_FALSE(calls->calls.empty());
    }
}

TEST(TestConvRepackFormulation, BackwardDataIsGemmIntoFp32ThenCol2im)
{
    auto shape = padded3x3();
    shape.dataType = conv_data_objects::DataType::HALF;
    const auto calls = plan(shape, ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(calls.has_value());
    EXPECT_EQ(calls->row, ConvGemmRow::IM2COL);
    EXPECT_TRUE(calls->before.empty());
    ASSERT_EQ(calls->after.size(), 1u);
    ASSERT_NE(stepArgs<Col2ImArgs>(calls->after[0]), nullptr);
    EXPECT_EQ(calls->after[0].dstUid, CONV_X_UID);
    EXPECT_EQ(calls->after[0].dataType, conv_data_objects::DataType::HALF);
    EXPECT_EQ(stepArgs<Col2ImArgs>(calls->after[0])->total, 2 * 4 * 5 * 7);

    const auto colUid = calls->after[0].srcUid;
    ASSERT_EQ(calls->workspace.size(), 1u);
    EXPECT_EQ(calls->workspace[0].uid, colUid);
    EXPECT_EQ(calls->workspace[0].bytes, static_cast<size_t>(2 * 5 * 7 * 4 * 9 * 4));
    for(const auto& call : calls->calls)
    {
        EXPECT_EQ(call.d.uid, colUid);
        EXPECT_EQ(call.d.dataType, conv_data_objects::DataType::FLOAT);
        EXPECT_EQ(call.a.dataType, conv_data_objects::DataType::HALF);
        EXPECT_EQ(call.b.dataType, conv_data_objects::DataType::HALF);
    }
}

TEST(TestConvRepackFormulation, BackwardWeightsChannelsLastNeedsNoStaging)
{
    const auto calls = plan(channelsLast(padded3x3()), ConvDirection::BACKWARD_WEIGHTS);
    ASSERT_TRUE(calls.has_value());
    ASSERT_EQ(calls->before.size(), 1u);
    EXPECT_NE(stepArgs<Im2ColArgs>(calls->before[0]), nullptr);
    EXPECT_EQ(calls->workspace.size(), 1u);
}

TEST(TestConvRepackFormulation, ChannelsFirstBackwardWeightsStagesDyChannelsLast)
{
    for(const auto& shape : {padded3x3(), ConvGemmShape{}})
    {
        const auto calls = plan(shape, ConvDirection::BACKWARD_WEIGHTS);
        ASSERT_TRUE(calls.has_value());
        EXPECT_EQ(calls->row, ConvGemmRow::IM2COL);
        ASSERT_EQ(calls->before.size(), 2u);
        EXPECT_NE(stepArgs<Im2ColArgs>(calls->before[0]), nullptr);
        const auto* copy = stepArgs<StridedCopyArgs>(calls->before[1]);
        ASSERT_NE(copy, nullptr);
        EXPECT_EQ(calls->before[1].srcUid, CONV_Y_UID);
        ASSERT_EQ(calls->workspace.size(), 2u);
        EXPECT_EQ(calls->before[1].dstUid, calls->workspace[1].uid);
        // dy (n, k, h, w) lands channels-last: k innermost.
        EXPECT_EQ(copy->dstStrides[1], 1);
        EXPECT_EQ(copy->dstStrides[3], shape.k);
    }
}

TEST(TestConvRepackFormulation, WorkspaceTensorsAreAlignedAndDisjoint)
{
    const auto calls = plan(padded3x3(), ConvDirection::BACKWARD_WEIGHTS);
    ASSERT_TRUE(calls.has_value());
    size_t end = 0;
    for(const auto& tensor : calls->workspace)
    {
        EXPECT_EQ(tensor.offset % 256, 0u);
        EXPECT_GE(tensor.offset, end);
        end = tensor.offset + tensor.bytes;
        for(const auto uid : {CONV_X_UID, CONV_W_UID, CONV_Y_UID})
        {
            EXPECT_NE(tensor.uid, uid);
        }
    }
    EXPECT_EQ(calls->workspaceBytes, end);
}

TEST(TestConvRepackFormulation, ColumnOrderFollowsTheFilterLayout)
{
    auto kcrs = padded3x3();
    auto krsc = padded3x3();
    krsc.wChannelsLast = true;
    for(const auto& [shape, inner] : {std::make_pair(kcrs, 0), std::make_pair(krsc, 1)})
    {
        const auto calls = plan(shape, ConvDirection::FORWARD);
        ASSERT_TRUE(calls.has_value());
        EXPECT_EQ(stepArgs<Im2ColArgs>(calls->before[0])->geometry.channelsInner, inner);
    }
}

TEST(TestConvRepackFormulation, GeometryPadsLeadingAxesAndCarriesTheMode)
{
    ConvGemmShape shape;
    shape.input = {11};
    shape.filter = {3};
    shape.prePadding = {2};
    shape.postPadding = {1};
    shape.stride = {2};
    shape.dilation = {1};
    shape.mode = conv_data_objects::ConvMode::CONVOLUTION;
    const auto calls = plan(shape, ConvDirection::FORWARD);
    ASSERT_TRUE(calls.has_value());
    const auto& geometry = stepArgs<Im2ColArgs>(calls->before[0])->geometry;
    for(const int axis : {0, 1})
    {
        EXPECT_EQ(geometry.input[axis], 1);
        EXPECT_EQ(geometry.filter[axis], 1);
        EXPECT_EQ(geometry.output[axis], 1);
        EXPECT_EQ(geometry.xStrides[2 + axis], 0);
    }
    EXPECT_EQ(geometry.input[2], 11);
    EXPECT_EQ(geometry.output[2], 6);
    EXPECT_EQ(geometry.prePadding[2], 2);
    EXPECT_EQ(geometry.stride[2], 2);
    EXPECT_EQ(geometry.flipFilter, 1);
}

TEST(TestConvRepackFormulation, ServesWhatTheDirectRowsDecline)
{
    ConvGemmShape strided1x1;
    strided1x1.stride = {2, 2};
    ConvGemmShape pointGroupedNhwc = channelsLast({});
    pointGroupedNhwc.input = pointGroupedNhwc.filter = {3, 4};
    pointGroupedNhwc.groups = 2;
    ConvGemmShape pointConvolution;
    pointConvolution.input = pointConvolution.filter = {3, 4};
    pointConvolution.mode = conv_data_objects::ConvMode::CONVOLUTION;
    for(const auto& shape : {padded3x3(), strided1x1, pointGroupedNhwc, pointConvolution})
    {
        for(const auto direction : ALL_DIRECTIONS)
        {
            EXPECT_FALSE(directGemmCalls(parse(shape, direction)).has_value());
            const auto calls = plan(shape, direction);
            ASSERT_TRUE(calls.has_value());
            EXPECT_EQ(calls->row, ConvGemmRow::IM2COL);
        }
    }
}

TEST(TestConvRepackFormulation, DeclinesAColumnMatrixOverTheWorkspaceCap)
{
    ConvGemmShape shape = channelsLast(padded3x3());
    shape.n = 64;
    shape.c = 512;
    shape.k = 64;
    shape.input = {128, 128};
    // 64 * 128 * 128 rows x 512 * 9 columns of fp32: far over the cap.
    for(const auto direction : ALL_DIRECTIONS)
    {
        EXPECT_FALSE(plan(shape, direction).has_value());
    }
    shape.n = 1;
    shape.c = 8;
    for(const auto direction : ALL_DIRECTIONS)
    {
        const auto calls = plan(shape, direction);
        ASSERT_TRUE(calls.has_value());
        EXPECT_LE(calls->workspaceBytes, MAX_REPACK_WORKSPACE_BYTES);
    }
}

TEST(TestConvRepackFormulation, DeclinesAFilterInNeitherColumnOrder)
{
    auto problem = parse(padded3x3(), ConvDirection::FORWARD);
    // (k, c, r, s) stored k, s, c, r: neither (c, r, s) nor (r, s, c) nests into one stride.
    problem.w.strides = {4 * 3 * 3, 3, 1, 4 * 3};
    EXPECT_FALSE(im2colGemmCalls(problem).has_value());
}

} // namespace
