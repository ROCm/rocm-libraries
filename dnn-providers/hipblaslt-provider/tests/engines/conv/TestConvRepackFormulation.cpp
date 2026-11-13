// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "common/ConvRepackFormulation.hpp"
#include "engines/conv/ConvGemmTestGraphs.hpp"

/**
 * @file TestConvRepackFormulation.cpp
 * @brief The IM2COL plans: which steps run around the GEMM in each direction and layout, the
 *        workspace they take, the column order and geometry handed to the kernels, the slicing
 *        of images to fit the workspace cap, and the shapes that are declined.
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

/// Stages that run GEMMs: one per slice of images (a final rounding stage runs none).
size_t gemmStages(const ConvGemmPlanCalls& calls)
{
    return static_cast<size_t>(
        std::count_if(calls.stages.begin(), calls.stages.end(), [](const ConvGemmStage& stage) {
            return !stage.calls.empty();
        }));
}

TEST(TestConvRepackFormulation, DirectRowsArePreferred)
{
    for(const auto direction : ALL_DIRECTIONS)
    {
        const auto calls = plan(channelsLast({}), direction);
        ASSERT_TRUE(calls.has_value());
        EXPECT_EQ(calls->row, ConvGemmRow::ONE_BY_ONE);
        EXPECT_FALSE(calls->hasRepack());
        EXPECT_EQ(calls->stages.size(), 1u);
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
        ASSERT_EQ(calls->stages.size(), 1u);
        const auto& stage = calls->stages[0];
        ASSERT_EQ(stage.before.size(), 1u);
        EXPECT_TRUE(stage.after.empty());
        const auto* args = stepArgs<Im2ColArgs>(stage.before[0]);
        ASSERT_NE(args, nullptr);
        EXPECT_EQ(stage.before[0].srcUid, CONV_X_UID);
        EXPECT_EQ(stage.before[0].dstUid, calls->workspace[0].uid);
        // rows (n * output positions) x columns (c * filter taps)
        EXPECT_EQ(args->total, 2 * 5 * 7 * 4 * 9);
        EXPECT_FALSE(stage.calls.empty());
    }
}

TEST(TestConvRepackFormulation, BackwardDataIsGemmIntoFp32ThenCol2im)
{
    auto shape = padded3x3();
    shape.dataType = conv_data_objects::DataType::HALF;
    const auto calls = plan(shape, ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(calls.has_value());
    EXPECT_EQ(calls->row, ConvGemmRow::IM2COL);
    ASSERT_EQ(calls->stages.size(), 1u);
    const auto& stage = calls->stages[0];
    EXPECT_TRUE(stage.before.empty());
    ASSERT_EQ(stage.after.size(), 1u);
    ASSERT_NE(stepArgs<Col2ImArgs>(stage.after[0]), nullptr);
    EXPECT_EQ(stage.after[0].dstUid, CONV_X_UID);
    EXPECT_EQ(stage.after[0].dataType, conv_data_objects::DataType::HALF);
    EXPECT_EQ(stepArgs<Col2ImArgs>(stage.after[0])->total, 2 * 4 * 5 * 7);

    const auto colUid = stage.after[0].srcUid;
    ASSERT_EQ(calls->workspace.size(), 1u);
    EXPECT_EQ(calls->workspace[0].uid, colUid);
    EXPECT_EQ(calls->workspace[0].bytes, static_cast<size_t>(2 * 5 * 7 * 4 * 9 * 4));
    for(const auto& call : stage.calls)
    {
        EXPECT_EQ(call.d.uid, colUid);
        EXPECT_EQ(call.d.dataType, conv_data_objects::DataType::FLOAT);
        EXPECT_EQ(call.a.dataType, conv_data_objects::DataType::HALF);
        EXPECT_EQ(call.b.dataType, conv_data_objects::DataType::HALF);
    }
}

TEST(TestConvRepackFormulation, DisjointTapsBackwardDataKeepsColumnsInDxType)
{
    // A strided 1x1 and a patchify backward data: every dx element gets at most one term, so the
    // GEMM writes its columns in dx's type and col2im copies them (no fp32 sum).
    ConvGemmShape shape;
    shape.stride = {2, 2};
    shape.input = {6, 8};
    shape.dataType = conv_data_objects::DataType::HALF;
    ConvGemmShape patchify = shape;
    patchify.filter = {3, 4};
    patchify.stride = {3, 4};
    patchify.input = {9, 12};
    for(const auto& candidate : {shape, patchify})
    {
        const auto calls = plan(candidate, ConvDirection::BACKWARD_DATA);
        ASSERT_TRUE(calls.has_value());
        EXPECT_EQ(calls->row, ConvGemmRow::IM2COL);
        ASSERT_FALSE(calls->stages.empty());
        for(const auto& call : calls->stages[0].calls)
        {
            EXPECT_EQ(call.d.dataType, conv_data_objects::DataType::HALF);
        }
        ASSERT_EQ(calls->stages[0].after.size(), 1u);
        EXPECT_NE(stepArgs<Col2ImArgs>(calls->stages[0].after[0]), nullptr);
    }
}

TEST(TestConvRepackFormulation, BackwardWeightsChannelsLastNeedsNoStaging)
{
    const auto calls = plan(channelsLast(padded3x3()), ConvDirection::BACKWARD_WEIGHTS);
    ASSERT_TRUE(calls.has_value());
    ASSERT_EQ(calls->stages.size(), 1u);
    ASSERT_EQ(calls->stages[0].before.size(), 1u);
    EXPECT_NE(stepArgs<Im2ColArgs>(calls->stages[0].before[0]), nullptr);
    EXPECT_EQ(calls->workspace.size(), 1u);
}

TEST(TestConvRepackFormulation, ColumnMatrixTakesTheLayoutOfY)
{
    for(const bool last : {false, true})
    {
        for(const auto direction : ALL_DIRECTIONS)
        {
            const auto calls = plan(last ? channelsLast(padded3x3()) : padded3x3(), direction);
            ASSERT_TRUE(calls.has_value());
            const auto& stage = calls->stages[0];
            const auto& step
                = direction == ConvDirection::BACKWARD_DATA ? stage.after[0] : stage.before[0];
            const auto& geometry = direction == ConvDirection::BACKWARD_DATA
                                       ? stepArgs<Col2ImArgs>(step)->geometry
                                       : stepArgs<Im2ColArgs>(step)->geometry;
            EXPECT_EQ(geometry.columnsFirst, last ? 0 : 1) << "channels last " << last;
        }
    }
}

TEST(TestConvRepackFormulation, AChannelsLastPreferenceOverridesY)
{
    for(const auto direction : ALL_DIRECTIONS)
    {
        const auto problem = parse(padded3x3(), direction); // channels-first y
        const auto followY = im2colGemmCalls(problem);
        const auto channelsLast
            = im2colGemmCalls(problem, MAX_REPACK_WORKSPACE_BYTES, ColumnLayout::CHANNELS_LAST);
        ASSERT_TRUE(followY.has_value());
        ASSERT_TRUE(channelsLast.has_value());
        EXPECT_TRUE(followY->columnsFirst);
        EXPECT_FALSE(channelsLast->columnsFirst);
    }
    // Channels-last y takes channels-last columns under either preference.
    const auto nhwc = im2colGemmCalls(parse(channelsLast(padded3x3()), ConvDirection::FORWARD));
    ASSERT_TRUE(nhwc.has_value());
    EXPECT_FALSE(nhwc->columnsFirst);
}

TEST(TestConvRepackFormulation, StagedChannelsLastRunsOneGemmOverEveryImage)
{
    // A channels-first strided 1x1: the other layouts batch a GEMM per image; staged
    // channels-last copies y channels-last so one GEMM covers both images' positions.
    ConvGemmShape strided1x1;
    strided1x1.stride = {2, 2};
    strided1x1.input = {6, 8};
    for(const auto direction : ALL_DIRECTIONS)
    {
        const auto problem = parse(strided1x1, direction);
        const auto followY = im2colGemmCalls(problem);
        const auto staged = im2colGemmCalls(
            problem, MAX_REPACK_WORKSPACE_BYTES, ColumnLayout::STAGED_CHANNELS_LAST);
        ASSERT_TRUE(followY.has_value());
        ASSERT_TRUE(staged.has_value());
        EXPECT_FALSE(staged->columnsFirst);
        ASSERT_EQ(gemmStages(*staged), 1u) << static_cast<int>(direction);
        ASSERT_EQ(staged->stages[0].calls.size(), 1u);
        const auto& call = staged->stages[0].calls[0];
        EXPECT_EQ(call.batchCount, 1);
        // The positions of both images (3 x 4 each) in one GEMM dimension.
        if(direction == ConvDirection::BACKWARD_WEIGHTS)
        {
            EXPECT_EQ(call.k, 2 * 3 * 4);
        }
        else
        {
            EXPECT_TRUE(call.m == 2 * 3 * 4 || call.n == 2 * 3 * 4);
        }
        // y staged channels-last next to the columns.
        EXPECT_EQ(staged->workspace.size(), followY->workspace.size() + 1);
    }
    // Channels-last y needs no staging: the same plan as channels-last columns.
    const auto nhwc = parse(channelsLast(strided1x1), ConvDirection::BACKWARD_DATA);
    const auto staged
        = im2colGemmCalls(nhwc, MAX_REPACK_WORKSPACE_BYTES, ColumnLayout::STAGED_CHANNELS_LAST);
    const auto plain
        = im2colGemmCalls(nhwc, MAX_REPACK_WORKSPACE_BYTES, ColumnLayout::CHANNELS_LAST);
    ASSERT_TRUE(staged.has_value());
    ASSERT_TRUE(plain.has_value());
    EXPECT_EQ(staged->workspaceBytes, plain->workspaceBytes);
}

TEST(TestConvRepackFormulation, ChannelsFirstBackwardWeightsBatchesImagesAfterIm2col)
{
    // Channels-first backward weights reduces over images, which no single channels-first GEMM
    // reads: one GEMM batched over the images writes fp32 partials, which a cast adds into w.
    const auto shape = padded3x3();
    const auto calls = plan(shape, ConvDirection::BACKWARD_WEIGHTS);
    ASSERT_TRUE(calls.has_value());
    EXPECT_EQ(calls->row, ConvGemmRow::IM2COL);
    ASSERT_EQ(calls->stages.size(), 1u);
    const auto& stage = calls->stages[0];
    // Only im2col before: dy is read as given, no staging copy.
    ASSERT_EQ(stage.before.size(), 1u);
    EXPECT_NE(stepArgs<Im2ColArgs>(stage.before[0]), nullptr);
    ASSERT_EQ(calls->workspace.size(), 2u); // columns, partials
    ASSERT_EQ(stage.calls.size(), 1u);
    EXPECT_EQ(stage.calls[0].batchCount, shape.n);
    EXPECT_EQ(stage.calls[0].d.uid, calls->workspace[1].uid);
    EXPECT_FALSE(stage.calls[0].accumulate);
    ASSERT_EQ(stage.after.size(), 1u);
    const auto* sum = stepArgs<StridedCastArgs>(stage.after[0]);
    ASSERT_NE(sum, nullptr);
    EXPECT_EQ(sum->batches, shape.n);
    EXPECT_EQ(stage.after[0].dstUid, CONV_W_UID);
}

TEST(TestConvRepackFormulation, GroupedChannelsFirstBackwardWeightsBatchesTheLongestAxis)
{
    // Grouped channels-first backward weights: per image a GEMM batched over the groups. The
    // images (8) outnumber the groups (2): one call per group, batched over the images, into
    // partials (image, group); a cast adds the images into each group's slice of w.
    ConvGemmShape shape = padded3x3();
    shape.n = 8;
    shape.c = 4;
    shape.k = 6;
    shape.groups = 2;
    const auto calls = plan(shape, ConvDirection::BACKWARD_WEIGHTS);
    ASSERT_TRUE(calls.has_value());
    EXPECT_EQ(calls->row, ConvGemmRow::IM2COL);
    const auto& stage = calls->stages.at(0);
    ASSERT_EQ(stage.calls.size(), 2u);
    ASSERT_EQ(calls->workspace.size(), 2u); // columns, partials
    const int64_t perGroup = 3 * (2 * 9);
    EXPECT_EQ(calls->workspace[1].bytes, static_cast<size_t>(8 * 2 * perGroup) * sizeof(float));
    for(size_t g = 0; g < stage.calls.size(); ++g)
    {
        const auto& call = stage.calls[g];
        EXPECT_EQ(call.batchCount, 8);
        EXPECT_EQ(call.d.uid, calls->workspace[1].uid);
        EXPECT_EQ(call.d.batchStride, 2 * perGroup);
        EXPECT_EQ(call.d.offset, static_cast<int64_t>(g) * perGroup);
        EXPECT_FALSE(call.accumulate);
    }
    ASSERT_EQ(stage.after.size(), 1u);
    const auto* sum = stepArgs<StridedCastArgs>(stage.after[0]);
    ASSERT_NE(sum, nullptr);
    EXPECT_EQ(sum->rank, 3);
    EXPECT_EQ(sum->dims[0], 2);
    EXPECT_EQ(sum->total, 2 * perGroup);
    EXPECT_EQ(sum->batches, 8);
    EXPECT_EQ(sum->batchStride, 2 * perGroup);
    EXPECT_EQ(stage.after[0].dstUid, CONV_W_UID);
}

TEST(TestConvRepackFormulation, LongReductionIntoSmallWeightsSplitsIntoChunks)
{
    // One image, 128 x 128 positions into a 4 x 9 dw: the reduction splits into 4 chunks of 4096,
    // batched; a cast adds them.
    ConvGemmShape shape = padded3x3();
    shape.n = 1;
    shape.c = 1;
    shape.k = 4;
    shape.input = {128, 128};
    const auto calls = plan(shape, ConvDirection::BACKWARD_WEIGHTS);
    ASSERT_TRUE(calls.has_value());
    const auto& stage = calls->stages.at(0);
    ASSERT_EQ(stage.calls.size(), 1u);
    EXPECT_EQ(stage.calls[0].batchCount, 4);
    EXPECT_EQ(stage.calls[0].k, 4096);
    const auto* sum = stepArgs<StridedCastArgs>(stage.after.at(0));
    ASSERT_NE(sum, nullptr);
    EXPECT_EQ(sum->batches, 4);
}

TEST(TestConvRepackFormulation, LongReductionIntoA16BitOutputSumsFp32Partials)
{
    // One output pixel over 65536 terms: forward, a filter as large as its 64 x 64 image of 16
    // channels; backward data, a 1x1 dx of one channel from 65536 output channels. Into a bf16 or
    // half output the reduction runs as 16 chunks of 4096 whose fp32 partials a cast sums in
    // order (narrowPartials); an fp32 output, or the default, keeps the single GEMM.
    ConvGemmShape forward;
    forward.n = 1;
    forward.c = forward.k = 16;
    forward.input = {64, 64};
    forward.filter = {64, 64};
    ConvGemmShape backward;
    backward.n = 1;
    backward.c = 1;
    backward.k = 65536;
    backward.input = {1, 1};
    for(const auto& [base, direction, outUid] :
        {std::tuple{forward, ConvDirection::FORWARD, CONV_Y_UID},
         std::tuple{backward, ConvDirection::BACKWARD_DATA, CONV_X_UID}})
    {
        for(const auto dataType : {conv_data_objects::DataType::BFLOAT16,
                                   conv_data_objects::DataType::HALF,
                                   conv_data_objects::DataType::FLOAT})
        {
            auto shape = base;
            shape.dataType = dataType;
            SCOPED_TRACE(std::to_string(static_cast<int>(direction)) + " "
                         + std::to_string(static_cast<int>(dataType)));
            const auto problem = parse(shape, direction);
            const auto single = convGemmCalls(problem);
            ASSERT_TRUE(single.has_value());
            ASSERT_EQ(single->stages.at(0).calls.size(), 1u);
            EXPECT_EQ(single->stages.at(0).calls[0].k, 65536);
            const auto calls = convGemmCalls(
                problem, MAX_REPACK_WORKSPACE_BYTES, ColumnLayout::FOLLOW_Y, false, true);
            ASSERT_TRUE(calls.has_value());
            const bool narrow = dataType != conv_data_objects::DataType::FLOAT;
            const auto& stage = calls->stages.at(0);
            ASSERT_EQ(stage.calls.size(), 1u);
            const auto& call = stage.calls[0];
            EXPECT_EQ(call.d.dataType, conv_data_objects::DataType::FLOAT);
            if(narrow)
            {
                EXPECT_EQ(call.k, 4096);
                EXPECT_EQ(call.batchCount, 16);
                EXPECT_NE(call.d.uid, outUid);
                ASSERT_EQ(stage.after.size(), 1u);
                const auto* sum = stepArgs<StridedCastArgs>(stage.after[0]);
                ASSERT_NE(sum, nullptr);
                EXPECT_EQ(sum->batches, 16);
                EXPECT_EQ(stage.after[0].dstUid, outUid);
                EXPECT_EQ(stage.after[0].dataType, dataType);
            }
            else
            {
                EXPECT_EQ(call.k, 65536);
                EXPECT_EQ(call.d.uid, outUid);
                EXPECT_TRUE(stage.after.empty());
            }
        }
    }
    // Below 65536 terms the single GEMM stays even with narrowPartials.
    ConvGemmShape shorter = forward;
    shorter.dataType = conv_data_objects::DataType::BFLOAT16;
    shorter.input = shorter.filter = {32, 32};
    const auto kept = convGemmCalls(parse(shorter, ConvDirection::FORWARD),
                                    MAX_REPACK_WORKSPACE_BYTES,
                                    ColumnLayout::FOLLOW_Y,
                                    false,
                                    true);
    ASSERT_TRUE(kept.has_value());
    EXPECT_EQ(kept->stages.at(0).calls.at(0).k, 16384);
}

TEST(TestConvRepackFormulation, SplitReductionFillsTheGpuWithFp32Partials)
{
    // A 1x1 channels-last 320 -> 320 backward weights over 8 x 128 x 128 positions: one GEMM of
    // 9 output tiles behind a 131072 reduction. Split, it is 32 chunks of 4096 (32 x 9 tiles
    // toward SPLIT_TARGET_TILES), fp32 partials summed into w; by default, the one GEMM.
    ConvGemmShape shape = channelsLast({});
    shape.n = 8;
    shape.c = shape.k = 320;
    shape.input = {128, 128};
    shape.dataType = conv_data_objects::DataType::BFLOAT16;
    const auto problem = parse(shape, ConvDirection::BACKWARD_WEIGHTS);
    const auto direct = convGemmCalls(problem);
    ASSERT_TRUE(direct.has_value());
    ASSERT_EQ(direct->stages.size(), 1u);
    EXPECT_EQ(direct->stages[0].calls.at(0).k, 131072);
    EXPECT_TRUE(direct->workspace.empty());

    const auto split
        = convGemmCalls(problem, MAX_REPACK_WORKSPACE_BYTES, ColumnLayout::FOLLOW_Y, true);
    ASSERT_TRUE(split.has_value());
    const auto& stage = split->stages.at(0);
    ASSERT_EQ(stage.calls.size(), 1u);
    EXPECT_EQ(stage.calls[0].batchCount, 32);
    EXPECT_EQ(stage.calls[0].k, 4096);
    EXPECT_EQ(stage.calls[0].d.dataType, conv_data_objects::DataType::FLOAT);
    const auto* sum = stepArgs<StridedCastArgs>(stage.after.at(0));
    ASSERT_NE(sum, nullptr);
    EXPECT_EQ(sum->batches, 32);
    EXPECT_EQ(split->workspaceBytes, 32u * 320 * 320 * sizeof(float));

    // Enough tiles already (4096 x 4096 is 1024): split leaves the plan as it is.
    shape.c = shape.k = 4096;
    shape.input = {16, 16};
    const auto wide = parse(shape, ConvDirection::BACKWARD_WEIGHTS);
    const auto wideSplit
        = convGemmCalls(wide, MAX_REPACK_WORKSPACE_BYTES, ColumnLayout::FOLLOW_Y, true);
    ASSERT_TRUE(wideSplit.has_value());
    EXPECT_EQ(wideSplit->workspaceBytes, convGemmCalls(wide)->workspaceBytes);
    EXPECT_EQ(wideSplit->stages.at(0).calls.at(0).k,
              convGemmCalls(wide)->stages.at(0).calls.at(0).k);
}

TEST(TestConvRepackFormulation, OneByOneChannelsFirstBackwardWeightsBatchesImages)
{
    // A 1x1 channels-first backward weights of several images: one strided-batched GEMM over the
    // images into fp32 partials (one call per group), then one cast adding them in image order.
    for(const int64_t groups : {1, 2})
    {
        ConvGemmShape shape;
        shape.n = 3;
        shape.c = 8;
        shape.k = 6;
        shape.groups = groups;
        shape.dataType = conv_data_objects::DataType::HALF;
        const auto calls = plan(shape, ConvDirection::BACKWARD_WEIGHTS);
        ASSERT_TRUE(calls.has_value());
        EXPECT_EQ(calls->row, ConvGemmRow::ONE_BY_ONE);
        ASSERT_EQ(calls->stages.size(), 1u);
        const auto& stage = calls->stages[0];
        EXPECT_TRUE(stage.before.empty());
        ASSERT_EQ(stage.calls.size(), static_cast<size_t>(groups));
        ASSERT_EQ(calls->workspace.size(), 1u);
        const auto partial = calls->workspace[0].uid;
        EXPECT_EQ(calls->workspace[0].bytes,
                  static_cast<size_t>(3 * 6 * (8 / groups)) * sizeof(float));
        for(const auto& call : stage.calls)
        {
            EXPECT_EQ(call.batchCount, 3);
            EXPECT_EQ(call.d.uid, partial);
            EXPECT_EQ(call.d.dataType, conv_data_objects::DataType::FLOAT);
            EXPECT_FALSE(call.accumulate);
        }
        ASSERT_EQ(stage.after.size(), 1u);
        const auto* sum = stepArgs<StridedCastArgs>(stage.after[0]);
        ASSERT_NE(sum, nullptr);
        EXPECT_EQ(sum->batches, 3);
        EXPECT_EQ(sum->batchStride, 6 * (8 / groups));
        EXPECT_EQ(stage.after[0].srcUid, partial);
        EXPECT_EQ(stage.after[0].dstUid, CONV_W_UID);
    }

    // One image needs no reduction: the direct row writes w.
    ConvGemmShape single;
    single.n = 1;
    const auto direct = plan(single, ConvDirection::BACKWARD_WEIGHTS);
    ASSERT_TRUE(direct.has_value());
    EXPECT_FALSE(direct->hasRepack());
}

TEST(TestConvRepackFormulation, OneByOneBackwardWeightsLayoutsAndChunks)
{
    ConvGemmShape shape;
    shape.n = 3;
    shape.c = 8;
    shape.k = 6;
    const auto problem = parse(shape, ConvDirection::BACKWARD_WEIGHTS);

    // Channels-last: x and dy copied channels-last, then the one GEMM writes w.
    const auto staged
        = convGemmCalls(problem, MAX_REPACK_WORKSPACE_BYTES, ColumnLayout::CHANNELS_LAST);
    ASSERT_TRUE(staged.has_value());
    ASSERT_EQ(staged->stages.size(), 1u);
    ASSERT_EQ(staged->stages[0].before.size(), 2u);
    EXPECT_NE(stepArgs<StridedCopyArgs>(staged->stages[0].before[0]), nullptr);
    EXPECT_NE(stepArgs<StridedCopyArgs>(staged->stages[0].before[1]), nullptr);
    ASSERT_EQ(staged->stages[0].calls.size(), 1u);
    EXPECT_EQ(staged->stages[0].calls[0].d.uid, CONV_W_UID);
    EXPECT_EQ(staged->stages[0].calls[0].k, 3 * 5 * 7);
    EXPECT_EQ(staged->workspace.size(), 2u);

    // Channels-first: the batched partials.
    const auto batched
        = convGemmCalls(problem, MAX_REPACK_WORKSPACE_BYTES, ColumnLayout::CHANNELS_FIRST);
    ASSERT_TRUE(batched.has_value());
    ASSERT_EQ(batched->stages[0].calls.size(), 1u);
    EXPECT_EQ(batched->stages[0].calls[0].batchCount, 3);

    // A long image splits into chunks, each a batch entry: 2 images of 128 x 128 pixels are
    // 4 chunks of 4096 each, the loop over images (2) the fewer.
    ConvGemmShape tall = shape;
    tall.n = 2;
    tall.input = {128, 128};
    const auto split = convGemmCalls(parse(tall, ConvDirection::BACKWARD_WEIGHTS),
                                     MAX_REPACK_WORKSPACE_BYTES,
                                     ColumnLayout::CHANNELS_FIRST);
    ASSERT_TRUE(split.has_value());
    ASSERT_EQ(split->stages[0].calls.size(), 2u);
    EXPECT_EQ(split->stages[0].calls[0].batchCount, 4);
    EXPECT_EQ(split->stages[0].calls[0].k, 4096);
    const auto* sum = stepArgs<StridedCastArgs>(split->stages[0].after.at(0));
    ASSERT_NE(sum, nullptr);
    EXPECT_EQ(sum->batches, 8);
}

TEST(TestConvRepackFormulation, NarrowPerImageBackwardWeightsSumsInFp32)
{
    // fp16 / bf16: the per-image GEMMs add into an fp32 accumulator, which one cast rounds into
    // w, rather than rounding into w after every image.
    for(const auto dataType :
        {conv_data_objects::DataType::HALF, conv_data_objects::DataType::BFLOAT16})
    {
        auto shape = padded3x3();
        shape.dataType = dataType;
        const auto calls = plan(shape, ConvDirection::BACKWARD_WEIGHTS);
        ASSERT_TRUE(calls.has_value());
        ASSERT_EQ(calls->stages.size(), 2u);
        for(const auto& call : calls->stages[0].calls)
        {
            EXPECT_EQ(call.d.dataType, conv_data_objects::DataType::FLOAT);
            EXPECT_NE(call.d.uid, CONV_W_UID);
        }
        const auto& round = calls->stages[1];
        ASSERT_EQ(round.before.size(), 1u);
        EXPECT_NE(stepArgs<StridedCastArgs>(round.before[0]), nullptr);
        EXPECT_EQ(round.before[0].dstUid, CONV_W_UID);
    }
}

TEST(TestConvRepackFormulation, ImagesThatDoNotNestStageDyForBackwardWeights)
{
    // Forward and backward data still read y one image at a time; backward weights reduces
    // over every image's pixels, which needs them in one stride, so dy is staged.
    ConvGemmShape shape = channelsLast(padded3x3());
    shape.n = 3;
    shape.imagePadding = 5;
    for(const auto direction : ALL_DIRECTIONS)
    {
        const auto calls = plan(shape, direction);
        ASSERT_TRUE(calls.has_value());
        ASSERT_EQ(calls->stages.size(), 1u);
        const auto& stage = calls->stages[0];
        long copies = 0;
        for(const auto* steps : {&stage.before, &stage.after})
        {
            copies += std::count_if(steps->begin(), steps->end(), [](const RepackStep& step) {
                return std::holds_alternative<StridedCopyArgs>(step.args);
            });
        }
        const bool staged = direction == ConvDirection::BACKWARD_WEIGHTS;
        EXPECT_EQ(copies, staged ? 1 : 0);
        EXPECT_EQ(calls->workspace.size(), staged ? 2u : 1u);
    }
}

TEST(TestConvRepackFormulation, WorkspaceTensorsAreAlignedAndDisjoint)
{
    ConvGemmShape staged = channelsLast(padded3x3());
    staged.imagePadding = 5;
    const auto calls = plan(staged, ConvDirection::BACKWARD_WEIGHTS);
    ASSERT_TRUE(calls.has_value());
    ASSERT_EQ(calls->workspace.size(), 2u);
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
        EXPECT_EQ(stepArgs<Im2ColArgs>(calls->stages[0].before[0])->geometry.channelsInner, inner);
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
    const auto& geometry = stepArgs<Im2ColArgs>(calls->stages[0].before[0])->geometry;
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

/// A channels-last 3x3 convolution of @p n images whose column matrix is far over the cap.
ConvGemmShape oversized(int64_t n)
{
    ConvGemmShape shape = channelsLast(padded3x3());
    shape.n = n;
    shape.c = 512;
    shape.k = 64;
    shape.input = {128, 128};
    return shape;
}

TEST(TestConvRepackFormulation, SlicesImagesToFitTheCap)
{
    // 64 images of 128 * 128 rows x 512 * 9 columns: about 302 MB of fp32 column matrix each.
    for(const auto direction : ALL_DIRECTIONS)
    {
        const auto shape = oversized(64);
        const auto calls = plan(shape, direction);
        ASSERT_TRUE(calls.has_value());
        EXPECT_LE(calls->workspaceBytes, MAX_REPACK_WORKSPACE_BYTES);
        ASSERT_GT(calls->stages.size(), 1u);

        const auto problem = parse(shape, direction);
        const int64_t xImageBytes = problem.x.strides[0] * 4;
        int64_t covered = 0;
        for(const auto& stage : calls->stages)
        {
            const auto& steps
                = direction == ConvDirection::BACKWARD_DATA ? stage.after : stage.before;
            ASSERT_FALSE(steps.empty());
            const auto& step = steps[0];
            const auto& geometry = direction == ConvDirection::BACKWARD_DATA
                                       ? stepArgs<Col2ImArgs>(step)->geometry
                                       : stepArgs<Im2ColArgs>(step)->geometry;
            // Each slice starts where the last ended, in x's bytes.
            const int64_t xOffset = direction == ConvDirection::BACKWARD_DATA ? step.dstByteOffset
                                                                              : step.srcByteOffset;
            EXPECT_EQ(xOffset, covered * xImageBytes);
            covered += geometry.n;
        }
        EXPECT_EQ(covered, 64);
    }
}

TEST(TestConvRepackFormulation, SlicedBackwardWeightsAccumulates)
{
    // fp32 sums into w itself; the first slice writes, the rest add.
    const auto fp32 = plan(oversized(64), ConvDirection::BACKWARD_WEIGHTS);
    ASSERT_TRUE(fp32.has_value());
    ASSERT_GT(fp32->stages.size(), 1u);
    for(size_t i = 0; i < fp32->stages.size(); ++i)
    {
        for(const auto& call : fp32->stages[i].calls)
        {
            EXPECT_EQ(call.accumulate, i > 0);
            EXPECT_EQ(call.d.uid, CONV_W_UID);
        }
    }

    // bf16 sums into an fp32 accumulator in the workspace, rounded into w by a final stage.
    auto shape = oversized(64);
    shape.dataType = conv_data_objects::DataType::BFLOAT16;
    const auto bf16 = plan(shape, ConvDirection::BACKWARD_WEIGHTS);
    ASSERT_TRUE(bf16.has_value());
    ASSERT_GT(bf16->stages.size(), 2u);
    const auto& round = bf16->stages.back();
    EXPECT_TRUE(round.calls.empty());
    ASSERT_EQ(round.before.size(), 1u);
    ASSERT_NE(stepArgs<StridedCastArgs>(round.before[0]), nullptr);
    EXPECT_EQ(round.before[0].dstUid, CONV_W_UID);
    const auto accumulator = round.before[0].srcUid;
    for(size_t i = 0; i + 1 < bf16->stages.size(); ++i)
    {
        for(const auto& call : bf16->stages[i].calls)
        {
            EXPECT_EQ(call.accumulate, i > 0);
            EXPECT_EQ(call.d.uid, accumulator);
            EXPECT_EQ(call.d.dataType, conv_data_objects::DataType::FLOAT);
        }
    }
}

TEST(TestConvRepackFormulation, UnslicedPlansWriteTheirOutputOnce)
{
    for(const auto direction : ALL_DIRECTIONS)
    {
        const auto calls = plan(channelsLast(padded3x3()), direction);
        ASSERT_TRUE(calls.has_value());
        ASSERT_EQ(calls->stages.size(), 1u);
        for(const auto& call : calls->stages[0].calls)
        {
            EXPECT_FALSE(call.accumulate);
        }
    }
}

TEST(TestConvRepackFormulation, ALowerCapSlicesSmallShapes)
{
    // Every direction, staged (channels-first wrw) and not, fp32 and bf16 (an accumulator).
    for(const auto dataType :
        {conv_data_objects::DataType::FLOAT, conv_data_objects::DataType::BFLOAT16})
    {
        for(const bool last : {false, true})
        {
            for(const auto direction : ALL_DIRECTIONS)
            {
                for(const int64_t n : {2, 4})
                {
                    auto shape = last ? channelsLast(padded3x3()) : padded3x3();
                    shape.n = n;
                    shape.dataType = dataType;
                    const auto problem = parse(shape, direction);
                    const auto whole = im2colGemmCalls(problem);
                    ASSERT_TRUE(whole.has_value());
                    ASSERT_EQ(gemmStages(*whole), 1u);
                    // One byte short of the whole is the tightest cap: alignment padding is
                    // counted once per workspace tensor, never per image.
                    const auto sliced = im2colGemmCalls(problem, whole->workspaceBytes - 1);
                    ASSERT_TRUE(sliced.has_value()) << "n " << n << " channels last " << last;
                    EXPECT_GT(gemmStages(*sliced), 1u);
                    EXPECT_LT(sliced->workspaceBytes, whole->workspaceBytes);
                }
            }
        }
    }
}

TEST(TestConvRepackFormulation, BandsAnImageOverTheCap)
{
    // One image of 512 * 512 rows x 512 * 9 columns is 2.4 GB of bf16 column matrix (4.8 GB of
    // fp32 for backward data): it runs in bands of rows along its height. Strided and dilated so
    // backward data's bands read overlapping output rows.
    ConvGemmShape shape = oversized(2);
    shape.input = {512, 512};
    shape.stride = {2, 1};
    shape.dilation = {2, 1};
    shape.prePadding = shape.postPadding = {2, 1};
    shape.dataType = conv_data_objects::DataType::BFLOAT16;
    for(const auto direction : ALL_DIRECTIONS)
    {
        const auto calls = plan(shape, direction);
        ASSERT_TRUE(calls.has_value()) << static_cast<int>(direction);
        EXPECT_LE(calls->workspaceBytes, MAX_REPACK_WORKSPACE_BYTES);
        const auto problem = parse(shape, direction);
        const int64_t pad = problem.prePadding[0];
        const int64_t stride = problem.stride[0];
        const int64_t reach = (problem.filter[0] - 1) * problem.dilation[0];
        const bool backwardData = direction == ConvDirection::BACKWARD_DATA;

        // Per image, the bands tile the banded axis (output rows, or dx rows for backward
        // data) exactly once, in order.
        int64_t image = 0;
        int64_t covered = 0;
        size_t bands = 0;
        for(const auto& stage : calls->stages)
        {
            if(stage.calls.empty())
            {
                continue; // the bf16 backward-weights rounding stage
            }
            ++bands;
            const auto& step = backwardData ? stage.after.at(0) : stage.before.at(0);
            const auto& geometry = backwardData ? stepArgs<Col2ImArgs>(step)->geometry
                                                : stepArgs<Im2ColArgs>(step)->geometry;
            EXPECT_EQ(geometry.n, 1);
            if(backwardData)
            {
                // dx rows [covered, covered + rows), read from output rows [outFirst, ...).
                const int64_t rows = geometry.input[1];
                EXPECT_EQ(step.dstByteOffset,
                          (image * problem.x.strides[0] + covered * problem.x.strides[2]) * 2);
                const int64_t outFirst = (pad + covered - geometry.prePadding[1]) / stride;
                EXPECT_EQ(pad + covered - outFirst * stride, geometry.prePadding[1]);
                for(int64_t i = covered; i < covered + rows; ++i)
                {
                    for(int64_t tap = 0; tap * problem.dilation[0] <= reach; ++tap)
                    {
                        const int64_t s = i + pad - tap * problem.dilation[0];
                        if(s >= 0 && s % stride == 0 && s / stride < problem.output[0])
                        {
                            EXPECT_GE(s / stride, outFirst) << "row " << i;
                            EXPECT_LT(s / stride, outFirst + geometry.output[1]) << "row " << i;
                        }
                    }
                }
                covered += rows;
                if(covered == problem.input[0])
                {
                    ++image;
                    covered = 0;
                }
            }
            else
            {
                EXPECT_EQ(step.srcByteOffset, image * problem.x.strides[0] * 2);
                EXPECT_EQ(geometry.prePadding[1], pad - covered * stride);
                covered += geometry.output[1];
                if(covered == problem.output[0])
                {
                    ++image;
                    covered = 0;
                }
            }
            for(const auto& call : stage.calls)
            {
                // Backward weights sums its bands; the rest write disjoint rows once.
                EXPECT_EQ(call.accumulate,
                          direction == ConvDirection::BACKWARD_WEIGHTS && bands > 1);
            }
        }
        EXPECT_EQ(image, 2);
        EXPECT_EQ(covered, 0);
        EXPECT_GT(bands, 2u);
    }
}

TEST(TestConvRepackFormulation, LongNarrowBackwardWeightsSumsInFp32)
{
    // A 1x1 channels-last fp16 backward weights over 16.7 M positions of 512 channels: 16 GiB of
    // x along the reduction, which rocBLAS would split, rounding each piece into fp16 w. (A
    // small output is chunked into fp32 partials instead; 512 x 512 is not small.)
    ConvGemmShape shape = channelsLast({});
    shape.n = 64;
    shape.c = shape.k = 512;
    shape.input = {512, 512};
    shape.dataType = conv_data_objects::DataType::HALF;
    const auto calls = plan(shape, ConvDirection::BACKWARD_WEIGHTS);
    ASSERT_TRUE(calls.has_value());
    ASSERT_EQ(calls->stages.size(), 2u);
    ASSERT_EQ(calls->workspace.size(), 1u);
    const auto accumulator = calls->workspace[0].uid;
    for(const auto& call : calls->stages[0].calls)
    {
        EXPECT_EQ(call.d.uid, accumulator);
        EXPECT_EQ(call.d.dataType, conv_data_objects::DataType::FLOAT);
    }
    const auto& round = calls->stages[1];
    EXPECT_TRUE(round.calls.empty());
    ASSERT_EQ(round.before.size(), 1u);
    ASSERT_NE(stepArgs<StridedCastArgs>(round.before[0]), nullptr);
    EXPECT_EQ(round.before[0].srcUid, accumulator);
    EXPECT_EQ(round.before[0].dstUid, CONV_W_UID);
    EXPECT_EQ(calls->workspace[0].bytes, 512u * 512 * sizeof(float));

    // fp32 needs no copy, and a short reduction writes fp16 w directly.
    shape.dataType = conv_data_objects::DataType::FLOAT;
    const auto fp32 = plan(shape, ConvDirection::BACKWARD_WEIGHTS);
    ASSERT_TRUE(fp32.has_value());
    EXPECT_EQ(fp32->stages.size(), 1u);
    shape.dataType = conv_data_objects::DataType::HALF;
    shape.input = {64, 64};
    const auto shortReduction = plan(shape, ConvDirection::BACKWARD_WEIGHTS);
    ASSERT_TRUE(shortReduction.has_value());
    EXPECT_EQ(shortReduction->stages.size(), 1u);
}

TEST(TestConvRepackFormulation, DeclinesWhenOneRowExceedsTheCap)
{
    // One output row of 2^20 positions x 512 * 9 columns is 9.7 GB of bf16 column matrix.
    ConvGemmShape shape = oversized(1);
    shape.input = {4, 1 << 20};
    shape.dataType = conv_data_objects::DataType::BFLOAT16;
    for(const auto direction : ALL_DIRECTIONS)
    {
        EXPECT_FALSE(plan(shape, direction).has_value());
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
