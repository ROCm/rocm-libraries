// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "common/ConvEpilogueFormulation.hpp"
#include "common/ConvRepackFormulation.hpp"
#include "engines/conv/ConvEpilogueTestGraphs.hpp"

/**
 * @file TestConvEpilogueFormulation.cpp
 * @brief Host-only checks of explicit-GEMM conv fusion: which fused graphs the engines read
 *        (the field-by-field gates are the shared parser's, tested with the hip-kernel-provider),
 *        and where each plan runs the epilogue -- in hipBLASLt's GEMM, in the plan's own last
 *        pass, or in one extra pass -- always one fp32 rounding into y.
 */
namespace
{

using namespace hipblaslt_plugin;
using namespace hipblaslt_plugin::test;
using conv_data_objects::DataType;

ConvGemmShape oneByOneNhwc()
{
    ConvGemmShape shape;
    shape.n = 2;
    shape.c = 16;
    shape.k = 24;
    shape.input = {5, 7};
    shape.xChannelsLast = shape.wChannelsLast = shape.yChannelsLast = true;
    shape.dataType = DataType::HALF;
    return shape;
}

ConvGemmShape threeByThree(bool channelsLast)
{
    ConvGemmShape shape = oneByOneNhwc();
    shape.filter = {3, 3};
    shape.prePadding = {1, 1};
    shape.postPadding = {1, 1};
    shape.xChannelsLast = shape.wChannelsLast = shape.yChannelsLast = channelsLast;
    return shape;
}

std::optional<ConvProblem> parse(const ConvGemmShape& shape, const std::vector<Step>& steps)
{
    const FusedConvGraph graph(shape, steps, DataType::HALF);
    return parseConvProblem(graph.graph(), ConvDirection::FORWARD);
}

std::optional<ConvGemmPlanCalls> fused(const ConvGemmShape& shape,
                                       const std::vector<Step>& steps,
                                       bool native,
                                       ColumnLayout layout = ColumnLayout::FOLLOW_Y)
{
    const auto problem = parse(shape, steps);
    if(!problem.has_value())
    {
        return std::nullopt;
    }
    auto calls = convGemmCalls(*problem, MAX_REPACK_WORKSPACE_BYTES, layout);
    if(!calls.has_value())
    {
        return std::nullopt;
    }
    return fuseEpilogue(*problem, std::move(*calls), native);
}

TEST(TestConvEpilogueFormulation, ParsesTheChainAndWritesItsOutput)
{
    const auto problem = parse(oneByOneNhwc(), {scale(), bias(DataType::HALF), relu()});
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->y.uid, epilogueOutputUid(3));
    EXPECT_EQ(problem->epilogue.ops.size(), 3U);
    EXPECT_EQ(problem->epilogue.operands.size(), 2U);
    EXPECT_EQ(problem->epilogue.convOutputUid, CONV_Y_UID);
}

TEST(TestConvEpilogueFormulation, AcceptsTheCudnnFrontendHalfGraph)
{
    // fp16_conv_bias_relu: NCHW half, intermediates half, bias half under a float node compute,
    // relu with lower clip 0 and slope 0 set explicitly.
    ConvGemmShape shape;
    shape.n = 8;
    shape.c = 32;
    shape.k = 64;
    shape.input = {16, 16};
    shape.filter = {3, 3};
    shape.prePadding = {1, 1};
    shape.postPadding = {1, 1};
    shape.dataType = DataType::HALF;
    const auto problem = parse(shape, {bias(DataType::HALF), relu(0.0F, std::nullopt, 0.0F)});
    ASSERT_TRUE(problem.has_value());
    const auto calls = fused(shape, {bias(DataType::HALF), relu(0.0F, std::nullopt, 0.0F)}, true);
    ASSERT_TRUE(calls.has_value());
    EXPECT_NE(calls->epilogueForm, EpilogueForm::NONE);
}

TEST(TestConvEpilogueFormulation, DeclinesAChainAfterBackwardData)
{
    const FusedConvGraph graph(
        oneByOneNhwc(), {relu()}, DataType::FLOAT, ConvDirection::BACKWARD_DATA);
    for(const auto direction :
        {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA, ConvDirection::BACKWARD_WEIGHTS})
    {
        EXPECT_FALSE(parseConvProblem(graph.graph(), direction).has_value());
    }
}

TEST(TestConvEpilogueFormulation, DeclinesWhatTheSharedParserDeclines)
{
    EXPECT_FALSE(parse(oneByOneNhwc(), {unary(PointwiseMode::EXP)}).has_value());
    Step leakyOnAdd = bias();
    leakyOnAdd.slope = 0.1F;
    EXPECT_FALSE(parse(oneByOneNhwc(), {leakyOnAdd}).has_value());
}

TEST(TestConvEpilogueFormulation, UnfusedPlanIsUnchanged)
{
    const ConvGemmGraph graph(oneByOneNhwc(), ConvDirection::FORWARD);
    const auto problem = parseConvProblem(graph.graph(), ConvDirection::FORWARD);
    ASSERT_TRUE(problem.has_value());
    ASSERT_TRUE(problem->epilogue.empty());
    auto calls = convGemmCalls(*problem);
    ASSERT_TRUE(calls.has_value());
    const auto before = *calls;
    const auto after = fuseEpilogue(*problem, std::move(*calls), true);
    ASSERT_TRUE(after.has_value());
    EXPECT_EQ(after->epilogueForm, EpilogueForm::NONE);
    EXPECT_EQ(after->stages.size(), before.stages.size());
    EXPECT_EQ(after->workspaceBytes, before.workspaceBytes);
    EXPECT_FALSE(after->stages[0].calls[0].epilogue.has_value());
}

// ---------------------------------------------------------------------------
// Where the epilogue runs
// ---------------------------------------------------------------------------

struct NativeCase
{
    std::string name;
    std::vector<Step> steps;
    int mode; ///< 0: no native form
};

std::vector<NativeCase> nativeCases()
{
    Step swish1 = unary(PointwiseMode::SWISH_FWD);
    swish1.swishBeta = 1.0F;
    Step swish2 = unary(PointwiseMode::SWISH_FWD);
    swish2.swishBeta = 2.0F;
    Step bf16Bias = bias(DataType::BFLOAT16);
    return {
        {"Bias", {bias(DataType::HALF)}, NATIVE_EPILOGUE_BIAS},
        {"Fp32Bias", {bias(DataType::FLOAT)}, NATIVE_EPILOGUE_BIAS},
        {"Sigmoid", {unary(PointwiseMode::SIGMOID_FWD)}, NATIVE_EPILOGUE_SIGMOID},
        // hipBLASLt's activations differ from the chain on NaN or infinity: the pass applies
        // them.
        {"Relu", {relu()}, 0},
        {"BiasRelu", {bias(DataType::HALF), relu(0.0F, std::nullopt, 0.0F)}, 0},
        {"BiasClamp", {bias(), relu(-1.0F, 6.0F)}, 0},
        {"ClippedRelu", {relu(std::nullopt, 6.0F)}, 0},
        {"BiasGeluTanh", {bias(), unary(PointwiseMode::GELU_APPROX_TANH_FWD)}, 0},
        {"BiasSilu", {bias(), unary(PointwiseMode::SWISH_FWD)}, 0},
        {"Swish1", {swish1}, 0},
        {"Swish2", {swish2}, 0},
        {"BiasSigmoid", {bias(), unary(PointwiseMode::SIGMOID_FWD)}, 0},
        {"ExactGelu", {bias(), unary(PointwiseMode::GELU_FWD)}, 0},
        {"Tanh", {unary(PointwiseMode::TANH_FWD)}, 0},
        {"Elu", {unary(PointwiseMode::ELU_FWD)}, 0},
        {"LeakyRelu", {relu(std::nullopt, std::nullopt, 0.1F)}, 0},
        {"ReluLowerClip", {relu(0.5F)}, 0},
        {"ScaleBiasRelu", {scale(), bias(), relu()}, 0},
        {"Bf16BiasOnHalf", {bf16Bias}, 0},
        {"BiasBias", {bias(), bias()}, 0},
    };
}

class TestConvEpilogueNative : public ::testing::TestWithParam<NativeCase>
{
};

TEST_P(TestConvEpilogueNative, MapsExactlyTheLibraryEpilogues)
{
    const auto& param = GetParam();
    const auto problem = parse(oneByOneNhwc(), param.steps);
    ASSERT_TRUE(problem.has_value());
    const auto native = nativeEpilogueFor(*problem);
    ASSERT_EQ(native.has_value(), param.mode != 0);
    if(native.has_value())
    {
        EXPECT_EQ(native->mode, param.mode);
    }

    // The 1x1 channels-last plan: the GEMM's rows are y's channels.
    const auto calls = fused(oneByOneNhwc(), param.steps, true);
    ASSERT_TRUE(calls.has_value());
    EXPECT_EQ(calls->epilogueForm,
              param.mode != 0 ? EpilogueForm::NATIVE : EpilogueForm::EXTRA_PASS);
    // rocBLAS: never native.
    const auto passes = fused(oneByOneNhwc(), param.steps, false);
    ASSERT_TRUE(passes.has_value());
    EXPECT_EQ(passes->epilogueForm, EpilogueForm::EXTRA_PASS);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvEpilogueNative,
                         ::testing::ValuesIn(nativeCases()),
                         [](const auto& info) { return info.param.name; });

TEST(TestConvEpilogueFormulation, NativeBiasRidesOnTheGemm)
{
    const auto problem = parse(oneByOneNhwc(), {bias(DataType::HALF)});
    ASSERT_TRUE(problem.has_value());
    const auto native = nativeEpilogueFor(*problem);
    ASSERT_TRUE(native.has_value());
    ASSERT_TRUE(native->biasUid.has_value());
    EXPECT_EQ(*native->biasUid, EPI_OPERAND_UID);
    EXPECT_EQ(native->biasType, DataType::HALF);

    const auto calls = fused(oneByOneNhwc(), {bias(DataType::HALF)}, true);
    ASSERT_TRUE(calls.has_value());
    ASSERT_EQ(calls->stages.size(), 1U);
    for(const auto& call : calls->stages[0].calls)
    {
        ASSERT_TRUE(call.epilogue.has_value());
        EXPECT_EQ(call.d.uid, epilogueOutputUid(1));
        EXPECT_EQ(call.m, 24) << "the GEMM's rows are the output channels";
    }
}

TEST(TestConvEpilogueFormulation, ImplicitColumnsChannelsLastIsNative)
{
    // im2col into a channels-last y: the GEMM writes y with channels as rows.
    const auto calls = fused(threeByThree(true), {bias(DataType::HALF)}, true);
    ASSERT_TRUE(calls.has_value());
    EXPECT_EQ(calls->epilogueForm, EpilogueForm::NATIVE);
}

TEST(TestConvEpilogueFormulation, GroupsTakeThePass)
{
    // hipBLASLt has no per-batch bias stride: grouped bias goes through the pass.
    ConvGemmShape shape = oneByOneNhwc();
    shape.groups = 2;
    const auto calls = fused(shape, {bias(DataType::HALF)}, true);
    ASSERT_TRUE(calls.has_value());
    EXPECT_EQ(calls->epilogueForm, EpilogueForm::EXTRA_PASS);
}

TEST(TestConvEpilogueFormulation, ChannelsFirstOneByOneTakesAnExtraPass)
{
    // Channels-first: the GEMM's rows are pixels, so the bias is not a row vector.
    ConvGemmShape shape = oneByOneNhwc();
    shape.xChannelsLast = shape.wChannelsLast = shape.yChannelsLast = false;
    const auto calls = fused(shape, {bias(DataType::HALF), relu()}, true);
    ASSERT_TRUE(calls.has_value());
    EXPECT_EQ(calls->epilogueForm, EpilogueForm::EXTRA_PASS);
    // y's fp32 copy in the workspace, and a last stage whose cast applies the epilogue.
    const auto& last = calls->stages.back();
    ASSERT_EQ(last.after.size(), 1U);
    EXPECT_TRUE(std::holds_alternative<StridedCastEpilogueArgs>(last.after[0].args));
    EXPECT_EQ(last.after[0].dstUid, epilogueOutputUid(2));
    EXPECT_EQ(calls->stages.front().calls.front().d.dataType, DataType::FLOAT);
    EXPECT_NE(calls->epilogue, nullptr);
    const auto& cast = std::get<StridedCastEpilogueArgs>(last.after[0].args);
    // y is dense and its fp32 copy shares its layout: one flat, coalesced walk.
    EXPECT_EQ(cast.epilogue.channelAxis, -1);
    EXPECT_EQ(cast.epilogue.flatChannelStride, 5 * 7);
    EXPECT_EQ(cast.epilogue.flatChannels, 24);
    EXPECT_EQ(cast.cast.rank, 1);
    EXPECT_EQ(cast.epilogue.ops, 2);
    EXPECT_EQ(cast.epilogue.op[0], CONV_REPACK_EPI_ADD_OPERAND);
    EXPECT_EQ(cast.epilogue.op[1], CONV_REPACK_EPI_RELU);
}

TEST(TestConvEpilogueFormulation, StagedChannelsLastFusesIntoItsCopy)
{
    // NCHW 3x3 under the staged channels-last layout: the copy back to y becomes the epilogue's
    // cast of an fp32 staging -- no extra pass.
    const auto calls = fused(threeByThree(false),
                             {bias(DataType::HALF), relu()},
                             true,
                             ColumnLayout::STAGED_CHANNELS_LAST);
    ASSERT_TRUE(calls.has_value());
    EXPECT_EQ(calls->epilogueForm, EpilogueForm::FUSED_PASS);
    int casts = 0;
    for(const auto& stage : calls->stages)
    {
        for(const auto& step : stage.after)
        {
            EXPECT_FALSE(std::holds_alternative<StridedCopyArgs>(step.args)
                         && step.dstUid == epilogueOutputUid(2))
                << "the copy into y is the epilogue's cast";
            casts += std::holds_alternative<StridedCastEpilogueArgs>(step.args) ? 1 : 0;
        }
    }
    EXPECT_GE(casts, 1);
}

TEST(TestConvEpilogueFormulation, ExtraPassAddsYsFp32Copy)
{
    ConvGemmShape shape = oneByOneNhwc();
    shape.xChannelsLast = shape.wChannelsLast = shape.yChannelsLast = false;
    const FusedConvGraph graph(shape, {relu()}, DataType::HALF);
    const auto problem = parseConvProblem(graph.graph(), ConvDirection::FORWARD);
    ASSERT_TRUE(problem.has_value());
    auto calls = convGemmCalls(*problem);
    ASSERT_TRUE(calls.has_value());
    const size_t before = calls->workspaceBytes;
    const auto after = fuseEpilogue(*problem, std::move(*calls), false);
    ASSERT_TRUE(after.has_value());
    const size_t yBytes = size_t{2} * 24 * 5 * 7 * sizeof(float);
    EXPECT_GE(after->workspaceBytes, before + yBytes);
}

TEST(TestConvEpilogueFormulation, Fp32OutputReductionsSplitInOrder)
{
    // The extra pass of a channels-first 3x3: the GEMM writes y's fp32 copy from fp16 operands.
    const auto calls = fused(threeByThree(false), {bias(DataType::HALF), relu()}, false);
    ASSERT_TRUE(calls.has_value());
    const auto& before = calls->stages.front().calls;
    ASSERT_EQ(before.size(), 1U);
    const GemmCall whole = before.front();
    ASSERT_EQ(whole.d.dataType, DataType::FLOAT);
    const auto split = splitFp32OutputReductions(*calls, 4);
    const auto& parts = split.stages.front().calls;
    ASSERT_EQ(parts.size(), 4U);
    int64_t covered = 0;
    for(size_t i = 0; i < parts.size(); ++i)
    {
        const auto& part = parts[i];
        EXPECT_EQ(part.accumulate, i > 0) << "each part after the first adds to the fp32 sum";
        EXPECT_EQ(part.d.uid, whole.d.uid);
        EXPECT_EQ(part.d.offset, whole.d.offset);
        EXPECT_EQ(part.m, whole.m);
        EXPECT_EQ(part.n, whole.n);
        EXPECT_EQ(part.a.offset,
                  whole.a.offset
                      + (whole.transposeA ? covered : covered * whole.a.leadingDimension));
        EXPECT_EQ(part.b.offset,
                  whole.b.offset
                      + (whole.transposeB ? covered * whole.b.leadingDimension : covered));
        covered += part.k;
    }
    EXPECT_EQ(covered, whole.k);
    // The epilogue pass after it is unchanged.
    EXPECT_EQ(split.stages.size(), calls->stages.size());
}

} // namespace
