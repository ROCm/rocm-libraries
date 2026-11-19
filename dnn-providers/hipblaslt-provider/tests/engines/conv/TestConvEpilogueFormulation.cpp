// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
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
    const FusedConvGraph graph(shape, steps, shape.dataType);
    return parseConvProblem(graph.graph(), ConvDirection::FORWARD);
}

/// The engines' GEMM epilogue support: hipBLASLt's, rocBLAS's (an addend only), the opt-in
/// library-activation engine's.
constexpr GemmEpilogueSupport HIPBLASLT{true, false};
constexpr GemmEpilogueSupport ROCBLAS{false, false};
constexpr GemmEpilogueSupport LIBRARY{true, true};

std::optional<ConvGemmPlanCalls> fused(const ConvGemmShape& shape,
                                       const std::vector<Step>& steps,
                                       std::optional<GemmEpilogueSupport> gemm,
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
    return fuseEpilogue(*problem, std::move(*calls), gemm);
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
    const auto calls
        = fused(shape, {bias(DataType::HALF), relu(0.0F, std::nullopt, 0.0F)}, HIPBLASLT);
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
    const auto after = fuseEpilogue(*problem, std::move(*calls), HIPBLASLT);
    ASSERT_TRUE(after.has_value());
    EXPECT_EQ(after->epilogueForm, EpilogueForm::NONE);
    EXPECT_EQ(after->stages.size(), before.stages.size());
    EXPECT_EQ(after->workspaceBytes, before.workspaceBytes);
    EXPECT_FALSE(after->stages[0].calls[0].epilogue.has_value());
}

// ---------------------------------------------------------------------------
// Where the epilogue runs
// ---------------------------------------------------------------------------

ConvGemmShape oneByOneNchw(DataType type = DataType::HALF)
{
    ConvGemmShape shape = oneByOneNhwc();
    shape.xChannelsLast = shape.wChannelsLast = shape.yChannelsLast = false;
    shape.dataType = type;
    return shape;
}

std::optional<ConvProblem> parseAs(const ConvGemmShape& shape, const std::vector<Step>& steps)
{
    const FusedConvGraph graph(shape, steps, DataType::FLOAT);
    return parseConvProblem(graph.graph(), ConvDirection::FORWARD);
}

/// Which GEMM form each chain takes: the bias, the activation hipBLASLt applies, the relu /
/// clamp tail; per support.
struct MappingCase
{
    std::string name;
    std::vector<Step> steps;
    DataType yType = DataType::HALF;
    /// gemmEpilogueFor under hipBLASLt's exact support: mapped, its activation, a tail.
    bool exact = false;
    int exactActivation = 0;
    bool tail = false;
    /// Under the library-activation support.
    bool library = false;
    int libraryActivation = 0;
    /// rocBLAS (an addend and the tail only).
    bool rocblas = false;
};

std::vector<MappingCase> mappingCases()
{
    Step swish1 = unary(PointwiseMode::SWISH_FWD);
    swish1.swishBeta = 1.0F;
    Step swish2 = unary(PointwiseMode::SWISH_FWD);
    swish2.swishBeta = 2.0F;
    const auto sigmoid = unary(PointwiseMode::SIGMOID_FWD);
    const auto geluTanh = unary(PointwiseMode::GELU_APPROX_TANH_FWD);
    constexpr int RELU = NATIVE_EPILOGUE_RELU;
    constexpr int CLAMP = NATIVE_EPILOGUE_CLAMP;
    constexpr int GELU = NATIVE_EPILOGUE_GELU;
    constexpr int SWISH = NATIVE_EPILOGUE_SWISH;
    constexpr int SIGMOID = NATIVE_EPILOGUE_SIGMOID;
    const auto h = DataType::HALF;
    const auto b = DataType::BFLOAT16;
    const auto f = DataType::FLOAT;
    return {
        // name, steps, y, exact, exactActivation, tail, library, libraryActivation, rocblas
        {"Bias", {bias(h)}, h, true, 0, false, true, 0, true},
        {"Fp32Bias", {bias(f)}, h, true, 0, false, true, 0, true},
        {"Bf16BiasOnBf16", {bias(b)}, b, true, 0, false, true, 0, true},
        {"Bf16BiasOnHalf", {bias(b)}, h, false, 0, false, false, 0, false},
        {"BiasRelu", {bias(h), relu()}, h, true, 0, true, true, RELU, true},
        {"Relu", {relu()}, h, true, 0, true, true, RELU, true},
        {"ExplicitZeroRelu", {relu(0.0F, std::nullopt, 0.0F)}, h, true, 0, true, true, RELU, true},
        {"BiasClampHalfExact", {bias(h), relu(-1.0F, 6.0F)}, h, true, 0, true, true, CLAMP, true},
        {"ClampBoundNotHalf", {bias(h), relu(0.0F, 0.1F)}, h, false, 0, false, true, CLAMP, false},
        {"ClampBoundNotBf16", {bias(b), relu(0.0F, 0.1F)}, b, false, 0, false, true, CLAMP, false},
        {"ClampBoundFp32", {bias(f), relu(0.0F, 0.1F)}, f, true, 0, true, true, CLAMP, true},
        {"ClampHalfTooLarge", {relu(0.0F, 70000.0F)}, h, false, 0, false, true, CLAMP, false},
        {"ClampHalfSubnormal", {relu(0.0F, 0x1p-20F)}, h, true, 0, true, true, CLAMP, true},
        {"ReluLowerClip", {relu(0.5F)}, h, true, 0, true, true, CLAMP, true},
        {"LeakyRelu",
         {relu(std::nullopt, std::nullopt, 0.1F)},
         h,
         false,
         0,
         false,
         false,
         0,
         false},
        {"Sigmoid16", {sigmoid}, h, true, SIGMOID, false, true, SIGMOID, false},
        {"SigmoidFp32", {sigmoid}, f, false, 0, false, true, SIGMOID, false},
        {"BiasSigmoid", {bias(h), sigmoid}, h, true, SIGMOID, false, true, SIGMOID, false},
        {"BiasGeluTanh", {bias(h), geluTanh}, h, false, 0, false, true, GELU, false},
        {"BiasSilu",
         {bias(h), unary(PointwiseMode::SWISH_FWD)},
         h,
         false,
         0,
         false,
         true,
         SWISH,
         false},
        {"Swish1", {swish1}, h, false, 0, false, true, SWISH, false},
        {"Swish2", {swish2}, h, false, 0, false, false, 0, false},
        {"ExactGelu",
         {bias(h), unary(PointwiseMode::GELU_FWD)},
         h,
         false,
         0,
         false,
         false,
         0,
         false},
        {"Tanh", {unary(PointwiseMode::TANH_FWD)}, h, false, 0, false, false, 0, false},
        {"Elu", {unary(PointwiseMode::ELU_FWD)}, h, false, 0, false, false, 0, false},
        {"ScaleBiasRelu", {scale(), bias(), relu()}, h, false, 0, false, false, 0, false},
        {"BiasBias", {bias(h), bias(h)}, h, false, 0, false, false, 0, false},
        {"ReluBias", {relu(), bias(h)}, h, false, 0, false, false, 0, false},
    };
}

class TestConvEpilogueGemm : public ::testing::TestWithParam<MappingCase>
{
};

TEST_P(TestConvEpilogueGemm, MapsWhatTheGemmComputesExactly)
{
    const auto& param = GetParam();
    ConvGemmShape shape = oneByOneNhwc();
    shape.dataType = param.yType;
    const auto problem = parseAs(shape, param.steps);
    ASSERT_TRUE(problem.has_value());
    const auto exact = gemmEpilogueFor(*problem, HIPBLASLT);
    ASSERT_EQ(exact.has_value(), param.exact);
    if(exact.has_value())
    {
        EXPECT_EQ(exact->activation, param.exactActivation);
        EXPECT_EQ(exact->clamp.has_value(), param.tail);
        EXPECT_EQ(exact->bias.has_value(), param.steps.front().mode == PointwiseMode::ADD);
    }
    const auto library = gemmEpilogueFor(*problem, LIBRARY);
    ASSERT_EQ(library.has_value(), param.library);
    if(library.has_value())
    {
        EXPECT_EQ(library->activation, param.libraryActivation);
        EXPECT_FALSE(library->clamp.has_value()) << "the library clamps in the GEMM";
    }
    EXPECT_EQ(usesLibraryActivation(*problem),
              param.library && param.libraryActivation != param.exactActivation);
    // rocBLAS: an addend and the tail, never a library epilogue.
    const auto rocblas = gemmEpilogueFor(*problem, ROCBLAS);
    ASSERT_EQ(rocblas.has_value(), param.rocblas);
    if(rocblas.has_value())
    {
        EXPECT_EQ(rocblas->activation, 0);
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvEpilogueGemm,
                         ::testing::ValuesIn(mappingCases()),
                         [](const auto& info) { return info.param.name; });

/// Every call writing y in @p plan.
std::vector<GemmCall> yCalls(const ConvGemmPlanCalls& plan, int64_t yUid)
{
    std::vector<GemmCall> calls;
    for(const auto& stage : plan.stages)
    {
        for(const auto& call : stage.calls)
        {
            if(call.d.uid == yUid)
            {
                calls.push_back(call);
            }
        }
    }
    return calls;
}

/// The plan's relu / clamp tail steps.
std::vector<ClampArgs> tails(const ConvGemmPlanCalls& plan)
{
    std::vector<ClampArgs> found;
    for(const auto& stage : plan.stages)
    {
        for(const auto* steps : {&stage.before, &stage.after})
        {
            for(const auto& step : *steps)
            {
                if(const auto* clamp = std::get_if<ClampArgs>(&step.args))
                {
                    EXPECT_EQ(step.srcUid, step.dstUid) << "the tail clamps y in place";
                    found.push_back(*clamp);
                }
            }
        }
    }
    return found;
}

TEST(TestConvEpilogueFormulation, NativeBiasRidesOnTheGemm)
{
    const auto problem = parse(oneByOneNhwc(), {bias(DataType::HALF)});
    ASSERT_TRUE(problem.has_value());
    const auto mapped = gemmEpilogueFor(*problem, HIPBLASLT);
    ASSERT_TRUE(mapped.has_value());
    ASSERT_TRUE(mapped->bias.has_value());
    EXPECT_EQ(mapped->bias->uid, EPI_OPERAND_UID);
    EXPECT_EQ(mapped->bias->dataType, DataType::HALF);

    const auto calls = fused(oneByOneNhwc(), {bias(DataType::HALF)}, HIPBLASLT);
    ASSERT_TRUE(calls.has_value());
    EXPECT_EQ(calls->epilogueForm, EpilogueForm::NATIVE);
    ASSERT_EQ(calls->stages.size(), 1U);
    for(const auto& call : calls->stages[0].calls)
    {
        ASSERT_TRUE(call.epilogue.has_value());
        EXPECT_EQ(call.epilogue->mode, NATIVE_EPILOGUE_BIAS);
        EXPECT_EQ(call.epilogue->biasUid, EPI_OPERAND_UID);
        EXPECT_FALSE(call.c.has_value());
        EXPECT_EQ(call.d.uid, epilogueOutputUid(1));
        EXPECT_EQ(call.m, 24) << "the GEMM's rows are the output channels";
    }
    EXPECT_FALSE(calls->hasRepack()) << "no workspace, no pass";
}

TEST(TestConvEpilogueFormulation, ChannelsLastBiasReluIsBiasInTheGemmThenTheTail)
{
    const auto calls = fused(oneByOneNhwc(), {bias(DataType::HALF), relu()}, HIPBLASLT);
    ASSERT_TRUE(calls.has_value());
    EXPECT_EQ(calls->epilogueForm, EpilogueForm::NATIVE);
    const auto written = yCalls(*calls, epilogueOutputUid(2));
    ASSERT_EQ(written.size(), 1U);
    ASSERT_TRUE(written[0].epilogue.has_value());
    EXPECT_EQ(written[0].epilogue->mode, NATIVE_EPILOGUE_BIAS) << "relu is not hipBLASLt's";
    EXPECT_EQ(written[0].d.dataType, DataType::HALF) << "y written once, in its type";
    const auto clamps = tails(*calls);
    ASSERT_EQ(clamps.size(), 1U);
    EXPECT_EQ(clamps[0].lower, 0.0F);
    EXPECT_TRUE(std::isinf(clamps[0].upper));
    EXPECT_EQ(clamps[0].total, int64_t{2} * 24 * 5 * 7);
    EXPECT_EQ(calls->workspaceBytes, 0U);
    // The library-activation engine: relu in the GEMM, no tail.
    const auto library = fused(oneByOneNhwc(), {bias(DataType::HALF), relu()}, LIBRARY);
    ASSERT_TRUE(library.has_value());
    const auto libraryCalls = yCalls(*library, epilogueOutputUid(2));
    ASSERT_EQ(libraryCalls.size(), 1U);
    EXPECT_EQ(libraryCalls[0].epilogue->mode, NATIVE_EPILOGUE_RELU_BIAS);
    EXPECT_TRUE(tails(*library).empty());
}

TEST(TestConvEpilogueFormulation, ChannelsFirstBiasIsABroadcastAddend)
{
    // Channels-first 1x1: the GEMM's rows are pixels and its columns the output channels, so
    // the bias is added from a matrix of it broadcast along the rows: one GEMM, one rounding.
    for(const auto& gemm : {HIPBLASLT, ROCBLAS})
    {
        const auto calls = fused(oneByOneNchw(), {bias(DataType::HALF), relu()}, gemm);
        ASSERT_TRUE(calls.has_value());
        EXPECT_EQ(calls->epilogueForm, EpilogueForm::NATIVE);
        const auto written = yCalls(*calls, epilogueOutputUid(2));
        ASSERT_EQ(written.size(), 1U);
        const auto& call = written[0];
        EXPECT_EQ(call.n, 24) << "the columns are the output channels";
        EXPECT_EQ(call.m, 5 * 7);
        EXPECT_FALSE(call.epilogue.has_value());
        ASSERT_TRUE(call.c.has_value());
        EXPECT_EQ(call.c->dataType, DataType::HALF);
        EXPECT_EQ(call.c->leadingDimension, call.m);
        EXPECT_EQ(call.c->batchStride, 0) << "one matrix for every image";
        EXPECT_FALSE(call.accumulate);
        EXPECT_EQ(call.d.dataType, DataType::HALF);
        // The fill: the bias broadcast along the matrix's rows.
        const auto& fill = calls->stages.front().before.front();
        const auto* broadcast = std::get_if<BroadcastArgs>(&fill.args);
        ASSERT_NE(broadcast, nullptr);
        EXPECT_EQ(fill.srcUid, EPI_OPERAND_UID);
        EXPECT_EQ(fill.dstUid, call.c->uid);
        EXPECT_EQ(fill.dataType, DataType::HALF);
        EXPECT_EQ(broadcast->rows, call.m);
        EXPECT_EQ(broadcast->channels, 24);
        EXPECT_EQ(broadcast->srcStride, 1);
        const auto tile
            = std::find_if(calls->workspace.begin(),
                           calls->workspace.end(),
                           [&](const WorkspaceTensor& t) { return t.uid == call.c->uid; });
        ASSERT_NE(tile, calls->workspace.end());
        EXPECT_GE(tile->bytes, size_t{24} * 5 * 7 * 2);
        EXPECT_EQ(tails(*calls).size(), 1U);
    }
    // The library-activation engine: the addend and hipBLASLt's relu.
    const auto library = fused(oneByOneNchw(), {bias(DataType::HALF), relu()}, LIBRARY);
    ASSERT_TRUE(library.has_value());
    const auto libraryCalls = yCalls(*library, epilogueOutputUid(2));
    ASSERT_EQ(libraryCalls.size(), 1U);
    ASSERT_TRUE(libraryCalls[0].epilogue.has_value());
    EXPECT_EQ(libraryCalls[0].epilogue->mode, NATIVE_EPILOGUE_RELU);
    EXPECT_FALSE(libraryCalls[0].epilogue->biasUid.has_value());
    EXPECT_TRUE(libraryCalls[0].c.has_value());
    EXPECT_TRUE(tails(*library).empty());
}

TEST(TestConvEpilogueFormulation, AddendTakesOnlyYsType)
{
    // An fp32 bias on a 16-bit y would round in a 16-bit addend: the pass instead. Along the
    // rows hipBLASLt takes an fp32 bias.
    EXPECT_EQ(fused(oneByOneNchw(), {bias(DataType::FLOAT)}, HIPBLASLT)->epilogueForm,
              EpilogueForm::EXTRA_PASS);
    EXPECT_EQ(fused(oneByOneNhwc(), {bias(DataType::FLOAT)}, HIPBLASLT)->epilogueForm,
              EpilogueForm::NATIVE);
    EXPECT_EQ(fused(oneByOneNchw(DataType::FLOAT), {bias(DataType::FLOAT)}, ROCBLAS)->epilogueForm,
              EpilogueForm::NATIVE);
}

TEST(TestConvEpilogueFormulation, NoAddendOnAVectorShapedGemm)
{
    // One output channel: the GEMM is a GEMV, which the libraries may run in place on d.
    ConvGemmShape shape = oneByOneNchw();
    shape.k = 1;
    EXPECT_EQ(fused(shape, {bias(DataType::HALF)}, ROCBLAS)->epilogueForm,
              EpilogueForm::EXTRA_PASS);
    EXPECT_EQ(fused(shape, {bias(DataType::HALF)}, HIPBLASLT)->epilogueForm,
              EpilogueForm::EXTRA_PASS);
}

TEST(TestConvEpilogueFormulation, RocblasTakesNoBiasAlongTheRows)
{
    // rocBLAS has no bias epilogue, and a bias along the rows is no matrix it can add.
    EXPECT_EQ(fused(oneByOneNhwc(), {bias(DataType::HALF)}, ROCBLAS)->epilogueForm,
              EpilogueForm::EXTRA_PASS);
    // A tail alone needs no GEMM epilogue: y written in its type, then clamped.
    const auto relu1 = fused(oneByOneNhwc(), {relu()}, ROCBLAS);
    ASSERT_TRUE(relu1.has_value());
    EXPECT_EQ(relu1->epilogueForm, EpilogueForm::NATIVE);
    EXPECT_EQ(tails(*relu1).size(), 1U);
    for(const auto& call : yCalls(*relu1, epilogueOutputUid(1)))
    {
        EXPECT_FALSE(call.epilogue.has_value());
        EXPECT_FALSE(call.c.has_value());
        EXPECT_EQ(call.d.dataType, DataType::HALF);
    }
}

TEST(TestConvEpilogueFormulation, SigmoidWithABiasNeedsTheAddend)
{
    // hipBLASLt has no sigmoid with a bias: the bias is an addend, so channels-last declines.
    const auto nchw = fused(
        oneByOneNchw(), {bias(DataType::HALF), unary(PointwiseMode::SIGMOID_FWD)}, HIPBLASLT);
    ASSERT_TRUE(nchw.has_value());
    EXPECT_EQ(nchw->epilogueForm, EpilogueForm::NATIVE);
    const auto written = yCalls(*nchw, epilogueOutputUid(2));
    ASSERT_EQ(written.size(), 1U);
    EXPECT_EQ(written[0].epilogue->mode, NATIVE_EPILOGUE_SIGMOID);
    EXPECT_TRUE(written[0].c.has_value());
    const auto nhwc = fused(
        oneByOneNhwc(), {bias(DataType::HALF), unary(PointwiseMode::SIGMOID_FWD)}, HIPBLASLT);
    EXPECT_EQ(nhwc->epilogueForm, EpilogueForm::EXTRA_PASS);
}

TEST(TestConvEpilogueFormulation, NativeDeclinesPerField)
{
    // Each field that rules the GEMM form out, against a chain it maps (bias + relu, half).
    ConvGemmShape grouped = oneByOneNhwc();
    grouped.groups = 2;
    EXPECT_EQ(fused(grouped, {bias(DataType::HALF), relu()}, HIPBLASLT)->epilogueForm,
              EpilogueForm::EXTRA_PASS)
        << "groups: no per-group bias";
    EXPECT_EQ(fused(oneByOneNhwc(),
                    {bias(DataType::HALF), relu(std::nullopt, std::nullopt, 0.5F)},
                    HIPBLASLT)
                  ->epilogueForm,
              EpilogueForm::EXTRA_PASS)
        << "slope";
    EXPECT_EQ(
        fused(oneByOneNhwc(), {bias(DataType::HALF), relu(0.0F, 0.3F)}, HIPBLASLT)->epilogueForm,
        EpilogueForm::EXTRA_PASS)
        << "upper bound not exact in y's type";
    EXPECT_EQ(fused(oneByOneNhwc(), {bias(DataType::BFLOAT16), relu()}, HIPBLASLT)->epilogueForm,
              EpilogueForm::EXTRA_PASS)
        << "bias type neither fp32 nor y's";
    Step deviceScalar = bias(DataType::HALF);
    deviceScalar.operand = OperandKind::DEVICE_SCALAR;
    EXPECT_EQ(fused(oneByOneNhwc(), {deviceScalar, relu()}, HIPBLASLT)->epilogueForm,
              EpilogueForm::EXTRA_PASS)
        << "bias not per channel";
    Step constantBias = bias();
    constantBias.operand = OperandKind::CONSTANT;
    constantBias.value = 0.5F;
    EXPECT_EQ(fused(oneByOneNhwc(), {constantBias, relu()}, HIPBLASLT)->epilogueForm,
              EpilogueForm::EXTRA_PASS)
        << "scalar bias";
    EXPECT_EQ(fused(oneByOneNhwc(), {bias(DataType::HALF), relu()}, std::nullopt)->epilogueForm,
              EpilogueForm::EXTRA_PASS)
        << "no GEMM epilogue asked for";
}

TEST(TestConvEpilogueFormulation, ImplicitColumnsChannelsLastIsNative)
{
    // im2col into a channels-last y: the GEMM writes y with channels as rows.
    const auto calls = fused(threeByThree(true), {bias(DataType::HALF)}, HIPBLASLT);
    ASSERT_TRUE(calls.has_value());
    EXPECT_EQ(calls->epilogueForm, EpilogueForm::NATIVE);
}

TEST(TestConvEpilogueFormulation, GroupsTakeThePass)
{
    // hipBLASLt has no per-batch bias stride: grouped bias goes through the pass.
    ConvGemmShape shape = oneByOneNhwc();
    shape.groups = 2;
    const auto calls = fused(shape, {bias(DataType::HALF)}, HIPBLASLT);
    ASSERT_TRUE(calls.has_value());
    EXPECT_EQ(calls->epilogueForm, EpilogueForm::EXTRA_PASS);
}

TEST(TestConvEpilogueFormulation, ChannelsFirstOneByOnePassIsOneExtraCast)
{
    // The epilogue in a pass (the benchmarking candidate against the GEMM's own).
    ConvGemmShape shape = oneByOneNhwc();
    shape.xChannelsLast = shape.wChannelsLast = shape.yChannelsLast = false;
    const auto calls = fused(shape, {bias(DataType::HALF), relu()}, std::nullopt);
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
                             HIPBLASLT,
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
    const auto after = fuseEpilogue(*problem, std::move(*calls), std::nullopt);
    ASSERT_TRUE(after.has_value());
    const size_t yBytes = size_t{2} * 24 * 5 * 7 * sizeof(float);
    EXPECT_GE(after->workspaceBytes, before + yBytes);
}

TEST(TestConvEpilogueFormulation, Fp32OutputReductionsSplitInOrder)
{
    // The extra pass of a channels-first 3x3: the GEMM writes y's fp32 copy from fp16 operands.
    const auto calls = fused(threeByThree(false), {bias(DataType::HALF), relu()}, std::nullopt);
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
