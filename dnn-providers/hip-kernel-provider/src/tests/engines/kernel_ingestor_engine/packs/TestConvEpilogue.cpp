// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphContentKey.hpp>

#include "compilation/KernelCompileOptions.hpp"
#include "compilation/Program.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvEpilogue.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvEpilogueTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestConvEpilogue.cpp
 * @brief Host-only checks of conv epilogue fusion: the graphs the epilogue parser accepts, one
 *        refusal per gated field, the generated code's identity (what the code-object key
 *        depends on), and the winner key of a fused graph. No device is used.
 */
namespace
{

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_epilogue_test;
using conv_naive::ConvDirection;
using conv_naive::Fusion;

std::optional<conv_naive::ConvNaiveProblem> parseFused(const FusedGraphSpec& spec,
                                                       Fusion fusion = Fusion::EPILOGUE)
{
    const GraphFixture fixture(buildFusedGraph(spec));
    return conv_naive::parseConvNaiveProblem(fixture.context(), ConvDirection::FORWARD, fusion);
}

ConvShape nhwcHalf()
{
    ConvShape shape;
    shape.channelsLast = true;
    shape.dataType = DataType::HALF;
    shape.prePadding = {1, 1};
    shape.postPadding = {1, 1};
    shape.filter = {3, 3};
    return shape;
}

// ---------------------------------------------------------------------------
// Accepted chains
// ---------------------------------------------------------------------------

struct ChainCase
{
    std::string name;
    std::vector<Step> steps;
    std::string pattern;
};

std::vector<ChainCase> chainCases()
{
    std::vector<ChainCase> cases;
    cases.push_back(
        {"BiasRelu", {bias(DataType::HALF), relu(0.0F, std::nullopt, 0.0F)}, "add.k.HALF|relu"});
    cases.push_back({"BiasOnly", {bias(DataType::HALF)}, "add.k.HALF"});
    cases.push_back({"ReluOnly", {relu()}, "relu"});
    cases.push_back({"ClippedRelu",
                     {bias(DataType::FLOAT), relu(std::nullopt, 6.0F)},
                     "add.k.FLOAT|relu.clamp"});
    cases.push_back({"Clamp", {relu(-1.0F, 1.0F)}, "relu.clamp"});
    cases.push_back({"LeakyRelu", {relu(std::nullopt, std::nullopt, 0.1F)}, "relu.leaky"});
    cases.push_back({"LeakyClamp", {relu(-1.0F, 1.0F, 0.2F)}, "relu.leaky.clamp"});
    cases.push_back({"ScaleBiasRelu",
                     {scale(DataType::FLOAT), bias(DataType::BFLOAT16), relu()},
                     "mul.k.FLOAT|add.k.BFLOAT16|relu"});
    cases.push_back({"Gelu", {bias(), unary(PointwiseMode::GELU_FWD)}, "add.k.FLOAT|gelu"});
    cases.push_back({"GeluTanh", {unary(PointwiseMode::GELU_APPROX_TANH_FWD)}, "gelu_tanh"});
    Step swish = unary(PointwiseMode::SWISH_FWD);
    swish.swishBeta = 1.5F;
    cases.push_back({"Swish", {bias(), swish}, "add.k.FLOAT|swish"});
    cases.push_back({"Silu", {unary(PointwiseMode::SWISH_FWD)}, "swish"});
    cases.push_back({"Sigmoid", {unary(PointwiseMode::SIGMOID_FWD)}, "sigmoid"});
    cases.push_back({"Tanh", {unary(PointwiseMode::TANH_FWD)}, "tanh"});
    Step elu = unary(PointwiseMode::ELU_FWD);
    elu.eluAlpha = 0.5F;
    cases.push_back({"Elu", {elu}, "elu"});
    cases.push_back({"Abs", {unary(PointwiseMode::ABS)}, "abs"});
    Step softplus = unary(PointwiseMode::SOFTPLUS_FWD);
    softplus.softplusBeta = 2.0F;
    cases.push_back({"Softplus", {softplus}, "softplus"});

    Step operandFirst = bias(DataType::HALF);
    operandFirst.operandFirst = true;
    cases.push_back({"BiasAsIn0", {operandFirst, relu()}, "add.k.HALF|relu"});

    Step constant = bias();
    constant.operand = OperandKind::CONSTANT;
    constant.value = 0.25F;
    cases.push_back({"ConstantScalarAdd", {constant}, "add.value"});

    Step runtime = scale();
    runtime.operand = OperandKind::RUNTIME;
    cases.push_back({"RuntimeScalarMul", {runtime, relu()}, "mul.value|relu"});

    Step deviceScalar = scale(DataType::HALF);
    deviceScalar.operand = OperandKind::DEVICE_SCALAR;
    cases.push_back({"DeviceScalarMul", {deviceScalar}, "mul.1.HALF"});

    // cuDNN-style: the bias node computed in the bias type.
    Step halfCompute = bias(DataType::HALF);
    halfCompute.compute = DataType::HALF;
    cases.push_back({"HalfComputeBias", {halfCompute, relu()}, "add.k.HALF|relu"});

    Step unsetCompute = relu();
    unsetCompute.compute = DataType::UNSET;
    cases.push_back({"UnsetCompute", {unsetCompute}, "relu"});
    return cases;
}

class TestConvEpilogueAccepts : public ::testing::TestWithParam<ChainCase>
{
};

TEST_P(TestConvEpilogueAccepts, ParsesChain)
{
    const auto& param = GetParam();
    const auto spec = fusedSpecFor(nhwcHalf(), param.steps, DataType::HALF);
    const auto problem = parseFused(spec);
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(conv_epilogue::patternOf(problem->epilogue), param.pattern);
    // The kernel writes the chain's output, in the convolution's y role.
    EXPECT_EQ(problem->operands.y, outputUid(param.steps.size()));
    EXPECT_EQ(problem->epilogue.convOutputUid, conv_naive_test::Y_UID);
    EXPECT_EQ(problem->epilogue.outputUid, outputUid(param.steps.size()));
}

TEST_P(TestConvEpilogueAccepts, DeclinedWithoutFusion)
{
    // Every engine that does not ask for the epilogue keeps declining fused graphs.
    const auto spec = fusedSpecFor(nhwcHalf(), GetParam().steps, DataType::HALF);
    EXPECT_FALSE(parseFused(spec, Fusion::NONE).has_value());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvEpilogueAccepts,
                         ::testing::ValuesIn(chainCases()),
                         [](const auto& info) { return info.param.name; });

TEST(TestConvEpilogue, SingleNodeGraphHasNoEpilogue)
{
    const auto spec = conv_naive_test::specFor(nhwcHalf(), ConvDirection::FORWARD);
    const GraphFixture fixture(conv_naive_test::buildGraph(spec));
    const auto problem = conv_naive::parseConvNaiveProblem(
        fixture.context(), ConvDirection::FORWARD, Fusion::EPILOGUE);
    ASSERT_TRUE(problem.has_value());
    EXPECT_TRUE(problem->epilogue.empty());
    EXPECT_EQ(problem->operands.y, conv_naive_test::Y_UID);
}

TEST(TestConvEpilogue, AcceptsTheCudnnFrontendHalfGraph)
{
    // fp16_conv_bias_relu: NCHW half, intermediates half, bias half with float node compute,
    // relu with lower clip 0 and slope 0 set explicitly. MIOpen's fused engine declines it.
    ConvShape shape;
    shape.n = 8;
    shape.c = 32;
    shape.k = 64;
    shape.input = {16, 16};
    shape.filter = {3, 3};
    shape.prePadding = {1, 1};
    shape.postPadding = {1, 1};
    shape.dataType = DataType::HALF;
    const auto spec = fusedSpecFor(
        shape, {bias(DataType::HALF), relu(0.0F, std::nullopt, 0.0F)}, DataType::HALF);
    const auto problem = parseFused(spec);
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(conv_epilogue::patternOf(problem->epilogue), "add.k.HALF|relu");
    ASSERT_EQ(problem->epilogue.scalars.size(), 3U);
    EXPECT_EQ(problem->epilogue.scalars[0].value, 0.0F);
    EXPECT_TRUE(std::isinf(problem->epilogue.scalars[1].value));
    EXPECT_EQ(problem->epilogue.scalars[2].value, 0.0F);
}

TEST(TestConvEpilogue, ValuesAreRuntimeSlots)
{
    Step constant = bias();
    constant.operand = OperandKind::CONSTANT;
    constant.value = -0.75F;
    Step runtime = scale();
    runtime.operand = OperandKind::RUNTIME;
    Step swish = unary(PointwiseMode::SWISH_FWD);
    swish.swishBeta = 1.25F;
    const auto spec = fusedSpecFor(nhwcHalf(), {constant, runtime, relu(-2.0F, 3.0F, 0.5F), swish});
    const auto problem = parseFused(spec);
    ASSERT_TRUE(problem.has_value());
    const auto& scalars = problem->epilogue.scalars;
    ASSERT_EQ(scalars.size(), 6U);
    EXPECT_EQ(scalars[0].value, -0.75F);
    EXPECT_FALSE(scalars[0].runtime.has_value());
    ASSERT_TRUE(scalars[1].runtime.has_value());
    EXPECT_EQ(scalars[1].runtime->uid, OPERAND_UID + 1);
    EXPECT_EQ(scalars[2].value, -2.0F);
    EXPECT_EQ(scalars[3].value, 3.0F);
    EXPECT_EQ(scalars[4].value, 0.5F);
    EXPECT_EQ(scalars[5].value, 1.25F);

    // Launch arguments: a runtime scalar is read from its host slot at execute.
    float supplied = 4.5F;
    const std::array<hipdnnPluginDeviceBuffer_t, 1> buffers{{{OPERAND_UID + 1, &supplied}}};
    const auto args = conv_epilogue::epilogueArgs(problem->epilogue, buffers.data(), 1, 7, 6);
    EXPECT_EQ(args.scalar[0], -0.75F);
    EXPECT_EQ(args.scalar[1], 4.5F);
    EXPECT_EQ(args.scalar[5], 1.25F);
    EXPECT_EQ(args.chanStride, 7);
    EXPECT_EQ(args.chans, 6);
}

// ---------------------------------------------------------------------------
// Refusals: one per gated field. Each starts from an accepted graph and changes one field.
// ---------------------------------------------------------------------------

struct DeclineCase
{
    std::string name;
    std::vector<Step> steps;
    std::function<void(FusedGraphSpec&)> mutate;
    /// The shared epilogue parser itself declines it (false: the convolution's own gates do,
    /// for the written output and the graph).
    bool parserLevel = true;
};

// The base chains: scale (per-channel) + bias (per-channel) + relu, and a unary activation.
std::vector<Step> sbr()
{
    return {scale(DataType::HALF), bias(DataType::HALF), relu()};
}

constexpr int64_t SCALE_OUT = OUT_UID;
constexpr int64_t BIAS_OUT = OUT_UID + 1;
constexpr int64_t FINAL_OUT = OUT_UID + 2;
constexpr int64_t SCALE_OPERAND = OPERAND_UID;
constexpr int64_t BIAS_OPERAND = OPERAND_UID + 1;

std::vector<DeclineCase> declineCases()
{
    std::vector<DeclineCase> cases;
    // Node kinds and order.
    cases.push_back(
        {"NodeNotPointwise", sbr(), [](FusedGraphSpec& s) { s.nodes[1].pointwise = false; }});
    cases.push_back({"ModeOutsideMenu", sbr(), [](FusedGraphSpec& s) {
                         s.nodes[2].mode = PointwiseMode::EXP;
                     }});
    cases.push_back(
        {"ModeSub", sbr(), [](FusedGraphSpec& s) { s.nodes[1].mode = PointwiseMode::SUB; }});
    cases.push_back({"ModeBackward", sbr(), [](FusedGraphSpec& s) {
                         s.nodes[2].mode = PointwiseMode::RELU_BWD;
                     }});
    const Step abs = unary(PointwiseMode::ABS);
    cases.push_back(
        {"ChainTooLong", {abs, abs, abs, abs, abs, abs, abs, abs}, [](FusedGraphSpec& s) {
             TensorEntry extra = entryOf(s, outputUid(8));
             entryOf(s, outputUid(8)).tensor.isVirtual = true;
             entryOf(s, outputUid(8)).tensor.dataType = DataType::FLOAT;
             extra.uid = OUT_UID + 50;
             s.tensors.push_back(extra);
             NodeSpec last = s.nodes.back();
             last.in0 = last.out;
             last.out = extra.uid;
             s.nodes.push_back(last);
         }});
    cases.push_back({"TooManyScalars",
                     {relu(), relu(), unary(PointwiseMode::SWISH_FWD)},
                     [](FusedGraphSpec& s) {
                         // 7 value slots, and a relu's 3 more: past the 8 the argument holds.
                         TensorEntry extra = entryOf(s, outputUid(3));
                         entryOf(s, outputUid(3)).tensor.isVirtual = true;
                         entryOf(s, outputUid(3)).tensor.dataType = DataType::FLOAT;
                         extra.uid = OUT_UID + 50;
                         s.tensors.push_back(extra);
                         NodeSpec last = s.nodes.front();
                         last.in0 = s.nodes.back().out;
                         last.out = extra.uid;
                         s.nodes.push_back(last);
                     }});
    Step hostValue = bias();
    hostValue.operand = OperandKind::CONSTANT;
    hostValue.value = 0.5F;
    cases.push_back(
        {"TooManyHostScalars", {relu(), relu(), hostValue, hostValue}, [](FusedGraphSpec& s) {
             // 8 value slots, and a fifth step's constant: past the 8 the argument holds.
             TensorEntry operand = entryOf(s, OPERAND_UID + 3);
             operand.uid = OPERAND_UID + 50;
             s.tensors.push_back(operand);
             TensorEntry extra = entryOf(s, outputUid(4));
             entryOf(s, outputUid(4)).tensor.isVirtual = true;
             entryOf(s, outputUid(4)).tensor.dataType = DataType::FLOAT;
             extra.uid = OUT_UID + 50;
             s.tensors.push_back(extra);
             NodeSpec last = s.nodes.back();
             last.in0 = last.out;
             last.in1 = operand.uid;
             last.out = extra.uid;
             s.nodes.push_back(last);
         }});
    cases.push_back({"TooManyOperands", {bias(), bias(), bias(), bias()}, [](FusedGraphSpec& s) {
                         TensorEntry operand = entryOf(s, OPERAND_UID);
                         operand.uid = OPERAND_UID + 50;
                         s.tensors.push_back(operand);
                         TensorEntry extra = entryOf(s, outputUid(4));
                         entryOf(s, outputUid(4)).tensor.isVirtual = true;
                         entryOf(s, outputUid(4)).tensor.dataType = DataType::FLOAT;
                         extra.uid = OUT_UID + 50;
                         s.tensors.push_back(extra);
                         NodeSpec last = s.nodes.back();
                         last.in0 = last.out;
                         last.in1 = operand.uid;
                         last.out = extra.uid;
                         s.nodes.push_back(last);
                     }});

    // Pointwise attributes.
    cases.push_back({"In2Set", sbr(), [](FusedGraphSpec& s) { s.nodes[1].in2 = BIAS_OPERAND; }});
    cases.push_back({"AxisSet", sbr(), [](FusedGraphSpec& s) { s.nodes[2].axis = 1; }});
    cases.push_back(
        {"ReluClipOnAdd", sbr(), [](FusedGraphSpec& s) { s.nodes[1].lowerClip = 0.0F; }});
    cases.push_back(
        {"ReluUpperOnMul", sbr(), [](FusedGraphSpec& s) { s.nodes[0].upperClip = 6.0F; }});
    cases.push_back({"ReluSlopeOnAdd", sbr(), [](FusedGraphSpec& s) { s.nodes[1].slope = 0.1F; }});
    cases.push_back(
        {"SwishBetaOnRelu", sbr(), [](FusedGraphSpec& s) { s.nodes[2].swishBeta = 1.0F; }});
    cases.push_back(
        {"EluAlphaOnRelu", sbr(), [](FusedGraphSpec& s) { s.nodes[2].eluAlpha = 1.0F; }});
    cases.push_back(
        {"SoftplusBetaOnRelu", sbr(), [](FusedGraphSpec& s) { s.nodes[2].softplusBeta = 1.0F; }});
    const float nan = std::numeric_limits<float>::quiet_NaN();
    cases.push_back(
        {"NanLowerClip", sbr(), [nan](FusedGraphSpec& s) { s.nodes[2].lowerClip = nan; }});
    cases.push_back(
        {"NanUpperClip", sbr(), [nan](FusedGraphSpec& s) { s.nodes[2].upperClip = nan; }});
    cases.push_back({"NanSlope", sbr(), [nan](FusedGraphSpec& s) { s.nodes[2].slope = nan; }});
    cases.push_back({"NanSwishBeta", {unary(PointwiseMode::SWISH_FWD)}, [nan](FusedGraphSpec& s) {
                         s.nodes[0].swishBeta = nan;
                     }});
    cases.push_back({"NanEluAlpha", {unary(PointwiseMode::ELU_FWD)}, [nan](FusedGraphSpec& s) {
                         s.nodes[0].eluAlpha = nan;
                     }});
    cases.push_back({"NonPositiveSoftplusBeta",
                     {unary(PointwiseMode::SOFTPLUS_FWD)},
                     [](FusedGraphSpec& s) { s.nodes[0].softplusBeta = 0.0F; }});

    // Node inputs and outputs.
    cases.push_back({"BinaryWithoutIn1", sbr(), [](FusedGraphSpec& s) { s.nodes[1].in1.reset(); }});
    cases.push_back(
        {"BinaryBothInputsChain", sbr(), [](FusedGraphSpec& s) { s.nodes[1].in1 = SCALE_OUT; }});
    cases.push_back({"BinaryChainNotAnInput", sbr(), [](FusedGraphSpec& s) {
                         s.nodes[1].in0 = BIAS_OPERAND;
                         s.nodes[1].in1 = SCALE_OPERAND;
                     }});
    cases.push_back({"OperandIsEarlierIntermediate", sbr(), [](FusedGraphSpec& s) {
                         s.nodes[1].in1 = conv_naive_test::Y_UID;
                     }});
    cases.push_back(
        {"UnaryInputNotChain", sbr(), [](FusedGraphSpec& s) { s.nodes[2].in0 = SCALE_OUT; }});
    cases.push_back(
        {"UnaryWithIn1", sbr(), [](FusedGraphSpec& s) { s.nodes[2].in1 = BIAS_OPERAND; }});
    cases.push_back({"OutputReusesChainTensor", sbr(), [](FusedGraphSpec& s) {
                         // The bias step writes the convolution's y again, which relu then reads.
                         s.nodes[1].out = conv_naive_test::Y_UID;
                         s.nodes[2].in0 = conv_naive_test::Y_UID;
                     }});
    cases.push_back({"OutputIsOperand",
                     sbr(),
                     [](FusedGraphSpec& s) { s.nodes[2].out = BIAS_OPERAND; },
                     false});
    cases.push_back({"OutputMissing", sbr(), [](FusedGraphSpec& s) { s.nodes[2].out = 999; }});
    cases.push_back({"OperandIsConvInput", sbr(), [](FusedGraphSpec& s) {
                         s.nodes[1].in1 = conv_naive_test::X_UID;
                     }});

    // Compute types.
    cases.push_back(
        {"ComputeDouble", sbr(), [](FusedGraphSpec& s) { s.nodes[2].compute = DataType::DOUBLE; }});
    cases.push_back(
        {"ComputeInt32", sbr(), [](FusedGraphSpec& s) { s.nodes[0].compute = DataType::INT32; }});
    cases.push_back({"ComputeBf16OverHalfOperands", sbr(), [](FusedGraphSpec& s) {
                         s.nodes[1].compute = DataType::BFLOAT16;
                     }});

    // Intermediates.
    cases.push_back(
        {"ConvOutputNotVirtual", sbr(), [](FusedGraphSpec& s) { s.conv.y.isVirtual = false; }});
    cases.push_back({"IntermediateNotVirtual", sbr(), [](FusedGraphSpec& s) {
                         entryOf(s, SCALE_OUT).tensor.isVirtual = false;
                     }});
    cases.push_back({"IntermediateDims", sbr(), [](FusedGraphSpec& s) {
                         entryOf(s, BIAS_OUT).tensor.dims[1] += 1;
                     }});
    cases.push_back({"IntermediateType", sbr(), [](FusedGraphSpec& s) {
                         entryOf(s, SCALE_OUT).tensor.dataType = DataType::INT8;
                     }});
    cases.push_back({"IntermediatePassByValue", sbr(), [](FusedGraphSpec& s) {
                         entryOf(s, SCALE_OUT).tensor.runtimePassByValue = true;
                     }});
    cases.push_back({"IntermediateRagged", sbr(), [](FusedGraphSpec& s) {
                         entryOf(s, BIAS_OUT).tensor.raggedOffsetUid = 77;
                     }});
    cases.push_back({"IntermediateRaggedMultiplier", sbr(), [](FusedGraphSpec& s) {
                         entryOf(s, BIAS_OUT).tensor.raggedOffsetMultiplier = 2;
                     }});
    cases.push_back({"IntermediateStridesRank", sbr(), [](FusedGraphSpec& s) {
                         entryOf(s, SCALE_OUT).tensor.strides.pop_back();
                     }});
    cases.push_back({"IntermediateNegativeStride", sbr(), [](FusedGraphSpec& s) {
                         entryOf(s, SCALE_OUT).tensor.strides[0] = -1;
                     }});
    cases.push_back(
        {"ConvOutputDims", sbr(), [](FusedGraphSpec& s) { s.conv.y.dims[2] += 1; }, false});

    // The written output.
    cases.push_back({"FinalOutputVirtual", sbr(), [](FusedGraphSpec& s) {
                         entryOf(s, FINAL_OUT).tensor.isVirtual = true;
                     }});
    cases.push_back({"FinalOutputDims",
                     sbr(),
                     [](FusedGraphSpec& s) { entryOf(s, FINAL_OUT).tensor.dims[3] += 1; },
                     false});
    cases.push_back(
        {"FinalOutputType",
         sbr(),
         [](FusedGraphSpec& s) { entryOf(s, FINAL_OUT).tensor.dataType = DataType::FLOAT; },
         false});

    // Operands.
    cases.push_back({"OperandVirtual", sbr(), [](FusedGraphSpec& s) {
                         entryOf(s, BIAS_OPERAND).tensor.isVirtual = true;
                     }});
    cases.push_back({"OperandChannelExtent", sbr(), [](FusedGraphSpec& s) {
                         entryOf(s, BIAS_OPERAND).tensor.dims[1] += 1;
                     }});
    cases.push_back({"OperandSpatialExtent", sbr(), [](FusedGraphSpec& s) {
                         auto& t = entryOf(s, BIAS_OPERAND).tensor;
                         t.dims[2] = t.dims[1]; // the channel count, on another axis
                         t.strides = conv_naive_test::packedStrides(t.dims, true);
                     }});
    cases.push_back({"OperandBatchExtent", sbr(), [](FusedGraphSpec& s) {
                         auto& t = entryOf(s, BIAS_OPERAND).tensor;
                         t.dims[0] = t.dims[1]; // the channel count, on another axis
                         t.strides = conv_naive_test::packedStrides(t.dims, true);
                     }});
    cases.push_back({"OperandRank", sbr(), [](FusedGraphSpec& s) {
                         auto& t = entryOf(s, BIAS_OPERAND).tensor;
                         t.dims.pop_back();
                         t.strides.pop_back();
                     }});
    cases.push_back({"OperandChannelStride", sbr(), [](FusedGraphSpec& s) {
                         entryOf(s, BIAS_OPERAND).tensor.strides[1] = 2;
                     }});
    cases.push_back({"OperandStridesRank", sbr(), [](FusedGraphSpec& s) {
                         entryOf(s, BIAS_OPERAND).tensor.strides.pop_back();
                     }});
    cases.push_back({"OperandNegativeStride", sbr(), [](FusedGraphSpec& s) {
                         entryOf(s, BIAS_OPERAND).tensor.strides[2] = -1;
                     }});
    cases.push_back({"OperandType", sbr(), [](FusedGraphSpec& s) {
                         entryOf(s, BIAS_OPERAND).tensor.dataType = DataType::INT32;
                     }});
    cases.push_back({"OperandAlignment", sbr(), [](FusedGraphSpec& s) {
                         entryOf(s, BIAS_OPERAND).tensor.alignment = 1;
                     }});
    cases.push_back({"OperandRagged", sbr(), [](FusedGraphSpec& s) {
                         entryOf(s, BIAS_OPERAND).tensor.raggedOffsetUid = 77;
                     }});
    cases.push_back({"OperandRaggedMultiplier", sbr(), [](FusedGraphSpec& s) {
                         entryOf(s, SCALE_OPERAND).tensor.raggedOffsetMultiplier = 2;
                     }});
    cases.push_back({"OperandPerChannelPassByValue", sbr(), [](FusedGraphSpec& s) {
                         entryOf(s, BIAS_OPERAND).tensor.runtimePassByValue = true;
                     }});
    cases.push_back({"OperandPerChannelConstant", sbr(), [](FusedGraphSpec& s) {
                         entryOf(s, BIAS_OPERAND).value = 1.0F;
                     }});
    cases.push_back({"OperandMissing", sbr(), [](FusedGraphSpec& s) { s.nodes[1].in1 = 998; }});
    Step constant = bias();
    constant.operand = OperandKind::CONSTANT;
    constant.value = 3.0F;
    cases.push_back({"ScalarValueIntegerType", {constant}, [](FusedGraphSpec& s) {
                         entryOf(s, OPERAND_UID).tensor.dataType = DataType::INT32;
                     }});

    // Graph and convolution.
    cases.push_back({"OverrideShape",
                     sbr(),
                     [](FusedGraphSpec& s) { s.conv.overrideShapeEnabled = true; },
                     false});
    return cases;
}

class TestConvEpilogueDeclines : public ::testing::TestWithParam<DeclineCase>
{
};

TEST_P(TestConvEpilogueDeclines, OneFieldChanged)
{
    const auto& param = GetParam();
    auto spec = fusedSpecFor(nhwcHalf(), param.steps, DataType::HALF);
    ASSERT_TRUE(parseFused(spec).has_value()) << "the unchanged graph must be accepted";
    param.mutate(spec);
    EXPECT_FALSE(parseFused(spec).has_value());
    if(param.parserLevel)
    {
        // The shared parser declines it on its own: no other gate is relied on.
        const GraphFixture fixture(buildFusedGraph(spec));
        EXPECT_FALSE(hipdnn_plugin_sdk::conv_epilogue::parseConvEpilogue(
                         fixture.context().graph, conv_naive_test::Y_UID, spec.conv.y.dims)
                         .has_value());
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvEpilogueDeclines,
                         ::testing::ValuesIn(declineCases()),
                         [](const auto& info) { return info.param.name; });

TEST(TestConvEpilogue, DeclinesFusedBackwardDirections)
{
    // Forward only: a backward-data graph whose dx feeds a pointwise chain is declined, though it
    // is well formed (dx and dy have one shape here: same padding, unit stride, C == K).
    ConvShape shape = nhwcHalf();
    shape.c = shape.k;
    auto conv = conv_naive_test::specFor(shape, ConvDirection::BACKWARD_DATA);
    const GraphFixture single(conv_naive_test::buildGraph(conv));
    ASSERT_TRUE(conv_naive::parseConvNaiveProblem(
                    single.context(), ConvDirection::BACKWARD_DATA, Fusion::EPILOGUE)
                    .has_value());
    FusedGraphSpec spec = fusedSpecFor(shape, {relu()}, DataType::FLOAT);
    spec.conv.direction = ConvDirection::BACKWARD_DATA;
    // dy is real; dx, which the convolution writes, heads the chain.
    spec.conv.y.isVirtual = false;
    spec.conv.y.dataType = shape.dataType;
    spec.conv.x.isVirtual = true;
    spec.conv.x.dataType = DataType::FLOAT;
    spec.nodes[0].in0 = conv_naive_test::X_UID;
    ASSERT_EQ(spec.conv.x.dims, spec.conv.y.dims);
    const GraphFixture fused(buildFusedGraph(spec));
    EXPECT_FALSE(conv_naive::parseConvNaiveProblem(
                     fused.context(), ConvDirection::BACKWARD_DATA, Fusion::EPILOGUE)
                     .has_value());
    // The forward direction of the same graph is not a forward convolution either.
    EXPECT_FALSE(
        conv_naive::parseConvNaiveProblem(fused.context(), ConvDirection::FORWARD, Fusion::EPILOGUE)
            .has_value());
}

// ---------------------------------------------------------------------------
// Naive's claim set: fusion never widens it. A fused graph is claimed exactly when the bare
// convolution is, over a grid of shapes on both sides of the workload budget.
// ---------------------------------------------------------------------------

TEST(TestConvEpilogue, NaiveClaimsFusedExactlyWhereItClaimsTheBareConv)
{
    const PackSymbols pack{"", "hipkernel.conv_naive_fwd.graph_match", "", "", "", "", "", "", ""};
    hipdnn_plugin_sdk::ingestor::DeviceProperties device = testDeviceProperties();
    device.gcnArchName = "gfx942";
    device.multiProcessorCount = 304;
    const std::vector<std::vector<Step>> chains{
        {bias(DataType::HALF)},
        {bias(DataType::HALF), relu()},
        {bias(DataType::HALF), unary(PointwiseMode::GELU_FWD)},
        {bias(DataType::HALF), unary(PointwiseMode::SWISH_FWD)}};
    int claimed = 0;
    int declined = 0;
    for(const int64_t n : {1, 8, 64})
    {
        for(const int64_t channels : {3, 32, 256})
        {
            for(const int64_t extent : {7, 56, 224})
            {
                for(const int64_t filter : {1, 3, 7})
                {
                    ConvShape shape = nhwcHalf();
                    shape.n = n;
                    shape.c = channels;
                    shape.k = channels * 2;
                    shape.input = {extent, extent};
                    shape.filter = {filter, filter};
                    shape.prePadding = {filter / 2, filter / 2};
                    shape.postPadding = {filter / 2, filter / 2};
                    const auto bare = conv_naive_test::specFor(shape, ConvDirection::FORWARD);
                    const GraphFixture bareFixture(conv_naive_test::buildGraph(bare), device);
                    const bool bareClaimed = matchesGraph(pack, bareFixture.context()).has_value();
                    (bareClaimed ? claimed : declined) += 1;
                    for(const auto& chain : chains)
                    {
                        const GraphFixture fused(buildFusedGraph(fusedSpecFor(shape, chain)),
                                                 device);
                        EXPECT_EQ(matchesGraph(pack, fused.context()).has_value(), bareClaimed)
                            << "n " << n << " c " << channels << " hw " << extent << " f " << filter
                            << ", " << chain.size() << " steps";
                    }
                }
            }
        }
    }
    // The grid reaches both sides of the budget.
    EXPECT_GT(claimed, 0);
    EXPECT_GT(declined, 0);
}

// ---------------------------------------------------------------------------
// Compile identity: the code-object key follows the pattern, never values or shapes.
// ---------------------------------------------------------------------------

std::string naiveKey(const conv_naive::ConvNaiveProblem& problem)
{
    compilation::KernelCompileOptions options("gfx942");
    conv_naive::addCompileOptions(options, problem);
    return compilation::Program::codeObjectKey("kernels/ConvNaive.cpp", options);
}

std::string keyOf(const FusedGraphSpec& spec)
{
    const auto problem = parseFused(spec);
    EXPECT_TRUE(problem.has_value());
    return problem.has_value() ? naiveKey(*problem) : std::string();
}

TEST(TestConvEpilogueKey, DiffersWithAndWithoutEpilogue)
{
    const auto bare = conv_naive_test::specFor(nhwcHalf(), ConvDirection::FORWARD);
    const GraphFixture fixture(conv_naive_test::buildGraph(bare));
    const auto unfused = conv_naive::parseConvNaiveProblem(
        fixture.context(), ConvDirection::FORWARD, Fusion::EPILOGUE);
    ASSERT_TRUE(unfused.has_value());
    const auto unfusedKey = naiveKey(*unfused);
    EXPECT_NE(unfusedKey, keyOf(fusedSpecFor(nhwcHalf(), {bias(DataType::HALF), relu()})));
    EXPECT_NE(unfusedKey, keyOf(fusedSpecFor(nhwcHalf(), {relu()})));

    // Fusion off adds nothing: the unfused options are exactly the element type's.
    compilation::KernelCompileOptions plain("gfx942");
    plain.add("CONV_NAIVE_TYPE", std::string("_Float16"));
    EXPECT_EQ(unfusedKey, compilation::Program::codeObjectKey("kernels/ConvNaive.cpp", plain));
}

TEST(TestConvEpilogueKey, DiffersBetweenPatterns)
{
    const auto biasRelu = keyOf(fusedSpecFor(nhwcHalf(), {bias(DataType::HALF), relu()}));
    EXPECT_NE(biasRelu, keyOf(fusedSpecFor(nhwcHalf(), {bias(DataType::FLOAT), relu()})));
    EXPECT_NE(
        biasRelu,
        keyOf(fusedSpecFor(nhwcHalf(), {bias(DataType::HALF), unary(PointwiseMode::GELU_FWD)})));
    EXPECT_NE(biasRelu, keyOf(fusedSpecFor(nhwcHalf(), {scale(DataType::HALF), relu()})));
    EXPECT_NE(biasRelu, keyOf(fusedSpecFor(nhwcHalf(), {bias(DataType::HALF)})));
    Step scalar = bias(DataType::HALF);
    scalar.operand = OperandKind::DEVICE_SCALAR;
    EXPECT_NE(biasRelu, keyOf(fusedSpecFor(nhwcHalf(), {scalar, relu()})));
}

TEST(TestConvEpilogueKey, EqualForSamePatternAcrossValuesAndShapes)
{
    const auto base = keyOf(fusedSpecFor(nhwcHalf(), {bias(DataType::HALF), relu()}));
    // Other values of one relu form: the lower clip of a relu, the bounds of a clamp, the slope
    // of a leaky relu.
    EXPECT_EQ(base, keyOf(fusedSpecFor(nhwcHalf(), {bias(DataType::HALF), relu(0.5F)})));
    const auto clamp
        = keyOf(fusedSpecFor(nhwcHalf(), {bias(DataType::HALF), relu(std::nullopt, 6.0F)}));
    EXPECT_EQ(clamp, keyOf(fusedSpecFor(nhwcHalf(), {bias(DataType::HALF), relu(-1.0F, 1.0F)})));
    const auto leaky = keyOf(
        fusedSpecFor(nhwcHalf(), {bias(DataType::HALF), relu(std::nullopt, std::nullopt, 0.1F)}));
    EXPECT_EQ(
        leaky,
        keyOf(fusedSpecFor(nhwcHalf(), {bias(DataType::HALF), relu(-2.0F, std::nullopt, 0.3F)})));
    // The forms themselves are different activations, so different code.
    EXPECT_NE(base, clamp);
    EXPECT_NE(base, leaky);
    EXPECT_NE(clamp, leaky);
    // Other shapes and layout: batch, channels, spatial, padding, channels-first.
    ConvShape other = nhwcHalf();
    other.n = 5;
    other.c = 12;
    other.k = 20;
    other.input = {13, 9};
    other.prePadding = {0, 0};
    other.postPadding = {0, 0};
    other.channelsLast = false;
    EXPECT_EQ(base, keyOf(fusedSpecFor(other, {bias(DataType::HALF), relu()})));
    // The bias as in_0 is the same pattern.
    Step first = bias(DataType::HALF);
    first.operandFirst = true;
    EXPECT_EQ(base, keyOf(fusedSpecFor(nhwcHalf(), {first, relu()})));
    // Scalar values: a constant and a runtime scalar of one type share code.
    Step constant = bias();
    constant.operand = OperandKind::CONSTANT;
    constant.value = 1.0F;
    Step runtime = bias();
    runtime.operand = OperandKind::RUNTIME;
    EXPECT_EQ(keyOf(fusedSpecFor(nhwcHalf(), {constant})), keyOf(fusedSpecFor(other, {runtime})));
}

TEST(TestConvEpilogueKey, WinnerKeyDiffersFromBareConv)
{
    const auto bareSpec = conv_naive_test::specFor(nhwcHalf(), ConvDirection::FORWARD);
    const GraphFixture bare(conv_naive_test::buildGraph(bareSpec));
    const GraphFixture fused(
        buildFusedGraph(fusedSpecFor(nhwcHalf(), {bias(DataType::HALF), relu()})));
    const GraphFixture fusedGelu(buildFusedGraph(
        fusedSpecFor(nhwcHalf(), {bias(DataType::HALF), unary(PointwiseMode::GELU_FWD)})));
    using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey;
    const GraphContentKey bareKey(bare.context().graph);
    const GraphContentKey fusedKey(fused.context().graph);
    EXPECT_NE(bareKey, fusedKey);
    EXPECT_NE(fusedKey, GraphContentKey(fusedGelu.context().graph));
    // The same fused graph built twice shares a record.
    const GraphFixture again(
        buildFusedGraph(fusedSpecFor(nhwcHalf(), {bias(DataType::HALF), relu()})));
    EXPECT_EQ(fusedKey, GraphContentKey(again.context().graph));
}

TEST(TestConvEpilogueKey, SignatureCheckTakesBothArities)
{
    using hipdnn_plugin_sdk::ingestor::KernelArgument;
    const std::vector<KernelArgument> unfused{{"by_value", 8, 0, ""}};
    const std::vector<KernelArgument> fused{
        {"by_value", 8, 0, ""}, {"by_value", static_cast<uint32_t>(sizeof(ConvEpiArgs)), 0, ""}};
    EXPECT_NO_THROW(requireSignatureMatch(unfused, unfused, "k", "k"));
    EXPECT_NO_THROW(requireSignatureMatch(fused, fused, "k", "k"));
    EXPECT_ANY_THROW(requireSignatureMatch(unfused, fused, "k", "k"));
    EXPECT_ANY_THROW(requireSignatureMatch(fused, unfused, "k", "k"));
}

std::string generatedCode(const conv_epilogue::ConvEpilogue& epilogue)
{
    compilation::KernelCompileOptions options("gfx942");
    conv_epilogue::addEpilogueDefines(options, epilogue);
    const std::vector<std::string> defines = options;
    for(const auto& define : defines)
    {
        if(define.find("CONV_EPI_CODE=") != std::string::npos)
        {
            return define;
        }
    }
    return {};
}

TEST(TestConvEpilogueKey, ActivationFormFollowsTheOutputType)
{
    // An fp32 output takes the few-ulp activations, a 16-bit output the faster form: different
    // generated code and code-object keys for the same chain, the same for a chain without one.
    ConvShape fp32 = nhwcHalf();
    fp32.dataType = DataType::FLOAT;
    ConvShape bf16 = nhwcHalf();
    bf16.dataType = DataType::BFLOAT16;
    for(const auto mode : {PointwiseMode::GELU_FWD,
                           PointwiseMode::GELU_APPROX_TANH_FWD,
                           PointwiseMode::SWISH_FWD,
                           PointwiseMode::SIGMOID_FWD,
                           PointwiseMode::TANH_FWD,
                           PointwiseMode::ELU_FWD,
                           PointwiseMode::SOFTPLUS_FWD})
    {
        const std::vector<Step> chain{bias(), unary(mode)};
        const auto wide = parseFused(fusedSpecFor(fp32, chain));
        const auto half = parseFused(fusedSpecFor(nhwcHalf(), chain));
        const auto brain = parseFused(fusedSpecFor(bf16, chain));
        ASSERT_TRUE(wide.has_value() && half.has_value() && brain.has_value());
        EXPECT_EQ(wide->epilogue.outputType, DataType::FLOAT);
        EXPECT_NE(generatedCode(wide->epilogue).find("<1>("), std::string::npos);
        EXPECT_NE(generatedCode(half->epilogue).find("<0>("), std::string::npos);
        EXPECT_EQ(generatedCode(half->epilogue), generatedCode(brain->epilogue));
        EXPECT_NE(generatedCode(wide->epilogue), generatedCode(half->epilogue));
        EXPECT_NE(conv_epilogue::patternOf(wide->epilogue),
                  conv_epilogue::patternOf(half->epilogue));
        EXPECT_NE(naiveKey(*wide), naiveKey(*half));
    }
    const std::vector<Step> reluChain{bias(), relu()};
    const auto wide = parseFused(fusedSpecFor(fp32, reluChain));
    const auto half = parseFused(fusedSpecFor(nhwcHalf(), reluChain));
    ASSERT_TRUE(wide.has_value() && half.has_value());
    EXPECT_EQ(generatedCode(wide->epilogue), generatedCode(half->epilogue));
}

TEST(TestConvEpilogueKey, GeneratedCodeIsCanonical)
{
    // Two parses of one pattern give byte-identical generated defines.
    const auto first
        = parseFused(fusedSpecFor(nhwcHalf(), {scale(), bias(DataType::HALF), relu()}));
    const auto second
        = parseFused(fusedSpecFor(nhwcHalf(), {scale(), bias(DataType::HALF), relu(1.0F)}));
    ASSERT_TRUE(first.has_value() && second.has_value());
    compilation::KernelCompileOptions a("gfx942");
    compilation::KernelCompileOptions b("gfx942");
    conv_epilogue::addEpilogueDefines(a, first->epilogue);
    conv_epilogue::addEpilogueDefines(b, second->epilogue);
    const std::vector<std::string> av = a;
    const std::vector<std::string> bv = b;
    EXPECT_EQ(av, bv);
    // And removing them leaves the unfused options.
    conv_epilogue::removeEpilogueDefines(a);
    const std::vector<std::string> removed = a;
    const std::vector<std::string> plain = compilation::KernelCompileOptions("gfx942");
    EXPECT_EQ(removed, plain);
}

TEST(TestConvEpilogueKey, AddingAndRemovingTheEpilogueRestoresTheCodeObjectKey)
{
    // A staging helper of a fused plan compiles with the epilogue defines removed: it must
    // reuse the unfused build's code object, so its options and key come back unchanged. Enough
    // pack defines that the define map rehashes on the way.
    const auto fused
        = parseFused(fusedSpecFor(nhwcHalf(), {scale(), bias(DataType::HALF), relu()}));
    ASSERT_TRUE(fused.has_value());
    compilation::KernelCompileOptions options("gfx942");
    for(int i = 0; i < 40; ++i)
    {
        options.add("CONV_TEST_DEFINE_" + std::to_string(i), i);
    }
    const std::vector<std::string> before = options;
    const auto keyBefore = compilation::Program::codeObjectKey("kernels/ConvIgemm.cpp", before);

    conv_epilogue::addEpilogueDefines(options, fused->epilogue);
    const std::vector<std::string> withEpilogue = options;
    EXPECT_NE(compilation::Program::codeObjectKey("kernels/ConvIgemm.cpp", withEpilogue),
              keyBefore);

    conv_epilogue::removeEpilogueDefines(options);
    const std::vector<std::string> after = options;
    EXPECT_EQ(after, before);
    EXPECT_EQ(compilation::Program::codeObjectKey("kernels/ConvIgemm.cpp", after), keyBefore);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_CONV
