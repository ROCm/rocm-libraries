// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>

#include "engines/kernel_ingestor_engine/packs/HipConvFwdBiasActFmaIgemmRdnaLaunchValues.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestHipConvFwdBiasActFmaIgemmRdnaPack.cpp
 * @brief Host-only checks of `hipkernel:hip_conv_fwd_bias_act_fma_igemm_rdna`: the
 *        conv + (bias +) activation graphs its matcher accepts and refuses, the tokens it binds,
 *        the one kernel its kernel matcher admits per graph, and its saved launch values. The
 *        device cases are IntegrationGpuHipConvFwdBiasActFmaIgemmRdna.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hip_kernel_provider::kernel_ingestor_engine::conv_rdna::fma_igemm::Activation;
using hip_kernel_provider::kernel_ingestor_engine::conv_rdna::fma_igemm::ActivationMode;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::SavedLaunchInputs;
using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr PackSymbols HIP_CONV_FWD_BIAS_ACT_FMA_IGEMM_RDNA{
    "hipkernel:hip_conv_fwd_bias_act_fma_igemm_rdna",
    "hipkernel.hip_conv_fwd_bias_act_fma_igemm_rdna.graph_match",
    "",
    "hipkernel.hip_conv_fwd_bias_act_fma_igemm_rdna.kernel_match",
    "hipkernel.hip_conv_fwd_bias_act_fma_igemm_rdna.score",
    "hipkernel.hip_conv_fwd_bias_act_fma_igemm_rdna.dispatch",
    "conv_fwd_bias_act.x.uid",
    "conv_fwd_bias_act.w.uid",
    "conv_fwd_bias_act.y.uid"};

constexpr int64_t X_UID = 1;
constexpr int64_t W_UID = 2;
constexpr int64_t CONV_OUT_UID = 3;
constexpr int64_t B_UID = 4;
constexpr int64_t SUM_UID = 5;
constexpr int64_t Y_UID = 6;

/// One conv + bias + activation graph. Spatial vectors have one entry per spatial dimension,
/// outermost first. The defaults are a 2-D NCHW fp32 conv, fp32 intermediates, a (1, K, 1, 1)
/// bias and a ReLU, which the engine serves.
struct GraphSpec
{
    data_objects::DataType dtype = data_objects::DataType::FLOAT;
    bool channelsLast = false;
    int64_t n = 2;
    int64_t c = 8;
    int64_t k = 32;
    int64_t groups = 1;
    std::vector<int64_t> in{9, 11};
    std::vector<int64_t> filter{3, 3};
    std::vector<int64_t> prePad{1, 1};
    std::vector<int64_t> postPad{1, 1};
    std::vector<int64_t> stride{1, 1};
    std::vector<int64_t> dilation{1, 1};

    data_objects::DataType convCompute = data_objects::DataType::FLOAT;
    /// The dtype of both intermediates; FLOAT unless set.
    std::optional<data_objects::DataType> intermediateDtype;
    bool convOutVirtual = true;
    bool sumVirtual = true;
    std::optional<data_objects::DataType> biasDtype;
    /// The bias shape; (1, K, 1...) unless set.
    std::optional<std::vector<int64_t>> biasDims;
    bool biasFirst = false;
    data_objects::PointwiseMode biasMode = data_objects::PointwiseMode::ADD;

    data_objects::PointwiseMode activation = data_objects::PointwiseMode::RELU_FWD;
    data_objects::DataType activationCompute = data_objects::DataType::FLOAT;
    std::optional<float> reluLowerClip;
    std::optional<float> reluUpperClip;
    std::optional<float> reluLowerClipSlope;
    std::optional<float> eluAlpha;
    std::optional<float> swishBeta;
    std::optional<float> softplusBeta;
    /// The activation reads the convolution output instead of the biased sum.
    bool activationReadsConvOut = false;
    /// No activation node: the conv + bias engine's graph.
    bool noActivation = false;
    /// No bias: conv -> activation (with noActivation, the conv node alone).
    bool noBias = false;
};

/// Strides of @p dims (NC-spatial order) packed channels-first or channels-last.
std::vector<int64_t> packedStrides(const std::vector<int64_t>& dims, bool channelsLast)
{
    std::vector<int64_t> strides(dims.size(), 1);
    if(!channelsLast)
    {
        for(size_t i = dims.size() - 1; i-- > 0;)
        {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        return strides;
    }
    int64_t expected = dims[1];
    for(size_t i = dims.size(); i-- > 2;)
    {
        strides[i] = expected;
        expected *= dims[i];
    }
    strides[0] = expected;
    return strides;
}

std::vector<int64_t> outputDims(const GraphSpec& spec)
{
    std::vector<int64_t> dims{spec.n, spec.k};
    for(size_t i = 0; i < spec.in.size(); ++i)
    {
        dims.push_back((spec.in[i] + spec.prePad[i] + spec.postPad[i]
                        - spec.dilation[i] * (spec.filter[i] - 1) - 1)
                           / spec.stride[i]
                       + 1);
    }
    return dims;
}

flatbuffers::Offset<data_objects::PointwiseAttributes>
    biasNode(flatbuffers::FlatBufferBuilder& builder, const GraphSpec& spec)
{
    data_objects::PointwiseAttributesBuilder attributes(builder);
    attributes.add_operation(spec.biasMode);
    attributes.add_in_0_tensor_uid(spec.biasFirst ? B_UID : CONV_OUT_UID);
    attributes.add_in_1_tensor_uid(spec.biasFirst ? CONV_OUT_UID : B_UID);
    attributes.add_out_0_tensor_uid(spec.noActivation ? Y_UID : SUM_UID);
    return attributes.Finish();
}

flatbuffers::Offset<data_objects::PointwiseAttributes>
    activationNode(flatbuffers::FlatBufferBuilder& builder, const GraphSpec& spec)
{
    data_objects::PointwiseAttributesBuilder attributes(builder);
    attributes.add_operation(spec.activation);
    attributes.add_in_0_tensor_uid(spec.activationReadsConvOut || spec.noBias ? CONV_OUT_UID
                                                                              : SUM_UID);
    attributes.add_out_0_tensor_uid(Y_UID);
    if(spec.reluLowerClip)
    {
        attributes.add_relu_lower_clip(*spec.reluLowerClip);
    }
    if(spec.reluUpperClip)
    {
        attributes.add_relu_upper_clip(*spec.reluUpperClip);
    }
    if(spec.reluLowerClipSlope)
    {
        attributes.add_relu_lower_clip_slope(*spec.reluLowerClipSlope);
    }
    if(spec.eluAlpha)
    {
        attributes.add_elu_alpha(*spec.eluAlpha);
    }
    if(spec.swishBeta)
    {
        attributes.add_swish_beta(*spec.swishBeta);
    }
    if(spec.softplusBeta)
    {
        attributes.add_softplus_beta(*spec.softplusBeta);
    }
    return attributes.Finish();
}

flatbuffers::FlatBufferBuilder buildGraph(const GraphSpec& spec)
{
    std::vector<int64_t> xDims{spec.n, spec.c};
    xDims.insert(xDims.end(), spec.in.begin(), spec.in.end());
    std::vector<int64_t> wDims{spec.k, spec.c / spec.groups};
    wDims.insert(wDims.end(), spec.filter.begin(), spec.filter.end());
    const auto yDims = outputDims(spec);
    std::vector<int64_t> defaultBias(yDims.size(), 1);
    defaultBias[1] = spec.k;
    const auto bDims = spec.biasDims.value_or(defaultBias);

    const auto xStrides = packedStrides(xDims, spec.channelsLast);
    const auto wStrides = packedStrides(wDims, spec.channelsLast);
    const auto yStrides = packedStrides(yDims, spec.channelsLast);
    const auto bStrides = packedStrides(bDims, spec.channelsLast);
    const auto intermediate = spec.intermediateDtype.value_or(data_objects::DataType::FLOAT);

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, X_UID, nullptr, spec.dtype, &xStrides, &xDims));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, W_UID, nullptr, spec.dtype, &wStrides, &wDims));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, CONV_OUT_UID, nullptr, intermediate, &yStrides, &yDims, spec.convOutVirtual));
    if(!spec.noBias)
    {
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder, B_UID, nullptr, spec.biasDtype.value_or(spec.dtype), &bStrides, &bDims));
    }
    if(!spec.noActivation && !spec.noBias)
    {
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder, SUM_UID, nullptr, intermediate, &yStrides, &yDims, spec.sumVirtual));
    }
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, Y_UID, nullptr, spec.dtype, &yStrides, &yDims));

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    auto conv = data_objects::CreateConvolutionFwdAttributesDirect(
        builder,
        X_UID,
        W_UID,
        CONV_OUT_UID,
        &spec.prePad,
        &spec.postPad,
        &spec.stride,
        &spec.dilation,
        data_objects::ConvMode::CROSS_CORRELATION);
    nodes.push_back(
        data_objects::CreateNodeDirect(builder,
                                       "conv_fwd",
                                       spec.convCompute,
                                       data_objects::NodeAttributes::ConvolutionFwdAttributes,
                                       conv.Union()));
    if(!spec.noBias)
    {
        auto bias = biasNode(builder, spec);
        nodes.push_back(
            data_objects::CreateNodeDirect(builder,
                                           "bias",
                                           data_objects::DataType::FLOAT,
                                           data_objects::NodeAttributes::PointwiseAttributes,
                                           bias.Union()));
    }
    if(!spec.noActivation)
    {
        auto activation = activationNode(builder, spec);
        nodes.push_back(
            data_objects::CreateNodeDirect(builder,
                                           "activation",
                                           spec.activationCompute,
                                           data_objects::NodeAttributes::PointwiseAttributes,
                                           activation.Union()));
    }

    auto name = builder.CreateString("hip_conv_fwd_bias_act_fma_igemm_rdna_test");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);
    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    builder.Finish(graphBuilder.Finish());
    return builder;
}

bool matches(const GraphSpec& spec)
{
    const GraphFixture fixture(buildGraph(spec));
    return matchesGraph(HIP_CONV_FWD_BIAS_ACT_FMA_IGEMM_RDNA, fixture.context()).has_value();
}

/// A 3-D variant of @p spec: depth 4 in front, a 3x3x3 filter, unit depth stride, dilation
/// and padding 1.
GraphSpec threeD(GraphSpec spec)
{
    spec.in.insert(spec.in.begin(), 4);
    spec.filter.insert(spec.filter.begin(), 3);
    spec.prePad.insert(spec.prePad.begin(), 1);
    spec.postPad.insert(spec.postPad.begin(), 1);
    spec.stride.insert(spec.stride.begin(), 1);
    spec.dilation.insert(spec.dilation.begin(), 1);
    return spec;
}

KernelDefinition fmaKernel(const std::string& dtype, const std::string& layout, int64_t tileN)
{
    KernelDefinition kernel;
    kernel.metadata = {{"kernel_family", std::string("fma_igemm")},
                       {"dtype", dtype},
                       {"layout", layout},
                       {"tile_n", tileN}};
    return kernel;
}

bool admits(const GraphSpec& spec, const KernelDefinition& kernel)
{
    const GraphFixture fixture(buildGraph(spec));
    return matchesKernel(HIP_CONV_FWD_BIAS_ACT_FMA_IGEMM_RDNA, fixture.context(), kernel);
}

// ---------------------------------------------------------------------------
// Graph matcher
// ---------------------------------------------------------------------------

TEST(TestHipConvFwdBiasActFmaIgemmRdnaGraphMatcher, AcceptsEveryServedActivation)
{
    for(const auto mode : {data_objects::PointwiseMode::RELU_FWD,
                           data_objects::PointwiseMode::SIGMOID_FWD,
                           data_objects::PointwiseMode::TANH_FWD,
                           data_objects::PointwiseMode::ELU_FWD,
                           data_objects::PointwiseMode::GELU_FWD,
                           data_objects::PointwiseMode::GELU_APPROX_TANH_FWD,
                           data_objects::PointwiseMode::SWISH_FWD,
                           data_objects::PointwiseMode::SOFTPLUS_FWD,
                           data_objects::PointwiseMode::IDENTITY})
    {
        GraphSpec spec;
        spec.activation = mode;
        EXPECT_TRUE(matches(spec)) << data_objects::EnumNamePointwiseMode(mode);
        spec.noBias = true;
        EXPECT_TRUE(matches(spec))
            << data_objects::EnumNamePointwiseMode(mode) << " without a bias";
    }

    GraphSpec clamp;
    clamp.reluLowerClip = 0.1F;
    clamp.reluUpperClip = 0.5F;
    EXPECT_TRUE(matches(clamp)) << "clamp";
    GraphSpec leaky;
    leaky.reluLowerClipSlope = 0.1F;
    EXPECT_TRUE(matches(leaky)) << "leaky ReLU";
    GraphSpec relu6;
    relu6.reluUpperClip = 6.0F;
    EXPECT_TRUE(matches(relu6)) << "ReLU6";
    GraphSpec elu;
    elu.activation = data_objects::PointwiseMode::ELU_FWD;
    elu.eluAlpha = 0.5F;
    EXPECT_TRUE(matches(elu)) << "ELU alpha 0.5";
}

TEST(TestHipConvFwdBiasActFmaIgemmRdnaGraphMatcher, AcceptsEveryServedShape)
{
    GraphSpec nhwc;
    nhwc.channelsLast = true;
    GraphSpec grouped;
    grouped.groups = 2;
    GraphSpec biasFirst;
    biasFirst.biasFirst = true;
    GraphSpec half;
    half.dtype = data_objects::DataType::HALF;
    GraphSpec halfIntermediates = half;
    halfIntermediates.intermediateDtype = data_objects::DataType::HALF;
    GraphSpec bfloat;
    bfloat.dtype = data_objects::DataType::BFLOAT16;
    GraphSpec rank3;
    rank3.in = {40};
    rank3.filter = {5};
    rank3.prePad = {2};
    rank3.postPad = {1};
    rank3.stride = {2};
    rank3.dilation = {3};

    EXPECT_TRUE(matches(GraphSpec{}));
    EXPECT_TRUE(matches(nhwc));
    EXPECT_TRUE(matches(grouped));
    EXPECT_TRUE(matches(biasFirst)) << "ADD(bias, conv)";
    EXPECT_TRUE(matches(half)) << "fp32 intermediates";
    EXPECT_TRUE(matches(halfIntermediates)) << "intermediates in the data dtype";
    EXPECT_TRUE(matches(bfloat));
    EXPECT_TRUE(matches(rank3));
    EXPECT_TRUE(matches(threeD(GraphSpec{})));
    EXPECT_TRUE(matches(threeD(nhwc)));

    GraphSpec noBias;
    noBias.noBias = true;
    GraphSpec noBiasHalfNhwc = noBias;
    noBiasHalfNhwc.dtype = data_objects::DataType::HALF;
    noBiasHalfNhwc.channelsLast = true;
    GraphSpec noBiasGrouped = noBias;
    noBiasGrouped.groups = 2;
    EXPECT_TRUE(matches(noBias)) << "conv -> activation";
    EXPECT_TRUE(matches(noBiasHalfNhwc)) << "conv -> activation, half NHWC";
    EXPECT_TRUE(matches(noBiasGrouped)) << "conv -> activation, grouped";
    EXPECT_TRUE(matches(threeD(noBias))) << "conv -> activation, 3-D";
}

TEST(TestHipConvFwdBiasActFmaIgemmRdnaGraphMatcher, RefusesWhatItCannotServe)
{
    GraphSpec depthwise;
    depthwise.k = 8;
    depthwise.groups = 8;
    EXPECT_FALSE(matches(depthwise)) << "depthwise";

    GraphSpec exp;
    exp.activation = data_objects::PointwiseMode::EXP;
    EXPECT_FALSE(matches(exp)) << "an activation the kernel does not apply";

    GraphSpec twoAdds;
    twoAdds.activation = data_objects::PointwiseMode::ADD;
    EXPECT_FALSE(matches(twoAdds)) << "a second ADD instead of an activation";

    GraphSpec mul;
    mul.biasMode = data_objects::PointwiseMode::MUL;
    EXPECT_FALSE(matches(mul)) << "MUL instead of ADD";

    GraphSpec halfActivation;
    halfActivation.activationCompute = data_objects::DataType::HALF;
    EXPECT_FALSE(matches(halfActivation)) << "an activation computed in half";

    GraphSpec halfConv;
    halfConv.convCompute = data_objects::DataType::HALF;
    EXPECT_FALSE(matches(halfConv)) << "a convolution computed in half";

    GraphSpec storedSum;
    storedSum.sumVirtual = false;
    EXPECT_FALSE(matches(storedSum)) << "a biased sum that is stored too";

    GraphSpec storedConvOut;
    storedConvOut.convOutVirtual = false;
    EXPECT_FALSE(matches(storedConvOut)) << "a convolution output that is stored too";

    GraphSpec otherIntermediate;
    otherIntermediate.dtype = data_objects::DataType::HALF;
    otherIntermediate.intermediateDtype = data_objects::DataType::BFLOAT16;
    EXPECT_FALSE(matches(otherIntermediate)) << "intermediates of a third dtype";

    GraphSpec skipsBias;
    skipsBias.activationReadsConvOut = true;
    EXPECT_FALSE(matches(skipsBias)) << "an activation that does not read the biased sum";

    GraphSpec convBiasOnly;
    convBiasOnly.noActivation = true;
    EXPECT_FALSE(matches(convBiasOnly)) << "conv + bias without an activation";

    GraphSpec convOnly;
    convOnly.noActivation = true;
    convOnly.noBias = true;
    EXPECT_FALSE(matches(convOnly)) << "the conv node alone";

    GraphSpec noBiasStoredConvOut;
    noBiasStoredConvOut.noBias = true;
    noBiasStoredConvOut.convOutVirtual = false;
    EXPECT_FALSE(matches(noBiasStoredConvOut)) << "conv -> activation, conv output stored too";

    GraphSpec noBiasExp;
    noBiasExp.noBias = true;
    noBiasExp.activation = data_objects::PointwiseMode::EXP;
    EXPECT_FALSE(matches(noBiasExp)) << "conv -> EXP";

    GraphSpec noBiasHalfActivation;
    noBiasHalfActivation.noBias = true;
    noBiasHalfActivation.activationCompute = data_objects::DataType::HALF;
    EXPECT_FALSE(matches(noBiasHalfActivation)) << "conv -> activation computed in half";

    GraphSpec fullTensor;
    fullTensor.biasDims = outputDims(fullTensor);
    EXPECT_FALSE(matches(fullTensor)) << "an ADD of a full tensor";

    GraphSpec wrongBiasDtype;
    wrongBiasDtype.dtype = data_objects::DataType::HALF;
    wrongBiasDtype.biasDtype = data_objects::DataType::FLOAT;
    EXPECT_FALSE(matches(wrongBiasDtype)) << "a bias of another dtype";

    GraphSpec softplusBetaZero;
    softplusBetaZero.activation = data_objects::PointwiseMode::SOFTPLUS_FWD;
    softplusBetaZero.softplusBeta = 0.0F;
    EXPECT_FALSE(matches(softplusBetaZero)) << "softplus with beta 0";

    GraphSpec nanClip;
    nanClip.reluUpperClip = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(matches(nanClip)) << "a NaN clip";
}

TEST(TestHipConvFwdBiasActFmaIgemmRdnaBinding, BindsTheOperandsTheBiasAndTheGroupedFlops)
{
    GraphSpec spec = threeD(GraphSpec{});
    spec.groups = 2;
    const GraphFixture fixture(buildGraph(spec));
    const auto bound = matchesGraph(HIP_CONV_FWD_BIAS_ACT_FMA_IGEMM_RDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());

    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_BIAS_ACT_FMA_IGEMM_RDNA.inputAToken), X_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_BIAS_ACT_FMA_IGEMM_RDNA.inputBToken), W_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_BIAS_ACT_FMA_IGEMM_RDNA.outputToken), Y_UID)
        << "the activation output, not a virtual intermediate";
    EXPECT_EQ(tryGetBoundInt(*bound, "conv_fwd_bias_act.b.uid"), B_UID);
    // 2 x N x K x (D x H x W out) x (C / groups) x taps; padding 1 keeps the output size.
    EXPECT_EQ(tryGetBoundInt(*bound, "conv_fwd_bias_act.flops"),
              int64_t{2} * 2 * 32 * 4 * 9 * 11 * (8 / 2) * 27);
}

TEST(TestHipConvFwdBiasActFmaIgemmRdnaBinding, BindsNoBiasForConvActivation)
{
    GraphSpec spec;
    spec.noBias = true;
    const GraphFixture fixture(buildGraph(spec));
    const auto bound = matchesGraph(HIP_CONV_FWD_BIAS_ACT_FMA_IGEMM_RDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());

    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_BIAS_ACT_FMA_IGEMM_RDNA.inputAToken), X_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_BIAS_ACT_FMA_IGEMM_RDNA.outputToken), Y_UID);
    EXPECT_FALSE(tryGetBoundInt(*bound, "conv_fwd_bias_act.b.uid").has_value());
}

// ---------------------------------------------------------------------------
// Kernel matcher
// ---------------------------------------------------------------------------

TEST(TestHipConvFwdBiasActFmaIgemmRdnaKernelMatcher, AdmitsOnlyTheGraphsDtypeLayoutAndTile)
{
    const GraphSpec spec; // fp32, channels-first, 32 output channels: tile 32.

    EXPECT_TRUE(admits(spec, fmaKernel("FLOAT", "CHANNELS_FIRST", 32)));
    EXPECT_FALSE(admits(spec, fmaKernel("HALF", "CHANNELS_FIRST", 32)));
    EXPECT_FALSE(admits(spec, fmaKernel("FLOAT", "CHANNELS_LAST", 32)));
    EXPECT_FALSE(admits(spec, fmaKernel("FLOAT", "CHANNELS_FIRST", 64)));

    GraphSpec nhwcHalf;
    nhwcHalf.channelsLast = true;
    nhwcHalf.dtype = data_objects::DataType::HALF;
    nhwcHalf.k = 100;
    EXPECT_TRUE(admits(nhwcHalf, fmaKernel("HALF", "CHANNELS_LAST", 128)));
    EXPECT_FALSE(admits(nhwcHalf, fmaKernel("HALF", "CHANNELS_LAST", 64)));
}

// ---------------------------------------------------------------------------
// Saved launch values
// ---------------------------------------------------------------------------

/// A problem whose every field differs, so a value read into the wrong field shows.
hip_kernel_provider::kernel_ingestor_engine::conv_rdna::ConvProblem distinctProblem()
{
    hip_kernel_provider::kernel_ingestor_engine::conv_rdna::ConvProblem problem;
    problem.n = 3;
    problem.c = 12;
    problem.k = 40;
    problem.groups = 4;
    problem.in = {5, 17, 19};
    problem.filter = {3, 5, 7};
    problem.out = {2, 9, 10};
    problem.pad = {1, 2, 3};
    problem.stride = {2, 2, 2};
    problem.dilation = {1, 2, 3};
    return problem;
}

/// A clamped leaky ReLU whose three parameters differ.
Activation distinctActivation()
{
    return Activation{ActivationMode::RELU, 0.25F, 6.0F, 0.125F};
}

SavedLaunchInputs savedValid(const Activation& activation = distinctActivation())
{
    return SavedLaunchInputs{
        std::string(HIP_CONV_FWD_BIAS_ACT_FMA_IGEMM_RDNA_DISPATCH_SYMBOL_V1),
        hipConvFwdBiasActFmaIgemmRdnaLaunchValues({21, 22, 24, 23}, distinctProblem(), activation)};
}

TEST(TestHipConvFwdBiasActFmaIgemmRdnaLaunchValues, RoundTripThroughTheirValues)
{
    const auto read = readHipConvFwdBiasActFmaIgemmRdnaLaunchInputs(savedValid());
    const auto expected = distinctProblem();

    EXPECT_EQ(read.binding.x, 21);
    EXPECT_EQ(read.binding.w, 22);
    EXPECT_EQ(read.binding.b, 24);
    EXPECT_EQ(read.binding.y, 23);
    EXPECT_EQ(read.problem.n, expected.n);
    EXPECT_EQ(read.problem.c, expected.c);
    EXPECT_EQ(read.problem.k, expected.k);
    EXPECT_EQ(read.problem.groups, expected.groups);
    EXPECT_EQ(read.problem.in, expected.in);
    EXPECT_EQ(read.problem.filter, expected.filter);
    EXPECT_EQ(read.problem.out, expected.out);
    EXPECT_EQ(read.problem.pad, expected.pad);
    EXPECT_EQ(read.problem.stride, expected.stride);
    EXPECT_EQ(read.problem.dilation, expected.dilation);
    EXPECT_EQ(read.activation.mode, ActivationMode::RELU);
    EXPECT_EQ(read.activation.p0, 0.25F);
    EXPECT_EQ(read.activation.p1, 6.0F);
    EXPECT_EQ(read.activation.p2, 0.125F);
}

TEST(TestHipConvFwdBiasActFmaIgemmRdnaLaunchValues, RoundTripEveryActivation)
{
    for(const auto mode : {ActivationMode::RELU,
                           ActivationMode::SIGMOID,
                           ActivationMode::TANH,
                           ActivationMode::ELU,
                           ActivationMode::GELU,
                           ActivationMode::GELU_APPROX_TANH,
                           ActivationMode::SWISH,
                           ActivationMode::SOFTPLUS,
                           ActivationMode::IDENTITY})
    {
        const auto read
            = readHipConvFwdBiasActFmaIgemmRdnaLaunchInputs(savedValid(Activation{mode, 1.5F}));
        EXPECT_EQ(read.activation.mode, mode) << static_cast<int>(mode);
        EXPECT_EQ(read.activation.p0, 1.5F) << static_cast<int>(mode);
    }
    // The largest float, the reference's default upper clip, survives the double round trip.
    const auto read = readHipConvFwdBiasActFmaIgemmRdnaLaunchInputs(savedValid(
        Activation{ActivationMode::RELU, 0.0F, std::numeric_limits<float>::max(), 0.0F}));
    EXPECT_EQ(read.activation.p1, std::numeric_limits<float>::max());
}

TEST(TestHipConvFwdBiasActFmaIgemmRdnaLaunchValues, RoundTripWithoutABias)
{
    HipConvFwdBiasActFmaIgemmRdnaBinding binding;
    binding.x = 21;
    binding.w = 22;
    binding.y = 23;
    const SavedLaunchInputs saved{
        std::string(HIP_CONV_FWD_BIAS_ACT_FMA_IGEMM_RDNA_DISPATCH_SYMBOL_V1),
        hipConvFwdBiasActFmaIgemmRdnaLaunchValues(
            binding, distinctProblem(), distinctActivation())};
    EXPECT_EQ(saved.values.count("b.uid"), 0U);
    EXPECT_EQ(saved.values.size(), 29U);

    const auto read = readHipConvFwdBiasActFmaIgemmRdnaLaunchInputs(saved);
    EXPECT_EQ(read.binding.x, 21);
    EXPECT_EQ(read.binding.w, 22);
    EXPECT_FALSE(read.binding.b.has_value());
    EXPECT_EQ(read.binding.y, 23);
    EXPECT_EQ(read.activation.mode, ActivationMode::RELU);
}

TEST(TestHipConvFwdBiasActFmaIgemmRdnaLaunchValues, SavesTheV1ValueNames)
{
    const auto values = savedValid().values;
    for(const auto* name :
        {"x.uid",      "w.uid",      "b.uid",         "y.uid",         "n",
         "c",          "k",          "groups",        "in_d",          "in_h",
         "in_w",       "filter_d",   "filter_h",      "filter_w",      "out_d",
         "out_h",      "out_w",      "pad_d",         "pad_h",         "pad_w",
         "stride_d",   "stride_h",   "stride_w",      "dilation_d",    "dilation_h",
         "dilation_w", "activation", "activation.p0", "activation.p1", "activation.p2"})
    {
        EXPECT_EQ(values.count(name), 1U) << name;
    }
    EXPECT_EQ(values.size(), 30U);
    EXPECT_EQ(std::get<std::string>(values.at("activation")), "RELU_FWD");
}

TEST(TestHipConvFwdBiasActFmaIgemmRdnaLaunchValues, RefusesMalformedValues)
{
    const auto refused = [](const SavedLaunchInputs& inputs, const char* what) {
        EXPECT_THROW(readHipConvFwdBiasActFmaIgemmRdnaLaunchInputs(inputs),
                     hipdnn_plugin_sdk::HipdnnPluginException)
            << what;
    };

    auto biasContract = savedValid();
    biasContract.dispatchSymbol = "hipkernel.hip_conv_fwd_bias_fma_igemm_rdna.dispatch.v1";
    refused(biasContract, "the conv + bias engine's contract");

    auto missingActivation = savedValid();
    missingActivation.values.erase("activation");
    refused(missingActivation, "no activation");

    auto missingParameter = savedValid();
    missingParameter.values.erase("activation.p2");
    refused(missingParameter, "no third activation parameter");

    auto exp = savedValid();
    exp.values["activation"] = std::string("EXP");
    refused(exp, "an activation the kernel does not apply");

    auto intActivation = savedValid();
    intActivation.values["activation"] = int64_t{0};
    refused(intActivation, "an activation saved as an integer");

    auto nanParameter = savedValid();
    nanParameter.values["activation.p1"] = std::nan("");
    refused(nanParameter, "a NaN parameter");

    auto hugeParameter = savedValid();
    hugeParameter.values["activation.p0"] = 1e300;
    refused(hugeParameter, "a parameter beyond the float range");

    refused(savedValid(Activation{ActivationMode::SOFTPLUS, 0.0F}), "softplus with beta 0");

    auto extra = savedValid();
    extra.values["tile_n"] = int64_t{32};
    refused(extra, "a value outside the contract");

    auto zeroExtent = savedValid();
    zeroExtent.values["in_w"] = int64_t{0};
    refused(zeroExtent, "a zero extent");
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
