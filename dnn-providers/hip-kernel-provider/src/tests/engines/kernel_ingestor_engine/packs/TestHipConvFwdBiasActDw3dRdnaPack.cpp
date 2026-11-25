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

#include "engines/kernel_ingestor_engine/packs/HipConvFwdBiasActDw3dRdnaLaunchValues.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestHipConvFwdBiasActDw3dRdnaPack.cpp
 * @brief Host-only checks of `hipkernel:hip_conv_fwd_bias_act_dw3d_rdna`: the depthwise 3-D
 *        conv + (bias +) activation graphs its matcher accepts and refuses, the tokens it binds,
 *        the one kernel its kernel matcher admits per graph, and its saved launch values. The
 *        device cases are IntegrationGpuHipConvFwdBiasActDw3dRdna.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hip_kernel_provider::kernel_ingestor_engine::conv_rdna::Activation;
using hip_kernel_provider::kernel_ingestor_engine::conv_rdna::ActivationMode;
using hip_kernel_provider::kernel_ingestor_engine::conv_rdna::ConvProblem;
using hip_kernel_provider::kernel_ingestor_engine::conv_rdna::dw3d::Geometry;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::SavedLaunchInputs;
using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr PackSymbols HIP_CONV_FWD_BIAS_ACT_DW3D_RDNA{
    "hipkernel:hip_conv_fwd_bias_act_dw3d_rdna",
    "hipkernel.hip_conv_fwd_bias_act_dw3d_rdna.graph_match",
    "",
    "hipkernel.hip_conv_fwd_bias_act_dw3d_rdna.kernel_match",
    "hipkernel.hip_conv_fwd_bias_act_dw3d_rdna.score",
    "hipkernel.hip_conv_fwd_bias_act_dw3d_rdna.dispatch",
    "conv_fwd_bias_act_dw3d.x.uid",
    "conv_fwd_bias_act_dw3d.w.uid",
    "conv_fwd_bias_act_dw3d.y.uid"};

constexpr int64_t X_UID = 1;
constexpr int64_t W_UID = 2;
constexpr int64_t CONV_OUT_UID = 3;
constexpr int64_t B_UID = 4;
constexpr int64_t SUM_UID = 5;
constexpr int64_t Y_UID = 6;

/// One depthwise conv + (bias +) activation graph. Spatial vectors have one entry per spatial
/// dimension, outermost first. The defaults are an NCDHW fp32 3x3x3 depthwise conv with 16
/// channels, fp32 intermediates, a (1, K, 1, 1, 1) bias and a ReLU, which the engine serves.
struct GraphSpec
{
    data_objects::DataType dtype = data_objects::DataType::FLOAT;
    bool channelsLast = false;
    int64_t n = 2;
    int64_t c = 16;
    int64_t k = 16;
    int64_t groups = 16;
    std::vector<int64_t> in{6, 9, 11};
    std::vector<int64_t> filter{3, 3, 3};
    std::vector<int64_t> prePad{1, 1, 1};
    std::vector<int64_t> postPad{1, 1, 1};
    std::vector<int64_t> stride{1, 1, 1};
    std::vector<int64_t> dilation{1, 1, 1};

    data_objects::DataType convCompute = data_objects::DataType::FLOAT;
    /// The dtype of the intermediates; FLOAT unless set.
    std::optional<data_objects::DataType> intermediateDtype;
    bool convOutVirtual = true;
    bool sumVirtual = true;
    bool biasFirst = false;

    data_objects::PointwiseMode activation = data_objects::PointwiseMode::RELU_FWD;
    data_objects::DataType activationCompute = data_objects::DataType::FLOAT;
    std::optional<float> reluUpperClip;
    std::optional<float> softplusBeta;
    /// No activation node: conv + bias.
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
    attributes.add_operation(data_objects::PointwiseMode::ADD);
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
    attributes.add_in_0_tensor_uid(spec.noBias ? CONV_OUT_UID : SUM_UID);
    attributes.add_out_0_tensor_uid(Y_UID);
    if(spec.reluUpperClip)
    {
        attributes.add_relu_upper_clip(*spec.reluUpperClip);
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
    std::vector<int64_t> bDims(yDims.size(), 1);
    bDims[1] = spec.k;

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
            builder, B_UID, nullptr, spec.dtype, &bStrides, &bDims));
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

    auto name = builder.CreateString("hip_conv_fwd_bias_act_dw3d_rdna_test");
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
    return matchesGraph(HIP_CONV_FWD_BIAS_ACT_DW3D_RDNA, fixture.context()).has_value();
}

KernelDefinition specializedKernel(const std::string& dtype,
                                   const std::string& layout,
                                   std::vector<int64_t> filter,
                                   int64_t stride,
                                   int64_t tileD,
                                   int64_t tileW,
                                   int64_t uniformK)
{
    KernelDefinition kernel;
    kernel.metadata = {{"kernel_family", std::string("dw3d_direct")},
                       {"variant", std::string("specialized")},
                       {"dtype", dtype},
                       {"layout", layout},
                       {"filter_d", filter[0]},
                       {"filter_h", filter[1]},
                       {"filter_w", filter[2]},
                       {"stride_h", stride},
                       {"stride_w", stride},
                       {"tile_d", tileD},
                       {"tile_w", tileW},
                       {"uniform_k", uniformK}};
    return kernel;
}

KernelDefinition genericKernel(const std::string& dtype, const std::string& layout)
{
    KernelDefinition kernel;
    kernel.metadata = {{"kernel_family", std::string("dw3d_direct")},
                       {"variant", std::string("generic")},
                       {"dtype", dtype},
                       {"layout", layout},
                       {"filter_d", int64_t{0}},
                       {"filter_h", int64_t{0}},
                       {"filter_w", int64_t{0}},
                       {"stride_h", int64_t{0}},
                       {"stride_w", int64_t{0}},
                       {"tile_d", int64_t{1}},
                       {"tile_w", int64_t{4}},
                       {"uniform_k", int64_t{0}}};
    return kernel;
}

bool admits(const GraphSpec& spec, const KernelDefinition& kernel)
{
    const GraphFixture fixture(buildGraph(spec));
    return matchesKernel(HIP_CONV_FWD_BIAS_ACT_DW3D_RDNA, fixture.context(), kernel);
}

// ---------------------------------------------------------------------------
// Graph matcher
// ---------------------------------------------------------------------------

TEST(TestHipConvFwdBiasActDw3dRdnaGraphMatcher, AcceptsEveryServedActivation)
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

    GraphSpec relu6;
    relu6.reluUpperClip = 6.0F;
    EXPECT_TRUE(matches(relu6)) << "ReLU6";
}

TEST(TestHipConvFwdBiasActDw3dRdnaGraphMatcher, AcceptsEveryServedShape)
{
    GraphSpec ndhwc;
    ndhwc.channelsLast = true;
    GraphSpec multiplier;
    multiplier.k = 32;
    GraphSpec biasFirst;
    biasFirst.biasFirst = true;
    GraphSpec half;
    half.dtype = data_objects::DataType::HALF;
    GraphSpec halfIntermediates = half;
    halfIntermediates.intermediateDtype = data_objects::DataType::HALF;
    GraphSpec bfloat;
    bfloat.dtype = data_objects::DataType::BFLOAT16;
    GraphSpec oddFilter;
    oddFilter.filter = {2, 3, 4};
    oddFilter.prePad = {0, 1, 2};
    oddFilter.postPad = {1, 1, 1};

    EXPECT_TRUE(matches(GraphSpec{}));
    EXPECT_TRUE(matches(ndhwc));
    EXPECT_TRUE(matches(multiplier)) << "K = 2 C";
    EXPECT_TRUE(matches(biasFirst)) << "ADD(bias, conv)";
    EXPECT_TRUE(matches(half)) << "fp32 intermediates";
    EXPECT_TRUE(matches(halfIntermediates)) << "intermediates in the data dtype";
    EXPECT_TRUE(matches(bfloat));
    EXPECT_TRUE(matches(oddFilter)) << "a filter served by the generic variant";

    GraphSpec noBiasHalfNdhwc;
    noBiasHalfNdhwc.noBias = true;
    noBiasHalfNdhwc.dtype = data_objects::DataType::HALF;
    noBiasHalfNdhwc.channelsLast = true;
    EXPECT_TRUE(matches(noBiasHalfNdhwc)) << "conv -> activation, half NDHWC";
}

TEST(TestHipConvFwdBiasActDw3dRdnaGraphMatcher, RefusesWhatItCannotServe)
{
    GraphSpec dense;
    dense.groups = 1;
    EXPECT_FALSE(matches(dense)) << "non-depthwise";

    GraphSpec grouped;
    grouped.groups = 4;
    EXPECT_FALSE(matches(grouped)) << "4 channels per group";

    GraphSpec twoD;
    twoD.in = {9, 11};
    twoD.filter = {3, 3};
    twoD.prePad = {1, 1};
    twoD.postPad = {1, 1};
    twoD.stride = {1, 1};
    twoD.dilation = {1, 1};
    EXPECT_FALSE(matches(twoD)) << "2-D depthwise";

    GraphSpec halfConv;
    halfConv.convCompute = data_objects::DataType::HALF;
    EXPECT_FALSE(matches(halfConv)) << "a convolution computed in half";

    GraphSpec halfActivation;
    halfActivation.activationCompute = data_objects::DataType::HALF;
    EXPECT_FALSE(matches(halfActivation)) << "an activation computed in half";

    GraphSpec exp;
    exp.activation = data_objects::PointwiseMode::EXP;
    EXPECT_FALSE(matches(exp)) << "an activation the kernel does not apply";

    GraphSpec storedSum;
    storedSum.sumVirtual = false;
    EXPECT_FALSE(matches(storedSum)) << "a biased sum that is stored too";

    GraphSpec storedConvOut;
    storedConvOut.noBias = true;
    storedConvOut.convOutVirtual = false;
    EXPECT_FALSE(matches(storedConvOut)) << "conv -> activation, conv output stored too";

    GraphSpec convBiasOnly;
    convBiasOnly.noActivation = true;
    EXPECT_FALSE(matches(convBiasOnly)) << "conv + bias without an activation";

    GraphSpec convOnly;
    convOnly.noActivation = true;
    convOnly.noBias = true;
    EXPECT_FALSE(matches(convOnly)) << "the conv node alone";

    GraphSpec softplusBetaZero;
    softplusBetaZero.activation = data_objects::PointwiseMode::SOFTPLUS_FWD;
    softplusBetaZero.softplusBeta = 0.0F;
    EXPECT_FALSE(matches(softplusBetaZero)) << "softplus with beta 0";
}

TEST(TestHipConvFwdBiasActDw3dRdnaBinding, BindsTheOperandsTheBiasAndTheFlops)
{
    const GraphSpec spec;
    const GraphFixture fixture(buildGraph(spec));
    const auto bound = matchesGraph(HIP_CONV_FWD_BIAS_ACT_DW3D_RDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());

    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_BIAS_ACT_DW3D_RDNA.inputAToken), X_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_BIAS_ACT_DW3D_RDNA.inputBToken), W_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_BIAS_ACT_DW3D_RDNA.outputToken), Y_UID)
        << "the activation output, not a virtual intermediate";
    EXPECT_EQ(tryGetBoundInt(*bound, "conv_fwd_bias_act_dw3d.b.uid"), B_UID);
    // 2 x N x K x (D x H x W out) x (C / groups) x taps; padding 1 keeps the output size.
    EXPECT_EQ(tryGetBoundInt(*bound, "conv_fwd_bias_act_dw3d.flops"),
              int64_t{2} * 2 * 16 * 6 * 9 * 11 * 1 * 27);
}

TEST(TestHipConvFwdBiasActDw3dRdnaBinding, BindsNoBiasForConvActivation)
{
    GraphSpec spec;
    spec.noBias = true;
    const GraphFixture fixture(buildGraph(spec));
    const auto bound = matchesGraph(HIP_CONV_FWD_BIAS_ACT_DW3D_RDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());

    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_BIAS_ACT_DW3D_RDNA.inputAToken), X_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_BIAS_ACT_DW3D_RDNA.outputToken), Y_UID);
    EXPECT_FALSE(tryGetBoundInt(*bound, "conv_fwd_bias_act_dw3d.b.uid").has_value());
}

// ---------------------------------------------------------------------------
// Kernel matcher
// ---------------------------------------------------------------------------

TEST(TestHipConvFwdBiasActDw3dRdnaKernelMatcher, AdmitsTheVariantOfTheUnfusedEngine)
{
    const GraphSpec spec; // NCDHW fp32 3x3x3 stride 1: tile 2 x 4, uniform-k.
    EXPECT_TRUE(admits(spec, specializedKernel("FLOAT", "CHANNELS_FIRST", {3, 3, 3}, 1, 2, 4, 1)));
    EXPECT_FALSE(admits(spec, specializedKernel("FLOAT", "CHANNELS_FIRST", {3, 3, 3}, 1, 1, 4, 1)))
        << "another depth tile";
    EXPECT_FALSE(admits(spec, specializedKernel("HALF", "CHANNELS_FIRST", {3, 3, 3}, 1, 2, 4, 1)))
        << "another dtype";
    EXPECT_FALSE(admits(spec, genericKernel("FLOAT", "CHANNELS_FIRST"))) << "the generic variant";

    GraphSpec noBias = spec;
    noBias.noBias = true;
    EXPECT_TRUE(admits(noBias, specializedKernel("FLOAT", "CHANNELS_FIRST", {3, 3, 3}, 1, 2, 4, 1)))
        << "the same variant without a bias";

    GraphSpec ndhwc;
    ndhwc.channelsLast = true;
    EXPECT_TRUE(admits(ndhwc, specializedKernel("FLOAT", "CHANNELS_LAST", {3, 3, 3}, 1, 1, 4, 0)));

    GraphSpec oddFilter;
    oddFilter.filter = {2, 3, 4};
    EXPECT_TRUE(admits(oddFilter, genericKernel("FLOAT", "CHANNELS_FIRST")));
    EXPECT_FALSE(admits(oddFilter, genericKernel("FLOAT", "CHANNELS_LAST")));
}

// ---------------------------------------------------------------------------
// Saved launch values
// ---------------------------------------------------------------------------

/// A problem whose every field differs, so a value read into the wrong field shows.
ConvProblem distinctProblem()
{
    ConvProblem problem;
    problem.spatialRank = 3;
    problem.n = 3;
    problem.c = 12;
    problem.k = 24;
    problem.groups = 12;
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

HipConvFwdBiasActDw3dRdnaBinding distinctBinding()
{
    HipConvFwdBiasActDw3dRdnaBinding binding;
    binding.x = 21;
    binding.w = 22;
    binding.b = 24;
    binding.y = 23;
    return binding;
}

SavedLaunchInputs savedValid(const Geometry& geometry = {false, 2, 8, true},
                             const Activation& activation = distinctActivation())
{
    return SavedLaunchInputs{std::string(HIP_CONV_FWD_BIAS_ACT_DW3D_RDNA_DISPATCH_SYMBOL_V1),
                             hipConvFwdBiasActDw3dRdnaLaunchValues(
                                 distinctBinding(), distinctProblem(), geometry, activation)};
}

TEST(TestHipConvFwdBiasActDw3dRdnaLaunchValues, RoundTripThroughTheirValues)
{
    const auto read = readHipConvFwdBiasActDw3dRdnaLaunchInputs(savedValid());
    const auto expected = distinctProblem();

    EXPECT_EQ(read.binding.x, 21);
    EXPECT_EQ(read.binding.w, 22);
    EXPECT_EQ(read.binding.b, 24);
    EXPECT_EQ(read.binding.y, 23);
    EXPECT_EQ(read.problem.n, expected.n);
    EXPECT_EQ(read.problem.c, expected.c);
    EXPECT_EQ(read.problem.k, expected.k);
    EXPECT_EQ(read.problem.groups, expected.c);
    EXPECT_EQ(read.problem.in, expected.in);
    EXPECT_EQ(read.problem.filter, expected.filter);
    EXPECT_EQ(read.problem.out, expected.out);
    EXPECT_EQ(read.problem.pad, expected.pad);
    EXPECT_EQ(read.problem.stride, expected.stride);
    EXPECT_EQ(read.problem.dilation, expected.dilation);
    EXPECT_FALSE(read.geometry.generic);
    EXPECT_EQ(read.geometry.tileD, 2);
    EXPECT_EQ(read.geometry.tileW, 8);
    EXPECT_TRUE(read.geometry.uniformK);
    EXPECT_EQ(read.activation.mode, ActivationMode::RELU);
    EXPECT_EQ(read.activation.p0, 0.25F);
    EXPECT_EQ(read.activation.p1, 6.0F);
    EXPECT_EQ(read.activation.p2, 0.125F);

    const auto generic = readHipConvFwdBiasActDw3dRdnaLaunchInputs(savedValid({true, 1, 4, false}));
    EXPECT_TRUE(generic.geometry.generic);
}

TEST(TestHipConvFwdBiasActDw3dRdnaLaunchValues, RoundTripEveryActivation)
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
        const auto read = readHipConvFwdBiasActDw3dRdnaLaunchInputs(
            savedValid({false, 2, 8, true}, Activation{mode, 1.5F}));
        EXPECT_EQ(read.activation.mode, mode) << static_cast<int>(mode);
        EXPECT_EQ(read.activation.p0, 1.5F) << static_cast<int>(mode);
    }
}

TEST(TestHipConvFwdBiasActDw3dRdnaLaunchValues, RoundTripWithoutABias)
{
    auto binding = distinctBinding();
    binding.b.reset();
    const SavedLaunchInputs saved{
        std::string(HIP_CONV_FWD_BIAS_ACT_DW3D_RDNA_DISPATCH_SYMBOL_V1),
        hipConvFwdBiasActDw3dRdnaLaunchValues(
            binding, distinctProblem(), {false, 2, 8, true}, distinctActivation())};
    EXPECT_EQ(saved.values.count("b.uid"), 0U);
    EXPECT_EQ(saved.values.size(), 32U);

    const auto read = readHipConvFwdBiasActDw3dRdnaLaunchInputs(saved);
    EXPECT_EQ(read.binding.x, 21);
    EXPECT_FALSE(read.binding.b.has_value());
    EXPECT_EQ(read.binding.y, 23);
    EXPECT_EQ(read.activation.mode, ActivationMode::RELU);
}

TEST(TestHipConvFwdBiasActDw3dRdnaLaunchValues, SavesTheV1ValueNames)
{
    const auto values = savedValid().values;
    for(const auto* name : {"x.uid",
                            "w.uid",
                            "b.uid",
                            "y.uid",
                            "n",
                            "c",
                            "k",
                            "in_d",
                            "in_h",
                            "in_w",
                            "filter_d",
                            "filter_h",
                            "filter_w",
                            "out_d",
                            "out_h",
                            "out_w",
                            "pad_d",
                            "pad_h",
                            "pad_w",
                            "stride_d",
                            "stride_h",
                            "stride_w",
                            "dilation_d",
                            "dilation_h",
                            "dilation_w",
                            "generic",
                            "tile_d",
                            "tile_w",
                            "uniform_k",
                            "activation",
                            "activation.p0",
                            "activation.p1",
                            "activation.p2"})
    {
        EXPECT_EQ(values.count(name), 1U) << name;
    }
    EXPECT_EQ(values.size(), 33U);
    EXPECT_EQ(std::get<std::string>(values.at("activation")), "RELU_FWD");
}

TEST(TestHipConvFwdBiasActDw3dRdnaLaunchValues, RefusesMalformedValues)
{
    const auto refused = [](const SavedLaunchInputs& inputs, const char* what) {
        EXPECT_THROW(readHipConvFwdBiasActDw3dRdnaLaunchInputs(inputs),
                     hipdnn_plugin_sdk::HipdnnPluginException)
            << what;
    };

    auto unfusedContract = savedValid();
    unfusedContract.dispatchSymbol = "hipkernel.hip_conv_fwd_dw3d_rdna.dispatch.v1";
    refused(unfusedContract, "the unfused engine's contract");

    auto missingActivation = savedValid();
    missingActivation.values.erase("activation");
    refused(missingActivation, "no activation");

    auto exp = savedValid();
    exp.values["activation"] = std::string("EXP");
    refused(exp, "an activation the kernel does not apply");

    auto nanParameter = savedValid();
    nanParameter.values["activation.p1"] = std::nan("");
    refused(nanParameter, "a NaN parameter");

    auto missingTile = savedValid();
    missingTile.values.erase("tile_w");
    refused(missingTile, "a missing geometry value");

    auto extra = savedValid();
    extra.values["groups"] = int64_t{12};
    refused(extra, "a value outside the contract");

    auto notMultiple = savedValid();
    notMultiple.values["k"] = int64_t{20};
    refused(notMultiple, "K not a multiple of C");

    refused(savedValid({false, 3, 4, true}), "a depth tile of 3");

    auto hugeGrid = savedValid();
    hugeGrid.values["n"] = int64_t{std::numeric_limits<int>::max()};
    refused(hugeGrid, "a grid beyond the launch limits");
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
