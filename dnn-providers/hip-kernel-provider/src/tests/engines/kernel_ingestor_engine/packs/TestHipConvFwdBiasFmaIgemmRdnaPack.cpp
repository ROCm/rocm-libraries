// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>

#include "engines/kernel_ingestor_engine/packs/HipConvFwdBiasFmaIgemmRdnaLaunchValues.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestHipConvFwdBiasFmaIgemmRdnaPack.cpp
 * @brief Host-only checks of `hipkernel:hip_conv_fwd_bias_fma_igemm_rdna`: the conv + bias
 *        graphs its matcher accepts and refuses, the tokens it binds, the one kernel its kernel
 *        matcher admits per graph, and its saved launch values. The device cases are
 *        IntegrationGpuHipConvFwdBiasFmaIgemmRdna.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::SavedLaunchInputs;
using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr PackSymbols HIP_CONV_FWD_BIAS_FMA_IGEMM_RDNA{
    "hipkernel:hip_conv_fwd_bias_fma_igemm_rdna",
    "hipkernel.hip_conv_fwd_bias_fma_igemm_rdna.graph_match",
    "",
    "hipkernel.hip_conv_fwd_bias_fma_igemm_rdna.kernel_match",
    "hipkernel.hip_conv_fwd_bias_fma_igemm_rdna.score",
    "hipkernel.hip_conv_fwd_bias_fma_igemm_rdna.dispatch",
    "conv_fwd_bias.x.uid",
    "conv_fwd_bias.w.uid",
    "conv_fwd_bias.y.uid"};

constexpr int64_t X_UID = 1;
constexpr int64_t W_UID = 2;
constexpr int64_t CONV_OUT_UID = 3;
constexpr int64_t B_UID = 4;
constexpr int64_t Y_UID = 5;
constexpr int64_t RELU_OUT_UID = 6;

/// One conv + bias graph. Spatial vectors have one entry per spatial dimension, outermost
/// first. The defaults are a 2-D NCHW fp32 conv with an fp32 conv output and a (1, K, 1, 1)
/// bias the engine serves.
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

    /// The convolution output's dtype; FLOAT unless set.
    std::optional<data_objects::DataType> convOutDtype;
    bool convOutVirtual = true;
    std::optional<data_objects::DataType> biasDtype;
    /// The bias shape; (1, K, 1...) unless set.
    std::optional<std::vector<int64_t>> biasDims;
    bool biasFirst = false;
    data_objects::PointwiseMode mode = data_objects::PointwiseMode::ADD;
    /// The ADD reads x instead of the convolution output.
    bool addReadsX = false;
    /// The conv node alone: the unfused engine's graph.
    bool convOnly = false;
    /// A relu after the ADD: the conv + bias + activation graph.
    bool reluAfter = false;
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
    pointwise(flatbuffers::FlatBufferBuilder& builder,
              data_objects::PointwiseMode mode,
              int64_t in0,
              std::optional<int64_t> in1,
              int64_t out)
{
    data_objects::PointwiseAttributesBuilder attributes(builder);
    attributes.add_operation(mode);
    attributes.add_in_0_tensor_uid(in0);
    if(in1.has_value())
    {
        attributes.add_in_1_tensor_uid(*in1);
    }
    attributes.add_out_0_tensor_uid(out);
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

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, X_UID, nullptr, spec.dtype, &xStrides, &xDims));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, W_UID, nullptr, spec.dtype, &wStrides, &wDims));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder,
        CONV_OUT_UID,
        nullptr,
        spec.convOutDtype.value_or(data_objects::DataType::FLOAT),
        &yStrides,
        &yDims,
        spec.convOutVirtual && !spec.convOnly));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, B_UID, nullptr, spec.biasDtype.value_or(spec.dtype), &bStrides, &bDims));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, Y_UID, nullptr, spec.dtype, &yStrides, &yDims, spec.reluAfter));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, RELU_OUT_UID, nullptr, spec.dtype, &yStrides, &yDims));

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
                                       data_objects::DataType::FLOAT,
                                       data_objects::NodeAttributes::ConvolutionFwdAttributes,
                                       conv.Union()));
    if(!spec.convOnly)
    {
        const int64_t convSide = spec.addReadsX ? X_UID : CONV_OUT_UID;
        auto add = spec.biasFirst ? pointwise(builder, spec.mode, B_UID, convSide, Y_UID)
                                  : pointwise(builder, spec.mode, convSide, B_UID, Y_UID);
        nodes.push_back(
            data_objects::CreateNodeDirect(builder,
                                           "bias",
                                           data_objects::DataType::FLOAT,
                                           data_objects::NodeAttributes::PointwiseAttributes,
                                           add.Union()));
    }
    if(spec.reluAfter)
    {
        auto relu = pointwise(
            builder, data_objects::PointwiseMode::RELU_FWD, Y_UID, std::nullopt, RELU_OUT_UID);
        nodes.push_back(
            data_objects::CreateNodeDirect(builder,
                                           "relu",
                                           data_objects::DataType::FLOAT,
                                           data_objects::NodeAttributes::PointwiseAttributes,
                                           relu.Union()));
    }

    auto name = builder.CreateString("hip_conv_fwd_bias_fma_igemm_rdna_test");
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
    return matchesGraph(HIP_CONV_FWD_BIAS_FMA_IGEMM_RDNA, fixture.context()).has_value();
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
    return matchesKernel(HIP_CONV_FWD_BIAS_FMA_IGEMM_RDNA, fixture.context(), kernel);
}

// ---------------------------------------------------------------------------
// Graph matcher
// ---------------------------------------------------------------------------

TEST(TestHipConvFwdBiasFmaIgemmRdnaGraphMatcher, AcceptsEveryServedShape)
{
    GraphSpec nhwc;
    nhwc.channelsLast = true;
    GraphSpec grouped;
    grouped.groups = 2;
    GraphSpec biasFirst;
    biasFirst.biasFirst = true;
    GraphSpec half;
    half.dtype = data_objects::DataType::HALF;
    GraphSpec halfConvOut = half;
    halfConvOut.convOutDtype = data_objects::DataType::HALF;
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
    EXPECT_TRUE(matches(half)) << "fp32 convolution output";
    EXPECT_TRUE(matches(halfConvOut)) << "convolution output in the data dtype";
    EXPECT_TRUE(matches(bfloat));
    EXPECT_TRUE(matches(rank3));
    EXPECT_TRUE(matches(threeD(GraphSpec{})));
    EXPECT_TRUE(matches(threeD(nhwc)));
}

TEST(TestHipConvFwdBiasFmaIgemmRdnaGraphMatcher, RefusesWhatItCannotServe)
{
    GraphSpec depthwise;
    depthwise.k = 8;
    depthwise.groups = 8;
    EXPECT_FALSE(matches(depthwise)) << "depthwise";

    GraphSpec mul;
    mul.mode = data_objects::PointwiseMode::MUL;
    EXPECT_FALSE(matches(mul)) << "MUL instead of ADD";

    GraphSpec fullTensor;
    fullTensor.biasDims = outputDims(fullTensor);
    EXPECT_FALSE(matches(fullTensor)) << "an ADD of a full tensor";

    GraphSpec wrongChannels;
    wrongChannels.biasDims = std::vector<int64_t>{1, 16, 1, 1};
    EXPECT_FALSE(matches(wrongChannels)) << "a bias of another channel count";

    GraphSpec wrongBiasDtype;
    wrongBiasDtype.dtype = data_objects::DataType::HALF;
    wrongBiasDtype.biasDtype = data_objects::DataType::FLOAT;
    EXPECT_FALSE(matches(wrongBiasDtype)) << "a bias of another dtype";

    GraphSpec otherConvOutDtype;
    otherConvOutDtype.dtype = data_objects::DataType::HALF;
    otherConvOutDtype.convOutDtype = data_objects::DataType::BFLOAT16;
    EXPECT_FALSE(matches(otherConvOutDtype)) << "a convolution output of a third dtype";

    GraphSpec storedConvOut;
    storedConvOut.convOutVirtual = false;
    EXPECT_FALSE(matches(storedConvOut)) << "a convolution output that is stored too";

    GraphSpec addReadsX;
    addReadsX.addReadsX = true;
    EXPECT_FALSE(matches(addReadsX)) << "an ADD that does not read the convolution output";

    GraphSpec convOnly;
    convOnly.convOnly = true;
    EXPECT_FALSE(matches(convOnly)) << "the conv node alone";

    GraphSpec reluAfter;
    reluAfter.reluAfter = true;
    EXPECT_FALSE(matches(reluAfter)) << "conv + bias + relu";
}

TEST(TestHipConvFwdBiasFmaIgemmRdnaBinding, BindsTheOperandsTheBiasAndTheGroupedFlops)
{
    GraphSpec spec = threeD(GraphSpec{});
    spec.groups = 2;
    const GraphFixture fixture(buildGraph(spec));
    const auto bound = matchesGraph(HIP_CONV_FWD_BIAS_FMA_IGEMM_RDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());

    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_BIAS_FMA_IGEMM_RDNA.inputAToken), X_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_BIAS_FMA_IGEMM_RDNA.inputBToken), W_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_BIAS_FMA_IGEMM_RDNA.outputToken), Y_UID)
        << "the ADD output, not the virtual convolution output";
    EXPECT_EQ(tryGetBoundInt(*bound, "conv_fwd_bias.b.uid"), B_UID);
    // 2 x N x K x (D x H x W out) x (C / groups) x taps; padding 1 keeps the output size.
    EXPECT_EQ(tryGetBoundInt(*bound, "conv_fwd_bias.flops"),
              int64_t{2} * 2 * 32 * 4 * 9 * 11 * (8 / 2) * 27);
}

// ---------------------------------------------------------------------------
// Kernel matcher
// ---------------------------------------------------------------------------

TEST(TestHipConvFwdBiasFmaIgemmRdnaKernelMatcher, AdmitsOnlyTheGraphsDtypeLayoutAndTile)
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

SavedLaunchInputs savedValid()
{
    return SavedLaunchInputs{
        std::string(HIP_CONV_FWD_BIAS_FMA_IGEMM_RDNA_DISPATCH_SYMBOL_V1),
        hipConvFwdBiasFmaIgemmRdnaLaunchValues({21, 22, 24, 23}, distinctProblem())};
}

TEST(TestHipConvFwdBiasFmaIgemmRdnaLaunchValues, RoundTripThroughTheirValues)
{
    const auto read = readHipConvFwdBiasFmaIgemmRdnaLaunchInputs(savedValid());
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
}

TEST(TestHipConvFwdBiasFmaIgemmRdnaLaunchValues, SavesTheV1ValueNames)
{
    const auto values = savedValid().values;
    for(const auto* name :
        {"x.uid",    "w.uid",    "b.uid",      "y.uid",      "n",         "c",        "k",
         "groups",   "in_d",     "in_h",       "in_w",       "filter_d",  "filter_h", "filter_w",
         "out_d",    "out_h",    "out_w",      "pad_d",      "pad_h",     "pad_w",    "stride_d",
         "stride_h", "stride_w", "dilation_d", "dilation_h", "dilation_w"})
    {
        EXPECT_EQ(values.count(name), 1U) << name;
    }
    EXPECT_EQ(values.size(), 26U);
}

TEST(TestHipConvFwdBiasFmaIgemmRdnaLaunchValues, RefusesMalformedValues)
{
    const auto refused = [](const SavedLaunchInputs& inputs, const char* what) {
        EXPECT_THROW(readHipConvFwdBiasFmaIgemmRdnaLaunchInputs(inputs),
                     hipdnn_plugin_sdk::HipdnnPluginException)
            << what;
    };

    auto unfusedContract = savedValid();
    unfusedContract.dispatchSymbol = "hipkernel.hip_conv_fwd_fma_igemm_rdna.dispatch.v1";
    refused(unfusedContract, "the unfused engine's contract");

    auto missingBias = savedValid();
    missingBias.values.erase("b.uid");
    refused(missingBias, "no bias uid");

    auto extra = savedValid();
    extra.values["tile_n"] = int64_t{32};
    refused(extra, "a value outside the contract");

    auto zeroExtent = savedValid();
    zeroExtent.values["in_w"] = int64_t{0};
    refused(zeroExtent, "a zero extent");

    auto indivisible = savedValid();
    indivisible.values["groups"] = int64_t{5};
    refused(indivisible, "groups that do not divide c and k");

    auto hugeGrid = savedValid();
    hugeGrid.values["n"] = int64_t{std::numeric_limits<int>::max()};
    refused(hugeGrid, "a grid beyond the launch limits");
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
