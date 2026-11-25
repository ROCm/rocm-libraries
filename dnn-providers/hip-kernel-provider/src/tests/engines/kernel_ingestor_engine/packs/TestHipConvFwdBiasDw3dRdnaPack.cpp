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

#include "engines/kernel_ingestor_engine/packs/HipConvFwdBiasDw3dRdnaLaunchValues.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestHipConvFwdBiasDw3dRdnaPack.cpp
 * @brief Host-only checks of `hipkernel:hip_conv_fwd_bias_dw3d_rdna`: the depthwise 3-D
 *        conv + bias graphs its matcher accepts and refuses, the tokens it binds, the one kernel
 *        its kernel matcher admits per graph, and its saved launch values. The device cases are
 *        IntegrationGpuHipConvFwdBiasDw3dRdna.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hip_kernel_provider::kernel_ingestor_engine::conv_rdna::ConvProblem;
using hip_kernel_provider::kernel_ingestor_engine::conv_rdna::dw3d::Geometry;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::SavedLaunchInputs;
using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr PackSymbols HIP_CONV_FWD_BIAS_DW3D_RDNA{
    "hipkernel:hip_conv_fwd_bias_dw3d_rdna",
    "hipkernel.hip_conv_fwd_bias_dw3d_rdna.graph_match",
    "",
    "hipkernel.hip_conv_fwd_bias_dw3d_rdna.kernel_match",
    "hipkernel.hip_conv_fwd_bias_dw3d_rdna.score",
    "hipkernel.hip_conv_fwd_bias_dw3d_rdna.dispatch",
    "conv_fwd_bias_dw3d.x.uid",
    "conv_fwd_bias_dw3d.w.uid",
    "conv_fwd_bias_dw3d.y.uid"};

constexpr int64_t X_UID = 1;
constexpr int64_t W_UID = 2;
constexpr int64_t CONV_OUT_UID = 3;
constexpr int64_t B_UID = 4;
constexpr int64_t Y_UID = 5;
constexpr int64_t RELU_OUT_UID = 6;

/// One depthwise conv + bias graph. Spatial vectors have one entry per spatial dimension,
/// outermost first. The defaults are an NCDHW fp32 3x3x3 depthwise conv with 16 channels, an
/// fp32 conv output and a (1, K, 1, 1, 1) bias, which the engine serves.
struct GraphSpec
{
    data_objects::DataType dtype = data_objects::DataType::FLOAT;
    data_objects::DataType computeType = data_objects::DataType::FLOAT;
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
                                       spec.computeType,
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

    auto name = builder.CreateString("hip_conv_fwd_bias_dw3d_rdna_test");
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
    return matchesGraph(HIP_CONV_FWD_BIAS_DW3D_RDNA, fixture.context()).has_value();
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
    return matchesKernel(HIP_CONV_FWD_BIAS_DW3D_RDNA, fixture.context(), kernel);
}

// ---------------------------------------------------------------------------
// Graph matcher
// ---------------------------------------------------------------------------

TEST(TestHipConvFwdBiasDw3dRdnaGraphMatcher, AcceptsEveryServedShape)
{
    GraphSpec ndhwc;
    ndhwc.channelsLast = true;
    GraphSpec multiplier;
    multiplier.k = 32;
    GraphSpec biasFirst;
    biasFirst.biasFirst = true;
    GraphSpec half;
    half.dtype = data_objects::DataType::HALF;
    GraphSpec halfConvOut = half;
    halfConvOut.convOutDtype = data_objects::DataType::HALF;
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
    EXPECT_TRUE(matches(half)) << "fp32 convolution output";
    EXPECT_TRUE(matches(halfConvOut)) << "convolution output in the data dtype";
    EXPECT_TRUE(matches(bfloat));
    EXPECT_TRUE(matches(oddFilter)) << "a filter served by the generic variant";
}

TEST(TestHipConvFwdBiasDw3dRdnaGraphMatcher, RefusesWhatItCannotServe)
{
    GraphSpec halfCompute;
    halfCompute.dtype = data_objects::DataType::HALF;
    halfCompute.computeType = data_objects::DataType::HALF;
    EXPECT_FALSE(matches(halfCompute)) << "half compute type";

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

    GraphSpec mul;
    mul.mode = data_objects::PointwiseMode::MUL;
    EXPECT_FALSE(matches(mul)) << "MUL instead of ADD";

    GraphSpec fullTensor;
    fullTensor.biasDims = outputDims(fullTensor);
    EXPECT_FALSE(matches(fullTensor)) << "an ADD of a full tensor";

    GraphSpec wrongChannels;
    wrongChannels.biasDims = std::vector<int64_t>{1, 8, 1, 1, 1};
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

TEST(TestHipConvFwdBiasDw3dRdnaBinding, BindsTheOperandsTheBiasAndTheFlops)
{
    const GraphSpec spec;
    const GraphFixture fixture(buildGraph(spec));
    const auto bound = matchesGraph(HIP_CONV_FWD_BIAS_DW3D_RDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());

    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_BIAS_DW3D_RDNA.inputAToken), X_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_BIAS_DW3D_RDNA.inputBToken), W_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_BIAS_DW3D_RDNA.outputToken), Y_UID)
        << "the ADD output, not the virtual convolution output";
    EXPECT_EQ(tryGetBoundInt(*bound, "conv_fwd_bias_dw3d.b.uid"), B_UID);
    // 2 x N x K x (D x H x W out) x (C / groups) x taps; padding 1 keeps the output size.
    EXPECT_EQ(tryGetBoundInt(*bound, "conv_fwd_bias_dw3d.flops"),
              int64_t{2} * 2 * 16 * 6 * 9 * 11 * 1 * 27);
}

// ---------------------------------------------------------------------------
// Kernel matcher
// ---------------------------------------------------------------------------

TEST(TestHipConvFwdBiasDw3dRdnaKernelMatcher, AdmitsTheVariantOfTheUnfusedEngine)
{
    const GraphSpec spec; // NCDHW fp32 3x3x3 stride 1: tile 2 x 4, uniform-k.
    EXPECT_TRUE(admits(spec, specializedKernel("FLOAT", "CHANNELS_FIRST", {3, 3, 3}, 1, 2, 4, 1)));
    EXPECT_FALSE(admits(spec, specializedKernel("FLOAT", "CHANNELS_FIRST", {3, 3, 3}, 1, 1, 4, 1)))
        << "another depth tile";
    EXPECT_FALSE(admits(spec, specializedKernel("HALF", "CHANNELS_FIRST", {3, 3, 3}, 1, 2, 4, 1)))
        << "another dtype";
    EXPECT_FALSE(admits(spec, genericKernel("FLOAT", "CHANNELS_FIRST"))) << "the generic variant";

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

SavedLaunchInputs savedValid(const Geometry& geometry = {false, 2, 8, true})
{
    return SavedLaunchInputs{
        std::string(HIP_CONV_FWD_BIAS_DW3D_RDNA_DISPATCH_SYMBOL_V1),
        hipConvFwdBiasDw3dRdnaLaunchValues({21, 22, 24, 23}, distinctProblem(), geometry)};
}

TEST(TestHipConvFwdBiasDw3dRdnaLaunchValues, RoundTripThroughTheirValues)
{
    const auto read = readHipConvFwdBiasDw3dRdnaLaunchInputs(savedValid());
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

    const auto generic = readHipConvFwdBiasDw3dRdnaLaunchInputs(savedValid({true, 1, 4, false}));
    EXPECT_TRUE(generic.geometry.generic);
}

TEST(TestHipConvFwdBiasDw3dRdnaLaunchValues, SavesTheV1ValueNames)
{
    const auto values = savedValid().values;
    for(const auto* name :
        {"x.uid",      "w.uid",    "b.uid",    "y.uid",    "n",          "c",
         "k",          "in_d",     "in_h",     "in_w",     "filter_d",   "filter_h",
         "filter_w",   "out_d",    "out_h",    "out_w",    "pad_d",      "pad_h",
         "pad_w",      "stride_d", "stride_h", "stride_w", "dilation_d", "dilation_h",
         "dilation_w", "generic",  "tile_d",   "tile_w",   "uniform_k"})
    {
        EXPECT_EQ(values.count(name), 1U) << name;
    }
    EXPECT_EQ(values.size(), 29U);
}

TEST(TestHipConvFwdBiasDw3dRdnaLaunchValues, RefusesMalformedValues)
{
    const auto refused = [](const SavedLaunchInputs& inputs, const char* what) {
        EXPECT_THROW(readHipConvFwdBiasDw3dRdnaLaunchInputs(inputs),
                     hipdnn_plugin_sdk::HipdnnPluginException)
            << what;
    };

    auto unfusedContract = savedValid();
    unfusedContract.dispatchSymbol = "hipkernel.hip_conv_fwd_dw3d_rdna.dispatch.v1";
    refused(unfusedContract, "the unfused engine's contract");

    auto missingBias = savedValid();
    missingBias.values.erase("b.uid");
    refused(missingBias, "no bias uid");

    auto extra = savedValid();
    extra.values["groups"] = int64_t{12};
    refused(extra, "a value outside the contract");

    auto zeroExtent = savedValid();
    zeroExtent.values["in_w"] = int64_t{0};
    refused(zeroExtent, "a zero extent");

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
