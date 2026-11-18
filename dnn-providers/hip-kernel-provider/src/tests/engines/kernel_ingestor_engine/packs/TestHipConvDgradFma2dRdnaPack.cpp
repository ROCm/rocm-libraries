// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>

#include "engines/kernel_ingestor_engine/packs/HipConvDgradFma2dRdnaLaunchValues.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestHipConvDgradFma2dRdnaPack.cpp
 * @brief Host-only checks of `hipkernel:hip_conv_dgrad_fma_2d_rdna`: the graphs its matcher
 *        accepts and refuses, the channel tile each problem takes, the one kernel its kernel
 *        matcher admits per graph, and its saved launch values. The device cases are
 *        IntegrationGpuHipConvDgradFma2dRdna.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hip_kernel_provider::kernel_ingestor_engine::conv_rdna::ConvProblem;
using hip_kernel_provider::kernel_ingestor_engine::conv_rdna::Layout;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::SavedLaunchInputs;
using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

// In the forward view the matcher binds dx as x and dy as y.
constexpr PackSymbols HIP_CONV_DGRAD_FMA_2D_RDNA{
    "hipkernel:hip_conv_dgrad_fma_2d_rdna",
    "hipkernel.hip_conv_dgrad_fma_2d_rdna.graph_match",
    "",
    "hipkernel.hip_conv_dgrad_fma_2d_rdna.kernel_match",
    "hipkernel.hip_conv_dgrad_fma_2d_rdna.score",
    "hipkernel.hip_conv_dgrad_fma_2d_rdna.dispatch",
    "conv_dgrad_fma_2d.y.uid",
    "conv_dgrad_fma_2d.w.uid",
    "conv_dgrad_fma_2d.x.uid"};

constexpr int64_t DY_UID = 1;
constexpr int64_t W_UID = 2;
constexpr int64_t DX_UID = 3;

/// One conv_dgrad graph in the forward view: dx is (n, c, in), w is (k, c / groups, filter),
/// dy is (n, k, out). Spatial vectors have one entry per spatial dimension, outermost first.
/// The defaults are an NCHW fp32 3 x 3 stride 2 problem with 64 dx and 32 dy channels.
struct GraphSpec
{
    data_objects::DataType dtype = data_objects::DataType::FLOAT;
    bool channelsLast = false;
    bool forward = false;
    int64_t n = 2;
    int64_t c = 64;
    int64_t k = 32;
    int64_t groups = 1;
    std::vector<int64_t> in{14, 16};
    std::vector<int64_t> filter{3, 3};
    std::vector<int64_t> prePad{1, 1};
    std::vector<int64_t> postPad{1, 1};
    std::vector<int64_t> stride{2, 2};
    std::vector<int64_t> dilation{1, 1};
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

/// The conv_dgrad graph of @p spec, or with spec.forward the conv_fprop of the same problem.
flatbuffers::FlatBufferBuilder buildGraph(const GraphSpec& spec)
{
    std::vector<int64_t> dxDims{spec.n, spec.c};
    dxDims.insert(dxDims.end(), spec.in.begin(), spec.in.end());
    std::vector<int64_t> wDims{spec.k, spec.c / spec.groups};
    wDims.insert(wDims.end(), spec.filter.begin(), spec.filter.end());
    const auto dyDims = outputDims(spec);
    const auto dxStrides = packedStrides(dxDims, spec.channelsLast);
    const auto wStrides = packedStrides(wDims, spec.channelsLast);
    const auto dyStrides = packedStrides(dyDims, spec.channelsLast);

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, DY_UID, nullptr, spec.dtype, &dyStrides, &dyDims));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, W_UID, nullptr, spec.dtype, &wStrides, &wDims));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, DX_UID, nullptr, spec.dtype, &dxStrides, &dxDims));

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    if(spec.forward)
    {
        // x is the dx tensor, y the dy tensor.
        auto attributes = data_objects::CreateConvolutionFwdAttributesDirect(
            builder,
            DX_UID,
            W_UID,
            DY_UID,
            &spec.prePad,
            &spec.postPad,
            &spec.stride,
            &spec.dilation,
            data_objects::ConvMode::CROSS_CORRELATION);
        nodes.push_back(
            data_objects::CreateNodeDirect(builder,
                                           "conv_fwd",
                                           spec.dtype,
                                           data_objects::NodeAttributes::ConvolutionFwdAttributes,
                                           attributes.Union()));
    }
    else
    {
        auto attributes = data_objects::CreateConvolutionBwdAttributesDirect(
            builder,
            DY_UID,
            W_UID,
            DX_UID,
            &spec.prePad,
            &spec.postPad,
            &spec.stride,
            &spec.dilation,
            data_objects::ConvMode::CROSS_CORRELATION);
        nodes.push_back(
            data_objects::CreateNodeDirect(builder,
                                           "conv_dgrad",
                                           spec.dtype,
                                           data_objects::NodeAttributes::ConvolutionBwdAttributes,
                                           attributes.Union()));
    }

    auto name = builder.CreateString("hip_conv_dgrad_fma_2d_rdna_test");
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
    return matchesGraph(HIP_CONV_DGRAD_FMA_2D_RDNA, fixture.context()).has_value();
}

/// The problem @p spec poses in the forward view (dims only; dtype and layout set), with the
/// spatial values right-aligned as convProblem keeps them.
ConvProblem problemOf(const GraphSpec& spec)
{
    ConvProblem problem;
    problem.spatialRank = static_cast<int>(spec.in.size());
    problem.dtype = spec.dtype;
    problem.layout = spec.channelsLast ? Layout::CHANNELS_LAST : Layout::CHANNELS_FIRST;
    problem.n = static_cast<int>(spec.n);
    problem.c = static_cast<int>(spec.c);
    problem.k = static_cast<int>(spec.k);
    problem.groups = static_cast<int>(spec.groups);
    const auto yDims = outputDims(spec);
    const size_t first = 3 - spec.in.size();
    for(size_t i = 0; i < spec.in.size(); ++i)
    {
        problem.in[first + i] = static_cast<int>(spec.in[i]);
        problem.filter[first + i] = static_cast<int>(spec.filter[i]);
        problem.out[first + i] = static_cast<int>(yDims[2 + i]);
        problem.pad[first + i] = static_cast<int>(spec.prePad[i]);
        problem.stride[first + i] = static_cast<int>(spec.stride[i]);
        problem.dilation[first + i] = static_cast<int>(spec.dilation[i]);
    }
    return problem;
}

KernelDefinition kernelOf(const std::string& dtype,
                          const std::string& layout,
                          int64_t tileN,
                          const std::string& family = "dgrad_fma_igemm")
{
    KernelDefinition kernel;
    kernel.metadata
        = {{"kernel_family", family}, {"dtype", dtype}, {"layout", layout}, {"tile_n", tileN}};
    return kernel;
}

bool admits(const GraphSpec& spec, const KernelDefinition& kernel)
{
    const GraphFixture fixture(buildGraph(spec));
    return matchesKernel(HIP_CONV_DGRAD_FMA_2D_RDNA, fixture.context(), kernel);
}

/// The 1-D shape hip_conv_dgrad_fma_1d_rdna takes: rank 4 with unit H and no H padding.
GraphSpec unitHeight()
{
    GraphSpec spec;
    spec.in = {1, 406};
    spec.filter = {1, 8};
    spec.prePad = {0, 0};
    spec.postPad = {0, 0};
    spec.stride = {1, 4};
    return spec;
}

// ---------------------------------------------------------------------------
// Graph matcher
// ---------------------------------------------------------------------------

TEST(TestHipConvDgradFma2dRdnaGraphMatcher, AcceptsTwoDimensionalDgrad)
{
    GraphSpec nhwc;
    nhwc.channelsLast = true;
    GraphSpec half;
    half.dtype = data_objects::DataType::HALF;
    GraphSpec bfloat;
    bfloat.dtype = data_objects::DataType::BFLOAT16;
    GraphSpec grouped;
    grouped.groups = 8;
    grouped.k = 64;
    GraphSpec twoPerGroup;
    twoPerGroup.groups = 32;
    twoPerGroup.k = 32;
    GraphSpec oneChannel;
    oneChannel.c = 1;
    GraphSpec dilated;
    dilated.dilation = {2, 3};
    dilated.stride = {1, 1};

    EXPECT_TRUE(matches(GraphSpec{}));
    EXPECT_TRUE(matches(nhwc));
    EXPECT_TRUE(matches(half));
    EXPECT_TRUE(matches(bfloat));
    EXPECT_TRUE(matches(grouped)) << "8 dx channels per group";
    EXPECT_TRUE(matches(twoPerGroup)) << "2 dx channels per group";
    EXPECT_TRUE(matches(oneChannel)) << "1 dx channel, groups == 1";
    EXPECT_TRUE(matches(dilated));

    // Shapes with a unit H axis the 1-D engine declines: a tall filter over a unit-H dy, and
    // H padding.
    auto tallFilter = unitHeight();
    tallFilter.in[0] = 3;
    tallFilter.filter[0] = 3;
    EXPECT_TRUE(matches(tallFilter)) << "H filter 3";
    auto padH = unitHeight();
    padH.prePad[0] = 1;
    padH.postPad[0] = 1;
    padH.stride[0] = 3;
    EXPECT_TRUE(matches(padH)) << "unit H with H padding";
}

TEST(TestHipConvDgradFma2dRdnaGraphMatcher, RefusesWhatItCannotServe)
{
    GraphSpec depthwise;
    depthwise.groups = 64;
    depthwise.k = 64;
    EXPECT_FALSE(matches(depthwise)) << "depthwise";

    GraphSpec multiplier;
    multiplier.groups = 64;
    multiplier.k = 128;
    EXPECT_FALSE(matches(multiplier)) << "one dx channel per group, two dy channels";

    EXPECT_FALSE(matches(unitHeight())) << "the 1-D shape";

    GraphSpec oneD;
    oneD.in = {406};
    oneD.filter = {8};
    oneD.prePad = {0};
    oneD.postPad = {0};
    oneD.stride = {4};
    oneD.dilation = {1};
    EXPECT_FALSE(matches(oneD)) << "rank 3";

    GraphSpec threeD;
    threeD.in = {4, 8, 10};
    threeD.filter = {3, 3, 3};
    threeD.prePad = {1, 1, 1};
    threeD.postPad = {1, 1, 1};
    threeD.stride = {1, 1, 1};
    threeD.dilation = {1, 1, 1};
    EXPECT_FALSE(matches(threeD)) << "3-D";

    GraphSpec forward;
    forward.forward = true;
    EXPECT_FALSE(matches(forward)) << "the same problem as conv_fprop";

    GraphSpec manyGroups;
    manyGroups.groups = 70000;
    manyGroups.c = int64_t{70000} * 2;
    manyGroups.k = 70000;
    EXPECT_FALSE(matches(manyGroups)) << "a grid y beyond 65535";

    GraphSpec manyPositions;
    manyPositions.n = int64_t{1} << 21;
    manyPositions.in = {32, 32};
    manyPositions.stride = {1, 1};
    EXPECT_FALSE(matches(manyPositions)) << "a grid x beyond the launch limit";
}

TEST(TestHipConvDgradFma2dRdnaBinding, BindsTheOperandsAndTheFlops)
{
    const GraphSpec spec;
    const GraphFixture fixture(buildGraph(spec));
    const auto bound = matchesGraph(HIP_CONV_DGRAD_FMA_2D_RDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_DGRAD_FMA_2D_RDNA.inputAToken), DY_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_DGRAD_FMA_2D_RDNA.inputBToken), W_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_DGRAD_FMA_2D_RDNA.outputToken), DX_UID);
    // 2 x N x K x (dy positions) x (C / groups) x taps; dy is 7 x 8.
    EXPECT_EQ(tryGetBoundInt(*bound, "conv_dgrad_fma_2d.flops"),
              int64_t{2} * 2 * 32 * 7 * 8 * 64 * 9);
}

// ---------------------------------------------------------------------------
// Channel tile
// ---------------------------------------------------------------------------

TEST(TestHipConvDgradFma2dRdnaTile, IsTheNarrowestThatHoldsOneGroup)
{
    // dx channels, groups -> tile
    const std::vector<std::pair<std::pair<int64_t, int64_t>, int64_t>> expected{{{3, 1}, 32},
                                                                                {{32, 1}, 32},
                                                                                {{33, 1}, 64},
                                                                                {{64, 1}, 64},
                                                                                {{65, 1}, 128},
                                                                                {{4096, 1}, 128},
                                                                                {{128, 2}, 64},
                                                                                {{64, 4}, 32},
                                                                                {{64, 32}, 32}};
    for(const auto& [shape, tile] : expected)
    {
        GraphSpec spec;
        spec.c = shape.first;
        spec.groups = shape.second;
        spec.k = 16 * shape.second;
        EXPECT_EQ(hipConvDgradFma2dRdnaTileN(problemOf(spec)), tile)
            << shape.first << " channels, " << shape.second << " groups";
    }
}

// ---------------------------------------------------------------------------
// Kernel matcher
// ---------------------------------------------------------------------------

TEST(TestHipConvDgradFma2dRdnaKernelMatcher, AdmitsOnlyTheKernelOfTheProblem)
{
    const GraphSpec spec; // NCHW fp32, 64 dx channels: tile 64.
    EXPECT_TRUE(admits(spec, kernelOf("FLOAT", "CHANNELS_FIRST", 64)));
    EXPECT_FALSE(admits(spec, kernelOf("FLOAT", "CHANNELS_FIRST", 128))) << "another tile";
    EXPECT_FALSE(admits(spec, kernelOf("FLOAT", "CHANNELS_FIRST", 32))) << "another tile";
    EXPECT_FALSE(admits(spec, kernelOf("HALF", "CHANNELS_FIRST", 64))) << "another dtype";
    EXPECT_FALSE(admits(spec, kernelOf("FLOAT", "CHANNELS_LAST", 64))) << "another layout";
    EXPECT_FALSE(admits(spec, kernelOf("FLOAT", "CHANNELS_FIRST", 64, "fma_igemm")))
        << "another family";

    GraphSpec nhwc;
    nhwc.channelsLast = true;
    nhwc.dtype = data_objects::DataType::BFLOAT16;
    nhwc.c = 200;
    EXPECT_TRUE(admits(nhwc, kernelOf("BFLOAT16", "CHANNELS_LAST", 128)));

    GraphSpec depthwise;
    depthwise.groups = 64;
    depthwise.k = 64;
    EXPECT_FALSE(admits(depthwise, kernelOf("FLOAT", "CHANNELS_FIRST", 32)))
        << "a graph the graph matcher refuses";
}

// ---------------------------------------------------------------------------
// Saved launch values
// ---------------------------------------------------------------------------

/// A problem whose every field differs, so a value read into the wrong field shows.
ConvProblem distinctProblem()
{
    ConvProblem problem;
    problem.spatialRank = 2;
    problem.n = 10;
    problem.c = 48;
    problem.k = 36;
    problem.groups = 12;
    problem.in = {1, 23, 29};
    problem.filter = {1, 3, 5};
    problem.out = {1, 11, 13};
    problem.pad = {0, 4, 6};
    problem.stride = {1, 2, 7};
    problem.dilation = {1, 8, 9};
    return problem;
}

SavedLaunchInputs savedValid()
{
    return SavedLaunchInputs{std::string(HIP_CONV_DGRAD_FMA_2D_RDNA_DISPATCH_SYMBOL_V1),
                             hipConvDgradFma2dRdnaLaunchValues({21, 22, 24}, distinctProblem())};
}

TEST(TestHipConvDgradFma2dRdnaLaunchValues, RoundTripThroughTheirValues)
{
    const auto read = readHipConvDgradFma2dRdnaLaunchInputs(savedValid());
    const auto expected = distinctProblem();

    EXPECT_EQ(read.binding.dy, 21);
    EXPECT_EQ(read.binding.w, 22);
    EXPECT_EQ(read.binding.dx, 24);
    EXPECT_EQ(read.problem.spatialRank, 2);
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

TEST(TestHipConvDgradFma2dRdnaLaunchValues, SavesTheV1ValueNames)
{
    const auto values = savedValid().values;
    for(const auto* name : {"dy.uid",
                            "w.uid",
                            "dx.uid",
                            "n",
                            "c",
                            "k",
                            "groups",
                            "in_h",
                            "in_w",
                            "filter_h",
                            "filter_w",
                            "out_h",
                            "out_w",
                            "pad_h",
                            "pad_w",
                            "stride_h",
                            "stride_w",
                            "dilation_h",
                            "dilation_w"})
    {
        EXPECT_EQ(values.count(name), 1U) << name;
    }
    EXPECT_EQ(values.size(), 19U);
}

TEST(TestHipConvDgradFma2dRdnaLaunchValues, RefusesMalformedValues)
{
    const auto refused = [](const SavedLaunchInputs& inputs, const char* what) {
        EXPECT_THROW(readHipConvDgradFma2dRdnaLaunchInputs(inputs),
                     hipdnn_plugin_sdk::HipdnnPluginException)
            << what;
    };

    auto otherContract = savedValid();
    otherContract.dispatchSymbol = "hipkernel.hip_conv_dgrad_fma_2d_rdna.dispatch.v2";
    refused(otherContract, "another contract");

    auto missing = savedValid();
    missing.values.erase("dilation_h");
    refused(missing, "a missing value");

    auto extra = savedValid();
    extra.values["in_d"] = int64_t{1};
    refused(extra, "a value outside the contract");

    auto notMultiple = savedValid();
    notMultiple.values["groups"] = int64_t{5};
    refused(notMultiple, "channels not a multiple of the group count");

    auto zeroStride = savedValid();
    zeroStride.values["stride_w"] = int64_t{0};
    refused(zeroStride, "stride 0");

    auto negativePad = savedValid();
    negativePad.values["pad_h"] = int64_t{-1};
    refused(negativePad, "negative padding");

    auto manyGroups = savedValid();
    manyGroups.values["groups"] = int64_t{70000};
    manyGroups.values["c"] = int64_t{70000} * 2;
    manyGroups.values["k"] = int64_t{70000};
    refused(manyGroups, "a grid y beyond 65535");

    auto manyPositions = savedValid();
    manyPositions.values["n"] = int64_t{1} << 21;
    manyPositions.values["in_h"] = int64_t{64};
    manyPositions.values["in_w"] = int64_t{112};
    refused(manyPositions, "a grid x beyond the launch limit");

    auto wrappingStrides = savedValid();
    wrappingStrides.values["stride_h"] = int64_t{1} << 30;
    wrappingStrides.values["stride_w"] = int64_t{1} << 30;
    refused(wrappingStrides, "strides whose product exceeds the grid");

    auto hugeFilter = savedValid();
    hugeFilter.values["filter_h"] = int64_t{50000};
    hugeFilter.values["filter_w"] = int64_t{50000};
    refused(hugeFilter, "more taps x K iterations than the kernel counts in int");
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
