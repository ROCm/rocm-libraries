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
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>

#include "engines/kernel_ingestor_engine/packs/HipConvFwdDw3dRdnaLaunchValues.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestHipConvFwdDw3dRdnaPack.cpp
 * @brief Host-only checks of `hipkernel:hip_conv_fwd_dw3d_rdna`: the graphs its matcher
 *        accepts and refuses, the variant each problem takes (specialized or generic, and its
 *        tile), the one kernel its kernel matcher admits per graph, and its saved launch
 *        values. The device cases are IntegrationGpuHipConvFwdDw3dRdna.
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

constexpr PackSymbols HIP_CONV_FWD_DW3D_RDNA{"hipkernel:hip_conv_fwd_dw3d_rdna",
                                             "hipkernel.hip_conv_fwd_dw3d_rdna.graph_match",
                                             "",
                                             "hipkernel.hip_conv_fwd_dw3d_rdna.kernel_match",
                                             "hipkernel.hip_conv_fwd_dw3d_rdna.score",
                                             "hipkernel.hip_conv_fwd_dw3d_rdna.dispatch",
                                             "conv_fwd_dw3d.x.uid",
                                             "conv_fwd_dw3d.w.uid",
                                             "conv_fwd_dw3d.y.uid"};

constexpr int64_t X_UID = 1;
constexpr int64_t W_UID = 2;
constexpr int64_t Y_UID = 3;

/// One conv-forward graph. Spatial vectors have one entry per spatial dimension, outermost
/// first. The defaults are an NCDHW fp32 3x3x3 depthwise conv with 16 channels.
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

flatbuffers::FlatBufferBuilder buildGraph(const GraphSpec& spec)
{
    std::vector<int64_t> xDims{spec.n, spec.c};
    xDims.insert(xDims.end(), spec.in.begin(), spec.in.end());
    std::vector<int64_t> wDims{spec.k, spec.c / spec.groups};
    wDims.insert(wDims.end(), spec.filter.begin(), spec.filter.end());
    const auto yDims = outputDims(spec);
    const auto xStrides = packedStrides(xDims, spec.channelsLast);
    const auto wStrides = packedStrides(wDims, spec.channelsLast);
    const auto yStrides = packedStrides(yDims, spec.channelsLast);

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, X_UID, nullptr, spec.dtype, &xStrides, &xDims));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, W_UID, nullptr, spec.dtype, &wStrides, &wDims));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, Y_UID, nullptr, spec.dtype, &yStrides, &yDims));

    auto attributes = data_objects::CreateConvolutionFwdAttributesDirect(
        builder,
        X_UID,
        W_UID,
        Y_UID,
        &spec.prePad,
        &spec.postPad,
        &spec.stride,
        &spec.dilation,
        data_objects::ConvMode::CROSS_CORRELATION);
    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(
        data_objects::CreateNodeDirect(builder,
                                       "conv_fwd",
                                       spec.dtype,
                                       data_objects::NodeAttributes::ConvolutionFwdAttributes,
                                       attributes.Union()));

    auto name = builder.CreateString("hip_conv_fwd_dw3d_rdna_test");
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
    return matchesGraph(HIP_CONV_FWD_DW3D_RDNA, fixture.context()).has_value();
}

/// The problem @p spec poses, as the engine reads it (dims only; dtype and layout set).
ConvProblem problemOf(const GraphSpec& spec)
{
    ConvProblem problem;
    problem.spatialRank = 3;
    problem.dtype = spec.dtype;
    problem.layout = spec.channelsLast ? Layout::CHANNELS_LAST : Layout::CHANNELS_FIRST;
    problem.n = static_cast<int>(spec.n);
    problem.c = static_cast<int>(spec.c);
    problem.k = static_cast<int>(spec.k);
    problem.groups = static_cast<int>(spec.groups);
    const auto yDims = outputDims(spec);
    for(size_t i = 0; i < 3; ++i)
    {
        problem.in[i] = static_cast<int>(spec.in[i]);
        problem.filter[i] = static_cast<int>(spec.filter[i]);
        problem.out[i] = static_cast<int>(yDims[2 + i]);
        problem.pad[i] = static_cast<int>(spec.prePad[i]);
        problem.stride[i] = static_cast<int>(spec.stride[i]);
        problem.dilation[i] = static_cast<int>(spec.dilation[i]);
    }
    return problem;
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
    return matchesKernel(HIP_CONV_FWD_DW3D_RDNA, fixture.context(), kernel);
}

// ---------------------------------------------------------------------------
// Graph matcher
// ---------------------------------------------------------------------------

TEST(TestHipConvFwdDw3dRdnaGraphMatcher, AcceptsDepthwise3d)
{
    GraphSpec ndhwc;
    ndhwc.channelsLast = true;
    GraphSpec multiplier;
    multiplier.k = 32;
    GraphSpec half;
    half.dtype = data_objects::DataType::HALF;
    GraphSpec bfloat;
    bfloat.dtype = data_objects::DataType::BFLOAT16;
    GraphSpec oddFilter;
    oddFilter.filter = {2, 3, 4};
    oddFilter.prePad = {0, 1, 2};
    oddFilter.postPad = {1, 1, 1};

    EXPECT_TRUE(matches(GraphSpec{}));
    EXPECT_TRUE(matches(ndhwc));
    EXPECT_TRUE(matches(multiplier)) << "K = 2 C";
    EXPECT_TRUE(matches(half));
    EXPECT_TRUE(matches(bfloat));
    EXPECT_TRUE(matches(oddFilter)) << "a filter served by the generic variant";
}

TEST(TestHipConvFwdDw3dRdnaGraphMatcher, RefusesWhatItCannotServe)
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

    GraphSpec singleChannel;
    singleChannel.c = 1;
    singleChannel.k = 1;
    singleChannel.groups = 1;
    EXPECT_FALSE(matches(singleChannel)) << "one channel is not a grouped convolution";
}

TEST(TestHipConvFwdDw3dRdnaBinding, BindsTheOperandsAndTheFlops)
{
    const GraphSpec spec;
    const GraphFixture fixture(buildGraph(spec));
    const auto bound = matchesGraph(HIP_CONV_FWD_DW3D_RDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_DW3D_RDNA.inputAToken), X_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_DW3D_RDNA.inputBToken), W_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_DW3D_RDNA.outputToken), Y_UID);
    // 2 x N x K x (D x H x W out) x (C / groups) x taps; padding 1 keeps the output size.
    EXPECT_EQ(tryGetBoundInt(*bound, "conv_fwd_dw3d.flops"),
              int64_t{2} * 2 * 16 * 6 * 9 * 11 * 1 * 27);
}

// ---------------------------------------------------------------------------
// Variant selection
// ---------------------------------------------------------------------------

TEST(TestHipConvFwdDw3dRdnaGeometry, PicksTheSpecializedTile)
{
    // NCDHW fp32 3x3x3: two depths, 4 columns, uniform-k.
    auto geometry = hipConvFwdDw3dRdnaGeometry(problemOf(GraphSpec{}));
    EXPECT_FALSE(geometry.generic);
    EXPECT_EQ(geometry.tileD, 2);
    EXPECT_EQ(geometry.tileW, 4);
    EXPECT_TRUE(geometry.uniformK);

    // NCDHW fp16 5x5x5: two depths, 8 columns for the 5-wide filter.
    GraphSpec half555;
    half555.dtype = data_objects::DataType::HALF;
    half555.filter = {5, 5, 5};
    half555.prePad = half555.postPad = {2, 2, 2};
    geometry = hipConvFwdDw3dRdnaGeometry(problemOf(half555));
    EXPECT_FALSE(geometry.generic);
    EXPECT_EQ(geometry.tileD, 2);
    EXPECT_EQ(geometry.tileW, 8);

    // A depth stride of 2 keeps the specialization with one depth.
    GraphSpec depthStride;
    depthStride.stride = {2, 1, 1};
    geometry = hipConvFwdDw3dRdnaGeometry(problemOf(depthStride));
    EXPECT_FALSE(geometry.generic);
    EXPECT_EQ(geometry.tileD, 1);

    // NDHWC: one depth, no uniform-k grid.
    GraphSpec ndhwc;
    ndhwc.channelsLast = true;
    geometry = hipConvFwdDw3dRdnaGeometry(problemOf(ndhwc));
    EXPECT_FALSE(geometry.generic);
    EXPECT_EQ(geometry.tileD, 1);
    EXPECT_FALSE(geometry.uniformK);
}

TEST(TestHipConvFwdDw3dRdnaGeometry, FallsBackToTheGenericVariant)
{
    GraphSpec oddFilter;
    oddFilter.filter = {2, 3, 4};
    EXPECT_TRUE(hipConvFwdDw3dRdnaGeometry(problemOf(oddFilter)).generic) << "filter 2x3x4";

    GraphSpec unequalStride;
    unequalStride.stride = {1, 1, 2};
    EXPECT_TRUE(hipConvFwdDw3dRdnaGeometry(problemOf(unequalStride)).generic)
        << "H / W stride 1 / 2";

    GraphSpec stride3;
    stride3.stride = {1, 3, 3};
    EXPECT_TRUE(hipConvFwdDw3dRdnaGeometry(problemOf(stride3)).generic) << "H / W stride 3";

    GraphSpec dilated;
    dilated.dilation = {1, 2, 2};
    EXPECT_TRUE(hipConvFwdDw3dRdnaGeometry(problemOf(dilated)).generic) << "H / W dilation 2";

    // N x K beyond the uniform-k grid: NCDHW falls back, NDHWC keeps its 1-D grid.
    GraphSpec wide;
    wide.n = 2;
    wide.c = wide.k = wide.groups = 33000;
    EXPECT_TRUE(hipConvFwdDw3dRdnaGeometry(problemOf(wide)).generic) << "N x K = 66000, NCDHW";
    wide.channelsLast = true;
    EXPECT_FALSE(hipConvFwdDw3dRdnaGeometry(problemOf(wide)).generic) << "N x K = 66000, NDHWC";
}

// ---------------------------------------------------------------------------
// Kernel matcher
// ---------------------------------------------------------------------------

TEST(TestHipConvFwdDw3dRdnaKernelMatcher, AdmitsOnlyTheVariantOfTheProblem)
{
    const GraphSpec spec; // NCDHW fp32 3x3x3 stride 1: tile 2 x 4, uniform-k.
    EXPECT_TRUE(admits(spec, specializedKernel("FLOAT", "CHANNELS_FIRST", {3, 3, 3}, 1, 2, 4, 1)));
    EXPECT_FALSE(admits(spec, specializedKernel("FLOAT", "CHANNELS_FIRST", {3, 3, 3}, 1, 1, 4, 1)))
        << "another depth tile";
    EXPECT_FALSE(admits(spec, specializedKernel("FLOAT", "CHANNELS_FIRST", {3, 3, 3}, 2, 2, 4, 1)))
        << "another stride";
    EXPECT_FALSE(admits(spec, specializedKernel("FLOAT", "CHANNELS_FIRST", {5, 5, 5}, 1, 2, 4, 1)))
        << "another filter";
    EXPECT_FALSE(admits(spec, specializedKernel("HALF", "CHANNELS_FIRST", {3, 3, 3}, 1, 2, 4, 1)))
        << "another dtype";
    EXPECT_FALSE(admits(spec, genericKernel("FLOAT", "CHANNELS_FIRST"))) << "the generic variant";

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

SavedLaunchInputs savedValid(HipConvFwdDw3dRdnaGeometry geometry = {false, 2, 8, true})
{
    return SavedLaunchInputs{
        std::string(HIP_CONV_FWD_DW3D_RDNA_DISPATCH_SYMBOL_V1),
        hipConvFwdDw3dRdnaLaunchValues({21, 22, 23}, distinctProblem(), geometry)};
}

TEST(TestHipConvFwdDw3dRdnaLaunchValues, RoundTripThroughTheirValues)
{
    const auto read = readHipConvFwdDw3dRdnaLaunchInputs(savedValid());
    const auto expected = distinctProblem();

    EXPECT_EQ(read.binding.x, 21);
    EXPECT_EQ(read.binding.w, 22);
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

    const auto generic = readHipConvFwdDw3dRdnaLaunchInputs(savedValid({true, 1, 4, false}));
    EXPECT_TRUE(generic.geometry.generic);
}

TEST(TestHipConvFwdDw3dRdnaLaunchValues, SavesTheV1ValueNames)
{
    const auto values = savedValid().values;
    for(const auto* name :
        {"x.uid",      "w.uid",      "y.uid",      "n",        "c",        "k",        "in_d",
         "in_h",       "in_w",       "filter_d",   "filter_h", "filter_w", "out_d",    "out_h",
         "out_w",      "pad_d",      "pad_h",      "pad_w",    "stride_d", "stride_h", "stride_w",
         "dilation_d", "dilation_h", "dilation_w", "generic",  "tile_d",   "tile_w",   "uniform_k"})
    {
        EXPECT_EQ(values.count(name), 1U) << name;
    }
    EXPECT_EQ(values.size(), 28U);
}

TEST(TestHipConvFwdDw3dRdnaLaunchValues, RefusesMalformedValues)
{
    const auto refused = [](const SavedLaunchInputs& inputs, const char* what) {
        EXPECT_THROW(readHipConvFwdDw3dRdnaLaunchInputs(inputs),
                     hipdnn_plugin_sdk::HipdnnPluginException)
            << what;
    };

    auto otherContract = savedValid();
    otherContract.dispatchSymbol = "hipkernel.hip_conv_fwd_dw3d_rdna.dispatch.v2";
    refused(otherContract, "another contract");

    auto missing = savedValid();
    missing.values.erase("tile_w");
    refused(missing, "a missing value");

    auto extra = savedValid();
    extra.values["groups"] = int64_t{12};
    refused(extra, "a value outside the contract");

    auto notMultiple = savedValid();
    notMultiple.values["k"] = int64_t{20};
    refused(notMultiple, "K not a multiple of C");

    refused(savedValid({false, 3, 4, true}), "a depth tile of 3");
    refused(savedValid({false, 2, 4, false}), "two depths without the uniform-k grid");
    refused(savedValid({true, 2, 4, false}), "a generic variant with two depths");

    auto hugeGrid = savedValid();
    hugeGrid.values["n"] = int64_t{std::numeric_limits<int>::max()};
    refused(hugeGrid, "a grid beyond the launch limits");
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
