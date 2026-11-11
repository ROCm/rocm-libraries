// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>

#include "engines/kernel_ingestor_engine/packs/HipConvFwdRdnaLaunchValues.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestHipConvFwdRdnaPack.cpp
 * @brief Host-only checks of `hipkernel:hip_conv_fwd_rdna`: the graphs its matcher accepts
 *        and refuses, the tokens it binds, the one kernel its kernel matcher admits per graph,
 *        and its saved launch values. The device cases are IntegrationGpuHipConvFwdRdna.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::MetadataValues;
using hipdnn_plugin_sdk::ingestor::SavedLaunchInputs;
using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr PackSymbols HIP_CONV_FWD_RDNA{"hipkernel:hip_conv_fwd_rdna",
                                        "hipkernel.hip_conv_fwd_rdna.graph_match",
                                        "",
                                        "hipkernel.hip_conv_fwd_rdna.kernel_match",
                                        "hipkernel.hip_conv_fwd_rdna.score",
                                        "hipkernel.hip_conv_fwd_rdna.dispatch",
                                        "conv_fwd.x.uid",
                                        "conv_fwd.w.uid",
                                        "conv_fwd.y.uid"};

constexpr int64_t X_UID = 1;
constexpr int64_t W_UID = 2;
constexpr int64_t Y_UID = 3;

/// One conv-forward graph. Spatial vectors have one entry per spatial dimension, outermost
/// first. The defaults are a 2-D NCHW fp32 conv the engine serves.
struct GraphSpec
{
    data_objects::DataType dtype = data_objects::DataType::FLOAT;
    std::optional<data_objects::DataType> wDtype;
    bool channelsLast = false;
    std::optional<bool> wChannelsLast;
    data_objects::ConvMode mode = data_objects::ConvMode::CROSS_CORRELATION;
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
    std::optional<int64_t> wChannelsOverride;
    std::optional<std::vector<int64_t>> yDimsOverride;
    bool xNotPacked = false;
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
    std::vector<int64_t> wDims{spec.k, spec.wChannelsOverride.value_or(spec.c / spec.groups)};
    wDims.insert(wDims.end(), spec.filter.begin(), spec.filter.end());
    const auto yDims = spec.yDimsOverride.value_or(outputDims(spec));

    auto xStrides = packedStrides(xDims, spec.channelsLast);
    if(spec.xNotPacked)
    {
        xStrides[0] *= 2;
    }
    const auto wStrides = packedStrides(wDims, spec.wChannelsLast.value_or(spec.channelsLast));
    const auto yStrides = packedStrides(yDims, spec.channelsLast);

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, X_UID, nullptr, spec.dtype, &xStrides, &xDims));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, W_UID, nullptr, spec.wDtype.value_or(spec.dtype), &wStrides, &wDims));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, Y_UID, nullptr, spec.dtype, &yStrides, &yDims));

    auto attributes = data_objects::CreateConvolutionFwdAttributesDirect(builder,
                                                                         X_UID,
                                                                         W_UID,
                                                                         Y_UID,
                                                                         &spec.prePad,
                                                                         &spec.postPad,
                                                                         &spec.stride,
                                                                         &spec.dilation,
                                                                         spec.mode);
    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(
        data_objects::CreateNodeDirect(builder,
                                       "conv_fwd",
                                       spec.dtype,
                                       data_objects::NodeAttributes::ConvolutionFwdAttributes,
                                       attributes.Union()));

    auto name = builder.CreateString("hip_conv_fwd_rdna_test");
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
    return matchesGraph(HIP_CONV_FWD_RDNA, fixture.context()).has_value();
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

KernelDefinition fmaKernel(const std::string& dtype,
                           const std::string& layout,
                           int64_t tileN,
                           const std::string& family = "fma_igemm")
{
    KernelDefinition kernel;
    kernel.metadata
        = {{"kernel_family", family}, {"dtype", dtype}, {"layout", layout}, {"tile_n", tileN}};
    return kernel;
}

bool admits(const GraphSpec& spec, const KernelDefinition& kernel)
{
    const GraphFixture fixture(buildGraph(spec));
    return matchesKernel(HIP_CONV_FWD_RDNA, fixture.context(), kernel);
}

// ---------------------------------------------------------------------------
// Graph matcher
// ---------------------------------------------------------------------------

TEST(TestHipConvFwdRdnaGraphMatcher, AcceptsEveryServedShape)
{
    GraphSpec nhwc;
    nhwc.channelsLast = true;
    GraphSpec grouped;
    grouped.groups = 2;
    GraphSpec asymmetric;
    asymmetric.prePad = {0, 2};
    asymmetric.postPad = {1, 0};
    GraphSpec strided;
    strided.stride = {2, 3};
    strided.dilation = {2, 1};
    GraphSpec conv1d;
    conv1d.in = {1, 40};
    conv1d.filter = {1, 5};
    conv1d.prePad = {0, 2};
    conv1d.postPad = {0, 2};
    GraphSpec half;
    half.dtype = data_objects::DataType::HALF;
    GraphSpec bfloat;
    bfloat.dtype = data_objects::DataType::BFLOAT16;

    EXPECT_TRUE(matches(GraphSpec{}));
    EXPECT_TRUE(matches(nhwc));
    EXPECT_TRUE(matches(grouped));
    EXPECT_TRUE(matches(asymmetric));
    EXPECT_TRUE(matches(strided));
    EXPECT_TRUE(matches(conv1d));
    EXPECT_TRUE(matches(half));
    EXPECT_TRUE(matches(bfloat));
    EXPECT_TRUE(matches(threeD(GraphSpec{})));
    EXPECT_TRUE(matches(threeD(nhwc)));
    EXPECT_TRUE(matches(threeD(grouped)));

    // Rank 3 (NCL / NLC), the 1-D form some frontends send instead of H = 1.
    GraphSpec rank3;
    rank3.in = {40};
    rank3.filter = {5};
    rank3.prePad = {2};
    rank3.postPad = {1};
    rank3.stride = {2};
    rank3.dilation = {3};
    EXPECT_TRUE(matches(rank3));
    rank3.channelsLast = true;
    EXPECT_TRUE(matches(rank3));
}

TEST(TestHipConvFwdRdnaGraphMatcher, RefusesWhatTheKernelCannotServe)
{
    GraphSpec depthwise;
    depthwise.c = 8;
    depthwise.k = 8;
    depthwise.groups = 8;
    EXPECT_FALSE(matches(depthwise)) << "depthwise";

    GraphSpec narrowGroups;
    narrowGroups.k = 16;
    narrowGroups.groups = 2;
    EXPECT_FALSE(matches(narrowGroups)) << "8 output channels per group";

    GraphSpec convolution;
    convolution.mode = data_objects::ConvMode::CONVOLUTION;
    EXPECT_FALSE(matches(convolution)) << "convolution mode";

    GraphSpec mixedDtype;
    mixedDtype.wDtype = data_objects::DataType::HALF;
    EXPECT_FALSE(matches(mixedDtype)) << "mixed dtypes";

    GraphSpec mixedLayout;
    mixedLayout.wChannelsLast = true;
    EXPECT_FALSE(matches(mixedLayout)) << "w channels-last, x and y channels-first";

    GraphSpec notPacked;
    notPacked.xNotPacked = true;
    EXPECT_FALSE(matches(notPacked)) << "non-packed x";

    GraphSpec wrongOutput;
    wrongOutput.yDimsOverride = std::vector<int64_t>{2, 32, 9, 10};
    EXPECT_FALSE(matches(wrongOutput)) << "y narrower than the output";

    GraphSpec indivisibleChannels;
    indivisibleChannels.wChannelsOverride = 3;
    EXPECT_FALSE(matches(indivisibleChannels)) << "x channels not a multiple of w channels";

    GraphSpec rank2;
    rank2.n = 16;
    rank2.in = {};
    rank2.filter = {};
    rank2.prePad = {};
    rank2.postPad = {};
    rank2.stride = {};
    rank2.dilation = {};
    EXPECT_FALSE(matches(rank2)) << "rank 2";
}

TEST(TestHipConvFwdRdnaBinding, BindsTheOperandsAndTheGroupedFlops)
{
    GraphSpec spec = threeD(GraphSpec{});
    spec.groups = 2;
    const GraphFixture fixture(buildGraph(spec));
    const auto bound = matchesGraph(HIP_CONV_FWD_RDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());

    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_RDNA.inputAToken), X_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_RDNA.inputBToken), W_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_RDNA.outputToken), Y_UID);
    // 2 x N x K x (D x H x W out) x (C / groups) x taps; padding 1 keeps the output size.
    EXPECT_EQ(tryGetBoundInt(*bound, "conv_fwd.flops"),
              int64_t{2} * 2 * 32 * 4 * 9 * 11 * (8 / 2) * 27);
}

// ---------------------------------------------------------------------------
// Kernel matcher
// ---------------------------------------------------------------------------

TEST(TestHipConvFwdRdnaKernelMatcher, AdmitsOnlyTheGraphsFamilyDtypeLayoutAndTile)
{
    const GraphSpec spec; // fp32, channels-first, 32 output channels: tile 32.

    EXPECT_TRUE(admits(spec, fmaKernel("FLOAT", "CHANNELS_FIRST", 32)));
    EXPECT_FALSE(admits(spec, fmaKernel("HALF", "CHANNELS_FIRST", 32)));
    EXPECT_FALSE(admits(spec, fmaKernel("FLOAT", "CHANNELS_LAST", 32)));
    EXPECT_FALSE(admits(spec, fmaKernel("FLOAT", "CHANNELS_FIRST", 64)));
    EXPECT_FALSE(admits(spec, fmaKernel("FLOAT", "CHANNELS_FIRST", 32, "direct_1d")));

    GraphSpec nhwcHalf;
    nhwcHalf.channelsLast = true;
    nhwcHalf.dtype = data_objects::DataType::HALF;
    EXPECT_TRUE(admits(nhwcHalf, fmaKernel("HALF", "CHANNELS_LAST", 32)));
    EXPECT_FALSE(admits(nhwcHalf, fmaKernel("HALF", "CHANNELS_FIRST", 32)));
}

class TestHipConvFwdRdnaTile : public ::testing::TestWithParam<std::pair<int64_t, int64_t>>
{
};

// The narrowest of 32, 64 and 128 that holds one group's output channels, and exactly one.
TEST_P(TestHipConvFwdRdnaTile, PicksTheNarrowestTileHoldingAGroup)
{
    const auto [outChannelsPerGroup, expectedTile] = GetParam();
    GraphSpec spec;
    spec.c = 16;
    spec.groups = 2;
    spec.k = outChannelsPerGroup * 2;
    for(const int64_t tile : {32, 64, 128})
    {
        EXPECT_EQ(admits(spec, fmaKernel("FLOAT", "CHANNELS_FIRST", tile)), tile == expectedTile)
            << outChannelsPerGroup << " output channels per group, tile " << tile;
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         TestHipConvFwdRdnaTile,
                         ::testing::Values(std::make_pair(16, 32),
                                           std::make_pair(32, 32),
                                           std::make_pair(33, 64),
                                           std::make_pair(64, 64),
                                           std::make_pair(65, 128),
                                           std::make_pair(200, 128)));

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
    return SavedLaunchInputs{std::string(HIP_CONV_FWD_RDNA_DISPATCH_SYMBOL_V1),
                             hipConvFwdRdnaLaunchValues({21, 22, 23}, distinctProblem())};
}

TEST(TestHipConvFwdRdnaLaunchValues, RoundTripThroughTheirValues)
{
    const auto read = readHipConvFwdRdnaLaunchInputs(savedValid());
    const auto expected = distinctProblem();

    EXPECT_EQ(read.binding.x, 21);
    EXPECT_EQ(read.binding.w, 22);
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

TEST(TestHipConvFwdRdnaLaunchValues, SavesTheV1ValueNames)
{
    const auto values = savedValid().values;
    for(const auto* name :
        {"x.uid",    "w.uid",      "y.uid",      "n",         "c",        "k",        "groups",
         "in_d",     "in_h",       "in_w",       "filter_d",  "filter_h", "filter_w", "out_d",
         "out_h",    "out_w",      "pad_d",      "pad_h",     "pad_w",    "stride_d", "stride_h",
         "stride_w", "dilation_d", "dilation_h", "dilation_w"})
    {
        EXPECT_EQ(values.count(name), 1U) << name;
    }
    EXPECT_EQ(values.size(), 25U);
}

TEST(TestHipConvFwdRdnaLaunchValues, RefusesMalformedValues)
{
    const auto refused = [](const SavedLaunchInputs& inputs, const char* what) {
        EXPECT_THROW(readHipConvFwdRdnaLaunchInputs(inputs),
                     hipdnn_plugin_sdk::HipdnnPluginException)
            << what;
    };

    auto otherContract = savedValid();
    otherContract.dispatchSymbol = "hipkernel.hip_conv_fwd_rdna.dispatch.v2";
    refused(otherContract, "another contract");

    auto missing = savedValid();
    missing.values.erase("stride_h");
    refused(missing, "a missing value");

    auto extra = savedValid();
    extra.values["tile_n"] = int64_t{32};
    refused(extra, "a value outside the contract");

    auto zeroExtent = savedValid();
    zeroExtent.values["in_w"] = int64_t{0};
    refused(zeroExtent, "a zero extent");

    auto negativePad = savedValid();
    negativePad.values["pad_h"] = int64_t{-1};
    refused(negativePad, "a negative padding");

    auto wrongType = savedValid();
    wrongType.values["k"] = std::string("40");
    refused(wrongType, "a string extent");

    auto indivisible = savedValid();
    indivisible.values["groups"] = int64_t{5};
    refused(indivisible, "groups that do not divide c and k");

    auto hugeGrid = savedValid();
    hugeGrid.values["n"] = int64_t{std::numeric_limits<int>::max()};
    refused(hugeGrid, "a grid beyond the launch limits");
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
