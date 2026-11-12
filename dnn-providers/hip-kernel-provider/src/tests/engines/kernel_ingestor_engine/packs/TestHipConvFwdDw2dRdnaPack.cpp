// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>

#include "engines/kernel_ingestor_engine/packs/HipConvFwdDw2dRdnaLaunchValues.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestHipConvFwdDw2dRdnaPack.cpp
 * @brief Host-only checks of `hipkernel:hip_conv_fwd_dw2d_rdna`: the graphs its matcher
 *        accepts and refuses, the variant each problem takes (specialized undilated or dilated,
 *        or generic), the one kernel its kernel matcher admits per graph, and its saved launch
 *        values. The device cases are IntegrationGpuHipConvFwdDw2dRdna.
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

constexpr PackSymbols HIP_CONV_FWD_DW2D_RDNA{"hipkernel:hip_conv_fwd_dw2d_rdna",
                                             "hipkernel.hip_conv_fwd_dw2d_rdna.graph_match",
                                             "",
                                             "hipkernel.hip_conv_fwd_dw2d_rdna.kernel_match",
                                             "hipkernel.hip_conv_fwd_dw2d_rdna.score",
                                             "hipkernel.hip_conv_fwd_dw2d_rdna.dispatch",
                                             "conv_fwd_dw2d.x.uid",
                                             "conv_fwd_dw2d.w.uid",
                                             "conv_fwd_dw2d.y.uid"};

constexpr int64_t X_UID = 1;
constexpr int64_t W_UID = 2;
constexpr int64_t Y_UID = 3;

/// One conv-forward graph. Spatial vectors have one entry per spatial dimension, outermost
/// first. The defaults are an NCHW fp32 3x3 depthwise conv with 32 channels and "same"
/// padding, which the specialized kernel serves.
struct GraphSpec
{
    data_objects::DataType dtype = data_objects::DataType::FLOAT;
    bool channelsLast = false;
    int64_t n = 2;
    int64_t c = 32;
    int64_t k = 32;
    int64_t groups = 32;
    std::vector<int64_t> in{12, 14};
    std::vector<int64_t> filter{3, 3};
    std::vector<int64_t> prePad{1, 1};
    std::vector<int64_t> postPad{1, 1};
    std::vector<int64_t> stride{1, 1};
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

    auto name = builder.CreateString("hip_conv_fwd_dw2d_rdna_test");
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
    return matchesGraph(HIP_CONV_FWD_DW2D_RDNA, fixture.context()).has_value();
}

/// The problem @p spec poses, as the engine reads it (dims only; dtype and layout set). The
/// spatial values are right-aligned, as convProblem keeps them.
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

KernelDefinition kernelOf(const std::string& variant,
                          const std::string& dtype,
                          const std::string& layout,
                          int64_t filterH,
                          int64_t filterW,
                          int64_t stride,
                          int64_t dilation)
{
    KernelDefinition kernel;
    kernel.metadata = {{"kernel_family", std::string("dw2d_direct")},
                       {"variant", variant},
                       {"dtype", dtype},
                       {"layout", layout},
                       {"filter_h", filterH},
                       {"filter_w", filterW},
                       {"stride", stride},
                       {"dilation", dilation}};
    return kernel;
}

KernelDefinition specializedKernel(
    const std::string& dtype, int64_t filterH, int64_t filterW, int64_t stride, int64_t dilation)
{
    return kernelOf("specialized", dtype, "CHANNELS_FIRST", filterH, filterW, stride, dilation);
}

KernelDefinition genericKernel(const std::string& dtype, const std::string& layout)
{
    return kernelOf("generic", dtype, layout, 0, 0, 0, 0);
}

bool admits(const GraphSpec& spec, const KernelDefinition& kernel)
{
    const GraphFixture fixture(buildGraph(spec));
    return matchesKernel(HIP_CONV_FWD_DW2D_RDNA, fixture.context(), kernel);
}

/// A 48x48 input with filter @p filter, stride @p stride and dilation @p dilation along both
/// axes, and "same" padding dilation x (filter - 1) / 2.
GraphSpec sameShape(std::vector<int64_t> filter, int64_t stride, int64_t dilation)
{
    GraphSpec spec;
    spec.in = {48, 48};
    spec.filter = filter;
    spec.stride = {stride, stride};
    spec.dilation = {dilation, dilation};
    spec.prePad = {dilation * (filter[0] - 1) / 2, dilation * (filter[1] - 1) / 2};
    spec.postPad = spec.prePad;
    return spec;
}

/// The depthwise conv1d of a Conformer block: length 100, filter 31, "same" padding.
GraphSpec conv1dK31()
{
    GraphSpec spec;
    spec.in = {100};
    spec.filter = {31};
    spec.prePad = {15};
    spec.postPad = {15};
    spec.stride = {1};
    spec.dilation = {1};
    return spec;
}

// ---------------------------------------------------------------------------
// Graph matcher
// ---------------------------------------------------------------------------

TEST(TestHipConvFwdDw2dRdnaGraphMatcher, AcceptsDepthwise2dAnd1d)
{
    GraphSpec nhwc;
    nhwc.channelsLast = true;
    GraphSpec multiplier;
    multiplier.k = 64;
    GraphSpec half;
    half.dtype = data_objects::DataType::HALF;
    GraphSpec bfloat;
    bfloat.dtype = data_objects::DataType::BFLOAT16;
    GraphSpec oddFilter;
    oddFilter.filter = {2, 4};
    oddFilter.prePad = {0, 2};

    EXPECT_TRUE(matches(GraphSpec{}));
    EXPECT_TRUE(matches(nhwc));
    EXPECT_TRUE(matches(multiplier)) << "K = 2 C";
    EXPECT_TRUE(matches(half));
    EXPECT_TRUE(matches(bfloat));
    EXPECT_TRUE(matches(oddFilter)) << "a filter served by the generic variant";
    EXPECT_TRUE(matches(conv1dK31())) << "1-D depthwise";
}

TEST(TestHipConvFwdDw2dRdnaGraphMatcher, RefusesWhatItCannotServe)
{
    GraphSpec dense;
    dense.groups = 1;
    EXPECT_FALSE(matches(dense)) << "non-depthwise";

    GraphSpec grouped;
    grouped.groups = 8;
    EXPECT_FALSE(matches(grouped)) << "4 channels per group";

    GraphSpec threeD;
    threeD.in = {4, 12, 14};
    threeD.filter = {3, 3, 3};
    threeD.prePad = {1, 1, 1};
    threeD.postPad = {1, 1, 1};
    threeD.stride = {1, 1, 1};
    threeD.dilation = {1, 1, 1};
    EXPECT_FALSE(matches(threeD)) << "3-D depthwise";

    GraphSpec singleChannel;
    singleChannel.c = 1;
    singleChannel.k = 1;
    singleChannel.groups = 1;
    EXPECT_FALSE(matches(singleChannel)) << "one channel is not a grouped convolution";
}

TEST(TestHipConvFwdDw2dRdnaBinding, BindsTheOperandsAndTheFlops)
{
    const GraphSpec spec;
    const GraphFixture fixture(buildGraph(spec));
    const auto bound = matchesGraph(HIP_CONV_FWD_DW2D_RDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_DW2D_RDNA.inputAToken), X_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_DW2D_RDNA.inputBToken), W_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_FWD_DW2D_RDNA.outputToken), Y_UID);
    // 2 x N x K x (H x W out) x (C / groups) x taps; padding 1 keeps the output size.
    EXPECT_EQ(tryGetBoundInt(*bound, "conv_fwd_dw2d.flops"), int64_t{2} * 2 * 32 * 12 * 14 * 1 * 9);
}

// ---------------------------------------------------------------------------
// Variant selection
// ---------------------------------------------------------------------------

TEST(TestHipConvFwdDw2dRdnaGeometry, PicksTheUndilatedSpecialization)
{
    const std::vector<std::pair<std::vector<int64_t>, int64_t>> shapes{
        {{3, 3}, 1}, {{3, 3}, 2}, {{5, 5}, 1}, {{5, 5}, 2}, {{7, 7}, 1}, {{7, 7}, 2}, {{1, 31}, 1}};
    for(const auto& [filter, stride] : shapes)
    {
        const auto geometry = hipConvFwdDw2dRdnaGeometry(problemOf(sameShape(filter, stride, 1)));
        EXPECT_FALSE(geometry.generic) << filter[0] << "x" << filter[1] << " stride " << stride;
        EXPECT_FALSE(geometry.laneOnDilationGrid);
        EXPECT_EQ(geometry.outPerLane, 2);
    }

    // Only the pre-padding is compiled in; a post-padding of 0 just shortens the output.
    auto postPad0 = sameShape({3, 3}, 2, 1);
    postPad0.postPad = {0, 0};
    EXPECT_FALSE(hipConvFwdDw2dRdnaGeometry(problemOf(postPad0)).generic) << "post-padding 0";

    // A 1-D conv is the 1xK case.
    EXPECT_FALSE(hipConvFwdDw2dRdnaGeometry(problemOf(conv1dK31())).generic) << "1-D, K = 31";
}

TEST(TestHipConvFwdDw2dRdnaGeometry, PicksTheDilatedSpecialization)
{
    const std::vector<std::pair<std::vector<int64_t>, int64_t>> shapes{
        {{3, 3}, 2}, {{3, 3}, 4}, {{3, 3}, 6}, {{3, 3}, 12}, {{3, 3}, 18}, {{5, 5}, 2}};
    for(const auto& [filter, dilation] : shapes)
    {
        auto spec = sameShape(filter, 1, dilation);
        auto geometry = hipConvFwdDw2dRdnaGeometry(problemOf(spec));
        EXPECT_FALSE(geometry.generic) << filter[0] << "x" << filter[1] << " dilation " << dilation;
        EXPECT_TRUE(geometry.laneOnDilationGrid);
        EXPECT_EQ(geometry.outPerLane, 2) << "fp32";

        spec.dtype = data_objects::DataType::HALF;
        EXPECT_EQ(hipConvFwdDw2dRdnaGeometry(problemOf(spec)).outPerLane, 4) << "fp16";
        spec.dtype = data_objects::DataType::BFLOAT16;
        EXPECT_EQ(hipConvFwdDw2dRdnaGeometry(problemOf(spec)).outPerLane, 4) << "bf16";
    }
}

TEST(TestHipConvFwdDw2dRdnaGeometry, FallsBackToTheGenericVariant)
{
    GraphSpec nhwc;
    nhwc.channelsLast = true;
    EXPECT_TRUE(hipConvFwdDw2dRdnaGeometry(problemOf(nhwc)).generic) << "NHWC";

    GraphSpec multiplier;
    multiplier.k = 64;
    EXPECT_TRUE(hipConvFwdDw2dRdnaGeometry(problemOf(multiplier)).generic) << "K = 2 C";

    GraphSpec pad0;
    pad0.prePad = {0, 0};
    EXPECT_TRUE(hipConvFwdDw2dRdnaGeometry(problemOf(pad0)).generic) << "pre-padding 0";

    GraphSpec padH1W0;
    padH1W0.prePad = {1, 0};
    EXPECT_TRUE(hipConvFwdDw2dRdnaGeometry(problemOf(padH1W0)).generic) << "pre-padding 1 / 0";

    GraphSpec strideH1W2;
    strideH1W2.stride = {1, 2};
    EXPECT_TRUE(hipConvFwdDw2dRdnaGeometry(problemOf(strideH1W2)).generic) << "stride 1 / 2";

    EXPECT_TRUE(hipConvFwdDw2dRdnaGeometry(problemOf(sameShape({9, 9}, 1, 1))).generic) << "9x9";
    EXPECT_TRUE(hipConvFwdDw2dRdnaGeometry(problemOf(sameShape({3, 3}, 3, 1))).generic)
        << "stride 3";
    EXPECT_TRUE(hipConvFwdDw2dRdnaGeometry(problemOf(sameShape({3, 3}, 1, 3))).generic)
        << "dilation 3";
    EXPECT_TRUE(hipConvFwdDw2dRdnaGeometry(problemOf(sameShape({3, 3}, 2, 2))).generic)
        << "dilation 2 at stride 2";
    EXPECT_TRUE(hipConvFwdDw2dRdnaGeometry(problemOf(sameShape({7, 7}, 1, 2))).generic)
        << "7x7 at dilation 2";
}

// ---------------------------------------------------------------------------
// Kernel matcher
// ---------------------------------------------------------------------------

TEST(TestHipConvFwdDw2dRdnaKernelMatcher, AdmitsOnlyTheVariantOfTheProblem)
{
    const GraphSpec spec; // NCHW fp32 3x3 stride 1 padding 1: the undilated specialization.
    EXPECT_TRUE(admits(spec, specializedKernel("FLOAT", 3, 3, 1, 1)));
    EXPECT_FALSE(admits(spec, specializedKernel("FLOAT", 3, 3, 2, 1))) << "another stride";
    EXPECT_FALSE(admits(spec, specializedKernel("FLOAT", 5, 5, 1, 1))) << "another filter";
    EXPECT_FALSE(admits(spec, specializedKernel("FLOAT", 3, 3, 1, 2))) << "another dilation";
    EXPECT_FALSE(admits(spec, specializedKernel("HALF", 3, 3, 1, 1))) << "another dtype";
    EXPECT_FALSE(admits(spec, genericKernel("FLOAT", "CHANNELS_FIRST"))) << "the generic variant";

    const auto dilated = sameShape({3, 3}, 1, 6);
    EXPECT_TRUE(admits(dilated, specializedKernel("FLOAT", 3, 3, 1, 6)));
    EXPECT_FALSE(admits(dilated, specializedKernel("FLOAT", 3, 3, 1, 1))) << "undilated";

    EXPECT_TRUE(admits(conv1dK31(), specializedKernel("FLOAT", 1, 31, 1, 1)));

    GraphSpec nhwc;
    nhwc.channelsLast = true;
    EXPECT_TRUE(admits(nhwc, genericKernel("FLOAT", "CHANNELS_LAST")));
    EXPECT_FALSE(admits(nhwc, genericKernel("FLOAT", "CHANNELS_FIRST")));
    EXPECT_FALSE(admits(nhwc, specializedKernel("FLOAT", 3, 3, 1, 1))) << "a specialization";
}

// ---------------------------------------------------------------------------
// Saved launch values
// ---------------------------------------------------------------------------

/// A problem whose every field differs, so a value read into the wrong field shows.
ConvProblem distinctProblem()
{
    ConvProblem problem;
    problem.spatialRank = 2;
    problem.n = 3;
    problem.c = 12;
    problem.k = 24;
    problem.groups = 12;
    problem.in = {1, 17, 19};
    problem.filter = {1, 5, 7};
    problem.out = {1, 9, 10};
    problem.pad = {0, 2, 3};
    problem.stride = {1, 2, 4};
    problem.dilation = {1, 3, 5};
    return problem;
}

SavedLaunchInputs savedValid(HipConvFwdDw2dRdnaGeometry geometry = {false, true, 4})
{
    return SavedLaunchInputs{
        std::string(HIP_CONV_FWD_DW2D_RDNA_DISPATCH_SYMBOL_V1),
        hipConvFwdDw2dRdnaLaunchValues({21, 22, 23}, distinctProblem(), geometry)};
}

TEST(TestHipConvFwdDw2dRdnaLaunchValues, RoundTripThroughTheirValues)
{
    const auto read = readHipConvFwdDw2dRdnaLaunchInputs(savedValid());
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
    EXPECT_TRUE(read.geometry.laneOnDilationGrid);
    EXPECT_EQ(read.geometry.outPerLane, 4);

    const auto undilated = readHipConvFwdDw2dRdnaLaunchInputs(savedValid({false, false, 2}));
    EXPECT_FALSE(undilated.geometry.laneOnDilationGrid);
    EXPECT_EQ(undilated.geometry.outPerLane, 2);

    const auto generic = readHipConvFwdDw2dRdnaLaunchInputs(savedValid({true, false, 4}));
    EXPECT_TRUE(generic.geometry.generic);
}

TEST(TestHipConvFwdDw2dRdnaLaunchValues, SavesTheV1ValueNames)
{
    const auto values = savedValid().values;
    for(const auto* name :
        {"x.uid",       "w.uid",      "y.uid",      "n",        "c",
         "k",           "in_h",       "in_w",       "filter_h", "filter_w",
         "out_h",       "out_w",      "pad_h",      "pad_w",    "stride_h",
         "stride_w",    "dilation_h", "dilation_w", "generic",  "lane_on_dilation_grid",
         "out_per_lane"})
    {
        EXPECT_EQ(values.count(name), 1U) << name;
    }
    EXPECT_EQ(values.size(), 21U);
}

TEST(TestHipConvFwdDw2dRdnaLaunchValues, RefusesMalformedValues)
{
    const auto refused = [](const SavedLaunchInputs& inputs, const char* what) {
        EXPECT_THROW(readHipConvFwdDw2dRdnaLaunchInputs(inputs),
                     hipdnn_plugin_sdk::HipdnnPluginException)
            << what;
    };

    auto otherContract = savedValid();
    otherContract.dispatchSymbol = "hipkernel.hip_conv_fwd_dw2d_rdna.dispatch.v2";
    refused(otherContract, "another contract");

    auto missing = savedValid();
    missing.values.erase("out_per_lane");
    refused(missing, "a missing value");

    auto extra = savedValid();
    extra.values["in_d"] = int64_t{1};
    refused(extra, "a value outside the contract");

    auto notMultiple = savedValid();
    notMultiple.values["k"] = int64_t{20};
    refused(notMultiple, "K not a multiple of C");

    refused(savedValid({false, false, 4}), "four outputs per lane off the dilation grid");
    refused(savedValid({false, true, 3}), "three outputs per lane");
    refused(savedValid({true, true, 4}), "a generic variant on the dilation grid");
    refused(savedValid({true, false, 2}), "a generic variant with two columns");

    // The specialized grid is capped; the generic one is not.
    auto hugeGrid = savedValid({true, false, 4});
    hugeGrid.values["n"] = int64_t{std::numeric_limits<int>::max()};
    refused(hugeGrid, "a grid beyond the launch limits");
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
