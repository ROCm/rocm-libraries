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

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>

#include "engines/kernel_ingestor_engine/packs/HipConvDgradDwRdnaLaunchValues.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestHipConvDgradDwRdnaPack.cpp
 * @brief Host-only checks of `hipkernel:hip_conv_dgrad_dw_rdna`: the graphs its matcher
 *        accepts and refuses, the variant each problem takes (specialized 1-D or generic), the
 *        one kernel its kernel matcher admits per graph, and its saved launch values. The
 *        device cases are IntegrationGpuHipConvDgradDwRdna.
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
constexpr PackSymbols HIP_CONV_DGRAD_DW_RDNA{"hipkernel:hip_conv_dgrad_dw_rdna",
                                             "hipkernel.hip_conv_dgrad_dw_rdna.graph_match",
                                             "",
                                             "hipkernel.hip_conv_dgrad_dw_rdna.kernel_match",
                                             "hipkernel.hip_conv_dgrad_dw_rdna.score",
                                             "hipkernel.hip_conv_dgrad_dw_rdna.dispatch",
                                             "conv_dgrad_dw.y.uid",
                                             "conv_dgrad_dw.w.uid",
                                             "conv_dgrad_dw.x.uid"};

constexpr int64_t DY_UID = 1;
constexpr int64_t W_UID = 2;
constexpr int64_t DX_UID = 3;

/// One conv_dgrad graph in the forward view: dx is (n, c, in), w is (k, c / groups, filter),
/// dy is (n, k, out). Spatial vectors have one entry per spatial dimension, outermost first.
/// The defaults are an NCL fp32 depthwise k = 12 stride 2 problem with 32 channels, which the
/// specialized kernel serves.
struct GraphSpec
{
    data_objects::DataType dtype = data_objects::DataType::FLOAT;
    bool channelsLast = false;
    bool forward = false;
    int64_t n = 2;
    int64_t c = 32;
    int64_t k = 32;
    int64_t groups = 32;
    std::vector<int64_t> in{210};
    std::vector<int64_t> filter{12};
    std::vector<int64_t> prePad{0};
    std::vector<int64_t> postPad{0};
    std::vector<int64_t> stride{2};
    std::vector<int64_t> dilation{1};
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

    auto name = builder.CreateString("hip_conv_dgrad_dw_rdna_test");
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
    return matchesGraph(HIP_CONV_DGRAD_DW_RDNA, fixture.context()).has_value();
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

KernelDefinition kernelOf(const std::string& variant,
                          const std::string& dtype,
                          const std::string& layout,
                          int64_t filterW,
                          int64_t stride)
{
    KernelDefinition kernel;
    kernel.metadata = {{"kernel_family", std::string("dgrad_dw_direct")},
                       {"variant", variant},
                       {"dtype", dtype},
                       {"layout", layout},
                       {"filter_w", filterW},
                       {"stride", stride}};
    return kernel;
}

KernelDefinition specializedKernel(const std::string& dtype, int64_t filterW, int64_t stride)
{
    return kernelOf("specialized", dtype, "CHANNELS_FIRST", filterW, stride);
}

KernelDefinition genericKernel(const std::string& dtype, const std::string& layout)
{
    return kernelOf("generic", dtype, layout, 0, 0);
}

bool admits(const GraphSpec& spec, const KernelDefinition& kernel)
{
    const GraphFixture fixture(buildGraph(spec));
    return matchesKernel(HIP_CONV_DGRAD_DW_RDNA, fixture.context(), kernel);
}

/// @p spec as rank 4 with a unit height axis.
GraphSpec withUnitHeight(GraphSpec spec)
{
    spec.in.insert(spec.in.begin(), 1);
    spec.filter.insert(spec.filter.begin(), 1);
    spec.prePad.insert(spec.prePad.begin(), 0);
    spec.postPad.insert(spec.postPad.begin(), 0);
    spec.stride.insert(spec.stride.begin(), 1);
    spec.dilation.insert(spec.dilation.begin(), 1);
    return spec;
}

/// A 2-D depthwise problem with a 3x3 filter at stride 2.
GraphSpec depthwise2d()
{
    GraphSpec spec;
    spec.in = {14, 16};
    spec.filter = {3, 3};
    spec.prePad = {1, 1};
    spec.postPad = {1, 1};
    spec.stride = {2, 2};
    spec.dilation = {1, 1};
    return spec;
}

/// A 3-D depthwise problem with a 3x3x3 filter.
GraphSpec depthwise3d()
{
    GraphSpec spec;
    spec.in = {4, 8, 10};
    spec.filter = {3, 3, 3};
    spec.prePad = {1, 1, 1};
    spec.postPad = {1, 1, 1};
    spec.stride = {1, 1, 1};
    spec.dilation = {1, 1, 1};
    return spec;
}

// ---------------------------------------------------------------------------
// Graph matcher
// ---------------------------------------------------------------------------

TEST(TestHipConvDgradDwRdnaGraphMatcher, AcceptsDepthwiseDgrad)
{
    GraphSpec nlc;
    nlc.channelsLast = true;
    GraphSpec multiplier;
    multiplier.k = 64;
    GraphSpec half;
    half.dtype = data_objects::DataType::HALF;
    GraphSpec bfloat;
    bfloat.dtype = data_objects::DataType::BFLOAT16;

    EXPECT_TRUE(matches(GraphSpec{}));
    EXPECT_TRUE(matches(nlc));
    EXPECT_TRUE(matches(multiplier)) << "K = 2 C";
    EXPECT_TRUE(matches(half));
    EXPECT_TRUE(matches(bfloat));
    EXPECT_TRUE(matches(withUnitHeight(GraphSpec{}))) << "rank 4, H == 1";
    EXPECT_TRUE(matches(depthwise2d())) << "2-D";
    EXPECT_TRUE(matches(depthwise3d())) << "3-D";
}

TEST(TestHipConvDgradDwRdnaGraphMatcher, RefusesWhatItCannotServe)
{
    GraphSpec dense;
    dense.groups = 1;
    EXPECT_FALSE(matches(dense)) << "non-depthwise";

    GraphSpec grouped;
    grouped.groups = 8;
    EXPECT_FALSE(matches(grouped)) << "4 channels per group";

    GraphSpec singleChannel;
    singleChannel.c = 1;
    singleChannel.k = 1;
    singleChannel.groups = 1;
    EXPECT_FALSE(matches(singleChannel)) << "one channel is not a grouped convolution";

    GraphSpec forward;
    forward.forward = true;
    EXPECT_FALSE(matches(forward)) << "the same problem as conv_fprop";
}

TEST(TestHipConvDgradDwRdnaBinding, BindsTheOperandsAndTheFlops)
{
    const GraphSpec spec;
    const GraphFixture fixture(buildGraph(spec));
    const auto bound = matchesGraph(HIP_CONV_DGRAD_DW_RDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_DGRAD_DW_RDNA.inputAToken), DY_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_DGRAD_DW_RDNA.inputBToken), W_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_DGRAD_DW_RDNA.outputToken), DX_UID);
    // 2 x N x K x (dy length) x (C / groups) x taps; dy is (210 - 12) / 2 + 1 = 100 long.
    EXPECT_EQ(tryGetBoundInt(*bound, "conv_dgrad_dw.flops"), int64_t{2} * 2 * 32 * 100 * 1 * 12);
}

// ---------------------------------------------------------------------------
// Variant selection
// ---------------------------------------------------------------------------

TEST(TestHipConvDgradDwRdnaGeometry, PicksTheSpecializationOfEveryBuiltShape)
{
    for(const auto& [filter, stride] :
        std::vector<std::pair<int64_t, int64_t>>{{12, 2}, {29, 2}, {43, 3}, {7, 1}})
    {
        GraphSpec spec;
        spec.in = {400};
        spec.filter = {filter};
        spec.stride = {stride};
        spec.prePad = {filter / 2};
        spec.postPad = {1};
        auto geometry = hipConvDgradDwRdnaGeometry(problemOf(spec));
        EXPECT_FALSE(geometry.generic) << "k " << filter << " stride " << stride;
        EXPECT_EQ(geometry.tileQ, 1) << "fp32";

        spec.dtype = data_objects::DataType::HALF;
        EXPECT_EQ(hipConvDgradDwRdnaGeometry(problemOf(spec)).tileQ, 2) << "fp16";
        spec.dtype = data_objects::DataType::BFLOAT16;
        EXPECT_EQ(hipConvDgradDwRdnaGeometry(problemOf(spec)).tileQ, 2) << "bf16";

        EXPECT_FALSE(hipConvDgradDwRdnaGeometry(problemOf(withUnitHeight(spec))).generic)
            << "rank 4, H == 1";
    }
}

TEST(TestHipConvDgradDwRdnaGeometry, FallsBackToTheGenericVariant)
{
    const auto generic = [](const GraphSpec& spec) {
        const auto geometry = hipConvDgradDwRdnaGeometry(problemOf(spec));
        return geometry.generic && geometry.tileQ == 1;
    };

    GraphSpec nlc;
    nlc.channelsLast = true;
    EXPECT_TRUE(generic(nlc)) << "NLC";

    GraphSpec multiplier;
    multiplier.k = 64;
    EXPECT_TRUE(generic(multiplier)) << "K = 2 C";

    GraphSpec dilated;
    dilated.dilation = {2};
    EXPECT_TRUE(generic(dilated)) << "dilation 2";

    GraphSpec otherStride;
    otherStride.stride = {3};
    EXPECT_TRUE(generic(otherStride)) << "k 12 at stride 3";

    GraphSpec otherFilter;
    otherFilter.filter = {9};
    EXPECT_TRUE(generic(otherFilter)) << "k 9";

    EXPECT_TRUE(generic(depthwise2d())) << "2-D";
    EXPECT_TRUE(generic(depthwise3d())) << "3-D";

    auto tallFilter = withUnitHeight(GraphSpec{});
    tallFilter.in[0] = 3;
    tallFilter.filter[0] = 3;
    EXPECT_TRUE(generic(tallFilter)) << "H filter 3";

    // A unit H filter at H stride 3 keeps dy one row high under H padding 1, but no tap
    // reaches dx's row: the specialized kernel would not see that.
    auto padH = withUnitHeight(GraphSpec{});
    padH.prePad[0] = 1;
    padH.postPad[0] = 1;
    padH.stride[0] = 3;
    EXPECT_TRUE(generic(padH)) << "H padding";

    GraphSpec wideBatch;
    wideBatch.n = 70000;
    EXPECT_TRUE(generic(wideBatch)) << "batch beyond the grid's z";

    GraphSpec wideChannels;
    wideChannels.c = wideChannels.k = wideChannels.groups = 70000;
    EXPECT_TRUE(generic(wideChannels)) << "channels beyond the grid's y";

    GraphSpec longRow;
    longRow.in = {std::numeric_limits<int>::max() - 100};
    EXPECT_TRUE(generic(longRow)) << "dx positions beyond int";
}

// ---------------------------------------------------------------------------
// Kernel matcher
// ---------------------------------------------------------------------------

TEST(TestHipConvDgradDwRdnaKernelMatcher, AdmitsOnlyTheVariantOfTheProblem)
{
    const GraphSpec spec; // NCL fp32 k 12 stride 2: the specialization.
    EXPECT_TRUE(admits(spec, specializedKernel("FLOAT", 12, 2)));
    EXPECT_FALSE(admits(spec, specializedKernel("FLOAT", 12, 3))) << "another stride";
    EXPECT_FALSE(admits(spec, specializedKernel("FLOAT", 29, 2))) << "another filter";
    EXPECT_FALSE(admits(spec, specializedKernel("HALF", 12, 2))) << "another dtype";
    EXPECT_FALSE(admits(spec, genericKernel("FLOAT", "CHANNELS_FIRST"))) << "the generic variant";

    GraphSpec nlc;
    nlc.channelsLast = true;
    EXPECT_TRUE(admits(nlc, genericKernel("FLOAT", "CHANNELS_LAST")));
    EXPECT_FALSE(admits(nlc, genericKernel("FLOAT", "CHANNELS_FIRST")));
    EXPECT_FALSE(admits(nlc, specializedKernel("FLOAT", 12, 2))) << "a specialization";

    EXPECT_TRUE(admits(depthwise3d(), genericKernel("FLOAT", "CHANNELS_FIRST")));
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
    problem.filter = {2, 5, 7};
    problem.out = {4, 9, 10};
    problem.pad = {1, 2, 3};
    problem.stride = {1, 2, 4};
    problem.dilation = {2, 3, 5};
    return problem;
}

SavedLaunchInputs savedValid(HipConvDgradDwRdnaGeometry geometry = {false, 2})
{
    return SavedLaunchInputs{
        std::string(HIP_CONV_DGRAD_DW_RDNA_DISPATCH_SYMBOL_V1),
        hipConvDgradDwRdnaLaunchValues({21, 22, 23}, distinctProblem(), geometry)};
}

TEST(TestHipConvDgradDwRdnaLaunchValues, RoundTripThroughTheirValues)
{
    const auto read = readHipConvDgradDwRdnaLaunchInputs(savedValid());
    const auto expected = distinctProblem();

    EXPECT_EQ(read.binding.dy, 21);
    EXPECT_EQ(read.binding.w, 22);
    EXPECT_EQ(read.binding.dx, 23);
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
    EXPECT_EQ(read.geometry.tileQ, 2);

    const auto generic = readHipConvDgradDwRdnaLaunchInputs(savedValid({true, 1}));
    EXPECT_TRUE(generic.geometry.generic);
    EXPECT_EQ(generic.geometry.tileQ, 1);
}

TEST(TestHipConvDgradDwRdnaLaunchValues, SavesTheV1ValueNames)
{
    const auto values = savedValid().values;
    for(const auto* name :
        {"dy.uid",     "w.uid",      "dx.uid",     "n",        "c",        "k",        "in_d",
         "in_h",       "in_w",       "filter_d",   "filter_h", "filter_w", "out_d",    "out_h",
         "out_w",      "pad_d",      "pad_h",      "pad_w",    "stride_d", "stride_h", "stride_w",
         "dilation_d", "dilation_h", "dilation_w", "generic",  "tile_q"})
    {
        EXPECT_EQ(values.count(name), 1U) << name;
    }
    EXPECT_EQ(values.size(), 26U);
}

TEST(TestHipConvDgradDwRdnaLaunchValues, RefusesMalformedValues)
{
    const auto refused = [](const SavedLaunchInputs& inputs, const char* what) {
        EXPECT_THROW(readHipConvDgradDwRdnaLaunchInputs(inputs),
                     hipdnn_plugin_sdk::HipdnnPluginException)
            << what;
    };

    auto otherContract = savedValid();
    otherContract.dispatchSymbol = "hipkernel.hip_conv_dgrad_dw_rdna.dispatch.v2";
    refused(otherContract, "another contract");

    auto missing = savedValid();
    missing.values.erase("tile_q");
    refused(missing, "a missing value");

    auto extra = savedValid();
    extra.values["groups"] = int64_t{12};
    refused(extra, "a value outside the contract");

    auto notMultiple = savedValid();
    notMultiple.values["k"] = int64_t{20};
    refused(notMultiple, "K not a multiple of C");

    refused(savedValid({false, 3}), "three phases per lane");
    refused(savedValid({true, 2}), "a generic variant with two phases per lane");

    auto wideChannels = savedValid();
    wideChannels.values["c"] = int64_t{70000};
    wideChannels.values["k"] = int64_t{70000};
    refused(wideChannels, "a specialized grid beyond the launch limits");

    auto longRow = savedValid();
    longRow.values["in_w"] = int64_t{std::numeric_limits<int>::max() - 100};
    refused(longRow, "specialized dx positions beyond int");

    auto wideGeneric = savedValid({true, 1});
    wideGeneric.values["c"] = int64_t{70000};
    wideGeneric.values["k"] = int64_t{70000};
    EXPECT_NO_THROW(readHipConvDgradDwRdnaLaunchInputs(wideGeneric))
        << "the generic grid walks dx in a grid-stride loop";
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
