// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>

#include "engines/kernel_ingestor_engine/packs/HipConvDgradFma1dRdnaLaunchValues.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestHipConvDgradFma1dRdnaPack.cpp
 * @brief Host-only checks of `hipkernel:hip_conv_dgrad_fma_1d_rdna`: the graphs its matcher
 *        accepts and refuses, the variant and channel tile each problem takes, the one kernel
 *        its kernel matcher admits per graph, and its saved launch values. The device cases are
 *        IntegrationGpuHipConvDgradFma1dRdna.
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
constexpr PackSymbols HIP_CONV_DGRAD_FMA_1D_RDNA{
    "hipkernel:hip_conv_dgrad_fma_1d_rdna",
    "hipkernel.hip_conv_dgrad_fma_1d_rdna.graph_match",
    "",
    "hipkernel.hip_conv_dgrad_fma_1d_rdna.kernel_match",
    "hipkernel.hip_conv_dgrad_fma_1d_rdna.score",
    "hipkernel.hip_conv_dgrad_fma_1d_rdna.dispatch",
    "conv_dgrad_fma_1d.y.uid",
    "conv_dgrad_fma_1d.w.uid",
    "conv_dgrad_fma_1d.x.uid"};

constexpr int64_t DY_UID = 1;
constexpr int64_t W_UID = 2;
constexpr int64_t DX_UID = 3;

/// One conv_dgrad graph in the forward view: dx is (n, c, in), w is (k, c / groups, filter),
/// dy is (n, k, out). Spatial vectors have one entry per spatial dimension, outermost first.
/// The defaults are an NCL fp32 k = 8 stride 4 problem with 64 dx and 32 dy channels, which
/// the specialized kernel serves.
struct GraphSpec
{
    data_objects::DataType dtype = data_objects::DataType::FLOAT;
    bool channelsLast = false;
    bool forward = false;
    int64_t n = 2;
    int64_t c = 64;
    int64_t k = 32;
    int64_t groups = 1;
    std::vector<int64_t> in{406};
    std::vector<int64_t> filter{8};
    std::vector<int64_t> prePad{0};
    std::vector<int64_t> postPad{0};
    std::vector<int64_t> stride{4};
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

    auto name = builder.CreateString("hip_conv_dgrad_fma_1d_rdna_test");
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
    return matchesGraph(HIP_CONV_DGRAD_FMA_1D_RDNA, fixture.context()).has_value();
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
                          int64_t stride,
                          int64_t channelsPerWave)
{
    KernelDefinition kernel;
    kernel.metadata = {{"kernel_family", std::string("dgrad_fma_1d")},
                       {"variant", variant},
                       {"dtype", dtype},
                       {"layout", layout},
                       {"filter_w", filterW},
                       {"stride", stride},
                       {"channels_per_wave", channelsPerWave}};
    return kernel;
}

KernelDefinition specializedKernel(const std::string& dtype,
                                   int64_t filterW,
                                   int64_t stride,
                                   int64_t channelsPerWave)
{
    return kernelOf("specialized", dtype, "CHANNELS_FIRST", filterW, stride, channelsPerWave);
}

KernelDefinition genericKernel(const std::string& dtype, const std::string& layout)
{
    return kernelOf("generic", dtype, layout, 0, 0, 0);
}

bool admits(const GraphSpec& spec, const KernelDefinition& kernel)
{
    const GraphFixture fixture(buildGraph(spec));
    return matchesKernel(HIP_CONV_DGRAD_FMA_1D_RDNA, fixture.context(), kernel);
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

/// @p spec with filter width @p filter, stride @p stride, @p channels dx channels and dtype
/// @p dtype.
GraphSpec shape(int64_t filter,
                int64_t stride,
                int64_t channels,
                data_objects::DataType dtype = data_objects::DataType::FLOAT)
{
    GraphSpec spec;
    spec.dtype = dtype;
    spec.filter = {filter};
    spec.stride = {stride};
    spec.c = channels;
    return spec;
}

// ---------------------------------------------------------------------------
// Graph matcher
// ---------------------------------------------------------------------------

TEST(TestHipConvDgradFma1dRdnaGraphMatcher, AcceptsOneDimensionalDgrad)
{
    GraphSpec nlc;
    nlc.channelsLast = true;
    GraphSpec grouped;
    grouped.groups = 4;
    GraphSpec half;
    half.dtype = data_objects::DataType::HALF;
    GraphSpec bfloat;
    bfloat.dtype = data_objects::DataType::BFLOAT16;

    EXPECT_TRUE(matches(GraphSpec{}));
    EXPECT_TRUE(matches(nlc));
    EXPECT_TRUE(matches(grouped)) << "16 dx channels per group";
    EXPECT_TRUE(matches(half));
    EXPECT_TRUE(matches(bfloat));
    EXPECT_TRUE(matches(withUnitHeight(GraphSpec{}))) << "rank 4, H == 1";
}

TEST(TestHipConvDgradFma1dRdnaGraphMatcher, RefusesWhatItCannotServe)
{
    GraphSpec depthwise;
    depthwise.groups = 64;
    depthwise.k = 64;
    EXPECT_FALSE(matches(depthwise)) << "depthwise";

    GraphSpec twoD;
    twoD.in = {14, 16};
    twoD.filter = {3, 3};
    twoD.prePad = {1, 1};
    twoD.postPad = {1, 1};
    twoD.stride = {2, 2};
    twoD.dilation = {1, 1};
    EXPECT_FALSE(matches(twoD)) << "2-D";

    auto tallFilter = withUnitHeight(GraphSpec{});
    tallFilter.in[0] = 3;
    tallFilter.filter[0] = 3;
    EXPECT_FALSE(matches(tallFilter)) << "H filter 3";

    // A unit H filter at H stride 3 keeps dy one row high under H padding 1, but no tap
    // reaches dx's row: not a 1-D problem.
    auto padH = withUnitHeight(GraphSpec{});
    padH.prePad[0] = 1;
    padH.postPad[0] = 1;
    padH.stride[0] = 3;
    EXPECT_FALSE(matches(padH)) << "H padding";

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
}

TEST(TestHipConvDgradFma1dRdnaBinding, BindsTheOperandsAndTheFlops)
{
    const GraphSpec spec;
    const GraphFixture fixture(buildGraph(spec));
    const auto bound = matchesGraph(HIP_CONV_DGRAD_FMA_1D_RDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_DGRAD_FMA_1D_RDNA.inputAToken), DY_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_DGRAD_FMA_1D_RDNA.inputBToken), W_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, HIP_CONV_DGRAD_FMA_1D_RDNA.outputToken), DX_UID);
    // 2 x N x K x (dy length) x (C / groups) x taps; dy is (406 - 8) / 4 + 1 = 100 long.
    EXPECT_EQ(tryGetBoundInt(*bound, "conv_dgrad_fma_1d.flops"),
              int64_t{2} * 2 * 32 * 100 * 64 * 8);
}

// ---------------------------------------------------------------------------
// Variant selection
// ---------------------------------------------------------------------------

TEST(TestHipConvDgradFma1dRdnaGeometry, PicksTheTileOfMIOpenForEveryShape)
{
    constexpr auto F32 = data_objects::DataType::FLOAT;
    constexpr auto F16 = data_objects::DataType::HALF;
    constexpr auto BF16 = data_objects::DataType::BFLOAT16;
    // filter, stride, dx channels, dtype -> q per lane, channels per wave
    const std::vector<std::tuple<int64_t, int64_t, int64_t, data_objects::DataType, int, int>>
        expected{{8, 4, 64, F32, 2, 8},
                 {8, 4, 33, F32, 2, 8},
                 {8, 4, 32, F32, 2, 4},
                 {8, 4, 16, F32, 2, 2},
                 {8, 4, 8, F32, 2, 1},
                 {4, 2, 512, F32, 2, 8},
                 {16, 8, 64, F32, 2, 4},
                 {16, 8, 16, F32, 2, 2},
                 {16, 8, 5, F32, 2, 1},
                 {9, 5, 1024, F32, 2, 4},
                 {16, 8, 64, F16, 2, 4},
                 {8, 4, 64, BF16, 2, 8},
                 {4, 2, 24, F16, 2, 4},
                 {9, 5, 512, F16, 4, 2},
                 {9, 5, 8, BF16, 4, 1},
                 {9, 5, 9, BF16, 4, 2}};
    for(const auto& [filter, stride, channels, dtype, q, cw] : expected)
    {
        const auto spec = shape(filter, stride, channels, dtype);
        const auto geometry = hipConvDgradFma1dRdnaGeometry(problemOf(spec));
        EXPECT_FALSE(geometry.generic) << filter << " " << stride << " " << channels;
        EXPECT_EQ(geometry.qPerLane, q) << filter << " " << stride << " " << channels;
        EXPECT_EQ(geometry.channelsPerWave, cw) << filter << " " << stride << " " << channels;
        EXPECT_FALSE(hipConvDgradFma1dRdnaGeometry(problemOf(withUnitHeight(spec))).generic)
            << "rank 4, H == 1";
    }
}

TEST(TestHipConvDgradFma1dRdnaGeometry, FallsBackToTheGenericVariant)
{
    const auto generic = [](const GraphSpec& spec) {
        const auto geometry = hipConvDgradFma1dRdnaGeometry(problemOf(spec));
        return geometry.generic && geometry.qPerLane == 1 && geometry.channelsPerWave == 1;
    };

    GraphSpec nlc;
    nlc.channelsLast = true;
    EXPECT_TRUE(generic(nlc)) << "NLC";

    GraphSpec grouped;
    grouped.groups = 2;
    EXPECT_TRUE(generic(grouped)) << "groups 2";

    GraphSpec dilated;
    dilated.dilation = {2};
    EXPECT_TRUE(generic(dilated)) << "dilation 2";

    EXPECT_TRUE(generic(shape(8, 2, 64))) << "k 8 at stride 2";
    EXPECT_TRUE(generic(shape(3, 1, 64))) << "k 3";

    GraphSpec wideBatch;
    wideBatch.n = 70000;
    EXPECT_TRUE(generic(wideBatch)) << "batch beyond the grid's z";

    GraphSpec longRow;
    longRow.in = {std::numeric_limits<int>::max() - 100};
    EXPECT_TRUE(generic(longRow)) << "dx positions beyond int";
}

// ---------------------------------------------------------------------------
// Kernel matcher
// ---------------------------------------------------------------------------

TEST(TestHipConvDgradFma1dRdnaKernelMatcher, AdmitsOnlyTheVariantOfTheProblem)
{
    const GraphSpec spec; // NCL fp32 k 8 stride 4, 64 dx channels: 8 channels per wave.
    EXPECT_TRUE(admits(spec, specializedKernel("FLOAT", 8, 4, 8)));
    EXPECT_FALSE(admits(spec, specializedKernel("FLOAT", 8, 4, 4))) << "another channel tile";
    EXPECT_FALSE(admits(spec, specializedKernel("FLOAT", 4, 2, 8))) << "another shape";
    EXPECT_FALSE(admits(spec, specializedKernel("HALF", 8, 4, 8))) << "another dtype";
    EXPECT_FALSE(admits(spec, genericKernel("FLOAT", "CHANNELS_FIRST"))) << "the generic variant";

    EXPECT_TRUE(admits(shape(8, 4, 16), specializedKernel("FLOAT", 8, 4, 2)));

    GraphSpec nlc;
    nlc.channelsLast = true;
    EXPECT_TRUE(admits(nlc, genericKernel("FLOAT", "CHANNELS_LAST")));
    EXPECT_FALSE(admits(nlc, genericKernel("FLOAT", "CHANNELS_FIRST")));
    EXPECT_FALSE(admits(nlc, specializedKernel("FLOAT", 8, 4, 8))) << "a specialization";
}

// ---------------------------------------------------------------------------
// Saved launch values
// ---------------------------------------------------------------------------

/// A problem whose every field differs, so a value read into the wrong field shows.
ConvProblem distinctProblem()
{
    ConvProblem problem;
    problem.spatialRank = 1;
    problem.n = 3;
    problem.c = 12;
    problem.k = 18;
    problem.groups = 6;
    problem.in = {1, 1, 97};
    problem.filter = {1, 1, 7};
    problem.out = {1, 1, 19};
    problem.pad = {0, 0, 2};
    problem.stride = {1, 1, 5};
    problem.dilation = {1, 1, 4};
    return problem;
}

SavedLaunchInputs savedValid(HipConvDgradFma1dRdnaGeometry geometry = {false, 4, 2})
{
    return SavedLaunchInputs{
        std::string(HIP_CONV_DGRAD_FMA_1D_RDNA_DISPATCH_SYMBOL_V1),
        hipConvDgradFma1dRdnaLaunchValues({21, 22, 23}, distinctProblem(), geometry)};
}

TEST(TestHipConvDgradFma1dRdnaLaunchValues, RoundTripThroughTheirValues)
{
    const auto read = readHipConvDgradFma1dRdnaLaunchInputs(savedValid());
    const auto expected = distinctProblem();

    EXPECT_EQ(read.binding.dy, 21);
    EXPECT_EQ(read.binding.w, 22);
    EXPECT_EQ(read.binding.dx, 23);
    EXPECT_EQ(read.problem.n, expected.n);
    EXPECT_EQ(read.problem.c, expected.c);
    EXPECT_EQ(read.problem.k, expected.k);
    EXPECT_EQ(read.problem.groups, expected.groups);
    EXPECT_EQ(read.problem.in[2], expected.in[2]);
    EXPECT_EQ(read.problem.filter[2], expected.filter[2]);
    EXPECT_EQ(read.problem.out[2], expected.out[2]);
    EXPECT_EQ(read.problem.pad[2], expected.pad[2]);
    EXPECT_EQ(read.problem.stride[2], expected.stride[2]);
    EXPECT_EQ(read.problem.dilation[2], expected.dilation[2]);
    EXPECT_FALSE(read.geometry.generic);
    EXPECT_EQ(read.geometry.qPerLane, 4);
    EXPECT_EQ(read.geometry.channelsPerWave, 2);

    const auto generic = readHipConvDgradFma1dRdnaLaunchInputs(savedValid({true, 1, 1}));
    EXPECT_TRUE(generic.geometry.generic);
}

TEST(TestHipConvDgradFma1dRdnaLaunchValues, SavesTheV1ValueNames)
{
    const auto values = savedValid().values;
    for(const auto* name : {"dy.uid",
                            "w.uid",
                            "dx.uid",
                            "n",
                            "c",
                            "k",
                            "groups",
                            "in_w",
                            "filter_w",
                            "out_w",
                            "pad_w",
                            "stride_w",
                            "dilation_w",
                            "generic",
                            "q_per_lane",
                            "channels_per_wave"})
    {
        EXPECT_EQ(values.count(name), 1U) << name;
    }
    EXPECT_EQ(values.size(), 16U);
}

TEST(TestHipConvDgradFma1dRdnaLaunchValues, RefusesMalformedValues)
{
    const auto refused = [](const SavedLaunchInputs& inputs, const char* what) {
        EXPECT_THROW(readHipConvDgradFma1dRdnaLaunchInputs(inputs),
                     hipdnn_plugin_sdk::HipdnnPluginException)
            << what;
    };

    auto otherContract = savedValid();
    otherContract.dispatchSymbol = "hipkernel.hip_conv_dgrad_fma_1d_rdna.dispatch.v2";
    refused(otherContract, "another contract");

    auto missing = savedValid();
    missing.values.erase("channels_per_wave");
    refused(missing, "a missing value");

    auto extra = savedValid();
    extra.values["in_h"] = int64_t{1};
    refused(extra, "a value outside the contract");

    auto notMultiple = savedValid();
    notMultiple.values["groups"] = int64_t{5};
    refused(notMultiple, "channels not a multiple of the group count");

    refused(savedValid({false, 3, 2}), "three q per lane");
    refused(savedValid({false, 2, 3}), "three channels per wave");
    refused(savedValid({true, 1, 2}), "a generic variant with a channel tile");

    auto wideBatch = savedValid();
    wideBatch.values["n"] = int64_t{70000};
    refused(wideBatch, "a specialized grid beyond the launch limits");

    auto longRow = savedValid();
    longRow.values["in_w"] = int64_t{std::numeric_limits<int>::max() - 100};
    refused(longRow, "specialized dx positions beyond int");

    auto wideGeneric = savedValid({true, 1, 1});
    wideGeneric.values["n"] = int64_t{70000};
    EXPECT_NO_THROW(readHipConvDgradFma1dRdnaLaunchInputs(wideGeneric))
        << "the generic grid walks dx in a grid-stride loop";
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
