// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr PackSymbols GFX950_CONV_FWD{"hipkernel:Gfx950ConvFwd",
                                      "hipkernel.gfx950_conv_fwd.graph_match",
                                      "",
                                      "hipkernel.gfx950_conv_fwd.kernel_match",
                                      "hipkernel.gfx950_conv_fwd.score",
                                      "hipkernel.gfx950_conv_fwd.dispatch",
                                      "gfx950_conv_fwd.x.uid",
                                      "gfx950_conv_fwd.w.uid",
                                      "gfx950_conv_fwd.y.uid"};

struct TensorSpec
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    data_objects::DataType dtype = data_objects::DataType::HALF;
    bool isVirtual = false;
    bool passByValue = false;
    bool constant = false;
    bool omitDims = false;
    bool omitStrides = false;
    std::optional<int64_t> raggedOffset = std::nullopt;
    int64_t alignment = 16;
};

struct GraphSpec
{
    std::array<TensorSpec, 3> tensors{{{{2, 32, 14, 14}, {6272, 1, 448, 32}},
                                       {{32, 32, 3, 3}, {288, 1, 96, 32}},
                                       {{2, 32, 14, 14}, {6272, 1, 448, 32}}}};
    std::vector<int64_t> stride{1, 1};
    std::vector<int64_t> dilation{1, 1};
    std::vector<int64_t> prePadding{1, 1};
    std::vector<int64_t> postPadding{1, 1};
    data_objects::DataType computeType = data_objects::DataType::FLOAT;
    data_objects::ConvMode mode = data_objects::ConvMode::CROSS_CORRELATION;
    bool omitStride = false;
    bool omitTensor = false;
    bool overrideShapes = false;
    unsigned int nodeCount = 1;
    int64_t xUid = 1;
    int64_t wUid = 2;
    int64_t yUid = 3;
};

DeviceProperties gfx950Properties(std::string arch = "gfx950")
{
    // Deliberately by value: matcher tests must run on CPU-only and non-gfx950 hosts.
    DeviceProperties properties;
    properties.gcnArchName = std::move(arch);
    properties.warpSize = 64;
    return properties;
}

flatbuffers::FlatBufferBuilder buildGraph(const GraphSpec& spec = {})
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    for(size_t i = 0; i < spec.tensors.size(); ++i)
    {
        if(spec.omitTensor && i == 0)
        {
            continue;
        }
        const auto& tensor = spec.tensors[i];
        const auto dims = tensor.omitDims ? flatbuffers::Offset<flatbuffers::Vector<int64_t>>()
                                          : builder.CreateVector(tensor.dims);
        const auto strides = tensor.omitStrides
                                 ? flatbuffers::Offset<flatbuffers::Vector<int64_t>>()
                                 : builder.CreateVector(tensor.strides);
        const auto value = tensor.constant
                               ? builder.CreateStruct(data_objects::Float16Value(1.0F)).Union()
                               : flatbuffers::Offset<void>();
        data_objects::TensorAttributesBuilder tensorBuilder(builder);
        tensorBuilder.add_uid(static_cast<int64_t>(i + 1));
        tensorBuilder.add_data_type(tensor.dtype);
        tensorBuilder.add_dims(dims);
        tensorBuilder.add_strides(strides);
        tensorBuilder.add_virtual_(tensor.isVirtual);
        tensorBuilder.add_is_runtime_pass_by_value(tensor.passByValue);
        tensorBuilder.add_alignment(tensor.alignment);
        if(tensor.constant)
        {
            tensorBuilder.add_value_type(data_objects::TensorValue::Float16Value);
            tensorBuilder.add_value(value);
        }
        if(tensor.raggedOffset)
        {
            tensorBuilder.add_ragged_offset_tensor_uid(*tensor.raggedOffset);
        }
        tensors.push_back(tensorBuilder.Finish());
    }
    const auto attributes = data_objects::CreateConvolutionFwdAttributesDirect(
        builder,
        spec.xUid,
        spec.wUid,
        spec.yUid,
        &spec.prePadding,
        &spec.postPadding,
        spec.omitStride ? nullptr : &spec.stride,
        &spec.dilation,
        spec.mode);
    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.reserve(spec.nodeCount);
    for(unsigned int i = 0; i < spec.nodeCount; ++i)
    {
        nodes.push_back(
            data_objects::CreateNodeDirect(builder,
                                           "gfx950_conv_fwd_test",
                                           spec.computeType,
                                           data_objects::NodeAttributes::ConvolutionFwdAttributes,
                                           attributes.Union()));
    }
    const auto tensorVector = builder.CreateVector(tensors);
    const auto nodeVector = builder.CreateVector(nodes);
    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_tensors(tensorVector);
    graphBuilder.add_nodes(nodeVector);
    graphBuilder.add_is_override_shape_enabled(spec.overrideShapes);
    builder.Finish(graphBuilder.Finish());
    return builder;
}

KernelDefinition smokeKernel(int64_t tileK = 64, const std::string& dtype = "fp16")
{
    KernelDefinition kernel;
    kernel.source.kind = KernelSourceKind::KPACK;
    kernel.metadata = {{"N", int64_t{2}},
                       {"C", int64_t{32}},
                       {"K", int64_t{32}},
                       {"Hi", int64_t{14}},
                       {"Wi", int64_t{14}},
                       {"Y", int64_t{3}},
                       {"X", int64_t{3}},
                       {"sH", int64_t{1}},
                       {"sW", int64_t{1}},
                       {"pH", int64_t{1}},
                       {"pW", int64_t{1}},
                       {"dH", int64_t{1}},
                       {"dW", int64_t{1}},
                       {"dtype", dtype},
                       {"tile_m", int64_t{64}},
                       {"tile_n", int64_t{64}},
                       {"tile_k", tileK},
                       {"warp_m", int64_t{2}},
                       {"warp_n", int64_t{2}},
                       {"warp_tile_m", int64_t{32}},
                       {"warp_tile_n", int64_t{32}},
                       {"warp_tile_k", int64_t{16}},
                       {"wave_size", int64_t{64}},
                       {"pipeline", std::string("mem")},
                       {"epilogue", std::string("cshuffle")},
                       {"layout", std::string("NHWC")},
                       {"groups", int64_t{1}},
                       // The descriptor loader completes these from their KMD defaults.
                       {"kernel_family", int64_t{0}},
                       {"direct_variant", std::string("none")},
                       {"block_w", int64_t{0}},
                       {"block_waves", int64_t{0}}};
    return kernel;
}

/// Groups are implicit in hipDNN: a [K, C / G, 3, 3] filter over the default 32-channel
/// input, with the matching [2, K, 14, 14] output.
GraphSpec groupedSpec(int64_t k, int64_t channelsPerGroup)
{
    GraphSpec spec;
    spec.tensors[1].dims = {k, channelsPerGroup, 3, 3};
    spec.tensors[1].strides = {9 * channelsPerGroup, 1, 3 * channelsPerGroup, channelsPerGroup};
    spec.tensors[2].dims = {2, k, 14, 14};
    spec.tensors[2].strides = {196 * k, 1, 14 * k, k};
    return spec;
}

KernelDefinition groupedKernel(int64_t k, int64_t groups)
{
    auto kernel = smokeKernel();
    kernel.metadata["K"] = k;
    kernel.metadata["groups"] = groups;
    return kernel;
}

/// A complete forward problem, so one description yields a graph and the kernels
/// compiled for it. groups divides c and k; the filter holds c / groups channels.
struct ConvShape
{
    int64_t n = 2;
    int64_t c = 32;
    int64_t k = 32;
    int64_t groups = 32;
    int64_t h = 14;
    int64_t w = 14;
    int64_t y = 3;
    int64_t x = 3;
    int64_t sH = 1;
    int64_t sW = 1;
    int64_t pH = 1;
    int64_t pW = 1;
    int64_t dH = 1;
    int64_t dW = 1;
};

ConvShape depthwiseShape(
    int64_t groups, int64_t h, int64_t w, int64_t filter, int64_t stride, int64_t pad)
{
    ConvShape shape;
    shape.c = shape.k = shape.groups = groups;
    shape.h = h;
    shape.w = w;
    shape.y = shape.x = filter;
    shape.sH = shape.sW = stride;
    shape.pH = shape.pW = pad;
    return shape;
}

GraphSpec convSpec(const ConvShape& shape)
{
    const auto cpg = shape.c / shape.groups;
    const auto ho = (shape.h + 2 * shape.pH - ((shape.y - 1) * shape.dH + 1)) / shape.sH + 1;
    const auto wo = (shape.w + 2 * shape.pW - ((shape.x - 1) * shape.dW + 1)) / shape.sW + 1;
    GraphSpec spec;
    spec.tensors[0].dims = {shape.n, shape.c, shape.h, shape.w};
    spec.tensors[0].strides = {shape.h * shape.w * shape.c, 1, shape.w * shape.c, shape.c};
    spec.tensors[1].dims = {shape.k, cpg, shape.y, shape.x};
    spec.tensors[1].strides = {shape.y * shape.x * cpg, 1, shape.x * cpg, cpg};
    spec.tensors[2].dims = {shape.n, shape.k, ho, wo};
    spec.tensors[2].strides = {ho * wo * shape.k, 1, wo * shape.k, shape.k};
    spec.stride = {shape.sH, shape.sW};
    spec.dilation = {shape.dH, shape.dW};
    spec.prePadding = spec.postPadding = {shape.pH, shape.pW};
    return spec;
}

/// An implicit-GEMM kernel compiled for exactly @p shape; cshuffle serves any K / groups.
KernelDefinition implicitKernel(const ConvShape& shape, int64_t tileK = 64)
{
    auto kernel = smokeKernel(tileK);
    for(const auto& [field, value] :
        std::vector<std::pair<const char*, int64_t>>{{"N", shape.n},
                                                     {"C", shape.c},
                                                     {"K", shape.k},
                                                     {"groups", shape.groups},
                                                     {"Hi", shape.h},
                                                     {"Wi", shape.w},
                                                     {"Y", shape.y},
                                                     {"X", shape.x},
                                                     {"sH", shape.sH},
                                                     {"sW", shape.sW},
                                                     {"pH", shape.pH},
                                                     {"pW", shape.pW},
                                                     {"dH", shape.dH},
                                                     {"dW", shape.dW}})
    {
        kernel.metadata[field] = value;
    }
    return kernel;
}

/// A direct depthwise kernel compiled for exactly @p shape, with the exact placeholders
/// gfx950_conv_fwd_direct_spec_for_request writes into the implicit-GEMM fields.
KernelDefinition directKernel(const ConvShape& shape,
                              const std::string& variant,
                              int64_t blockW,
                              int64_t blockWaves)
{
    auto kernel = implicitKernel(shape, 0);
    for(const auto* field : {"tile_m",
                             "tile_n",
                             "tile_k",
                             "warp_m",
                             "warp_n",
                             "warp_tile_m",
                             "warp_tile_n",
                             "warp_tile_k"})
    {
        kernel.metadata[field] = int64_t{0};
    }
    kernel.metadata["pipeline"] = std::string("none");
    kernel.metadata["epilogue"] = std::string("none");
    kernel.metadata["kernel_family"] = int64_t{1};
    kernel.metadata["direct_variant"] = variant;
    kernel.metadata["block_w"] = blockW;
    kernel.metadata["block_waves"] = blockWaves;
    return kernel;
}

KernelDefinition spatialKernel(const ConvShape& shape, int64_t blockWaves = 1)
{
    return directKernel(shape, "spatial", 0, blockWaves);
}

KernelDefinition stdKernel(const ConvShape& shape, int64_t blockW = 4, int64_t blockWaves = 1)
{
    return directKernel(shape, "std", blockW, blockWaves);
}

TEST(TestGfx950ConvFwdGraphMatcher, AcceptsBothStorageTypesAndBindsActualTensorUids)
{
    for(const auto dtype : {data_objects::DataType::HALF, data_objects::DataType::BFLOAT16})
    {
        GraphSpec spec;
        for(auto& tensor : spec.tensors)
        {
            tensor.dtype = dtype;
        }
        const GraphFixture fixture(buildGraph(spec), gfx950Properties());
        const auto bound = matchesGraph(GFX950_CONV_FWD, fixture.context());
        ASSERT_TRUE(bound);
        EXPECT_EQ(tryGetBoundInt(*bound, GFX950_CONV_FWD.inputAToken), 1);
        EXPECT_EQ(tryGetBoundInt(*bound, GFX950_CONV_FWD.inputBToken), 2);
        EXPECT_EQ(tryGetBoundInt(*bound, GFX950_CONV_FWD.outputToken), 3);
    }
}

TEST(TestGfx950ConvFwdGraphMatcher, ArchGateUnderstandsFeatureSuffixes)
{
    const GraphFixture accepted(buildGraph(), gfx950Properties("gfx950:sramecc+:xnack-"));
    EXPECT_TRUE(matchesGraph(GFX950_CONV_FWD, accepted.context()));
    for(const auto* arch : {"gfx942", "gfx9500", "gfx95", "", "prefix-gfx950"})
    {
        SCOPED_TRACE(arch);
        const GraphFixture refused(buildGraph(), gfx950Properties(arch));
        EXPECT_FALSE(matchesGraph(GFX950_CONV_FWD, refused.context()));
    }
    auto properties = gfx950Properties();
    properties.warpSize = 32;
    const GraphFixture refused(buildGraph(), properties);
    EXPECT_FALSE(matchesGraph(GFX950_CONV_FWD, refused.context()));
}

TEST(TestGfx950ConvFwdGraphMatcher, AcceptsUnitExtentAxesWithArbitraryStrides)
{
    GraphSpec spec;
    spec.tensors[0].dims = {1, 1, 14, 14};
    spec.tensors[0].strides = {0, 196, 14, 1};
    spec.tensors[1].dims = {1, 1, 3, 3};
    spec.tensors[1].strides = {9, 9, 3, 1};
    spec.tensors[2].dims = {1, 1, 14, 14};
    spec.tensors[2].strides = {196, 196, 14, 1};
    const GraphFixture fixture(buildGraph(spec), gfx950Properties());
    EXPECT_TRUE(matchesGraph(GFX950_CONV_FWD, fixture.context()));
}

TEST(TestGfx950ConvFwdGraphMatcher, AllowsOneByOneStridedDilatedAndNonSquareProblems)
{
    const std::vector<std::function<void(GraphSpec&)>> cases{
        [](auto& s) {
            s.tensors[1].dims = {32, 32, 1, 1};
            s.tensors[1].strides = {32, 1, 32, 32};
        },
        [](auto& s) { s.stride = {2, 3}; },
        [](auto& s) { s.dilation = {2, 1}; },
        [](auto& s) {
            s.tensors[0].dims = {2, 32, 15, 19};
            s.tensors[0].strides = {9120, 1, 608, 32};
        }};
    for(const auto& change : cases)
    {
        GraphSpec spec;
        change(spec);
        // The frontend will infer the changed output dimensions before preparation.
        spec.tensors[2].omitDims = true;
        spec.tensors[2].omitStrides = true;
        const GraphFixture fixture(buildGraph(spec), gfx950Properties());
        EXPECT_TRUE(matchesGraph(GFX950_CONV_FWD, fixture.context()));
    }
}

struct RefusalCase
{
    const char* name;
    std::function<void(GraphSpec&)> change;
};

TEST(TestGfx950ConvFwdGraphMatcher, RefusesUnsupportedGraphsBeforeReadingTheirGeometry)
{
    const std::vector<RefusalCase> cases{
        {"fused graph", [](auto& s) { s.nodeCount = 2; }},
        {"empty graph", [](auto& s) { s.nodeCount = 0; }},
        {"runtime shape overrides", [](auto& s) { s.overrideShapes = true; }},
        {"convolution mode", [](auto& s) { s.mode = data_objects::ConvMode::CONVOLUTION; }},
        {"half accumulation", [](auto& s) { s.computeType = data_objects::DataType::HALF; }},
        {"unset accumulation", [](auto& s) { s.computeType = data_objects::DataType::UNSET; }},
        {"missing tensor", [](auto& s) { s.omitTensor = true; }},
        {"dangling tensor", [](auto& s) { s.xUid = 999; }},
        {"aliased output uid", [](auto& s) { s.yUid = s.xUid; }},
        {"filter channels do not divide input channels",
         [](auto& s) {
             s.tensors[1].dims = {32, 12, 3, 3};
             s.tensors[1].strides = {108, 1, 36, 12};
         }},
        {"output channels do not divide into groups", [](auto& s) { s = groupedSpec(30, 8); }},
        {"grouped pointwise",
         [](auto& s) {
             s = groupedSpec(32, 8);
             s.tensors[1].dims = {32, 8, 1, 1};
             s.tensors[1].strides = {8, 1, 8, 8};
             s.prePadding = s.postPadding = {0, 0};
         }},
        {"depthwise pointwise",
         [](auto& s) {
             s = groupedSpec(32, 1);
             s.tensors[1].dims = {32, 1, 1, 1};
             s.tensors[1].strides = {1, 1, 1, 1};
             s.prePadding = s.postPadding = {0, 0};
         }},
        {"NCHW input", [](auto& s) { s.tensors[0].strides = {6272, 196, 14, 1}; }},
        {"KCYX filter", [](auto& s) { s.tensors[1].strides = {288, 9, 3, 1}; }},
        {"nonpacked input", [](auto& s) { s.tensors[0].strides[0] += 32; }},
        {"negative stride", [](auto& s) { s.tensors[0].strides[2] = -448; }},
        {"rank three input", [](auto& s) { s.tensors[0].dims.pop_back(); }},
        {"rank three filter", [](auto& s) { s.tensors[1].dims.pop_back(); }},
        {"rank five input", [](auto& s) { s.tensors[0].dims.push_back(1); }},
        {"missing input dims", [](auto& s) { s.tensors[0].omitDims = true; }},
        {"missing filter strides", [](auto& s) { s.tensors[1].omitStrides = true; }},
        {"zero extent", [](auto& s) { s.tensors[0].dims[2] = 0; }},
        {"negative extent", [](auto& s) { s.tensors[1].dims[3] = -1; }},
        {"overflow extent",
         [](auto& s) { s.tensors[0].dims[0] = std::numeric_limits<int64_t>::max(); }},
        {"missing stride", [](auto& s) { s.omitStride = true; }},
        {"wrong stride rank", [](auto& s) { s.stride = {1}; }},
        {"zero stride", [](auto& s) { s.stride[0] = 0; }},
        {"negative dilation", [](auto& s) { s.dilation[1] = -1; }},
        {"wrong dilation rank", [](auto& s) { s.dilation = {1, 1, 1}; }},
        {"negative padding", [](auto& s) { s.prePadding[0] = s.postPadding[0] = -1; }},
        {"asymmetric padding", [](auto& s) { s.postPadding[0] = 2; }},
        {"empty padding", [](auto& s) { s.prePadding.clear(); }},
        {"empty output extent", [](auto& s) { s.dilation = {100, 100}; }}};
    for(const auto& test : cases)
    {
        SCOPED_TRACE(test.name);
        GraphSpec spec;
        test.change(spec);
        const GraphFixture fixture(buildGraph(spec), gfx950Properties());
        EXPECT_FALSE(matchesGraph(GFX950_CONV_FWD, fixture.context()));
    }
    const GraphFixture pointwise(buildPointwiseGraph(), gfx950Properties());
    EXPECT_FALSE(matchesGraph(GFX950_CONV_FWD, pointwise.context()));
}

TEST(TestGfx950ConvFwdGraphMatcher, AcceptsGroupedAndDepthwiseFilters)
{
    // G=4, depthwise (G=C), and depthwise with a channel multiplier of 2.
    for(const auto& [k, channelsPerGroup] :
        std::vector<std::pair<int64_t, int64_t>>{{32, 8}, {32, 1}, {64, 1}})
    {
        SCOPED_TRACE(channelsPerGroup);
        SCOPED_TRACE(k);
        const GraphFixture fixture(buildGraph(groupedSpec(k, channelsPerGroup)),
                                   gfx950Properties());
        EXPECT_TRUE(matchesGraph(GFX950_CONV_FWD, fixture.context()));
    }
}

TEST(TestGfx950ConvFwdGraphMatcher, GroupedOneByOneIsAcceptedOnlyOffThePointwiseShortcut)
{
    // A 1x1 filter with padding or a non-unit stride is not rocKE's pointwise shortcut,
    // so the grouped kernel indexes it correctly.
    const std::vector<std::function<void(GraphSpec&)>> cases{
        [](auto&) {},
        [](auto& s) {
            s.stride = {2, 1};
            s.prePadding = s.postPadding = {0, 0};
        },
        [](auto& s) {
            s.stride = {1, 2};
            s.prePadding = s.postPadding = {0, 0};
        },
        [](auto& s) { s.prePadding = s.postPadding = {0, 0}; }};
    for(size_t i = 0; i < cases.size(); ++i)
    {
        SCOPED_TRACE(i);
        auto spec = groupedSpec(32, 8);
        spec.tensors[1].dims = {32, 8, 1, 1};
        spec.tensors[1].strides = {8, 1, 8, 8};
        cases[i](spec);
        spec.tensors[2].omitDims = true;
        spec.tensors[2].omitStrides = true;
        const GraphFixture fixture(buildGraph(spec), gfx950Properties());
        // Only the last case, unit stride and no padding, takes the pointwise shortcut.
        EXPECT_EQ(matchesGraph(GFX950_CONV_FWD, fixture.context()).has_value(),
                  i + 1 != cases.size());
    }
}

TEST(TestGfx950ConvFwdGraphMatcher, GroupCountIsBoundedByGridZ)
{
    for(const auto groups : {int64_t{65535}, int64_t{65536}})
    {
        SCOPED_TRACE(groups);
        GraphSpec spec;
        spec.tensors[0].dims = {1, groups, 1, 1};
        spec.tensors[0].strides = {groups, 1, groups, groups};
        spec.tensors[1].dims = {groups, 1, 3, 3};
        spec.tensors[1].strides = {9, 1, 3, 1};
        spec.tensors[2].omitDims = true;
        spec.tensors[2].omitStrides = true;
        const GraphFixture fixture(buildGraph(spec), gfx950Properties());
        EXPECT_EQ(matchesGraph(GFX950_CONV_FWD, fixture.context()).has_value(), groups == 65535);
    }
}

TEST(TestGfx950ConvFwdGraphMatcher, RejectsUnsupportedStorageOnEveryOperand)
{
    const std::vector<std::function<void(TensorSpec&)>> changes{
        [](auto& t) { t.dtype = data_objects::DataType::FLOAT; },
        [](auto& t) { t.dtype = data_objects::DataType::BFLOAT16; },
        [](auto& t) { t.isVirtual = true; },
        [](auto& t) { t.passByValue = true; },
        [](auto& t) { t.constant = true; },
        [](auto& t) { t.raggedOffset = 7; },
        [](auto& t) { t.alignment = 8; }};
    for(size_t operand = 0; operand < 3; ++operand)
    {
        SCOPED_TRACE(operand);
        for(const auto& change : changes)
        {
            GraphSpec spec;
            change(spec.tensors[operand]);
            const GraphFixture fixture(buildGraph(spec), gfx950Properties());
            EXPECT_FALSE(matchesGraph(GFX950_CONV_FWD, fixture.context()));
        }
    }
}

TEST(TestGfx950ConvFwdKernelMatcher, BothTileKVariantsMatchOnlyTheirStorageType)
{
    for(const auto dtype : {data_objects::DataType::HALF, data_objects::DataType::BFLOAT16})
    {
        GraphSpec spec;
        for(auto& tensor : spec.tensors)
        {
            tensor.dtype = dtype;
        }
        const GraphFixture fixture(buildGraph(spec), gfx950Properties());
        const std::string name = dtype == data_objects::DataType::HALF ? "fp16" : "bf16";
        const std::string otherName = name == "fp16" ? "bf16" : "fp16";
        for(const auto tileK : {64, 128})
        {
            EXPECT_TRUE(
                matchesKernel(GFX950_CONV_FWD, fixture.context(), smokeKernel(tileK, name)));
            EXPECT_FALSE(
                matchesKernel(GFX950_CONV_FWD, fixture.context(), smokeKernel(tileK, otherName)));
        }
    }
}

TEST(TestGfx950ConvFwdKernelMatcher, EveryBakedShapeAttributeMustMatch)
{
    const GraphFixture fixture(buildGraph(), gfx950Properties());
    for(const auto* field :
        {"N", "C", "K", "Hi", "Wi", "Y", "X", "sH", "sW", "pH", "pW", "dH", "dW"})
    {
        SCOPED_TRACE(field);
        auto kernel = smokeKernel();
        kernel.metadata[field] = kernel.getIntMetadata(field) + 1;
        EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
    }
}

TEST(TestGfx950ConvFwdKernelMatcher, MissingOrWronglyTypedMetadataDeclinesCleanly)
{
    const GraphFixture fixture(buildGraph(), gfx950Properties());
    const auto original = smokeKernel();
    for(const auto& field : original.metadata)
    {
        SCOPED_TRACE(field.first);
        auto kernel = original;
        kernel.metadata.erase(field.first);
        EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
        kernel.metadata[field.first] = std::vector<int64_t>{1};
        EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
    }
}

TEST(TestGfx950ConvFwdKernelMatcher, GroupCountMustMatchTheGraph)
{
    const GraphFixture grouped(buildGraph(groupedSpec(32, 8)), gfx950Properties());
    EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD, grouped.context(), groupedKernel(32, 4)));
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, grouped.context(), smokeKernel()));
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, grouped.context(), groupedKernel(32, 2)));
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, grouped.context(), groupedKernel(32, 8)));

    const GraphFixture depthwise(buildGraph(groupedSpec(64, 1)), gfx950Properties());
    EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD, depthwise.context(), groupedKernel(64, 32)));
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, depthwise.context(), groupedKernel(64, 1)));

    const GraphFixture ungrouped(buildGraph(), gfx950Properties());
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, ungrouped.context(), groupedKernel(32, 4)));
}

TEST(TestGfx950ConvFwdKernelMatcher, AcceptsSweptTuningForTheMatchedGeometry)
{
    // The build validated each variant against its geometry; the matcher admits any
    // tuning whose launch it can compute, including the non-dispatcher pipelines.
    const GraphFixture fixture(buildGraph(), gfx950Properties());
    for(const auto* pipeline : {"mem", "compv3", "compv4", "basic"})
    {
        SCOPED_TRACE(pipeline);
        auto kernel = smokeKernel();
        kernel.metadata["tile_m"] = int64_t{256};
        kernel.metadata["tile_n"] = int64_t{32};
        kernel.metadata["tile_k"] = int64_t{32};
        kernel.metadata["warp_m"] = int64_t{8};
        kernel.metadata["warp_n"] = int64_t{1};
        kernel.metadata["warp_tile_m"] = int64_t{16};
        kernel.metadata["warp_tile_n"] = int64_t{16};
        kernel.metadata["warp_tile_k"] = int64_t{32};
        kernel.metadata["pipeline"] = std::string(pipeline);
        EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
    }
}

TEST(TestGfx950ConvFwdKernelMatcher, DeclinesUnreviewedLaunchContractAndUnpackagedCode)
{
    const GraphFixture fixture(buildGraph(), gfx950Properties());
    for(const auto* field : {"tile_m",
                             "tile_n",
                             "tile_k",
                             "warp_m",
                             "warp_n",
                             "warp_tile_m",
                             "warp_tile_n",
                             "warp_tile_k"})
    {
        SCOPED_TRACE(field);
        auto kernel = smokeKernel();
        kernel.metadata[field] = int64_t{0};
        EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
        kernel.metadata.erase(field);
        EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
        kernel.metadata[field] = std::string("64");
        EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
    }
    auto kernel = smokeKernel();
    kernel.metadata["wave_size"] = int64_t{32};
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
    // 8 x 4 waves of 64 is 2048 threads, past the 1024-thread workgroup limit.
    kernel = smokeKernel();
    kernel.metadata["warp_m"] = int64_t{8};
    kernel.metadata["warp_n"] = int64_t{4};
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
    kernel = smokeKernel();
    kernel.metadata.erase("pipeline");
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
    kernel = smokeKernel();
    kernel.metadata["pipeline"] = std::string("wavelet");
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
    kernel = smokeKernel();
    kernel.metadata["epilogue"] = std::string("relu");
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
    kernel = smokeKernel();
    kernel.metadata["groups"] = int64_t{2};
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
    kernel = smokeKernel();
    kernel.metadata["layout"] = std::string("NCHW");
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
    kernel = smokeKernel();
    for(const auto kind : {KernelSourceKind::EMBEDDED_SOURCE,
                           KernelSourceKind::ROCKE_BUILDER,
                           KernelSourceKind::HSACO_FILE})
    {
        kernel.source.kind = kind;
        EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
    }
}

TEST(TestGfx950ConvFwdKernelMatcher, DefaultEpilogueRequiresScalarOutputStores)
{
    // is_valid_spec_for_problem accepts K=31 (auto vec_c=1) and rejects K=32
    // (auto vec_c=8) for default epilogues, for both dtypes and tile_k choices.
    for(const auto dtype : {data_objects::DataType::HALF, data_objects::DataType::BFLOAT16})
    {
        const std::string name = dtype == data_objects::DataType::HALF ? "fp16" : "bf16";
        SCOPED_TRACE(name);
        for(const auto k : {int64_t{31}, int64_t{32}})
        {
            SCOPED_TRACE(k);
            GraphSpec spec;
            for(auto& tensor : spec.tensors)
            {
                tensor.dtype = dtype;
            }
            spec.tensors[1].dims[0] = k;
            spec.tensors[2].dims[1] = k;
            spec.tensors[2].strides = {k * 14 * 14, 1, k * 14, k};
            const GraphFixture fixture(buildGraph(spec), gfx950Properties());
            for(const auto tileK : {64, 128})
            {
                SCOPED_TRACE(tileK);
                auto kernel = smokeKernel(tileK, name);
                kernel.metadata["K"] = k;
                kernel.metadata["epilogue"] = std::string("default");
                EXPECT_EQ(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel), k == 31);
            }
        }
    }
}

TEST(TestGfx950ConvFwdKernelMatcher, DefaultEpilogueFollowsOutputChannelsPerGroup)
{
    // rocKE derives vec_c from K / G, not K: an even K with an odd per-group count
    // (depthwise K=32, G=32; K=96, G=32) still stores scalars.
    for(const auto& [k, channelsPerGroup, accepted] :
        std::vector<std::tuple<int64_t, int64_t, bool>>{
            {32, 1, true}, {96, 1, true}, {64, 1, false}, {32, 8, false}, {96, 8, false}})
    {
        const auto groups = 32 / channelsPerGroup;
        SCOPED_TRACE(groups);
        SCOPED_TRACE(k);
        const GraphFixture fixture(buildGraph(groupedSpec(k, channelsPerGroup)),
                                   gfx950Properties());
        auto kernel = groupedKernel(k, groups);
        EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
        kernel.metadata["epilogue"] = std::string("default");
        EXPECT_EQ(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel), accepted);
    }
}

TEST(TestGfx950ConvFwdScore, FallbackPrefersDispatcherDefault)
{
    const GraphFixture fixture(buildGraph(), gfx950Properties());
    EXPECT_GT(scoreKernel(GFX950_CONV_FWD, fixture.context(), smokeKernel(64)),
              scoreKernel(GFX950_CONV_FWD, fixture.context(), smokeKernel(128)));
    // Any swept tuning ranks below both dispatcher variants, so an unbenchmarked run
    // serves what rocKE's dispatcher would.
    for(const auto& change :
        std::vector<std::pair<const char*, MetadataValue>>{{"tile_m", int64_t{256}},
                                                           {"warp_m", int64_t{4}},
                                                           {"warp_tile_k", int64_t{32}},
                                                           {"pipeline", std::string("compv4")}})
    {
        SCOPED_TRACE(change.first);
        auto swept = smokeKernel(64);
        swept.metadata[change.first] = change.second;
        EXPECT_GT(scoreKernel(GFX950_CONV_FWD, fixture.context(), smokeKernel(128)),
                  scoreKernel(GFX950_CONV_FWD, fixture.context(), swept));
    }
}

TEST(TestGfx950ConvFwdKernelMatcher, DirectKernelsMatchTheGuardedDepthwiseCatalogShapes)
{
    // Every direct arm of the synthetic catalog, for both storage types, beside the two
    // implicit-GEMM arms that serve the same graph.
    struct Case
    {
        ConvShape shape;
        std::vector<KernelDefinition> kernels;
    };
    const auto g32 = depthwiseShape(32, 14, 14, 3, 1, 1);
    const auto g32s2 = depthwiseShape(32, 17, 17, 3, 2, 1);
    const auto g5 = depthwiseShape(5, 9, 9, 7, 1, 3);
    const auto g96 = depthwiseShape(96, 14, 14, 3, 1, 1);
    const auto g96odd = depthwiseShape(96, 17, 17, 3, 2, 1);
    const auto g96even = depthwiseShape(96, 16, 18, 3, 2, 1);
    const auto g64 = depthwiseShape(64, 70, 40, 3, 1, 1);
    const auto g64s2 = depthwiseShape(64, 70, 40, 3, 2, 1);
    const auto g5large = depthwiseShape(5, 56, 24, 17, 1, 8);
    const std::vector<Case> cases{{g32, {spatialKernel(g32), stdKernel(g32)}},
                                  {g32s2, {spatialKernel(g32s2), stdKernel(g32s2)}},
                                  {g5, {spatialKernel(g5), stdKernel(g5)}},
                                  {g96, {stdKernel(g96), stdKernel(g96, 3, 2)}},
                                  {g96odd, {stdKernel(g96odd)}},
                                  {g96even, {stdKernel(g96even)}},
                                  {g64, {stdKernel(g64), stdKernel(g64, 32, 1)}},
                                  {g64s2, {stdKernel(g64s2), stdKernel(g64s2, 32, 1)}},
                                  {g5large, {spatialKernel(g5large)}}};
    for(size_t i = 0; i < cases.size(); ++i)
    {
        SCOPED_TRACE(i);
        for(const auto dtype : {data_objects::DataType::HALF, data_objects::DataType::BFLOAT16})
        {
            auto spec = convSpec(cases[i].shape);
            for(auto& tensor : spec.tensors)
            {
                tensor.dtype = dtype;
            }
            const std::string name = dtype == data_objects::DataType::HALF ? "fp16" : "bf16";
            const std::string otherName = name == "fp16" ? "bf16" : "fp16";
            const GraphFixture fixture(buildGraph(spec), gfx950Properties());
            ASSERT_TRUE(matchesGraph(GFX950_CONV_FWD, fixture.context()));
            auto kernels = cases[i].kernels;
            kernels.push_back(implicitKernel(cases[i].shape, 64));
            kernels.push_back(implicitKernel(cases[i].shape, 128));
            for(auto kernel : kernels)
            {
                kernel.metadata["dtype"] = name;
                EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
                kernel.metadata["dtype"] = otherName;
                EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
            }
        }
    }
}

TEST(TestGfx950ConvFwdKernelMatcher, DirectKernelsMustMatchEveryBakedShapeAttribute)
{
    const auto shape = depthwiseShape(32, 14, 14, 3, 1, 1);
    const GraphFixture fixture(buildGraph(convSpec(shape)), gfx950Properties());
    for(const auto& base : {spatialKernel(shape), stdKernel(shape)})
    {
        ASSERT_TRUE(matchesKernel(GFX950_CONV_FWD, fixture.context(), base));
        for(const auto* field :
            {"N", "C", "K", "Hi", "Wi", "Y", "X", "sH", "sW", "pH", "pW", "dH", "dW", "groups"})
        {
            SCOPED_TRACE(field);
            auto kernel = base;
            kernel.metadata[field] = kernel.getIntMetadata(field) + 1;
            EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
        }
        for(const auto& field : base.metadata)
        {
            SCOPED_TRACE(field.first);
            auto kernel = base;
            kernel.metadata.erase(field.first);
            EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
            kernel.metadata[field.first] = std::vector<int64_t>{1};
            EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
        }
    }
}

TEST(TestGfx950ConvFwdKernelMatcher, DirectKernelsDeclineEveryGraphOutsideTheGuardedContract)
{
    // Each graph is valid and served by implicit GEMM; a direct kernel compiled for the
    // exact same metadata is still refused, so the guard is what declines it.
    const auto base = depthwiseShape(32, 14, 14, 3, 1, 1);
    const std::vector<std::pair<const char*, std::function<void(ConvShape&)>>> cases{
        // Wrong-answer path 1: the runtime H loop writes past Ho into the next image.
        {"pad 0", [](auto& s) { s.pH = s.pW = 0; }},
        {"pad 0 stride 2",
         [](auto& s) {
             s.h = s.w = 16;
             s.pH = s.pW = 0;
             s.sH = s.sW = 2;
         }},
        // Wrong-answer path 2: output rows the input stream never reaches stay unwritten.
        {"pad 2", [](auto& s) { s.pH = s.pW = 2; }},
        {"pad 2 stride 2",
         [](auto& s) {
             s.h = s.w = 16;
             s.pH = s.pW = 2;
             s.sH = s.sW = 2;
         }},
        {"width only uncovered",
         [](auto& s) {
             s.w = 15;
             s.x = 1;
         }},
        // Contract refusals. Apart from dilation, which the guard does not model, each
        // shape is row-covered on both axes.
        {"sH != sW",
         [](auto& s) {
             s.w = 15;
             s.sW = 2;
         }},
        {"pH != pW",
         [](auto& s) {
             s.x = 1;
             s.pW = 0;
         }},
        {"dilation 2",
         [](auto& s) {
             s.dH = s.dW = 2;
             s.pH = s.pW = 2;
         }},
        {"dilation H only",
         [](auto& s) {
             s.dH = 2;
             s.pH = 2;
         }},
        {"channel multiplier", [](auto& s) { s.k = 64; }},
        {"grouped, not depthwise", [](auto& s) { s.groups = 4; }},
        {"grouped with one output per group",
         [](auto& s) {
             s.groups = 8;
             s.k = 8;
         }},
        {"dense", [](auto& s) { s.groups = 1; }}};
    for(const auto& [name, change] : cases)
    {
        SCOPED_TRACE(name);
        auto shape = base;
        change(shape);
        const GraphFixture fixture(buildGraph(convSpec(shape)), gfx950Properties());
        ASSERT_TRUE(matchesGraph(GFX950_CONV_FWD, fixture.context()));
        EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD, fixture.context(), implicitKernel(shape)));
        for(const auto& kernel : {spatialKernel(shape), stdKernel(shape)})
        {
            EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
        }
    }
}

TEST(TestGfx950ConvFwdKernelMatcher, DirectKernelsDeclineAGridZPastTheImageLimit)
{
    // z carries N for both direct variants; implicit GEMM carries groups there instead.
    for(const auto n : {int64_t{65535}, int64_t{65536}})
    {
        SCOPED_TRACE(n);
        auto shape = depthwiseShape(2, 3, 3, 3, 1, 1);
        shape.n = n;
        const GraphFixture fixture(buildGraph(convSpec(shape)), gfx950Properties());
        EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD, fixture.context(), implicitKernel(shape)));
        EXPECT_EQ(matchesKernel(GFX950_CONV_FWD, fixture.context(), spatialKernel(shape)),
                  n == 65535);
        EXPECT_EQ(matchesKernel(GFX950_CONV_FWD, fixture.context(), stdKernel(shape)), n == 65535);
    }
}

TEST(TestGfx950ConvFwdKernelMatcher, DirectKernelsRequireExactImplicitGemmPlaceholders)
{
    const auto shape = depthwiseShape(32, 14, 14, 3, 1, 1);
    const GraphFixture fixture(buildGraph(convSpec(shape)), gfx950Properties());
    const std::vector<std::pair<const char*, MetadataValue>> violations{
        {"tile_m", int64_t{64}},
        {"tile_n", int64_t{64}},
        {"tile_k", int64_t{64}},
        {"tile_k", int64_t{128}},
        {"tile_k", int64_t{-1}},
        {"warp_m", int64_t{1}},
        {"warp_n", int64_t{1}},
        {"warp_tile_m", int64_t{32}},
        {"warp_tile_n", int64_t{32}},
        {"warp_tile_k", int64_t{16}},
        {"wave_size", int64_t{32}},
        {"pipeline", std::string("mem")},
        {"pipeline", std::string("")},
        {"epilogue", std::string("cshuffle")},
        {"epilogue", std::string("default")},
        {"layout", std::string("NCHW")}};
    for(const auto& base : {spatialKernel(shape), stdKernel(shape)})
    {
        for(const auto& [field, value] : violations)
        {
            SCOPED_TRACE(field);
            auto kernel = base;
            kernel.metadata[field] = value;
            EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
        }
    }
}

TEST(TestGfx950ConvFwdKernelMatcher, FamilyAndVariantFieldsMustAgree)
{
    const auto shape = depthwiseShape(32, 14, 14, 3, 1, 1);
    const GraphFixture fixture(buildGraph(convSpec(shape)), gfx950Properties());
    const auto implicit = implicitKernel(shape);
    ASSERT_TRUE(matchesKernel(GFX950_CONV_FWD, fixture.context(), implicit));

    // Unknown families and variants are refused whatever the other fields say.
    for(const auto family : {int64_t{-1}, int64_t{2}, int64_t{100}})
    {
        SCOPED_TRACE(family);
        for(auto kernel : {implicit, spatialKernel(shape), stdKernel(shape)})
        {
            kernel.metadata["kernel_family"] = family;
            EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
        }
    }
    for(const auto* variant : {"", "STD", "direct_depthwise", "implicit_gemm"})
    {
        SCOPED_TRACE(variant);
        for(auto kernel : {implicit, stdKernel(shape)})
        {
            kernel.metadata["direct_variant"] = std::string(variant);
            EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
        }
    }
    // A string-typed family is never parsed.
    auto stringFamily = stdKernel(shape);
    stringFamily.metadata["kernel_family"] = std::string("1");
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), stringFamily));

    // An implicit-GEMM kernel must leave every direct field at its KMD default.
    for(const auto& [field, value] : std::vector<std::pair<const char*, MetadataValue>>{
            {"direct_variant", std::string("std")},
            {"direct_variant", std::string("spatial")},
            {"block_w", int64_t{4}},
            {"block_w", int64_t{-1}},
            {"block_waves", int64_t{1}}})
    {
        SCOPED_TRACE(field);
        auto kernel = implicit;
        kernel.metadata[field] = value;
        EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
    }
    // A direct kernel needs a direct variant, and a direct arm cannot pose as family 0.
    auto noVariant = stdKernel(shape);
    noVariant.metadata["direct_variant"] = std::string("none");
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), noVariant));
    for(auto kernel : {spatialKernel(shape), stdKernel(shape)})
    {
        kernel.metadata["kernel_family"] = int64_t{0};
        EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
    }
    // Nor can an implicit-GEMM tuning pose as family 1.
    auto relabeled = implicit;
    relabeled.metadata["kernel_family"] = int64_t{1};
    relabeled.metadata["direct_variant"] = std::string("std");
    relabeled.metadata["block_w"] = int64_t{4};
    relabeled.metadata["block_waves"] = int64_t{1};
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), relabeled));
}

TEST(TestGfx950ConvFwdKernelMatcher, DirectVariantTuningBounds)
{
    const auto g32 = depthwiseShape(32, 14, 14, 3, 1, 1);
    const GraphFixture fixture(buildGraph(convSpec(g32)), gfx950Properties());
    for(const auto blockWaves : {int64_t{1}, int64_t{16}})
    {
        SCOPED_TRACE(blockWaves);
        EXPECT_TRUE(
            matchesKernel(GFX950_CONV_FWD, fixture.context(), spatialKernel(g32, blockWaves)));
        EXPECT_TRUE(
            matchesKernel(GFX950_CONV_FWD, fixture.context(), stdKernel(g32, 4, blockWaves)));
    }
    for(const auto blockWaves : {int64_t{0}, int64_t{-1}, int64_t{17}})
    {
        SCOPED_TRACE(blockWaves);
        EXPECT_FALSE(
            matchesKernel(GFX950_CONV_FWD, fixture.context(), spatialKernel(g32, blockWaves)));
        EXPECT_FALSE(
            matchesKernel(GFX950_CONV_FWD, fixture.context(), stdKernel(g32, 4, blockWaves)));
    }
    // std needs a positive block width; spatial derives its own and must carry 0.
    EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD, fixture.context(), stdKernel(g32, 1)));
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), stdKernel(g32, 0)));
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), stdKernel(g32, -4)));
    EXPECT_FALSE(
        matchesKernel(GFX950_CONV_FWD, fixture.context(), directKernel(g32, "spatial", 2, 1)));

    // Spatial serves G < 64 only; std serves either side of it.
    for(const auto groups : {int64_t{63}, int64_t{64}, int64_t{96}})
    {
        SCOPED_TRACE(groups);
        const auto shape = depthwiseShape(groups, 14, 14, 3, 1, 1);
        const GraphFixture wide(buildGraph(convSpec(shape)), gfx950Properties());
        EXPECT_EQ(matchesKernel(GFX950_CONV_FWD, wide.context(), spatialKernel(shape)),
                  groups < 64);
        EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD, wide.context(), stdKernel(shape)));
    }
}

TEST(TestGfx950ConvFwdKernelMatcher, ForcedKnobValuesSelectOnlyTheirFamily)
{
    // GenericPlanBuilder::applyKnobFilter keeps a kernel when each forced knob equals its
    // completed integer metadata. Because kernelFits refuses every kernel whose
    // kernel_family, direct fields and tile_k disagree, a forced value can only ever
    // leave kernels of one family.
    const auto shape = depthwiseShape(32, 14, 14, 3, 1, 1);
    const GraphFixture fixture(buildGraph(convSpec(shape)), gfx950Properties());
    std::vector<KernelDefinition> catalog{implicitKernel(shape, 64),
                                          implicitKernel(shape, 128),
                                          spatialKernel(shape),
                                          stdKernel(shape)};
    // Inconsistent descriptors that a filter could otherwise keep.
    auto directTileK = stdKernel(shape);
    directTileK.metadata["tile_k"] = int64_t{64};
    auto implicitFamily1 = implicitKernel(shape, 64);
    implicitFamily1.metadata["kernel_family"] = int64_t{1};
    auto directFamily0 = spatialKernel(shape);
    directFamily0.metadata["kernel_family"] = int64_t{0};
    catalog.push_back(directTileK);
    catalog.push_back(implicitFamily1);
    catalog.push_back(directFamily0);

    const auto survivors = [&](const char* knob, int64_t value) {
        std::vector<std::string> variants;
        for(const auto& kernel : catalog)
        {
            if(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel)
               && kernel.getIntMetadata(knob) == value)
            {
                variants.push_back(std::get<std::string>(kernel.metadata.at("direct_variant")) + "/"
                                   + std::to_string(kernel.getIntMetadata("tile_k")));
            }
        }
        return variants;
    };
    using Names = std::vector<std::string>;
    EXPECT_EQ(survivors("kernel_family", 0), (Names{"none/64", "none/128"}));
    EXPECT_EQ(survivors("kernel_family", 1), (Names{"spatial/0", "std/0"}));
    EXPECT_EQ(survivors("tile_k", 0), (Names{"spatial/0", "std/0"}));
    EXPECT_EQ(survivors("tile_k", 64), (Names{"none/64"}));
    EXPECT_EQ(survivors("tile_k", 128), (Names{"none/128"}));
    EXPECT_TRUE(survivors("kernel_family", 2).empty());
}

TEST(TestGfx950ConvFwdScore, FallbackRanksTheDefaultDirectArmFirstThenOtherDirectArms)
{
    // G < 64: spatial with one wave is the default arm; G >= 64: std block_w 4, one wave.
    const auto g32 = depthwiseShape(32, 14, 14, 3, 1, 1);
    const GraphFixture small(buildGraph(convSpec(g32)), gfx950Properties());
    const auto scoreOf = [](const GraphFixture& fixture, const KernelDefinition& kernel) {
        EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
        return scoreKernel(GFX950_CONV_FWD, fixture.context(), kernel);
    };
    const auto defaultSmall = scoreOf(small, spatialKernel(g32));
    for(const auto& other : {spatialKernel(g32, 2), stdKernel(g32), stdKernel(g32, 3, 2)})
    {
        const auto otherScore = scoreOf(small, other);
        EXPECT_GT(defaultSmall, otherScore);
        EXPECT_GT(otherScore, scoreOf(small, implicitKernel(g32, 64)));
    }
    EXPECT_GT(scoreOf(small, implicitKernel(g32, 64)), scoreOf(small, implicitKernel(g32, 128)));

    const auto g96 = depthwiseShape(96, 14, 14, 3, 1, 1);
    const GraphFixture large(buildGraph(convSpec(g96)), gfx950Properties());
    const auto defaultLarge = scoreOf(large, stdKernel(g96));
    for(const auto& other : {stdKernel(g96, 3, 2), stdKernel(g96, 4, 2), stdKernel(g96, 8, 1)})
    {
        const auto otherScore = scoreOf(large, other);
        EXPECT_GT(defaultLarge, otherScore);
        EXPECT_GT(otherScore, scoreOf(large, implicitKernel(g96, 64)));
    }
    EXPECT_GT(scoreOf(large, implicitKernel(g96, 64)), scoreOf(large, implicitKernel(g96, 128)));
}

TEST(TestGfx950ConvFwdDispatch, DirectKernelPassesTheContractAndReachesCodeLoading)
{
    const auto shape = depthwiseShape(96, 14, 14, 3, 1, 1);
    const GraphFixture fixture(buildGraph(convSpec(shape)), gfx950Properties());
    const auto bound = matchesGraph(GFX950_CONV_FWD, fixture.context());
    ASSERT_TRUE(bound);
    const auto& handler = dispatchHandler(GFX950_CONV_FWD);
    EXPECT_EQ(handler.workspaceBytes(fixture.context(), *bound, stdKernel(shape)), 0U);
    try
    {
        // The descriptor names no archive, so loading is the first step that can fail.
        handler.prepare(fixture.context(), *bound, stdKernel(shape));
        FAIL() << "Preparation loaded a kernel from an empty archive path";
    }
    catch(const std::exception& error)
    {
        const std::string message = error.what();
        EXPECT_EQ(message.find("convolution contract"), std::string::npos) << message;
        EXPECT_EQ(message.find("packed NHWK output"), std::string::npos) << message;
    }

    // A guarded graph refuses the same arm before any code loads.
    auto unpadded = shape;
    unpadded.pH = unpadded.pW = 0;
    const GraphFixture guarded(buildGraph(convSpec(unpadded)), gfx950Properties());
    const auto guardedBound = matchesGraph(GFX950_CONV_FWD, guarded.context());
    ASSERT_TRUE(guardedBound);
    try
    {
        handler.prepare(guarded.context(), *guardedBound, stdKernel(unpadded));
        FAIL() << "Preparation accepted a direct kernel outside the row-coverage guard";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        EXPECT_NE(std::string(error.what()).find("convolution contract"), std::string::npos)
            << error.what();
    }
}

TEST(TestGfx950ConvFwdDispatch, RegisteredHandlerNeedsNoWorkspaceAndChecksBindings)
{
    const GraphFixture fixture(buildGraph(), gfx950Properties());
    const auto bound = matchesGraph(GFX950_CONV_FWD, fixture.context());
    ASSERT_TRUE(bound);
    const auto& handler = dispatchHandler(GFX950_CONV_FWD);
    EXPECT_EQ(handler.workspaceBytes(fixture.context(), *bound, smokeKernel()), 0U);
    EXPECT_THROW(handler.prepare(fixture.context(), {}, smokeKernel()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    auto wrongBound = *bound;
    wrongBound[std::string(GFX950_CONV_FWD.outputToken)] = int64_t{1};
    EXPECT_THROW(handler.prepare(fixture.context(), wrongBound, smokeKernel()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestGfx950ConvFwdDispatch, GroupedGraphPassesTheContractAndReachesCodeLoading)
{
    const GraphFixture fixture(buildGraph(groupedSpec(64, 1)), gfx950Properties());
    const auto bound = matchesGraph(GFX950_CONV_FWD, fixture.context());
    ASSERT_TRUE(bound);
    const auto& handler = dispatchHandler(GFX950_CONV_FWD);
    try
    {
        // The descriptor names no archive, so loading is the first step that can fail.
        handler.prepare(fixture.context(), *bound, groupedKernel(64, 32));
        FAIL() << "Preparation loaded a kernel from an empty archive path";
    }
    catch(const std::exception& error)
    {
        const std::string message = error.what();
        EXPECT_EQ(message.find("convolution contract"), std::string::npos) << message;
        EXPECT_EQ(message.find("packed NHWK output"), std::string::npos) << message;
    }
    // A groups=1 kernel for the same geometry is refused before any code loads.
    try
    {
        handler.prepare(fixture.context(), *bound, groupedKernel(64, 1));
        FAIL() << "Preparation accepted a groups=1 kernel for a grouped graph";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        EXPECT_NE(std::string(error.what()).find("convolution contract"), std::string::npos)
            << error.what();
    }
}

TEST(TestGfx950ConvFwdDispatch, ModuleCacheIsExportedAndClearedWithTheOtherPacks)
{
    namespace engine = hip_kernel_provider::kernel_ingestor_engine;
    engine::resetIngestorModuleCachesForTesting();
    EXPECT_EQ(engine::gfx950ConvFwdKpackModuleCache().size(), 0U);
}

TEST(TestGfx950ConvFwdDispatch, OutputValidationWaitsForPreparationAndPrecedesCodeLoading)
{
    const std::vector<std::function<void(TensorSpec&)>> cases{
        [](auto& t) {
            t.omitDims = true;
            t.omitStrides = true;
        },
        [](auto& t) { t.strides = {6272, 196, 14, 1}; },
        [](auto& t) { t.dims = {2, 32, 14}; },
        [](auto& t) { t.dims[2] = 0; },
        [](auto& t) { t.dims[0] = 1; },
        [](auto& t) {
            t.dims[1] = 16;
            t.strides = {3136, 1, 224, 16};
        }};
    for(const auto& change : cases)
    {
        GraphSpec spec;
        change(spec.tensors[2]);
        const GraphFixture fixture(buildGraph(spec), gfx950Properties());
        const auto bound = matchesGraph(GFX950_CONV_FWD, fixture.context());
        ASSERT_TRUE(bound);
        const auto& handler = dispatchHandler(GFX950_CONV_FWD);
        try
        {
            // An empty archive path must never be reached in any of these cases.
            handler.prepare(fixture.context(), *bound, smokeKernel());
            FAIL() << "Preparation accepted an invalid output tensor";
        }
        catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
        {
            EXPECT_NE(std::string(error.what()).find("packed NHWK output"), std::string::npos);
        }
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
