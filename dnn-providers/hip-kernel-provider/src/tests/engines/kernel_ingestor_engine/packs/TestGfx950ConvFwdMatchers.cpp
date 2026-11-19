// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
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
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

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
/// gfx950_conv_fwd_aot_problem_for_request writes into the implicit-GEMM fields.
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

KernelDefinition stdKernel(const ConvShape& shape, int64_t blockW = 4, int64_t blockWaves = 1)
{
    return directKernel(shape, "std", blockW, blockWaves);
}

// ---- hipkernel:Gfx950ConvFwdAot: shape-generic AOT instances ----

constexpr PackSymbols GFX950_CONV_FWD_AOT{"hipkernel:Gfx950ConvFwdAot",
                                          "hipkernel.gfx950_conv_fwd_aot.graph_match",
                                          "",
                                          "hipkernel.gfx950_conv_fwd_aot.kernel_match",
                                          "hipkernel.gfx950_conv_fwd_aot.score",
                                          "hipkernel.gfx950_conv_fwd_aot.dispatch",
                                          "gfx950_conv_fwd.x.uid",
                                          "gfx950_conv_fwd.w.uid",
                                          "gfx950_conv_fwd.y.uid"};

struct AotWidths
{
    int64_t a = 8;
    int64_t b = 8;
    int64_t c = 8;
};

/// An instance as make_conv_aot_config.py emits it: tuning and capability fields only,
/// no problem extents.
KernelDefinition aotKernel(AotWidths widths = {},
                           int64_t grouped = 1,
                           const std::string& dtype = "fp16",
                           int64_t tileM = 64,
                           int64_t tileN = 64)
{
    KernelDefinition kernel;
    kernel.name = "aot_instance";
    kernel.source.kind = KernelSourceKind::KPACK;
    kernel.metadata = {{"dtype", dtype},
                       {"layout", std::string("NHWC")},
                       {"tile_m", tileM},
                       {"tile_n", tileN},
                       {"tile_k", int64_t{64}},
                       {"warp_m", int64_t{2}},
                       {"warp_n", int64_t{2}},
                       {"warp_tile_m", int64_t{32}},
                       {"warp_tile_n", int64_t{32}},
                       {"warp_tile_k", int64_t{16}},
                       {"wave_size", int64_t{64}},
                       {"pipeline", std::string("mem")},
                       {"epilogue", std::string("cshuffle")},
                       {"vector_size_a", widths.a},
                       {"vector_size_b", widths.b},
                       {"vector_size_c", widths.c},
                       {"unroll_k", int64_t{0}},
                       {"async_dma", int64_t{0}},
                       {"async_chunk_a", int64_t{0}},
                       {"async_chunk_b", int64_t{0}},
                       {"grouped", grouped},
                       {"instance_id", std::string(40, 'a')},
                       // The descriptor loader completes these from their KMD defaults.
                       {"kernel_family", int64_t{0}},
                       {"direct_variant", std::string("none")},
                       {"filter_h", int64_t{0}},
                       {"filter_w", int64_t{0}},
                       {"stride_h", int64_t{0}},
                       {"stride_w", int64_t{0}},
                       {"pad_h", int64_t{0}},
                       {"pad_w", int64_t{0}},
                       {"dilation_h", int64_t{0}},
                       {"dilation_w", int64_t{0}},
                       {"block_w", int64_t{0}},
                       {"block_waves", int64_t{0}},
                       {"block_h", int64_t{0}},
                       {"unroll_rows", int64_t{0}},
                       {"block_q", int64_t{0}},
                       {"block_groups", int64_t{0}},
                       {"double_buffer", int64_t{0}},
                       {"cpg", int64_t{0}},
                       {"kpg", int64_t{0}},
                       {"tile_h", int64_t{0}},
                       {"tile_w", int64_t{0}},
                       {"tile_k_out", int64_t{0}},
                       {"ck", int64_t{0}},
                       {"waves_m", int64_t{0}},
                       {"waves_n", int64_t{0}},
                       {"atom", std::string("none")},
                       {"swizzle_wgm", int64_t{0}},
                       {"iglp", int64_t{-1}},
                       {"waves_per_eu", int64_t{0}},
                       {"group_merge", int64_t{1}},
                       {"split_c", int64_t{1}},
                       {"prefetch_rows", int64_t{0}},
                       {"w_fold", int64_t{0}}};
    return kernel;
}

/// A direct depthwise instance as make_conv_aot_config.py emits it: the identity's
/// placeholders in the implicit-GEMM fields, then what the binary compiles in. A
/// "same"-padded square filter, as the direct sweep builds.
KernelDefinition aotDirectKernel(const std::string& variant,
                                 int64_t filter = 3,
                                 int64_t stride = 1,
                                 int64_t blockW = 4,
                                 int64_t blockWaves = 1,
                                 const std::string& dtype = "fp16")
{
    auto kernel = aotKernel({0, 0, 0}, 0, dtype, 0, 0);
    for(const auto* field :
        {"tile_k", "warp_m", "warp_n", "warp_tile_m", "warp_tile_n", "warp_tile_k"})
    {
        kernel.metadata[field] = int64_t{0};
    }
    kernel.metadata["pipeline"] = std::string("direct");
    kernel.metadata["epilogue"] = std::string("direct");
    kernel.metadata["kernel_family"] = int64_t{1};
    kernel.metadata["direct_variant"] = variant;
    kernel.metadata["filter_h"] = kernel.metadata["filter_w"] = filter;
    kernel.metadata["stride_h"] = kernel.metadata["stride_w"] = stride;
    kernel.metadata["pad_h"] = kernel.metadata["pad_w"] = (filter - 1) / 2;
    kernel.metadata["dilation_h"] = kernel.metadata["dilation_w"] = int64_t{1};
    kernel.metadata["block_w"] = variant == "spatial" ? int64_t{0} : blockW;
    kernel.metadata["block_waves"] = blockWaves;
    return kernel;
}

/// A tiled, col, grouped or non-grouped instance as make_conv_aot_config.py emits it:
/// aotDirectKernel's placeholders and compiled values, block_w and block_waves cleared,
/// then @p knobs. The values are knobs of rocKE's direct sweep for the variant.
KernelDefinition aotVariantKernel(const std::string& variant,
                                  int64_t filter,
                                  int64_t stride,
                                  int64_t pad,
                                  const std::vector<std::pair<const char*, MetadataValue>>& knobs)
{
    auto kernel = aotDirectKernel(variant, filter, stride, 0, 0);
    kernel.metadata["pad_h"] = kernel.metadata["pad_w"] = pad;
    for(const auto& [field, value] : knobs)
    {
        kernel.metadata[field] = value;
    }
    return kernel;
}

KernelDefinition aotTiledKernel(int64_t filter = 3, int64_t stride = 1)
{
    return aotVariantKernel("tiled",
                            filter,
                            stride,
                            (filter - 1) / 2,
                            {{"block_w", int64_t{7}},
                             {"block_h", int64_t{2}},
                             {"block_waves", int64_t{1}},
                             {"unroll_rows", int64_t{1}}});
}

KernelDefinition aotColKernel(int64_t filter = 5, int64_t stride = 2, int64_t pad = 0)
{
    return aotVariantKernel(
        "col",
        filter,
        stride,
        pad,
        {{"block_w", int64_t{2}}, {"block_h", int64_t{8}}, {"block_waves", int64_t{1}}});
}

KernelDefinition aotGroupedKernel(int64_t cpg = 8)
{
    return aotVariantKernel("grouped",
                            3,
                            1,
                            1,
                            {{"block_h", int64_t{16}},
                             {"block_q", int64_t{16}},
                             {"block_groups", int64_t{4}},
                             {"double_buffer", int64_t{1}},
                             {"cpg", cpg},
                             {"kpg", cpg}});
}

KernelDefinition aotNongroupedKernel(int64_t stride = 1)
{
    return aotVariantKernel("nongrouped",
                            3,
                            stride,
                            1,
                            {{"tile_h", int64_t{8}},
                             {"tile_w", int64_t{32}},
                             {"tile_k_out", int64_t{64}},
                             {"ck", int64_t{16}},
                             {"waves_m", int64_t{2}},
                             {"waves_n", int64_t{2}},
                             {"atom", std::string("32x32x16")},
                             {"swizzle_wgm", int64_t{8}},
                             {"iglp", int64_t{0}},
                             {"waves_per_eu", int64_t{0}}});
}

ConvShape denseShape(int64_t c, int64_t k, int64_t h, int64_t w, int64_t stride = 1)
{
    ConvShape shape;
    shape.groups = 1;
    shape.c = c;
    shape.k = k;
    shape.h = h;
    shape.w = w;
    shape.sH = shape.sW = stride;
    return shape;
}

ConvShape groupedShape(int64_t groups, int64_t cpg)
{
    ConvShape shape;
    shape.groups = groups;
    shape.c = shape.k = groups * cpg;
    return shape;
}

TEST(TestGfx950ConvFwdAotKernelMatcher, OneInstanceServesEveryShapeItsCapabilitiesAdmit)
{
    // No geometry field exists to compare: the same descriptor claims every shape.
    const auto instance = aotKernel();
    // ConvShape fields: n, c, k, groups, h, w, y, x, sH, sW, pH, pW, dH, dW.
    const std::vector<ConvShape> shapes{{2, 32, 32, 1, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1},
                                        {1, 32, 64, 1, 8, 8, 1, 1, 1, 1, 0, 0, 1, 1},
                                        {2, 32, 64, 1, 17, 19, 3, 3, 2, 2, 1, 1, 1, 1},
                                        {1, 32, 32, 1, 15, 17, 3, 3, 1, 1, 2, 2, 2, 2},
                                        {2, 32, 64, 1, 11, 17, 3, 5, 1, 2, 1, 2, 1, 1},
                                        {4, 64, 128, 1, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1},
                                        {2, 64, 64, 2, 13, 13, 3, 3, 1, 1, 1, 1, 1, 1}};
    for(size_t i = 0; i < shapes.size(); ++i)
    {
        SCOPED_TRACE(i);
        const GraphFixture fixture(buildGraph(convSpec(shapes[i])), gfx950Properties());
        ASSERT_TRUE(matchesGraph(GFX950_CONV_FWD_AOT, fixture.context()));
        EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), instance));
    }
}

TEST(TestGfx950ConvFwdAotKernelMatcher, BakedWidthsMustDivideThePerGroupChannelRuns)
{
    // A and B run along C / groups, D along K / groups.
    ConvShape shape;
    shape.groups = 1;
    shape.c = 12;
    shape.k = 18;
    const GraphFixture fixture(buildGraph(convSpec(shape)), gfx950Properties());
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), aotKernel({8, 8, 2})));
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), aotKernel({4, 4, 4})));
    EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), aotKernel({4, 2, 2})));
    EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), aotKernel({4, 4, 2})));
    EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), aotKernel({1, 1, 1})));

    // Depthwise: one channel per group admits only scalar accesses.
    const GraphFixture depthwise(buildGraph(convSpec(depthwiseShape(32, 14, 14, 3, 1, 1))),
                                 gfx950Properties());
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, depthwise.context(), aotKernel({2, 2, 1})));
    EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, depthwise.context(), aotKernel({1, 1, 1})));
}

TEST(TestGfx950ConvFwdAotKernelMatcher, AGroupedGraphNeedsAGroupedBinary)
{
    ConvShape grouped;
    grouped.c = grouped.k = 64;
    grouped.groups = 2;
    const GraphFixture fixture(buildGraph(convSpec(grouped)), gfx950Properties());
    EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), aotKernel({}, 1)));
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), aotKernel({}, 0)));

    ConvShape dense;
    dense.groups = 1;
    const GraphFixture denseFixture(buildGraph(convSpec(dense)), gfx950Properties());
    EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, denseFixture.context(), aotKernel({}, 0)));
    EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, denseFixture.context(), aotKernel({}, 1)));
}

TEST(TestGfx950ConvFwdAotGraphMatcher, GroupedPointwiseIsServedByGroupedInstances)
{
    ConvShape shape;
    shape.groups = 4;
    shape.y = shape.x = 1;
    shape.pH = shape.pW = 0;
    const GraphFixture fixture(buildGraph(convSpec(shape)), gfx950Properties());
    ASSERT_TRUE(matchesGraph(GFX950_CONV_FWD_AOT, fixture.context()));
    // An instance is built against a 3x3 probe, so it is never the pointwise GEMM.
    EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), aotKernel()));
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), aotKernel({}, 0)));
}

TEST(TestGfx950ConvFwdAotKernelMatcher, AsyncChunksMustDivideThePerGroupInputRun)
{
    ConvShape shape;
    shape.groups = 1;
    shape.c = 24;
    const GraphFixture fixture(buildGraph(convSpec(shape)), gfx950Properties());
    auto kernel = aotKernel();
    kernel.metadata["async_dma"] = int64_t{1};
    kernel.metadata["async_chunk_a"] = int64_t{8};
    kernel.metadata["async_chunk_b"] = int64_t{8};
    EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), kernel));
    kernel.metadata["async_chunk_b"] = int64_t{16};
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), kernel));
    // An async binary without recorded chunks cannot be checked, so it is refused.
    kernel.metadata["async_chunk_a"] = kernel.metadata["async_chunk_b"] = int64_t{0};
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), kernel));
}

TEST(TestGfx950ConvFwdAotKernelMatcher, RefusesOutOfContractOrIncompleteMetadata)
{
    const GraphFixture fixture(buildGraph(), gfx950Properties());
    ASSERT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), aotKernel()));
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), aotKernel({}, 1, "bf16")));

    const std::vector<std::pair<const char*, MetadataValue>> outOfContract{
        {"pipeline", std::string("wavelet")},
        {"epilogue", std::string("none")},
        {"layout", std::string("NCHW")},
        {"wave_size", int64_t{32}},
        {"unroll_k", int64_t{2}},
        {"grouped", int64_t{-1}},
        {"vector_size_c", int64_t{0}},
        {"tile_m", int64_t{0}},
        {"async_chunk_a", int64_t{-1}}};
    for(const auto& [field, value] : outOfContract)
    {
        SCOPED_TRACE(field);
        auto kernel = aotKernel();
        kernel.metadata[field] = value;
        EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), kernel));
    }
    for(const auto& field : aotKernel().metadata)
    {
        if(field.first == "instance_id")
        {
            continue; // Tracking only; the launch never reads it.
        }
        SCOPED_TRACE(field.first);
        auto kernel = aotKernel();
        kernel.metadata.erase(field.first);
        EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), kernel));
    }
    auto embedded = aotKernel();
    embedded.source.kind = KernelSourceKind::EMBEDDED_SOURCE;
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), embedded));
}

TEST(TestGfx950ConvFwdAotScore, PrefersLessTilePaddingAndRanksRefusalsLast)
{
    // K = 32: a 32-wide N tile launches no padding columns, a 128-wide one 96 of them.
    const GraphFixture fixture(buildGraph(), gfx950Properties());
    const auto snug
        = scoreKernel(GFX950_CONV_FWD_AOT, fixture.context(), aotKernel({}, 1, "fp16", 64, 32));
    const auto padded
        = scoreKernel(GFX950_CONV_FWD_AOT, fixture.context(), aotKernel({}, 1, "fp16", 64, 128));
    EXPECT_GT(snug, padded);
    EXPECT_GT(padded, 0.0);
    EXPECT_EQ(scoreKernel(GFX950_CONV_FWD_AOT, fixture.context(), aotKernel({}, 1, "fp16", 64, 32)),
              snug);

    ConvShape shape;
    shape.groups = 1;
    shape.c = 12;
    const GraphFixture refused(buildGraph(convSpec(shape)), gfx950Properties());
    EXPECT_LT(scoreKernel(GFX950_CONV_FWD_AOT, refused.context(), aotKernel()), 0.0);
}

TEST(TestGfx950ConvFwdAotDirectKernelMatcher, ServesAnyImageBatchAndGroupCountWithItsFilter)
{
    // N, Hi, Wi and the group count are kernel arguments; one binary serves them all.
    const auto spatial = aotDirectKernel("spatial");
    const auto std3 = aotDirectKernel("std");
    for(const auto& shape : {depthwiseShape(32, 14, 14, 3, 1, 1),
                             depthwiseShape(5, 9, 21, 3, 1, 1),
                             depthwiseShape(63, 7, 7, 3, 1, 1),
                             depthwiseShape(8, 64, 64, 3, 1, 1)})
    {
        SCOPED_TRACE(shape.groups);
        auto batched = shape;
        batched.n = 3;
        const GraphFixture fixture(buildGraph(convSpec(batched)), gfx950Properties());
        EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), spatial));
        EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), std3));
        // This engine does not admit the other family's exact-shape descriptors.
        EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), stdKernel(shape)));
    }
    // Spatial serves G < 64 only; std serves either side of it.
    for(const auto groups : {int64_t{64}, int64_t{96}, int64_t{256}})
    {
        SCOPED_TRACE(groups);
        const GraphFixture fixture(buildGraph(convSpec(depthwiseShape(groups, 14, 14, 3, 1, 1))),
                                   gfx950Properties());
        EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), spatial));
        EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), std3));
    }
    // The filter, stride and padding are compiled in.
    const GraphFixture strided(buildGraph(convSpec(depthwiseShape(96, 17, 17, 3, 2, 1))),
                               gfx950Properties());
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, strided.context(), std3));
    EXPECT_TRUE(
        matchesKernel(GFX950_CONV_FWD_AOT, strided.context(), aotDirectKernel("std", 3, 2)));
    const GraphFixture wide(buildGraph(convSpec(depthwiseShape(96, 21, 15, 7, 2, 3))),
                            gfx950Properties());
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, wide.context(), aotDirectKernel("std", 5, 2)));
    EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, wide.context(), aotDirectKernel("std", 7, 2)));
    auto padded = aotDirectKernel("std", 7, 2);
    padded.metadata["pad_h"] = padded.metadata["pad_w"] = int64_t{2};
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, wide.context(), padded));
    // Only the dtype it was built for.
    EXPECT_FALSE(matchesKernel(
        GFX950_CONV_FWD_AOT, wide.context(), aotDirectKernel("std", 7, 2, 4, 1, "bf16")));
}

TEST(TestGfx950ConvFwdAotDirectKernelMatcher, DeclinesEveryGraphOutsideTheGuardedContract)
{
    // The direct guard: each graph is served by an implicit-GEMM instance, while a direct
    // instance compiled for its filter, stride and padding is not.
    const auto base = depthwiseShape(32, 14, 14, 3, 1, 1);
    const std::vector<std::pair<const char*, std::function<void(ConvShape&)>>> cases{
        {"sH != sW",
         [](auto& s) {
             s.w = 15;
             s.sW = 2;
         }},
        {"dilation 2",
         [](auto& s) {
             s.dH = s.dW = 2;
             s.pH = s.pW = 2;
         }},
        {"channel multiplier", [](auto& s) { s.k = 64; }},
        {"grouped, not depthwise", [](auto& s) { s.groups = 4; }},
        {"dense", [](auto& s) { s.groups = 1; }}};
    for(const auto& [name, change] : cases)
    {
        SCOPED_TRACE(name);
        auto shape = base;
        change(shape);
        const GraphFixture fixture(buildGraph(convSpec(shape)), gfx950Properties());
        ASSERT_TRUE(matchesGraph(GFX950_CONV_FWD_AOT, fixture.context()));
        EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), aotKernel({1, 1, 1})));
        for(const auto& kernel : {aotDirectKernel("spatial"), aotDirectKernel("std")})
        {
            EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), kernel));
        }
    }
    // z carries N for both direct variants.
    for(const auto n : {int64_t{65535}, int64_t{65536}})
    {
        SCOPED_TRACE(n);
        auto shape = depthwiseShape(2, 3, 3, 3, 1, 1);
        shape.n = n;
        const GraphFixture fixture(buildGraph(convSpec(shape)), gfx950Properties());
        EXPECT_EQ(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), aotDirectKernel("spatial")),
                  n == 65535);
    }
}

TEST(TestGfx950ConvFwdAotDirectKernelMatcher, FamilyFieldsAndPlaceholdersMustAgree)
{
    const auto shape = depthwiseShape(32, 14, 14, 3, 1, 1);
    const GraphFixture fixture(buildGraph(convSpec(shape)), gfx950Properties());
    const auto implicit = aotKernel({1, 1, 1});
    ASSERT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), implicit));
    for(const auto& base : {aotDirectKernel("spatial"), aotDirectKernel("std")})
    {
        ASSERT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), base));
        // A direct instance holds the direct identity's placeholders exactly.
        const std::vector<std::pair<const char*, MetadataValue>> violations{
            {"tile_m", int64_t{64}},
            {"tile_k", int64_t{64}},
            {"vector_size_a", int64_t{1}},
            {"async_chunk_b", int64_t{8}},
            {"grouped", int64_t{1}},
            {"unroll_k", int64_t{1}},
            {"pipeline", std::string("none")},
            {"epilogue", std::string("cshuffle")},
            {"wave_size", int64_t{32}},
            {"layout", std::string("NCHW")},
            {"kernel_family", int64_t{2}},
            {"kernel_family", int64_t{0}},
            {"kernel_family", std::string("1")},
            {"direct_variant", std::string("none")},
            {"direct_variant", std::string("STD")},
            {"dilation_h", int64_t{2}},
            {"block_waves", int64_t{0}},
            {"block_waves", int64_t{17}}};
        for(const auto& [field, value] : violations)
        {
            SCOPED_TRACE(field);
            auto kernel = base;
            kernel.metadata[field] = value;
            EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), kernel));
        }
        for(const auto& field : base.metadata)
        {
            if(field.first == "instance_id")
            {
                continue; // Tracking only; the launch never reads it.
            }
            SCOPED_TRACE(field.first);
            auto kernel = base;
            kernel.metadata.erase(field.first);
            EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), kernel));
        }
    }
    // std needs a positive block width; spatial derives its own and must carry 0.
    auto spatialWidth = aotDirectKernel("spatial");
    spatialWidth.metadata["block_w"] = int64_t{4};
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), spatialWidth));
    EXPECT_FALSE(
        matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), aotDirectKernel("std", 3, 1, 0)));

    // An implicit-GEMM instance leaves every direct field at its KMD default.
    for(const auto* field :
        {"filter_h", "stride_w", "pad_h", "dilation_w", "block_w", "block_waves"})
    {
        SCOPED_TRACE(field);
        auto kernel = implicit;
        kernel.metadata[field] = int64_t{1};
        EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), kernel));
    }
    auto variant = implicit;
    variant.metadata["direct_variant"] = std::string("std");
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), variant));
    // Nor can an implicit-GEMM tuning pose as family 1.
    auto relabeled = aotDirectKernel("std");
    for(const auto* field : {"tile_m",
                             "tile_n",
                             "tile_k",
                             "warp_m",
                             "warp_n",
                             "warp_tile_m",
                             "warp_tile_n",
                             "warp_tile_k",
                             "vector_size_a",
                             "vector_size_b",
                             "vector_size_c"})
    {
        relabeled.metadata[field] = implicit.metadata.at(field);
    }
    relabeled.metadata["pipeline"] = std::string("mem");
    relabeled.metadata["epilogue"] = std::string("cshuffle");
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), relabeled));
}

TEST(TestGfx950ConvFwdAotScore, DirectInstancesRankFirstAndTheGraphsDefaultArmHighest)
{
    // G < 64: spatial with one wave is the default arm; G >= 64: std block_w 4, one wave.
    // An instance has no build group count, so the graph's decides.
    const auto scoreOf = [](const GraphFixture& fixture, const KernelDefinition& kernel) {
        EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), kernel));
        return scoreKernel(GFX950_CONV_FWD_AOT, fixture.context(), kernel);
    };
    const GraphFixture small(buildGraph(convSpec(depthwiseShape(32, 14, 14, 3, 1, 1))),
                             gfx950Properties());
    const auto smallDefault = scoreOf(small, aotDirectKernel("spatial"));
    const auto implicitSmall = scoreOf(small, aotKernel({1, 1, 1}));
    for(const auto& other : {aotDirectKernel("spatial", 3, 1, 0, 2),
                             aotDirectKernel("std"),
                             aotDirectKernel("std", 3, 1, 16, 4)})
    {
        const auto otherScore = scoreOf(small, other);
        EXPECT_GT(smallDefault, otherScore);
        EXPECT_GT(otherScore, implicitSmall);
    }

    const GraphFixture large(buildGraph(convSpec(depthwiseShape(96, 14, 14, 3, 1, 1))),
                             gfx950Properties());
    const auto largeDefault = scoreOf(large, aotDirectKernel("std"));
    for(const auto& other :
        {aotDirectKernel("std", 3, 1, 16, 1), aotDirectKernel("std", 3, 1, 4, 2)})
    {
        const auto otherScore = scoreOf(large, other);
        EXPECT_GT(largeDefault, otherScore);
        EXPECT_GT(otherScore, scoreOf(large, aotKernel({1, 1, 1})));
    }
    // Refused graphs rank last.
    const GraphFixture dense(buildGraph(), gfx950Properties());
    EXPECT_LT(scoreKernel(GFX950_CONV_FWD_AOT, dense.context(), aotDirectKernel("std")), 0.0);
}

TEST(TestGfx950ConvFwdAotDispatch, DirectInstancePassesTheContractAndReachesCodeLoading)
{
    const auto shape = depthwiseShape(96, 14, 14, 3, 1, 1);
    const GraphFixture fixture(buildGraph(convSpec(shape)), gfx950Properties());
    const auto bound = matchesGraph(GFX950_CONV_FWD_AOT, fixture.context());
    ASSERT_TRUE(bound);
    const auto& handler = dispatchHandler(GFX950_CONV_FWD_AOT);
    EXPECT_EQ(handler.workspaceBytes(fixture.context(), *bound, aotDirectKernel("std")), 0U);
    try
    {
        // The descriptor names no archive, so loading is the first step that can fail.
        handler.prepare(fixture.context(), *bound, aotDirectKernel("std"));
        FAIL() << "Preparation loaded a kernel from an empty archive path";
    }
    catch(const std::exception& error)
    {
        const std::string message = error.what();
        EXPECT_EQ(message.find("convolution contract"), std::string::npos) << message;
        EXPECT_EQ(message.find("packed NHWK output"), std::string::npos) << message;
    }
    // A spatial instance cannot serve 96 groups, so it is refused before any code loads.
    EXPECT_THROW(handler.prepare(fixture.context(), *bound, aotDirectKernel("spatial")),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestGfx950ConvFwdAotDirectKernelMatcher, EachDirectVariantServesItsOwnProblemClass)
{
    const auto matches = [](const ConvShape& shape, const KernelDefinition& kernel) {
        const GraphFixture fixture(buildGraph(convSpec(shape)), gfx950Properties());
        return matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), kernel);
    };
    // tiled: depthwise with its compiled filter, here 15x15, which std cannot unroll.
    EXPECT_TRUE(matches(depthwiseShape(96, 17, 19, 15, 1, 7), aotTiledKernel(15)));
    EXPECT_TRUE(matches(depthwiseShape(5, 33, 7, 3, 1, 1), aotTiledKernel()));
    EXPECT_FALSE(matches(depthwiseShape(96, 17, 19, 13, 1, 6), aotTiledKernel(15)));
    // col: "valid" padding too; a block no wider than the output.
    EXPECT_TRUE(matches(depthwiseShape(24, 13, 17, 5, 2, 0), aotColKernel()));
    EXPECT_FALSE(matches(depthwiseShape(24, 13, 5, 5, 2, 0), aotColKernel()));
    EXPECT_FALSE(matches(depthwiseShape(24, 13, 17, 5, 2, 2), aotColKernel()));
    // grouped: its compiled channels per group, block_groups dividing the groups.
    EXPECT_TRUE(matches(groupedShape(8, 8), aotGroupedKernel(8)));
    EXPECT_FALSE(matches(groupedShape(8, 8), aotGroupedKernel(4)));
    EXPECT_FALSE(matches(groupedShape(6, 8), aotGroupedKernel(8)));
    // nongrouped: a dense graph with C a multiple of 16 and 8, K of 32, and Wo = 96.
    EXPECT_TRUE(matches(denseShape(64, 96, 10, 96), aotNongroupedKernel()));
    EXPECT_TRUE(matches(denseShape(16, 32, 3, 32), aotNongroupedKernel()));
    EXPECT_TRUE(matches(denseShape(64, 96, 10, 191, 2), aotNongroupedKernel(2)));
    EXPECT_FALSE(matches(denseShape(24, 96, 10, 96), aotNongroupedKernel()));
    EXPECT_FALSE(matches(denseShape(64, 80, 10, 96), aotNongroupedKernel()));
    EXPECT_FALSE(matches(denseShape(64, 96, 10, 96, 2), aotNongroupedKernel()));
    auto groupedDense = denseShape(64, 96, 10, 96);
    groupedDense.groups = 2;
    EXPECT_FALSE(matches(groupedDense, aotNongroupedKernel()));
    // A depthwise graph is no grouped or non-grouped kernel's.
    const GraphFixture depthwise(buildGraph(convSpec(depthwiseShape(32, 14, 14, 3, 1, 1))),
                                 gfx950Properties());
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, depthwise.context(), aotGroupedKernel()));
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, depthwise.context(), aotNongroupedKernel()));
    for(const auto& kernel : {aotTiledKernel(), aotColKernel(3, 1, 1)})
    {
        EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, depthwise.context(), kernel));
    }
}

TEST(TestGfx950ConvFwdAotDirectKernelMatcher, BatchesPastTheMaskedAccessOffsetRunInPieces)
{
    // An image is 2 * 1024 * 96 * 256 bytes in and out, so 42 fit below the masked-access
    // offset and 85 run as 29, 28 and 28: whole, the activation passes the i32 ABI.
    auto big = denseShape(256, 256, 1024, 96);
    big.n = 85;
    const GraphFixture fixture(buildGraph(convSpec(big)), gfx950Properties());
    const auto bound = matchesGraph(GFX950_CONV_FWD_AOT, fixture.context());
    ASSERT_TRUE(bound);
    EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), aotNongroupedKernel()));
    // Only a non-grouped or grouped-halo kernel launches per chunk, and not split-C, whose
    // workspace would span the whole batch.
    auto split = aotNongroupedKernel();
    split.metadata["split_c"] = int64_t{2};
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), split));
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), aotKernel()));
    // 42 images launch once, and so does implicit GEMM.
    auto one = big;
    one.n = 42;
    const GraphFixture single(buildGraph(convSpec(one)), gfx950Properties());
    EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, single.context(), aotNongroupedKernel()));
    EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, single.context(), aotKernel()));

    const auto& handler = dispatchHandler(GFX950_CONV_FWD_AOT);
    EXPECT_EQ(handler.workspaceBytes(fixture.context(), *bound, aotNongroupedKernel()), 0U);
    try
    {
        // The descriptor names no archive, so loading is the first step that can fail.
        handler.prepare(fixture.context(), *bound, aotNongroupedKernel());
        FAIL() << "Preparation loaded a kernel from an empty archive path";
    }
    catch(const std::exception& error)
    {
        const std::string message = error.what();
        EXPECT_EQ(message.find("convolution contract"), std::string::npos) << message;
    }
}

TEST(TestGfx950ConvFwdAotDirectKernelMatcher, EachVariantStatesOnlyItsOwnKnobs)
{
    // Every block knob, its KMD default and a value no variant here holds.
    const std::vector<std::tuple<const char*, MetadataValue, MetadataValue>> knobs{
        {"block_w", int64_t{0}, int64_t{5}},
        {"block_waves", int64_t{0}, int64_t{3}},
        {"block_h", int64_t{0}, int64_t{5}},
        {"unroll_rows", int64_t{0}, int64_t{1}},
        {"block_q", int64_t{0}, int64_t{48}},
        {"block_groups", int64_t{0}, int64_t{2}},
        {"double_buffer", int64_t{0}, int64_t{1}},
        {"cpg", int64_t{0}, int64_t{16}},
        {"kpg", int64_t{0}, int64_t{16}},
        {"tile_h", int64_t{0}, int64_t{16}},
        {"tile_w", int64_t{0}, int64_t{64}},
        {"tile_k_out", int64_t{0}, int64_t{32}},
        {"ck", int64_t{0}, int64_t{32}},
        {"waves_m", int64_t{0}, int64_t{1}},
        {"waves_n", int64_t{0}, int64_t{4}},
        {"atom", std::string("none"), std::string("16x16x32")},
        {"swizzle_wgm", int64_t{0}, int64_t{4}},
        {"iglp", int64_t{-1}, int64_t{1}},
        {"waves_per_eu", int64_t{0}, int64_t{3}},
        {"group_merge", int64_t{1}, int64_t{2}},
        {"split_c", int64_t{1}, int64_t{2}}};
    // Each variant's own knobs: AOT_DIRECT_KNOBS in convolution_forward.py.
    const std::vector<std::tuple<ConvShape, KernelDefinition, std::vector<std::string>>> cases{
        {depthwiseShape(32, 14, 14, 3, 1, 1),
         aotTiledKernel(),
         {"block_w", "block_h", "block_waves", "unroll_rows"}},
        {depthwiseShape(24, 13, 17, 5, 2, 0),
         aotColKernel(),
         {"block_w", "block_h", "block_waves"}},
        {groupedShape(8, 8),
         aotGroupedKernel(),
         {"block_h", "block_q", "block_groups", "double_buffer", "cpg", "kpg"}},
        {denseShape(64, 96, 10, 96),
         aotNongroupedKernel(),
         {"tile_h",
          "tile_w",
          "tile_k_out",
          "ck",
          "waves_m",
          "waves_n",
          "atom",
          "swizzle_wgm",
          "iglp",
          "waves_per_eu",
          "split_c"}},
    };
    for(const auto& [shape, base, own] : cases)
    {
        SCOPED_TRACE(std::get<std::string>(base.metadata.at("direct_variant")));
        const GraphFixture fixture(buildGraph(convSpec(shape)), gfx950Properties());
        ASSERT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), base));
        // A knob of another variant, set, is malformed metadata.
        for(const auto& [field, unset, value] : knobs)
        {
            if(std::find(own.begin(), own.end(), field) != own.end())
            {
                continue;
            }
            EXPECT_EQ(base.metadata.at(field), unset) << field;
            SCOPED_TRACE(field);
            auto kernel = base;
            kernel.metadata[field] = value;
            EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), kernel));
        }
        // Every field is required; the loader completes omitted ones from the KMD.
        for(const auto& field : base.metadata)
        {
            if(field.first == "instance_id")
            {
                continue;
            }
            SCOPED_TRACE(field.first);
            auto kernel = base;
            kernel.metadata.erase(field.first);
            EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), kernel));
        }
        // The implicit-GEMM placeholders hold for every variant.
        auto relabeled = base;
        relabeled.metadata["tile_k"] = int64_t{64};
        EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), relabeled));
    }
    // Compile-only knobs must still be well formed.
    const GraphFixture dense(buildGraph(convSpec(denseShape(64, 96, 10, 96))), gfx950Properties());
    for(const auto& [field, value] :
        std::vector<std::pair<const char*, MetadataValue>>{{"atom", std::string("8x8x8")},
                                                           {"iglp", int64_t{-2}},
                                                           {"waves_per_eu", int64_t{-1}},
                                                           {"swizzle_wgm", int64_t{0}}})
    {
        SCOPED_TRACE(field);
        auto kernel = aotNongroupedKernel();
        kernel.metadata[field] = value;
        EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, dense.context(), kernel));
    }
    const GraphFixture depthwise(buildGraph(convSpec(depthwiseShape(32, 14, 14, 3, 1, 1))),
                                 gfx950Properties());
    auto flag = aotTiledKernel();
    flag.metadata["unroll_rows"] = int64_t{2};
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, depthwise.context(), flag));
    // An implicit-GEMM instance leaves every knob at its KMD default too.
    for(const auto& [field, unset, value] : knobs)
    {
        SCOPED_TRACE(field);
        auto kernel = aotKernel({1, 1, 1});
        ASSERT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, depthwise.context(), kernel));
        kernel.metadata[field] = value;
        EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, depthwise.context(), kernel));
    }
}

TEST(TestGfx950ConvFwdAotScore, GroupedAndNongroupedDirectRankBelowImplicitGemm)
{
    // Depthwise kernels rank above implicit GEMM; the grouped and non-grouped direct
    // kernels take a placeholder below every implicit-GEMM instance (and above a
    // refusal), so an unbenchmarked dense or grouped graph stays on implicit GEMM.
    const auto scoreOf = [](const ConvShape& shape, const KernelDefinition& kernel) {
        const GraphFixture fixture(buildGraph(convSpec(shape)), gfx950Properties());
        EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, fixture.context(), kernel));
        return scoreKernel(GFX950_CONV_FWD_AOT, fixture.context(), kernel);
    };
    const auto dense = denseShape(64, 96, 10, 96);
    const auto nongrouped = scoreOf(dense, aotNongroupedKernel());
    EXPECT_LT(nongrouped, scoreOf(dense, aotKernel({1, 1, 1})));
    EXPECT_LT(nongrouped, scoreOf(dense, aotKernel({1, 1, 1}, 1, "fp16", 256, 256)));
    EXPECT_GT(nongrouped, -1.0);
    const auto grouped = groupedShape(8, 8);
    EXPECT_LT(scoreOf(grouped, aotGroupedKernel()), scoreOf(grouped, aotKernel({1, 1, 1})));
    const auto depthwise = depthwiseShape(32, 14, 14, 3, 1, 1);
    const auto implicit = scoreOf(depthwise, aotKernel({1, 1, 1}));
    const auto tiled = scoreOf(depthwise, aotTiledKernel());
    EXPECT_GT(tiled, implicit);
    EXPECT_LT(tiled, scoreOf(depthwise, aotDirectKernel("spatial")));
    EXPECT_GT(scoreOf(depthwise, aotColKernel(3, 1, 1)), implicit);
}

TEST(TestGfx950ConvFwdAotDispatch, EveryDirectVariantPassesTheContractAndReachesCodeLoading)
{
    for(const auto& [shape, kernel] : std::vector<std::pair<ConvShape, KernelDefinition>>{
            {depthwiseShape(32, 14, 14, 3, 1, 1), aotTiledKernel()},
            {depthwiseShape(24, 13, 17, 5, 2, 0), aotColKernel()},
            {groupedShape(8, 8), aotGroupedKernel()},
            {denseShape(64, 96, 10, 96), aotNongroupedKernel()},
        })
    {
        SCOPED_TRACE(std::get<std::string>(kernel.metadata.at("direct_variant")));
        const GraphFixture fixture(buildGraph(convSpec(shape)), gfx950Properties());
        const auto bound = matchesGraph(GFX950_CONV_FWD_AOT, fixture.context());
        ASSERT_TRUE(bound);
        const auto& handler = dispatchHandler(GFX950_CONV_FWD_AOT);
        EXPECT_EQ(handler.workspaceBytes(fixture.context(), *bound, kernel), 0U);
        try
        {
            // The descriptor names no archive, so loading is the first step that can fail.
            handler.prepare(fixture.context(), *bound, kernel);
            FAIL() << "Preparation loaded a kernel from an empty archive path";
        }
        catch(const std::exception& error)
        {
            const std::string message = error.what();
            EXPECT_EQ(message.find("convolution contract"), std::string::npos) << message;
        }
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
