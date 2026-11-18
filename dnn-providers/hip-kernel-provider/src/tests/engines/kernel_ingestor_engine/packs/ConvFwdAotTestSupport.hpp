// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/// @file ConvFwdAotTestSupport.hpp
/// What the forward AOT engines' host tests (hipkernel:Gfx950ConvFwdAot and
/// hipkernel:Gfx942ConvFwdAot) build their graphs, devices and kernels from: a forward
/// problem as hipDNN graph tensors and a node, an instance as make_conv_aot_config.py
/// packages it (the kpack kernel the descriptor loader delivers), a wave64 device of any
/// architecture, and each engine's native symbols and binding tokens. The engines share
/// their hooks (Gfx950ConvFwdNative.cpp); only the device they accept and the names
/// differ. The tables the IngestorGenerator tests parse stay in each suite.
namespace hip_kernel_provider::kernel_ingestor_engine::conv_fwd_aot_test
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::DeviceProperties;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::KernelSourceKind;
using hipdnn_plugin_sdk::ingestor::MetadataValue;
using testing::PackSymbols;

/// One forward AOT engine's contract: the pack's symbols and public binding tokens, the
/// private problem token its graph match binds, and the architecture it serves.
struct FwdEngine
{
    PackSymbols pack;
    std::string_view problemToken;
    std::string_view arch;
};

inline constexpr FwdEngine GFX950_FWD{{"hipkernel:Gfx950ConvFwdAot",
                                       "hipkernel.gfx950_conv_fwd_aot.graph_match",
                                       "",
                                       "hipkernel.gfx950_conv_fwd_aot.kernel_match",
                                       "hipkernel.gfx950_conv_fwd_aot.score",
                                       "hipkernel.gfx950_conv_fwd_aot.dispatch",
                                       "gfx950_conv_fwd.x.uid",
                                       "gfx950_conv_fwd.w.uid",
                                       "gfx950_conv_fwd.y.uid"},
                                      "_gfx950_conv_fwd.problem",
                                      "gfx950"};

inline constexpr FwdEngine GFX942_FWD{{"hipkernel:Gfx942ConvFwdAot",
                                       "hipkernel.gfx942_conv_fwd_aot.graph_match",
                                       "",
                                       "hipkernel.gfx942_conv_fwd_aot.kernel_match",
                                       "hipkernel.gfx942_conv_fwd_aot.score",
                                       "hipkernel.gfx942_conv_fwd_aot.dispatch",
                                       "gfx942_conv_fwd.x.uid",
                                       "gfx942_conv_fwd.w.uid",
                                       "gfx942_conv_fwd.y.uid"},
                                      "_gfx942_conv_fwd.problem",
                                      "gfx942"};

/// A device of @p arch at @p warpSize, by value: the matcher tests run on CPU-only hosts
/// and on devices of any architecture.
inline DeviceProperties deviceProperties(std::string arch, int warpSize = 64)
{
    DeviceProperties properties;
    properties.gcnArchName = std::move(arch);
    properties.warpSize = warpSize;
    return properties;
}

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

/// @p spec as a serialized graph of @p spec.nodeCount nodes named @p nodeName.
inline flatbuffers::FlatBufferBuilder buildGraph(const GraphSpec& spec = {},
                                                 const char* nodeName = "conv_fwd_aot_test")
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
                                           nodeName,
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

inline ConvShape depthwiseShape(
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

inline ConvShape denseShape(int64_t c, int64_t k, int64_t h, int64_t w, int64_t stride = 1)
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

inline ConvShape groupedShape(int64_t groups, int64_t cpg)
{
    ConvShape shape;
    shape.groups = groups;
    shape.c = shape.k = groups * cpg;
    return shape;
}

/// @p shape as packed NHWC tensors of @p dtype.
inline GraphSpec convSpec(const ConvShape& shape,
                          data_objects::DataType dtype = data_objects::DataType::HALF)
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
    for(auto& tensor : spec.tensors)
    {
        tensor.dtype = dtype;
    }
    spec.stride = {shape.sH, shape.sW};
    spec.dilation = {shape.dH, shape.dW};
    spec.prePadding = spec.postPadding = {shape.pH, shape.pW};
    return spec;
}

/// A packed NHWC forward graph for @p p, (N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW,
/// groups), in @p dtype, its tensors uids 1 (x), 2 (w) and 3 (y).
inline flatbuffers::FlatBufferBuilder convGraph(const std::array<int64_t, 14>& p,
                                                data_objects::DataType dtype
                                                = data_objects::DataType::HALF,
                                                const char* nodeName = "conv_fwd_aot_test")
{
    const auto [n, c, k, hi, wi, y, x, sH, sW, pH, pW, dH, dW, groups] = p;
    const auto cpg = c / groups;
    const auto ho = (hi + 2 * pH - ((y - 1) * dH + 1)) / sH + 1;
    const auto wo = (wi + 2 * pW - ((x - 1) * dW + 1)) / sW + 1;
    const std::array<std::vector<int64_t>, 3> dims{
        std::vector<int64_t>{n, c, hi, wi}, {k, cpg, y, x}, {n, k, ho, wo}};
    const std::array<std::vector<int64_t>, 3> strides{
        std::vector<int64_t>{hi * wi * c, 1, wi * c, c},
        {y * x * cpg, 1, x * cpg, cpg},
        {ho * wo * k, 1, wo * k, k}};

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    for(size_t i = 0; i < dims.size(); ++i)
    {
        const auto dimsOffset = builder.CreateVector(dims[i]);
        const auto stridesOffset = builder.CreateVector(strides[i]);
        data_objects::TensorAttributesBuilder tensor(builder);
        tensor.add_uid(static_cast<int64_t>(i + 1));
        tensor.add_data_type(dtype);
        tensor.add_dims(dimsOffset);
        tensor.add_strides(stridesOffset);
        tensor.add_alignment(16);
        tensors.push_back(tensor.Finish());
    }
    const std::vector<int64_t> stride{sH, sW};
    const std::vector<int64_t> dilation{dH, dW};
    const std::vector<int64_t> padding{pH, pW};
    const auto attributes = data_objects::CreateConvolutionFwdAttributesDirect(
        builder,
        1,
        2,
        3,
        &padding,
        &padding,
        &stride,
        &dilation,
        data_objects::ConvMode::CROSS_CORRELATION);
    const std::vector<flatbuffers::Offset<data_objects::Node>> nodes{
        data_objects::CreateNodeDirect(builder,
                                       nodeName,
                                       data_objects::DataType::FLOAT,
                                       data_objects::NodeAttributes::ConvolutionFwdAttributes,
                                       attributes.Union())};
    const auto tensorVector = builder.CreateVector(tensors);
    const auto nodeVector = builder.CreateVector(nodes);
    data_objects::GraphBuilder graph(builder);
    graph.add_tensors(tensorVector);
    graph.add_nodes(nodeVector);
    builder.Finish(graph.Finish());
    return builder;
}

/// The graph data type of a catalog dtype name ("fp16" or "bf16").
inline data_objects::DataType dataTypeOf(std::string_view dtype)
{
    return dtype == "bf16" ? data_objects::DataType::BFLOAT16 : data_objects::DataType::HALF;
}

struct AotWidths
{
    int64_t a = 8;
    int64_t b = 8;
    int64_t c = 8;
};

/// An instance as make_conv_aot_config.py emits it: tuning and capability fields only,
/// no problem extents.
inline KernelDefinition aotKernel(AotWidths widths = {},
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
inline KernelDefinition aotDirectKernel(const std::string& variant,
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
inline KernelDefinition
    aotVariantKernel(const std::string& variant,
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

inline KernelDefinition aotTiledKernel(int64_t filter = 3, int64_t stride = 1)
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

inline KernelDefinition aotColKernel(int64_t filter = 5, int64_t stride = 2, int64_t pad = 0)
{
    return aotVariantKernel(
        "col",
        filter,
        stride,
        pad,
        {{"block_w", int64_t{2}}, {"block_h", int64_t{8}}, {"block_waves", int64_t{1}}});
}

inline KernelDefinition aotGroupedKernel(int64_t cpg = 8)
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

/// A non-grouped instance; @p atom is its MFMA atom (gfx950's 32x32x16 by default; gfx942
/// builds 32x32x8 and 16x16x16).
inline KernelDefinition aotNongroupedKernel(int64_t stride = 1,
                                            const std::string& atom = "32x32x16")
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
                             {"atom", atom},
                             {"swizzle_wgm", int64_t{8}},
                             {"iglp", int64_t{0}},
                             {"waves_per_eu", int64_t{0}}});
}

/// One implicit-GEMM forward identity of rocKE's AOT sweep (fp16, grouped), as
/// make_conv_aot_config.py packages it. The fields the kernel matcher and the candidate
/// filter read, in this order: epilogue, vector_size_a/b/c, tile_m/n/k, warp_m/n,
/// warp_tile_m/n/k, pipeline, unroll_k, async_dma, async_chunk_a/b.
struct AotSample
{
    const char* epilogue;
    std::array<int64_t, 11> values;
    const char* pipeline;
    std::array<int64_t, 4> loop;
};

/// @p sample as its kpack kernel, every KMD field present.
inline KernelDefinition sampleKernel(const AotSample& sample)
{
    const auto& v = sample.values;
    auto kernel = aotKernel({v[0], v[1], v[2]}, 1, "fp16", v[3], v[4]);
    for(const auto& [field, value] :
        std::vector<std::pair<const char*, int64_t>>{{"tile_k", v[5]},
                                                     {"warp_m", v[6]},
                                                     {"warp_n", v[7]},
                                                     {"warp_tile_m", v[8]},
                                                     {"warp_tile_n", v[9]},
                                                     {"warp_tile_k", v[10]},
                                                     {"unroll_k", sample.loop[0]},
                                                     {"async_dma", sample.loop[1]},
                                                     {"async_chunk_a", sample.loop[2]},
                                                     {"async_chunk_b", sample.loop[3]}})
    {
        kernel.metadata[field] = value;
    }
    kernel.metadata["pipeline"] = std::string(sample.pipeline);
    kernel.metadata["epilogue"] = std::string(sample.epilogue);
    return kernel;
}

/// A direct std depthwise instance compiled for a 3x3, stride 1, "same"-padded filter.
inline KernelDefinition directSampleKernel()
{
    auto kernel
        = sampleKernel({"direct", {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, "direct", {0, 0, 0, 0}});
    kernel.metadata["grouped"] = int64_t{0};
    kernel.metadata["kernel_family"] = int64_t{1};
    kernel.metadata["direct_variant"] = std::string("std");
    kernel.metadata["filter_h"] = kernel.metadata["filter_w"] = int64_t{3};
    kernel.metadata["stride_h"] = kernel.metadata["stride_w"] = int64_t{1};
    kernel.metadata["pad_h"] = kernel.metadata["pad_w"] = int64_t{1};
    kernel.metadata["dilation_h"] = kernel.metadata["dilation_w"] = int64_t{1};
    kernel.metadata["block_w"] = int64_t{4};
    kernel.metadata["block_waves"] = int64_t{1};
    return kernel;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_fwd_aot_test

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
