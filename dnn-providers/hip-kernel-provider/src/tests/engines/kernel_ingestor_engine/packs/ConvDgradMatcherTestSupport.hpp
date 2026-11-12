// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <ostream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>

#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ConvDgradTestCatalog.hpp"

/// @file ConvDgradMatcherTestSupport.hpp
/// What the dgrad AOT engines' matcher tests build their graphs and kernels from: a
/// backward-data problem as hipDNN graph tensors and a node, and a catalog instance as
/// the kpack kernel the descriptor loader delivers. Shared by the gfx950, gfx942 and
/// gfx1250 suites; the arch-specific device properties and symbols stay in each.
namespace hip_kernel_provider::kernel_ingestor_engine::dgrad_matcher_test
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace catalog = dgrad_test_catalog;
using hipdnn_plugin_sdk::ingestor::DescriptorId;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::KernelSourceKind;

struct TensorSpec
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    data_objects::DataType dtype = data_objects::DataType::HALF;
    bool isVirtual = false;
    bool omitDims = false;
    int64_t alignment = 16;
};

/// dims [N, C, H, W] with packed NHWC strides.
inline TensorSpec nhwc(int64_t n, int64_t c, int64_t h, int64_t w)
{
    return TensorSpec{{n, c, h, w}, {h * w * c, 1, w * c, c}};
}

/// A complete backward-data problem: groups divides c and k; the filter holds c / groups
/// channels; Hi and Wi are dX's.
struct DgradShape
{
    int64_t n = 2;
    int64_t c = 32;
    int64_t k = 32;
    int64_t groups = 1;
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

inline DgradShape shapeOf(const std::array<int64_t, 14>& v)
{
    return DgradShape{
        v[0], v[1], v[2], v[13], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12]};
}

/// A shape as a test failure prints it.
inline std::ostream& operator<<(std::ostream& out, const DgradShape& s)
{
    return out << "N" << s.n << " C" << s.c << " K" << s.k << " G" << s.groups << " " << s.h << "x"
               << s.w << " f" << s.y << "x" << s.x << " s" << s.sH << "x" << s.sW << " p" << s.pH
               << "x" << s.pW << " d" << s.dH << "x" << s.dW;
}

struct GraphSpec
{
    /// dY (uid 1), W (uid 2), dX (uid 3).
    std::array<TensorSpec, 3> tensors;
    std::vector<int64_t> stride{1, 1};
    std::vector<int64_t> dilation{1, 1};
    std::vector<int64_t> prePadding{1, 1};
    std::vector<int64_t> postPadding{1, 1};
    data_objects::DataType computeType = data_objects::DataType::FLOAT;
    data_objects::ConvMode mode = data_objects::ConvMode::CROSS_CORRELATION;
    bool forwardNode = false;
    unsigned int nodeCount = 1;
    int64_t dyUid = 1;
    int64_t wUid = 2;
    int64_t dxUid = 3;
};

/// dY's extent by Python's floor rule, for shapes the tests know are valid.
inline int64_t
    outputExtent(int64_t in, int64_t filter, int64_t stride, int64_t pad, int64_t dilation)
{
    return (in + 2 * pad - dilation * (filter - 1) - 1) / stride + 1;
}

inline GraphSpec graphSpec(const DgradShape& s,
                           data_objects::DataType dtype = data_objects::DataType::HALF)
{
    GraphSpec spec;
    const auto ho = outputExtent(s.h, s.y, s.sH, s.pH, s.dH);
    const auto wo = outputExtent(s.w, s.x, s.sW, s.pW, s.dW);
    spec.tensors[0] = nhwc(s.n, s.k, ho, wo);
    spec.tensors[1] = nhwc(s.k, s.c / s.groups, s.y, s.x);
    spec.tensors[2] = nhwc(s.n, s.c, s.h, s.w);
    for(auto& tensor : spec.tensors)
    {
        tensor.dtype = dtype;
    }
    spec.stride = {s.sH, s.sW};
    spec.dilation = {s.dH, s.dW};
    spec.prePadding = spec.postPadding = {s.pH, s.pW};
    return spec;
}

/// The graph of @p spec, its node(s) named @p nodeName.
inline flatbuffers::FlatBufferBuilder buildGraph(const GraphSpec& spec,
                                                 std::string_view nodeName = "conv_dgrad_test")
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    for(size_t i = 0; i < spec.tensors.size(); ++i)
    {
        const auto& tensor = spec.tensors[i];
        const auto dims = tensor.omitDims ? flatbuffers::Offset<flatbuffers::Vector<int64_t>>()
                                          : builder.CreateVector(tensor.dims);
        const auto strides = tensor.omitDims ? flatbuffers::Offset<flatbuffers::Vector<int64_t>>()
                                             : builder.CreateVector(tensor.strides);
        data_objects::TensorAttributesBuilder tensorBuilder(builder);
        tensorBuilder.add_uid(static_cast<int64_t>(i + 1));
        tensorBuilder.add_data_type(tensor.dtype);
        tensorBuilder.add_dims(dims);
        tensorBuilder.add_strides(strides);
        tensorBuilder.add_virtual_(tensor.isVirtual);
        tensorBuilder.add_alignment(tensor.alignment);
        tensors.push_back(tensorBuilder.Finish());
    }
    flatbuffers::Offset<void> attributes;
    auto attributesType = data_objects::NodeAttributes::ConvolutionBwdAttributes;
    if(spec.forwardNode)
    {
        attributesType = data_objects::NodeAttributes::ConvolutionFwdAttributes;
        attributes = data_objects::CreateConvolutionFwdAttributesDirect(builder,
                                                                        spec.dxUid,
                                                                        spec.wUid,
                                                                        spec.dyUid,
                                                                        &spec.prePadding,
                                                                        &spec.postPadding,
                                                                        &spec.stride,
                                                                        &spec.dilation,
                                                                        spec.mode)
                         .Union();
    }
    else
    {
        attributes = data_objects::CreateConvolutionBwdAttributesDirect(builder,
                                                                        spec.dyUid,
                                                                        spec.wUid,
                                                                        spec.dxUid,
                                                                        &spec.prePadding,
                                                                        &spec.postPadding,
                                                                        &spec.stride,
                                                                        &spec.dilation,
                                                                        spec.mode)
                         .Union();
    }
    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.reserve(spec.nodeCount);
    for(unsigned int i = 0; i < spec.nodeCount; ++i)
    {
        nodes.push_back(data_objects::CreateNodeDirect(
            builder, std::string(nodeName).c_str(), spec.computeType, attributesType, attributes));
    }
    const auto tensorVector = builder.CreateVector(tensors);
    const auto nodeVector = builder.CreateVector(nodes);
    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_tensors(tensorVector);
    graphBuilder.add_nodes(nodeVector);
    builder.Finish(graphBuilder.Finish());
    return builder;
}

inline DescriptorId idOf(uint32_t value, uint8_t tag)
{
    DescriptorId id{};
    id[0] = tag;
    id[12] = static_cast<uint8_t>(value >> 24U);
    id[13] = static_cast<uint8_t>(value >> 16U);
    id[14] = static_cast<uint8_t>(value >> 8U);
    id[15] = static_cast<uint8_t>(value);
    return id;
}

/// A kpack kernel carrying @p entry's complete metadata, as the descriptor loader delivers
/// it (every field present, omitted ones completed from the KMD defaults), in pack @p pack,
/// named @p namePrefix + dtype + variant + instance id.
inline KernelDefinition
    kernelOf(const catalog::Entry& entry, uint8_t pack, std::string_view namePrefix)
{
    KernelDefinition kernel;
    const auto& id = entry.text("instance_id");
    kernel.kernelId = idOf(static_cast<uint32_t>(std::hash<std::string>{}(id + entry.dtype)), pack);
    kernel.packId = idOf(0, pack);
    kernel.source.kind = KernelSourceKind::KPACK;
    kernel.name
        = std::string(namePrefix) + entry.dtype + "_" + entry.text("direct_variant") + "_" + id;
    for(const auto& [name, value] : entry.fields)
    {
        if(const auto* integer = std::get_if<int64_t>(&value))
        {
            kernel.metadata[name] = *integer;
        }
        else
        {
            kernel.metadata[name] = std::get<std::string>(value);
        }
    }
    return kernel;
}

/// Runs the KernelPrepareFn registered as @p kernelMatcher over @p kernels as the engine's
/// shared state does at load: each carries its prepared payload (and a gfx950 helper is
/// filed for the MFMA pipelines of its own pack).
inline void prepareKernels(std::vector<KernelDefinition>& kernels, std::string_view kernelMatcher)
{
    registerNativeIngestorSymbols();
    const auto prepareFn = hipdnn_plugin_sdk::ingestor::KernelPrepareRegistry::tryResolve(
        std::string(kernelMatcher));
    ASSERT_NE(prepareFn, nullptr) << kernelMatcher;
    for(auto& kernel : kernels)
    {
        kernel.prepared = prepareFn(kernel);
        kernel.preparedBy = prepareFn;
    }
}

/// @p target's @p dtype catalog as pack @p pack, named as its bundle names it and
/// prepared by the KernelPrepareFn registered as @p kernelMatcher, in catalog order.
inline std::vector<KernelDefinition> catalogPack(const std::string& dtype,
                                                 uint8_t pack,
                                                 const conv_dgrad_aot::Target& target,
                                                 std::string_view kernelMatcher)
{
    const auto prefix = std::string(target.arch) + "_conv_dgrad_aot_";
    std::vector<KernelDefinition> kernels;
    for(const auto& entry : catalog::catalog(dtype, target))
    {
        kernels.push_back(kernelOf(entry, pack, prefix));
    }
    prepareKernels(kernels, kernelMatcher);
    return kernels;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::dgrad_matcher_test
