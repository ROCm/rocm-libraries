// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "core/Handle.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace asm_sdpa_engine
{

// Where a test graph's attention scale comes from. A pure runtime pass-by-value
// scale tensor is not listed: its value is known only at execute.
enum class AttnScaleSource
{
    ATTRIBUTE, // attn_scale_value
    CONSTANT_TENSOR, // compile-time constant pass-by-value scale tensor
    RUNTIME_TENSOR_WITH_DEFAULT, // runtime pass-by-value scale tensor with a default value
};

inline std::vector<std::pair<AttnScaleSource, std::string>> attnScaleSources()
{
    return {
        {AttnScaleSource::ATTRIBUTE, "attn_scale_value"},
        {AttnScaleSource::CONSTANT_TENSOR, "constant scale tensor"},
        {AttnScaleSource::RUNTIME_TENSOR_WITH_DEFAULT, "runtime scale tensor default"},
    };
}

// Scales the kernels mishandle: not a finite, positive, normal float. On gfx942,
// 0, -0.0, -1.2, -0.088 and 1e-40 gave NaN or Inf outputs or gradients, and +-inf
// gave NaN everywhere.
inline std::vector<std::pair<float, std::string>> unsupportedAttnScales()
{
    return {
        {0.0F, "0"},
        {-0.0F, "-0.0"},
        {-1.2F, "-1.2"},
        {-0.088388F, "-1/sqrt(128)"},
        {-std::numeric_limits<float>::min(), "-FLT_MIN"},
        {1e-40F, "subnormal 1e-40"},
        {std::nextafter(std::numeric_limits<float>::min(), 0.0F), "largest subnormal"},
        {std::numeric_limits<float>::denorm_min(), "smallest subnormal"},
        {std::numeric_limits<float>::quiet_NaN(), "NaN"},
        {std::numeric_limits<float>::infinity(), "+inf"},
        {-std::numeric_limits<float>::infinity(), "-inf"},
    };
}

// Finite, positive, normal scales, down to a tiny normal one (1e-37 is correct on
// gfx942).
inline std::vector<std::pair<float, std::string>> supportedAttnScales()
{
    return {
        {1e-37F, "tiny normal 1e-37"},
        {0.088388F, "1/sqrt(128)"},
        {1.0F, "1"},
    };
}

// A test case name: which scale, from which source.
inline std::string attnScaleCaseName(const std::string& scaleName, const std::string& sourceName)
{
    return scaleName + " as " + sourceName;
}

// Copy of a forward or backward SDPA graph with its attention scale replaced: no
// attn_scale_value, and scaleTensor (given the next free uid) as the scale tensor.
inline flatbuffers::FlatBufferBuilder withScaleTensor(
    const flatbuffers::FlatBufferBuilder& base,
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> scaleTensor)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    GraphT graph;
    GetGraph(base.GetBufferPointer())->UnPackTo(&graph);

    int64_t scaleUid = 1;
    for(const auto& tensor : graph.tensors)
    {
        scaleUid = std::max(scaleUid, tensor->uid + 1);
    }
    scaleTensor->uid = scaleUid;
    graph.tensors.push_back(std::move(scaleTensor));

    const auto setScaleTensor = [scaleUid](auto* attributes) {
        attributes->attn_scale_value = flatbuffers::nullopt;
        attributes->scale_tensor_uid = scaleUid;
    };
    auto& node = graph.nodes.front()->attributes;
    if(auto* forward = node.AsSdpaAttributes())
    {
        setScaleTensor(forward);
    }
    else if(auto* backward = node.AsSdpaBackwardAttributes())
    {
        setScaleTensor(backward);
    }
    else
    {
        throw std::logic_error("withScaleTensor: the first node is not an SDPA node");
    }

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Graph::Pack(builder, &graph));
    return builder;
}

// Copy of a forward or backward SDPA graph whose attention scale is `scale`, from
// `source`. A scale tensor is a [1] FLOAT pass-by-value tensor.
inline flatbuffers::FlatBufferBuilder
    withAttnScale(const flatbuffers::FlatBufferBuilder& base, AttnScaleSource source, float scale)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    if(source != AttnScaleSource::ATTRIBUTE)
    {
        auto tensor = std::make_unique<TensorAttributesT>();
        tensor->name = "attn_scale";
        tensor->data_type = DataType::FLOAT;
        tensor->dims = {1};
        tensor->strides = {1};
        tensor->value.Set(Float32Value(scale));
        tensor->is_runtime_pass_by_value = source == AttnScaleSource::RUNTIME_TENSOR_WITH_DEFAULT;
        return withScaleTensor(base, std::move(tensor));
    }

    GraphT graph;
    GetGraph(base.GetBufferPointer())->UnPackTo(&graph);

    const auto setScaleValue = [scale](auto* attributes) {
        attributes->attn_scale_value = scale;
        attributes->scale_tensor_uid = flatbuffers::nullopt;
    };
    auto& node = graph.nodes.front()->attributes;
    if(auto* forward = node.AsSdpaAttributes())
    {
        setScaleValue(forward);
    }
    else if(auto* backward = node.AsSdpaBackwardAttributes())
    {
        setScaleValue(backward);
    }
    else
    {
        throw std::logic_error("withAttnScale: the first node is not an SDPA node");
    }

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Graph::Pack(builder, &graph));
    return builder;
}

// Copy of a forward or backward SDPA graph whose scale tensor is a constant
// FP8_E4M3 tensor, a dtype the plugin SDK cannot read as a scalar operand.
inline flatbuffers::FlatBufferBuilder
    withUnreadableScaleTensor(const flatbuffers::FlatBufferBuilder& base)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    auto tensor = std::make_unique<TensorAttributesT>();
    tensor->name = "attn_scale";
    tensor->data_type = DataType::FP8_E4M3;
    tensor->dims = {1};
    tensor->strides = {1};
    tensor->value.Set(Float8Value(0x38)); // 1.0 in FP8_E4M3
    return withScaleTensor(base, std::move(tensor));
}

// Runs plan.execute with the pure runtime pass-by-value scale tensor scaleUid
// holding `scale`, every uid in tensorUids pointing at host memory, and a host
// workspace of workspaceSize bytes (none when 0). Only for plans that throw before
// any kernel reads that memory. Returns the status execute throws with, or
// HIPDNN_PLUGIN_STATUS_SUCCESS when it returns.
template <typename PlanT>
hipdnnPluginStatus_t executeWithRuntimeScale(const PlanT& plan,
                                             const std::vector<int64_t>& tensorUids,
                                             int64_t scaleUid,
                                             float scale,
                                             size_t workspaceSize)
{
    std::vector<std::byte> tensorMemory(16);
    std::vector<std::byte> workspace(workspaceSize);
    float hostScale = scale;
    std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers;
    deviceBuffers.reserve(tensorUids.size() + 1);
    for(const int64_t uid : tensorUids)
    {
        deviceBuffers.push_back({uid, tensorMemory.data()});
    }
    deviceBuffers.push_back({scaleUid, &hostScale});

    const Handle handle;
    try
    {
        plan.execute(handle,
                     deviceBuffers.data(),
                     static_cast<uint32_t>(deviceBuffers.size()),
                     workspace.empty() ? nullptr : workspace.data());
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        return e.getStatus();
    }
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

} // namespace asm_sdpa_engine
