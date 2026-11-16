// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include <algorithm>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace hipdnn_test_sdk::utilities::sdpa
{

// Post-process a serialized SDPA graph. For a domain-agnostic ragged mutator see
// makeGraphWithRaggedTensor in FlatbufferGraphTestUtils.hpp.

namespace detail
{
inline hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesT&
    getSdpaAttributes(hipdnn_flatbuffers_sdk::data_objects::GraphT& graph)
{
    if(graph.nodes.empty())
    {
        throw std::runtime_error("SdpaGraphMutators: graph has no nodes");
    }
    auto* attrs = graph.nodes.front()->attributes.AsSdpaAttributes();
    if(attrs == nullptr)
    {
        throw std::runtime_error("SdpaGraphMutators: first node is not an SDPA node");
    }
    return *attrs;
}

inline int64_t maxUid(const hipdnn_flatbuffers_sdk::data_objects::GraphT& graph)
{
    int64_t maxId = 0;
    for(const auto& tensor : graph.tensors)
    {
        maxId = std::max(maxId, tensor->uid);
    }
    return maxId;
}

inline int64_t getQTensorUid(const hipdnn_flatbuffers_sdk::data_objects::GraphT& graph)
{
    if(graph.nodes.empty())
    {
        throw std::runtime_error("SdpaGraphMutators: graph has no nodes");
    }
    const auto& attributes = graph.nodes.front()->attributes;
    if(const auto* fwd = attributes.AsSdpaAttributes())
    {
        return fwd->q_tensor_uid;
    }
    if(const auto* bwd = attributes.AsSdpaBackwardAttributes())
    {
        return bwd->q_tensor_uid;
    }
    throw std::runtime_error("SdpaGraphMutators: first node is not an SDPA node");
}

inline const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT&
    getTensor(const hipdnn_flatbuffers_sdk::data_objects::GraphT& graph, int64_t uid)
{
    for(const auto& tensor : graph.tensors)
    {
        if(tensor->uid == uid)
        {
            return *tensor;
        }
    }
    throw std::runtime_error("SdpaGraphMutators: tensor uid " + std::to_string(uid) + " not found");
}

inline int64_t appendIndexTensor(hipdnn_flatbuffers_sdk::data_objects::GraphT& graph,
                                 const std::string& name,
                                 const std::vector<int64_t>& dims = {1},
                                 const std::vector<int64_t>& strides = {1})
{
    auto tensor = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT>();
    tensor->uid = maxUid(graph) + 1;
    tensor->name = name;
    tensor->data_type = hipdnn_flatbuffers_sdk::data_objects::DataType::INT32;
    tensor->dims = dims;
    tensor->strides = strides;
    const int64_t uid = tensor->uid;
    graph.tensors.push_back(std::move(tensor));
    return uid;
}

inline flatbuffers::FlatBufferBuilder
    repack(const hipdnn_flatbuffers_sdk::data_objects::GraphT& graph)
{
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(hipdnn_flatbuffers_sdk::data_objects::CreateGraph(builder, &graph));
    return builder;
}
} // namespace detail

/// Mark the named tensors ragged with INT32 [B+1, 1, 1, 1] offset tensors, where B is
/// Q's batch. Q and O share a "qo" offset, K and V share a "kv" offset, and any other
/// name gets its own. Each ragged tensor of rank >= 3 gets
/// ragged_offset_multiplier = strides[2] (the SDPA sequence stride).
inline flatbuffers::FlatBufferBuilder withRaggedTensors(flatbuffers::FlatBufferBuilder&& builder,
                                                        const std::set<std::string>& names)
{
    auto graph = hipdnn_flatbuffers_sdk::data_objects::UnPackGraph(builder.GetBufferPointer());

    const int64_t batch = detail::getTensor(*graph, detail::getQTensorUid(*graph)).dims.at(0);
    const auto appendOffset = [&](const std::string& offsetName) {
        return detail::appendIndexTensor(*graph, offsetName, {batch + 1, 1, 1, 1}, {1, 1, 1, 1});
    };

    // Appending to graph->tensors while iterating it below would invalidate the loop.
    std::unordered_map<std::string, int64_t> offsetForName;
    if(names.count("q") != 0 || names.count("o") != 0)
    {
        const int64_t qoOffsetUid = appendOffset("qo_ragged_offset");
        offsetForName["q"] = qoOffsetUid;
        offsetForName["o"] = qoOffsetUid;
    }
    if(names.count("k") != 0 || names.count("v") != 0)
    {
        const int64_t kvOffsetUid = appendOffset("kv_ragged_offset");
        offsetForName["k"] = kvOffsetUid;
        offsetForName["v"] = kvOffsetUid;
    }
    for(const auto& name : names)
    {
        if(offsetForName.count(name) == 0)
        {
            offsetForName[name] = appendOffset(name + "_ragged_offset");
        }
    }

    for(auto& tensor : graph->tensors)
    {
        if(names.count(tensor->name) != 0)
        {
            tensor->ragged_offset_tensor_uid = offsetForName.at(tensor->name);
            if(tensor->strides.size() >= 3)
            {
                tensor->ragged_offset_multiplier = tensor->strides[2];
            }
        }
    }

    return detail::repack(*graph);
}

/// Attach a seq-len tensor and set seq_len_q_tensor_uid on the SDPA node.
inline flatbuffers::FlatBufferBuilder withSeqLenQ(flatbuffers::FlatBufferBuilder&& builder)
{
    auto graph = hipdnn_flatbuffers_sdk::data_objects::UnPackGraph(builder.GetBufferPointer());
    detail::getSdpaAttributes(*graph).seq_len_q_tensor_uid
        = detail::appendIndexTensor(*graph, "seq_len_q");
    return detail::repack(*graph);
}

/// Attach a seq-len tensor and set seq_len_kv_tensor_uid on the SDPA node.
inline flatbuffers::FlatBufferBuilder withSeqLenKv(flatbuffers::FlatBufferBuilder&& builder)
{
    auto graph = hipdnn_flatbuffers_sdk::data_objects::UnPackGraph(builder.GetBufferPointer());
    detail::getSdpaAttributes(*graph).seq_len_kv_tensor_uid
        = detail::appendIndexTensor(*graph, "seq_len_kv");
    return detail::repack(*graph);
}

} // namespace hipdnn_test_sdk::utilities::sdpa
