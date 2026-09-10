// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include <algorithm>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace hipdnn_test_sdk::utilities::sdpa
{

// SDPA-specific graph mutators that post-process a serialized SDPA graph (e.g. one
// built by createValidSdpaFwdGraph/createSdpaBwdGraph) by unpacking, editing the native
// GraphT, and re-emitting. This keeps ragged/seq-len wiring out of the shared
// createValidSdpaFwdGraph helper. For a domain-agnostic ragged mutator see
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

inline int64_t appendIndexTensor(hipdnn_flatbuffers_sdk::data_objects::GraphT& graph,
                                 const std::string& name)
{
    auto tensor = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT>();
    tensor->uid = maxUid(graph) + 1;
    tensor->name = name;
    tensor->data_type = hipdnn_flatbuffers_sdk::data_objects::DataType::INT32;
    tensor->dims = {1};
    tensor->strides = {1};
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

/// Mark the named tensors ragged. Q and O share a "qo" ragged offset, K and V
/// share a "kv" offset (matching buildPlan's qo=Q / kv=K wiring); any other name
/// gets its own offset aux. isApplicable only inspects ragged_offset_tensor_uid
/// presence, so the aux dims are not load-bearing.
inline flatbuffers::FlatBufferBuilder withRaggedTensors(flatbuffers::FlatBufferBuilder&& builder,
                                                        const std::set<std::string>& names)
{
    auto graph = hipdnn_flatbuffers_sdk::data_objects::UnPackGraph(builder.GetBufferPointer());

    // Append every offset aux up front: mutating graph->tensors while iterating it
    // below would invalidate the loop.
    std::unordered_map<std::string, int64_t> offsetForName;
    if(names.count("q") != 0 || names.count("o") != 0)
    {
        const int64_t qoOffsetUid = detail::appendIndexTensor(*graph, "qo_ragged_offset");
        offsetForName["q"] = qoOffsetUid;
        offsetForName["o"] = qoOffsetUid;
    }
    if(names.count("k") != 0 || names.count("v") != 0)
    {
        const int64_t kvOffsetUid = detail::appendIndexTensor(*graph, "kv_ragged_offset");
        offsetForName["k"] = kvOffsetUid;
        offsetForName["v"] = kvOffsetUid;
    }
    for(const auto& name : names)
    {
        if(offsetForName.count(name) == 0)
        {
            offsetForName[name] = detail::appendIndexTensor(*graph, name + "_ragged_offset");
        }
    }

    for(auto& tensor : graph->tensors)
    {
        if(names.count(tensor->name) != 0)
        {
            tensor->ragged_offset_tensor_uid = offsetForName.at(tensor->name);
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
