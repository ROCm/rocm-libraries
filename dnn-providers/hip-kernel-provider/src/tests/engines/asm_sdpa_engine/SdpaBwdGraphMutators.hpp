// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace asm_sdpa_engine
{

// Copies of a backward SDPA graph with exactly one new backward-only field set, so a decline
// test can show that one arm, and no other, flips a graph the engine accepts. The tensors the
// new uid fields name are added with the dims the frontend contract requires. Their uid is 0:
// the graph builders number tensors from 1, so a decline must test that the field is present,
// not that its value is nonzero. Each helper throws if the base graph already uses uid 0, so a
// new tensor can never alias another one.
constexpr int64_t K_SDPA_BWD_AUX_UID = 0;

namespace detail
{

inline flatbuffers::FlatBufferBuilder mutateSdpaBwd(
    const flatbuffers::FlatBufferBuilder& base,
    const std::function<void(hipdnn_flatbuffers_sdk::data_objects::GraphT&,
                             hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributesT&)>&
        mutate)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    GraphT graph;
    GetGraph(base.GetBufferPointer())->UnPackTo(&graph);

    auto* sdpa = graph.nodes.front()->attributes.AsSdpaBackwardAttributes();
    mutate(graph, *sdpa);

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Graph::Pack(builder, &graph));
    return builder;
}

inline std::vector<int64_t> tensorDims(const hipdnn_flatbuffers_sdk::data_objects::GraphT& graph,
                                       int64_t uid)
{
    for(const auto& tensor : graph.tensors)
    {
        if(tensor->uid == uid)
        {
            return tensor->dims;
        }
    }
    throw std::logic_error("tensorDims: tensor uid " + std::to_string(uid) + " not in graph");
}

inline void addAuxTensor(hipdnn_flatbuffers_sdk::data_objects::GraphT& graph,
                         const std::string& name,
                         const std::vector<int64_t>& dims)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    for(const auto& tensor : graph.tensors)
    {
        if(tensor->uid == K_SDPA_BWD_AUX_UID)
        {
            throw std::logic_error("addAuxTensor: tensor uid 0 is already in use");
        }
    }

    auto tensor = std::make_unique<TensorAttributesT>();
    tensor->uid = K_SDPA_BWD_AUX_UID;
    tensor->name = name;
    tensor->data_type = DataType::FLOAT;
    tensor->dims = dims;
    tensor->strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    graph.tensors.push_back(std::move(tensor));
}

} // namespace detail

// sink_token: [1, H_q, 1, 1], FLOAT input.
inline flatbuffers::FlatBufferBuilder addBwdSinkToken(const flatbuffers::FlatBufferBuilder& base)
{
    return detail::mutateSdpaBwd(base, [](auto& graph, auto& sdpa) {
        const auto qDims = detail::tensorDims(graph, sdpa.q_tensor_uid);
        detail::addAuxTensor(graph, "sink_token", {1, qDims[1], 1, 1});
        sdpa.sink_token_tensor_uid = K_SDPA_BWD_AUX_UID;
    });
}

// dsink_token: [1, H_q, 1, 1], FLOAT output. Set without a sink on purpose: the engine declines
// the dSink field itself, so this must not depend on the sink arm.
inline flatbuffers::FlatBufferBuilder addBwdDsinkToken(const flatbuffers::FlatBufferBuilder& base)
{
    return detail::mutateSdpaBwd(base, [](auto& graph, auto& sdpa) {
        const auto qDims = detail::tensorDims(graph, sdpa.q_tensor_uid);
        detail::addAuxTensor(graph, "dsink_token", {1, qDims[1], 1, 1});
        sdpa.dsink_token_tensor_uid = K_SDPA_BWD_AUX_UID;
    });
}

// rng_dump: [B, H_q, S_q, S_kv], FLOAT output. Set without dropout on purpose: the engine
// declines the dump field itself, so this must not depend on the dropout arm.
inline flatbuffers::FlatBufferBuilder addBwdRngDump(const flatbuffers::FlatBufferBuilder& base)
{
    return detail::mutateSdpaBwd(base, [](auto& graph, auto& sdpa) {
        const auto qDims = detail::tensorDims(graph, sdpa.q_tensor_uid);
        const auto kDims = detail::tensorDims(graph, sdpa.k_tensor_uid);
        detail::addAuxTensor(graph, "rng_dump", {qDims[0], qDims[1], qDims[2], kDims[2]});
        sdpa.rng_dump_tensor_uid = K_SDPA_BWD_AUX_UID;
    });
}

inline flatbuffers::FlatBufferBuilder
    withBwdDeterministicAlgorithm(const flatbuffers::FlatBufferBuilder& base)
{
    return detail::mutateSdpaBwd(base,
                                 [](auto&, auto& sdpa) { sdpa.is_deterministic_algorithm = true; });
}

// Frontend validation rejects max_total_seq_len without a ragged tensor, so no frontend-built
// graph reaches the engine like this; the flatbuffer is built directly.
inline flatbuffers::FlatBufferBuilder withBwdMaxTotalSeqLen(
    const flatbuffers::FlatBufferBuilder& base, int64_t maxTotalSeqLenQ, int64_t maxTotalSeqLenKv)
{
    return detail::mutateSdpaBwd(base, [=](auto&, auto& sdpa) {
        sdpa.max_total_seq_len_q = maxTotalSeqLenQ;
        sdpa.max_total_seq_len_kv = maxTotalSeqLenKv;
    });
}

} // namespace asm_sdpa_engine
