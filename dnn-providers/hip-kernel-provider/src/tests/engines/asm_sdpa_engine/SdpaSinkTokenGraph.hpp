// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace asm_sdpa_engine
{

// Copy of a forward SDPA graph whose SDPA node also carries a sink_token tensor
// ([1, num_heads, 1, 1], FLOAT). The sink uid is 0: the graph builders number tensors from 1, so
// a decline must test that the field is present, not that its value is nonzero. Throws if the
// base graph already uses uid 0, so the sink can never alias another tensor.
constexpr int64_t K_SINK_UID = 0;

inline flatbuffers::FlatBufferBuilder addSinkToken(const flatbuffers::FlatBufferBuilder& base)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    GraphT graph;
    GetGraph(base.GetBufferPointer())->UnPackTo(&graph);

    auto* sdpa = graph.nodes.front()->attributes.AsSdpaAttributes();
    int64_t numHeads = 1;
    for(const auto& tensor : graph.tensors)
    {
        if(tensor->uid == K_SINK_UID)
        {
            throw std::logic_error("addSinkToken: tensor uid 0 is already in use");
        }
        if(tensor->uid == sdpa->q_tensor_uid)
        {
            numHeads = tensor->dims[1];
        }
    }

    auto sink = std::make_unique<TensorAttributesT>();
    sink->uid = K_SINK_UID;
    sink->name = "sink_token";
    sink->data_type = DataType::FLOAT;
    sink->dims = {1, numHeads, 1, 1};
    sink->strides = {numHeads, 1, 1, 1};
    graph.tensors.push_back(std::move(sink));
    sdpa->sink_token_tensor_uid = K_SINK_UID;

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Graph::Pack(builder, &graph));
    return builder;
}

} // namespace asm_sdpa_engine
