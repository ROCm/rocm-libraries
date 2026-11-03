// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

struct GraphTest
{
    std::shared_ptr<flatbuffers::DetachedBuffer> buffer;
    std::string message;

    GraphTest(flatbuffers::FlatBufferBuilder&& builder, std::string inMessage)
        : buffer(std::make_shared<flatbuffers::DetachedBuffer>(builder.Release()))
        , message(std::move(inMessage))
    {
    }

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper() const
    {
        return hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(buffer->data(),
                                                                          buffer->size());
    }
};

// Unpack a serialized graph, apply `mutate`, and re-serialize it.
inline flatbuffers::FlatBufferBuilder
    mutateGraph(flatbuffers::FlatBufferBuilder&& builder,
                const std::function<void(hipdnn_flatbuffers_sdk::data_objects::GraphT&)>& mutate)
{
    auto graph = hipdnn_flatbuffers_sdk::data_objects::UnPackGraph(builder.GetBufferPointer());
    mutate(*graph);
    flatbuffers::FlatBufferBuilder mutated;
    mutated.Finish(hipdnn_flatbuffers_sdk::data_objects::CreateGraph(mutated, graph.get()));
    return mutated;
}

inline hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT&
    findTensor(hipdnn_flatbuffers_sdk::data_objects::GraphT& graph, const std::string& name)
{
    for(auto& tensor : graph.tensors)
    {
        if(tensor->name == name)
        {
            return *tensor;
        }
    }
    throw std::runtime_error("No tensor named " + name);
}
