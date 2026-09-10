// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace hip_kernel_provider
{

/// Owns a tiny serialized graph containing a ragged tensor and exposes a
/// UID-to-TensorAttributes map suitable for stubbing MockGraph::getTensorMap().
/// The map points into the owned buffer, so keep this fixture alive for the
/// duration of the test.
class RaggedTensorMapFixture
{
public:
    RaggedTensorMapFixture()
    {
        using namespace hipdnn_flatbuffers_sdk::data_objects;

        const std::vector<int64_t> dims{1};
        std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
        tensors.push_back(CreateTensorAttributesDirect(
            _builder, 1, "ragged_offset", DataType::INT32, &dims, &dims));
        tensors.push_back(CreateTensorAttributesDirect(_builder,
                                                       2,
                                                       "ragged",
                                                       DataType::BFLOAT16,
                                                       &dims,
                                                       &dims,
                                                       false,
                                                       TensorValue::NONE,
                                                       0,
                                                       false,
                                                       flatbuffers::Optional<int64_t>(1)));

        _builder.Finish(CreateGraphDirect(
            _builder, "ragged", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors));

        const auto* graph = GetGraph(_builder.GetBufferPointer());
        for(const auto* tensor : *graph->tensors())
        {
            _tensorMap[tensor->uid()] = tensor;
        }
    }

    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap() const
    {
        return _tensorMap;
    }

private:
    flatbuffers::FlatBufferBuilder _builder;
    std::unordered_map<int64_t, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>
        _tensorMap;
};

} // namespace hip_kernel_provider
