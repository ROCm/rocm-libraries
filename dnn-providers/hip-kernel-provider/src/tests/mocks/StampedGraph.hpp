// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>
#include <string_view>

#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>

namespace hip_kernel_provider
{

/// A node-less graph whose min_required_engine_api_version is @p floor, or unstamped
/// when @p floor is empty, as hipDNN stamps a graph from the features it uses.
class StampedGraph : public hipdnn_test_sdk::utilities::MockGraph
{
public:
    explicit StampedGraph(std::optional<std::string_view> floor)
    {
        hipdnn_flatbuffers_sdk::data_objects::EngineApiVersion version{};
        if(floor.has_value())
        {
            version = hipdnn_plugin_sdk::toEngineApiVersion(
                hipdnn_data_sdk::utilities::Version{*floor});
        }
        auto name = _buffer.CreateString("stamped_graph");
        hipdnn_flatbuffers_sdk::data_objects::GraphBuilder graph(_buffer);
        graph.add_name(name);
        graph.add_min_required_engine_api_version(floor.has_value() ? &version : nullptr);
        _buffer.Finish(graph.Finish());
        ON_CALL(*this, getGraph())
            .WillByDefault(::testing::ReturnRef(
                *flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::Graph>(
                    _buffer.GetBufferPointer())));
    }

private:
    flatbuffers::FlatBufferBuilder _buffer;
};

} // namespace hip_kernel_provider
