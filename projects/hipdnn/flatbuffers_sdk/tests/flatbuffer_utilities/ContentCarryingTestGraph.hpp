// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>

namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities::testing
{

using GraphId = hipdnn_flatbuffers_sdk::utilities::UuidBytes;

/// A graph fake carrying real content, for tests proving `GraphContentKey` discriminates
/// on it; `TestGraph` is content-empty by construction. Every field the key compares
/// is independently settable through `Spec`.
class ContentCarryingTestGraph : public hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph
{
public:
    struct TensorSpec
    {
        int64_t uid = 1;
        hipdnn_flatbuffers_sdk::data_objects::DataType dataType
            = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
        std::vector<int64_t> dims{4, 8};
        std::vector<int64_t> strides{8, 1};
        std::optional<int64_t> raggedOffsetTensorUid = std::nullopt;
    };
    struct NodeSpec
    {
        std::string name = "pointwise";
        hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType
            = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation
            = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD;
        int64_t in0TensorUid = 1;
        int64_t out0TensorUid = 2;
        std::optional<int64_t> in1TensorUid = std::nullopt;
    };

    /// Describes one valid two-tensor, one-node graph by default.
    struct Spec
    {
        std::optional<GraphId> graphId;
        std::string name = "content_graph";
        hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType
            = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
        hipdnn_flatbuffers_sdk::data_objects::DataType ioDataType
            = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
        hipdnn_flatbuffers_sdk::data_objects::DataType intermediateDataType
            = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
        std::optional<int64_t> preferredEngineId;
        bool isOverrideShapeEnabled = false;
        /// Derived from isOverrideShapeEnabled and tensor facts in production
        /// (PluginVersionConstants.hpp:58-93); settable here to reproduce that.
        std::optional<hipdnn_data_sdk::utilities::Version> minRequiredApiVersion;
        std::vector<TensorSpec> tensors{TensorSpec{1}, TensorSpec{2}};
        std::vector<NodeSpec> nodes{NodeSpec{}};
    };

    /// Two constructors, not a `Spec{}` default argument: a default argument is not a
    /// complete-class context and cannot see `Spec`'s member initializers.
    ContentCarryingTestGraph()
        : ContentCarryingTestGraph(Spec{})
    {
    }

    explicit ContentCarryingTestGraph(Spec spec)
        : _spec(std::move(spec))
    {
        build();
    }

    const hipdnn_flatbuffers_sdk::data_objects::Graph& getGraph() const override
    {
        return *flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::Graph>(
            _builder.GetBufferPointer());
    }

    bool isValid() const override
    {
        return true;
    }

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::SerializedBlobView bytes() const override
    {
        return {_builder.GetBufferPointer(), _builder.GetSize()};
    }

    uint32_t nodeCount() const override
    {
        return static_cast<uint32_t>(_spec.nodes.size());
    }

    bool hasOnlySupportedAttributes(
        std::set<hipdnn_flatbuffers_sdk::data_objects::NodeAttributes> /*supported*/) const override
    {
        return true;
    }

    const hipdnn_flatbuffers_sdk::data_objects::Node& getNode(uint32_t index) const override
    {
        const auto* nodes = getGraph().nodes();
        if(nodes == nullptr || index >= nodes->size())
        {
            throw std::out_of_range("ContentCarryingTestGraph: node index out of range");
        }
        return *nodes->Get(index);
    }

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper&
        getNodeWrapper(uint32_t /*index*/) const override
    {
        throw std::logic_error("ContentCarryingTestGraph carries no node wrappers");
    }

    const std::vector<std::unique_ptr<hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper>>&
        nodeWrappers() const override
    {
        throw std::logic_error("ContentCarryingTestGraph carries no node wrappers");
    }

    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        getTensorMap() const override
    {
        return _tensorMap;
    }

private:
    void build()
    {
        using namespace hipdnn_flatbuffers_sdk::data_objects;

        std::vector<flatbuffers::Offset<TensorAttributes>> tensorOffsets;
        tensorOffsets.reserve(_spec.tensors.size());
        for(const auto& tensor : _spec.tensors)
        {
            auto dims = _builder.CreateVector(tensor.dims);
            auto strides = _builder.CreateVector(tensor.strides);
            TensorAttributesBuilder tensorBuilder(_builder);
            tensorBuilder.add_uid(tensor.uid);
            tensorBuilder.add_data_type(tensor.dataType);
            tensorBuilder.add_dims(dims);
            tensorBuilder.add_strides(strides);
            if(tensor.raggedOffsetTensorUid.has_value())
            {
                tensorBuilder.add_ragged_offset_tensor_uid(*tensor.raggedOffsetTensorUid);
            }
            tensorOffsets.push_back(tensorBuilder.Finish());
        }

        std::vector<flatbuffers::Offset<Node>> nodeOffsets;
        nodeOffsets.reserve(_spec.nodes.size());
        for(const auto& node : _spec.nodes)
        {
            PointwiseAttributesBuilder attributesBuilder(_builder);
            attributesBuilder.add_operation(node.operation);
            attributesBuilder.add_in_0_tensor_uid(node.in0TensorUid);
            attributesBuilder.add_out_0_tensor_uid(node.out0TensorUid);
            if(node.in1TensorUid.has_value())
            {
                attributesBuilder.add_in_1_tensor_uid(*node.in1TensorUid);
            }
            const auto attributes = attributesBuilder.Finish();

            auto nodeName = _builder.CreateString(node.name);
            NodeBuilder nodeBuilder(_builder);
            nodeBuilder.add_name(nodeName);
            nodeBuilder.add_compute_data_type(node.computeDataType);
            nodeBuilder.add_attributes_type(NodeAttributes::PointwiseAttributes);
            nodeBuilder.add_attributes(attributes.Union());
            nodeOffsets.push_back(nodeBuilder.Finish());
        }

        auto tensors = _builder.CreateVector(tensorOffsets);
        auto nodes = _builder.CreateVector(nodeOffsets);
        auto name = _builder.CreateString(_spec.name);

        std::optional<hipdnn_flatbuffers_sdk::data_objects::Uuid> uuid;
        if(_spec.graphId.has_value())
        {
            uuid = hipdnn_flatbuffers_sdk::utilities::toFlatbufferUuid(*_spec.graphId);
        }

        hipdnn_flatbuffers_sdk::data_objects::EngineApiVersion apiVersion{};
        if(_spec.minRequiredApiVersion.has_value())
        {
            apiVersion = hipdnn_plugin_sdk::toEngineApiVersion(*_spec.minRequiredApiVersion);
        }

        GraphBuilder graphBuilder(_builder);
        graphBuilder.add_name(name);
        graphBuilder.add_compute_data_type(_spec.computeDataType);
        graphBuilder.add_io_data_type(_spec.ioDataType);
        graphBuilder.add_intermediate_data_type(_spec.intermediateDataType);
        graphBuilder.add_tensors(tensors);
        graphBuilder.add_nodes(nodes);
        graphBuilder.add_is_override_shape_enabled(_spec.isOverrideShapeEnabled);
        if(_spec.preferredEngineId.has_value())
        {
            graphBuilder.add_preferred_engine_id(*_spec.preferredEngineId);
        }
        if(_spec.minRequiredApiVersion.has_value())
        {
            graphBuilder.add_min_required_engine_api_version(&apiVersion);
        }
        if(uuid.has_value())
        {
            graphBuilder.add_id(&uuid.value());
        }
        _builder.Finish(graphBuilder.Finish());

        const auto* built = getGraph().tensors();
        if(built != nullptr)
        {
            for(uint32_t index = 0; index < built->size(); ++index)
            {
                const auto* tensor = built->Get(index);
                _tensorMap.emplace(tensor->uid(), tensor);
            }
        }
    }

    Spec _spec;
    flatbuffers::FlatBufferBuilder _builder;
    std::unordered_map<int64_t, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>
        _tensorMap;
};

/// A content-empty graph double: a valid, keyable graph carrying no tensors or nodes.
/// A separate fixture from `hipdnn_plugin_sdk::ingestor::testing::TestGraph` so this
/// file needs no dependency on its `GraphId`/`toEngineApiVersion` plumbing.
class EmptyTestGraph : public hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph
{
public:
    explicit EmptyTestGraph(std::optional<GraphId> graphId = std::nullopt)
    {
        auto name = _builder.CreateString("empty_test_graph");
        hipdnn_flatbuffers_sdk::data_objects::GraphBuilder graphBuilder(_builder);
        graphBuilder.add_name(name);
        if(graphId.has_value())
        {
            const auto uuid = hipdnn_flatbuffers_sdk::utilities::toFlatbufferUuid(*graphId);
            graphBuilder.add_id(&uuid);
        }
        _builder.Finish(graphBuilder.Finish());
    }

    const hipdnn_flatbuffers_sdk::data_objects::Graph& getGraph() const override
    {
        return *flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::Graph>(
            _builder.GetBufferPointer());
    }

    bool isValid() const override
    {
        return true;
    }

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::SerializedBlobView bytes() const override
    {
        return {_builder.GetBufferPointer(), _builder.GetSize()};
    }

    uint32_t nodeCount() const override
    {
        return 0;
    }

    bool hasOnlySupportedAttributes(
        std::set<hipdnn_flatbuffers_sdk::data_objects::NodeAttributes> /*supported*/) const override
    {
        return true;
    }

    const hipdnn_flatbuffers_sdk::data_objects::Node& getNode(uint32_t /*index*/) const override
    {
        throw std::logic_error("EmptyTestGraph carries no nodes");
    }

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper&
        getNodeWrapper(uint32_t /*index*/) const override
    {
        throw std::logic_error("EmptyTestGraph carries no nodes");
    }

    const std::vector<std::unique_ptr<hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper>>&
        nodeWrappers() const override
    {
        throw std::logic_error("EmptyTestGraph carries no nodes");
    }

    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        getTensorMap() const override
    {
        return _tensors;
    }

private:
    flatbuffers::FlatBufferBuilder _builder;
    std::unordered_map<int64_t, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>
        _tensors;
};

inline GraphId makeGraphId(uint8_t seed)
{
    GraphId id{};
    id.fill(seed);
    id[6] = static_cast<uint8_t>((id[6] & 0x0fU) | 0x40U);
    id[8] = static_cast<uint8_t>((id[8] & 0x3fU) | 0x80U);
    return id;
}

} // namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities::testing
