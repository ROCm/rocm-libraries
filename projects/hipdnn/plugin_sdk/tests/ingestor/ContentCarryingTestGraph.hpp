// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

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

#include "KernelIngestorTestFixtures.hpp"

namespace hipdnn_plugin_sdk::ingestor::testing
{

/// A graph fake that actually carries content, for the tests that must prove
/// `GraphContentKey` discriminates on it.
///
/// `TestGraph` (KernelIngestorTestFixtures.hpp) cannot serve here: it reports
/// `nodeCount() == 0`, throws from every node accessor, and hands back an empty tensor
/// map, so every instance is structurally identical and "changing a field changes the
/// key" is unassertable against it. It stays exactly as it is -- the matcher and
/// state-manager tests depend on that emptiness, and widening a shared fixture for one
/// story's needs would put new content in front of all of them.
///
/// Every field the key compares is independently settable through `Spec`, so a test can
/// vary exactly one and hold the rest fixed. That one-field-at-a-time property is what
/// makes the mutation tests a real pin rather than a smoke test.
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
    };
    struct NodeSpec
    {
        std::string name = "pointwise";
        hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType
            = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
        /// Two independently variable attribute fields, so a test can prove the key
        /// discriminates inside the union payload and not merely on its discriminant.
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation
            = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD;
        int64_t in0TensorUid = 1;
        int64_t out0TensorUid = 2;
    };

    /// Defaults describe one valid two-tensor, one-node graph. A test names only the
    /// field it is varying.
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
        std::vector<TensorSpec> tensors{TensorSpec{1}, TensorSpec{2}};
        std::vector<NodeSpec> nodes{NodeSpec{}};
    };

    /// Two constructors rather than one with a `Spec{}` default argument: a default
    /// argument is not a complete-class context, so naming `Spec{}` there while
    /// `ContentCarryingTestGraph`'s own body is still open cannot see `Spec`'s default
    /// member initializers.
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

} // namespace hipdnn_plugin_sdk::ingestor::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
