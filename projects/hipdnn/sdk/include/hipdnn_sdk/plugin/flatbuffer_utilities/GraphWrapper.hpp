// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <memory>

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/PluginException.hpp>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/NodeWrapper.hpp>

namespace hipdnn_plugin
{

/*
 * The IGraph interface expects that any implementations have the graph sorted in topological order.
 * The graph must also have no cycles and be fully connected(no orphan nodes).  We also expect
 * that the all tensors in the graph have unique uids.
*/
class IGraph
{
public:
    virtual ~IGraph() = default;

    virtual const hipdnn_sdk::data_objects::Graph& getGraph() const = 0;
    virtual bool isValid() const = 0;
    virtual uint32_t nodeCount() const = 0;
    virtual bool hasOnlySupportedAttributes(
        std::set<hipdnn_sdk::data_objects::NodeAttributes> supportedAttributes) const
        = 0;
    virtual const hipdnn_sdk::data_objects::Node& getNode(uint32_t index) const = 0;
    virtual const INodeWrapper& getNodeWrapper(uint32_t index) const = 0;
    virtual const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
        getTensorMap() const
        = 0;
};

class GraphWrapper : public IGraph
{
public:
    explicit GraphWrapper(const void* buffer, size_t size)
    {
        if(buffer != nullptr)
        {
            flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer), size);
            if(verifier.VerifyBuffer<hipdnn_sdk::data_objects::Graph>())
            {
                _shallowGraph = flatbuffers::GetRoot<hipdnn_sdk::data_objects::Graph>(buffer);
            }
        }
    }

    const hipdnn_sdk::data_objects::Graph& getGraph() const override
    {
        throwIfNotValid();
        return *_shallowGraph;
    }

    bool isValid() const override
    {
        return _shallowGraph != nullptr;
    }

    uint32_t nodeCount() const override
    {
        throwIfNotValid();
        auto nodes = _shallowGraph->nodes();
        if(nodes == nullptr)
        {
            return 0;
        }
        return static_cast<uint32_t>(nodes->size());
    }

    bool hasOnlySupportedAttributes(
        std::set<hipdnn_sdk::data_objects::NodeAttributes> supportedAttributes) const override
    {
        throwIfNotValid();

        auto nodes = _shallowGraph->nodes();
        if(nodes == nullptr)
        {
            return true; // No nodes means no unsupported attributes
        }

        return std::all_of(nodes->begin(), nodes->end(), [&](const auto node) {
            return supportedAttributes.find(node->attributes_type()) != supportedAttributes.end();
        });
    }

    const hipdnn_sdk::data_objects::Node& getNode(uint32_t index) const override
    {
        throwIfNotValid();

        auto nodes = _shallowGraph->nodes();
        if(nodes == nullptr)
        {
            throw std::out_of_range("No nodes in graph");
        }

        if(index >= nodes->size())
        {
            throw std::out_of_range("Index out of range for graph nodes");
        }

        return *nodes->Get(index);
    }

    const INodeWrapper& getNodeWrapper(uint32_t index) const override
    {
        throwIfNotValid();

        if(_nodeWrappers.empty())
        {
            auto nodes = _shallowGraph->nodes();
            if(nodes == nullptr)
            {
                throw std::out_of_range("No nodes in graph");
            }

            _nodeWrappers.reserve(nodes->size());
            for(const auto node : *nodes)
            {
                _nodeWrappers.push_back(std::make_unique<NodeWrapper>(node));
            }
        }

        if(index >= _nodeWrappers.size())
        {
            throw std::out_of_range("Index out of range for graph nodes");
        }
        return *_nodeWrappers[index];
    }

    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
        getTensorMap() const override
    {
        throwIfNotValid();

        if(!_tensorMap.empty())
        {
            return _tensorMap;
        }

        auto tensors = _shallowGraph->tensors();
        if(tensors != nullptr)
        {
            for(const auto tensor : *tensors)
            {
                _tensorMap[tensor->uid()] = tensor;
            }
        }

        return _tensorMap;
    }

private:
    void throwIfNotValid() const
    {
        if(!isValid())
        {
            throw hipdnn_plugin::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "Graph is not valid");
        }
    }

    // Pointer to the flatbuffer representation of the graph. We do not own this memory
    // as were just reading from the buffer passed during construction.
    const hipdnn_sdk::data_objects::Graph* _shallowGraph = nullptr;

    //lazy init state;
    mutable std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>
        _tensorMap;

    mutable std::vector<std::unique_ptr<INodeWrapper>> _nodeWrappers;
};

}
