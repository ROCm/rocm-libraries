// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <memory>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities
{

class INodeWrapper
{
public:
    virtual ~INodeWrapper() = default;
    virtual bool isValid() const = 0;
    virtual const hipdnn_flatbuffers_sdk::data_objects::Node& node() const = 0;

    virtual const void* attributes() const = 0;
    virtual hipdnn_flatbuffers_sdk::data_objects::NodeAttributes attributesType() const = 0;
    virtual std::string name() const = 0;
    virtual hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType() const = 0;

    template <typename T>
    const T& attributesAs() const
    {
        if(attributesType()
           != hipdnn_flatbuffers_sdk::data_objects::NodeAttributesTraits<T>::enum_value)
        {
            throw std::invalid_argument("Node attributes are not of the expected type");
        }

        auto* attr = attributes();
        if(attr == nullptr)
        {
            throw std::invalid_argument("Node attributes are null");
        }

        return *static_cast<const T*>(attr);
    }
};

class NodeWrapper : public INodeWrapper
{
public:
    explicit NodeWrapper(const hipdnn_flatbuffers_sdk::data_objects::Node* node)
        : _shallowNode(node)
    {
        throwIfNotValid();
    }

    bool isValid() const override
    {
        return _shallowNode != nullptr;
    }

    const hipdnn_flatbuffers_sdk::data_objects::Node& node() const override
    {
        return *_shallowNode;
    }

    const void* attributes() const override
    {
        return _shallowNode->attributes();
    }

    hipdnn_flatbuffers_sdk::data_objects::NodeAttributes attributesType() const override
    {
        return _shallowNode->attributes_type();
    }

    std::string name() const override
    {
        const auto& n = node();
        return n.name() != nullptr ? n.name()->str() : "";
    }

    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType() const override
    {
        return _shallowNode->compute_data_type();
    }

private:
    void throwIfNotValid() const
    {
        if(!isValid())
        {
            throw std::invalid_argument("Node is null");
        }
    }

    const hipdnn_flatbuffers_sdk::data_objects::Node* _shallowNode = nullptr;
};

} // namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities
