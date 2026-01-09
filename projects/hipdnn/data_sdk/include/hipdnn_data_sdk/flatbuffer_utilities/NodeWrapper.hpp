// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <memory>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>

namespace hipdnn_plugin_sdk
{

class INodeWrapper
{
public:
    virtual ~INodeWrapper() = default;
    virtual bool isValid() const = 0;
    virtual const hipdnn_data_sdk::data_objects::Node& node() const = 0;

    virtual const void* attributes() const = 0;
    virtual hipdnn_data_sdk::data_objects::NodeAttributes attributesType() const = 0;
    virtual const std::type_info& attributesClassType() const = 0;
    virtual std::string name() const = 0;
    virtual hipdnn_data_sdk::data_objects::DataType computeDataType() const = 0;

    template <typename T>
    const T& attributesAs() const
    {
        if(attributesClassType() != typeid(T))
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
    explicit NodeWrapper(const hipdnn_data_sdk::data_objects::Node* node)
        : _shallowNode(node)
    {
        throwIfNotValid();
    }

    bool isValid() const override
    {
        return _shallowNode != nullptr;
    }

    const hipdnn_data_sdk::data_objects::Node& node() const override
    {
        return *_shallowNode;
    }

    const void* attributes() const override
    {
        return _shallowNode->attributes();
    }

    hipdnn_data_sdk::data_objects::NodeAttributes attributesType() const override
    {
        return _shallowNode->attributes_type();
    }

    const std::type_info& attributesClassType() const override
    {
        switch(attributesType())
        {
        case hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes:
            return typeid(hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes);
        case hipdnn_data_sdk::data_objects::NodeAttributes::PointwiseAttributes:
            return typeid(hipdnn_data_sdk::data_objects::PointwiseAttributes);
        case hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes:
            return typeid(hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes);
        case hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormAttributes:
            return typeid(hipdnn_data_sdk::data_objects::BatchnormAttributes);
        case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes:
            return typeid(hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes);
        case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes:
            return typeid(hipdnn_data_sdk::data_objects::ConvolutionBwdAttributes);
        case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes:
            return typeid(hipdnn_data_sdk::data_objects::ConvolutionWrwAttributes);
        case hipdnn_data_sdk::data_objects::NodeAttributes::MatmulAttributes:
            return typeid(hipdnn_data_sdk::data_objects::MatmulAttributes);
        default:
            throw std::invalid_argument("Node attributes type is not recognized");
        }
    }

    std::string name() const override
    {
        const auto& n = node();
        return n.name() != nullptr ? n.name()->str() : "";
    }

    hipdnn_data_sdk::data_objects::DataType computeDataType() const override
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

    const hipdnn_data_sdk::data_objects::Node* _shallowNode = nullptr;
};
}
