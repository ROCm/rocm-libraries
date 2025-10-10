// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <memory>

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/PluginException.hpp>

namespace hipdnn_plugin
{

class INodeWrapper
{
public:
    virtual ~INodeWrapper() = default;
    virtual bool isValid() const = 0;
    virtual const hipdnn_sdk::data_objects::Node& node() const = 0;

    virtual const void* attributes() const = 0;
    virtual hipdnn_sdk::data_objects::NodeAttributes attributesType() const = 0;
    virtual const std::type_info& attributesClassType() const = 0;

    template <typename T>
    const T& attributesAs() const
    {
        if(attributesClassType() != typeid(T))
        {
            throw hipdnn_plugin::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "Node attributes are not of the expected type");
        }

        auto* attr = attributes();
        if(attr == nullptr)
        {
            throw hipdnn_plugin::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "Node attributes are null");
        }

        return *static_cast<const T*>(attr);
    }
};

class NodeWrapper : public INodeWrapper
{
public:
    explicit NodeWrapper(const hipdnn_sdk::data_objects::Node* node)
        : _shallowNode(node)
    {
        throwIfNotValid();
    }

    bool isValid() const override
    {
        return _shallowNode != nullptr;
    }

    const hipdnn_sdk::data_objects::Node& node() const override
    {
        throwIfNotValid();
        return *_shallowNode;
    }

    const void* attributes() const override
    {
        throwIfNotValid();
        return _shallowNode->attributes();
    }

    hipdnn_sdk::data_objects::NodeAttributes attributesType() const override
    {
        throwIfNotValid();
        return _shallowNode->attributes_type();
    }

    const std::type_info& attributesClassType() const override
    {
        throwIfNotValid();
        switch(attributesType())
        {
        case hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes:
            return typeid(hipdnn_sdk::data_objects::BatchnormInferenceAttributes);
        case hipdnn_sdk::data_objects::NodeAttributes::PointwiseAttributes:
            return typeid(hipdnn_sdk::data_objects::PointwiseAttributes);
        case hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes:
            return typeid(hipdnn_sdk::data_objects::BatchnormBackwardAttributes);
        case hipdnn_sdk::data_objects::NodeAttributes::BatchnormAttributes:
            return typeid(hipdnn_sdk::data_objects::BatchnormAttributes);
        case hipdnn_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes:
            return typeid(hipdnn_sdk::data_objects::ConvolutionFwdAttributes);
        case hipdnn_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes:
            return typeid(hipdnn_sdk::data_objects::ConvolutionBwdAttributes);
        case hipdnn_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes:
            return typeid(hipdnn_sdk::data_objects::ConvolutionWrwAttributes);
        default:
            throw hipdnn_plugin::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "Node attributes type is not recognized");
        }
    }

private:
    void throwIfNotValid() const
    {
        if(!isValid())
        {
            throw hipdnn_plugin::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "Node is null");
        }
    }

    const hipdnn_sdk::data_objects::Node* _shallowNode = nullptr;
};
}
