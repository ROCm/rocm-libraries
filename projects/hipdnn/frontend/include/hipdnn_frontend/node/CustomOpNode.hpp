// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/CustomOpAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>

namespace hipdnn_frontend::graph
{
class CustomOpNode : public INode
{
public:
    CustomOpAttributes attributes;

    CustomOpNode(CustomOpAttributes&& customOpAttributes, const GraphAttributes& graphAttrs)
        : INode(graphAttrs)
        , attributes(std::move(customOpAttributes))
    {
    }

    std::string getNodeName() const override
    {
        return attributes.get_name();
    }

    void gather_hipdnn_tensors(
        std::unordered_set<std::shared_ptr<TensorAttributes>>& allTensors) const override
    {
        for(const auto& tensor : attributes.inputs)
        {
            if(tensor)
            {
                allTensors.insert(tensor);
            }
        }

        for(const auto& tensor : attributes.outputs)
        {
            if(tensor)
            {
                allTensors.insert(tensor);
            }
        }
    }

    std::vector<std::shared_ptr<TensorAttributes>> getNodeInputTensorAttributes() const override
    {
        std::vector<std::shared_ptr<TensorAttributes>> result;
        for(const auto& tensor : attributes.inputs)
        {
            if(tensor)
            {
                result.push_back(tensor);
            }
        }
        return result;
    }

    std::vector<std::shared_ptr<TensorAttributes>> getNodeOutputTensorAttributes() const override
    {
        std::vector<std::shared_ptr<TensorAttributes>> result;
        for(const auto& tensor : attributes.outputs)
        {
            if(tensor)
            {
                result.push_back(tensor);
            }
        }
        return result;
    }

    Error pre_validate_node() const override
    {
        for(size_t i = 0; i < attributes.inputs.size(); ++i)
        {
            if(!attributes.inputs[i])
            {
                return {ErrorCode::INVALID_VALUE,
                        "Node " + attributes.name + " has null input tensor at index "
                            + std::to_string(i)};
            }
        }

        for(size_t i = 0; i < attributes.outputs.size(); ++i)
        {
            if(!attributes.outputs[i])
            {
                return {ErrorCode::INVALID_VALUE,
                        "Node " + attributes.name + " has null output tensor at index "
                            + std::to_string(i)};
            }
        }

        return {ErrorCode::OK, ""};
    }

    Error infer_properties_node() override
    {
        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));
        return {};
    }

    Error post_validate_node() const override
    {
        if(attributes.custom_op_id.empty())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "Node " + attributes.name + " does not have a custom_op_id set"};
        }

        if(attributes.compute_data_type == DataType::NOT_SET)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "Node " + attributes.name + " does not have a compute_data_type set"};
        }

        for(const auto& tensorAttr : getNodeOutputTensorAttributes())
        {
            HIPDNN_CHECK_ERROR(tensorAttr->validate());
        }

        return {ErrorCode::OK, ""};
    }

    flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>
        pack_node(flatbuffers::FlatBufferBuilder& builder) const override
    {
        return hipdnn_data_sdk::data_objects::CreateNodeDirect(
            builder,
            attributes.get_name().c_str(),
            toSdkType(attributes.compute_data_type),
            hipdnn_data_sdk::data_objects::NodeAttributes::CustomOpAttributes,
            attributes.pack_attributes(builder).Union());
    }
};
} // namespace hipdnn_frontend::graph
