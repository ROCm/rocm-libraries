// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/CustomOpAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/detail/CustomOpPacker.hpp>
#include <hipdnn_frontend/detail/CustomOpUnpacker.hpp>

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

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        CustomOpAttributes customOpAttr;
        HIPDNN_CHECK_ERROR(detail::unpackCustomOpOperation(opDesc, tensorMap, customOpAttr));
        attributes = std::move(customOpAttr);
        return {};
    }

    NodeType getNodeType() const override
    {
        return NodeType::CUSTOM_OP;
    }

    std::string getNodeName() const override
    {
        return attributes.get_name();
    }

    void gather_hipdnn_tensors(detail::OrderedTensorSet& allTensors) const override
    {
        for(const auto& tensor : attributes.inputs)
        {
            detail::insertTensorWithAux(allTensors, tensor);
        }

        for(const auto& tensor : attributes.outputs)
        {
            detail::insertTensorWithAux(allTensors, tensor);
        }
    }

    /// @brief Logical equality: same node type and logically equal CustomOpAttributes.
    bool logicallyEquals(const INode& other) const override
    {
        if(getNodeType() != other.getNodeType())
        {
            return false;
        }

        return attributes.logicallyEquals(static_cast<const CustomOpNode&>(other).attributes);
    }

    /// @brief Strict equality: same node type and strictly equal CustomOpAttributes.
    bool operator==(const INode& other) const override
    {
        if(getNodeType() != other.getNodeType())
        {
            return false;
        }

        return attributes == static_cast<const CustomOpNode&>(other).attributes;
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
                    "Node " + attributes.name
                        + " does not have a compute_data_type set. Set the graph data types "
                          "before adding nodes: a node copies them when it is added."};
        }

        for(const auto& tensorAttr : getNodeOutputTensorAttributes())
        {
            HIPDNN_CHECK_ERROR(tensorAttr->validate());
        }

        return {ErrorCode::OK, ""};
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createCustomOpOperation(attributes, tensorDescs, operations);
    }
};
} // namespace hipdnn_frontend::graph
