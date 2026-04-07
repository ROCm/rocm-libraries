// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <functional>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/NodeType.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hipdnn_frontend::graph
{

class INode
{
public:
    GraphAttributes graph_attributes; // NOLINT(readability-identifier-naming)
    INode(GraphAttributes attributes)
        : graph_attributes(std::move(attributes))
    {
    }
    virtual ~INode() = default;

    // Disable copy operations
    INode(const INode&) = delete;
    INode& operator=(const INode&) = delete;

    // Enable move operations
    INode(INode&&) = default;
    INode& operator=(INode&&) = default;

    virtual Error pre_validate_node() const // NOLINT(readability-identifier-naming)
    {
        return {};
    }
    virtual Error infer_properties_node() // NOLINT(readability-identifier-naming)
    {
        return {};
    }
    virtual Error post_validate_node() const // NOLINT(readability-identifier-naming)
    {
        return {};
    }
    virtual std::string getNodeName() const
    {
        return {};
    }

    virtual NodeType getNodeType() const
    {
        return NodeType::UNKNOWN;
    }

    virtual void
        // NOLINTNEXTLINE(readability-identifier-naming)
        gather_hipdnn_tensors(
            [[maybe_unused]] std::unordered_set<std::shared_ptr<TensorAttributes>>& allTensors)
            const
    {
    }

    /// Unpacks operation attributes from a backend descriptor into this node.
    /// Subclasses that support unpacking from the C-API must override this.
    // NOLINTNEXTLINE(readability-identifier-naming)
    virtual Error unpack_from_descriptor(
        [[maybe_unused]] hipdnnBackendDescriptor_t opDesc,
        [[maybe_unused]] std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap)
    {
        auto nodeName = getNodeName();
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "unpack_from_descriptor not implemented for node"
                    + (nodeName.empty() ? std::string{} : ": " + nodeName)};
    }

    // Creates backend operation descriptor(s) for this node using the C-API.
    // Tensor descriptors are deduplicated by UID in tensorDescs.
    // Container nodes (e.g. Graph) do not override this — they delegate to child nodes.
    // NOLINTNEXTLINE(readability-identifier-naming)
    virtual Error create_operation(
        [[maybe_unused]] std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>&
            tensorDescs,
        [[maybe_unused]] std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "create_operation not implemented for node: " + getNodeName()};
    }

    virtual std::vector<std::shared_ptr<TensorAttributes>> getNodeInputTensorAttributes() const
    {
        return {};
    }

    virtual std::vector<std::shared_ptr<TensorAttributes>> getNodeOutputTensorAttributes() const
    {
        return {};
    }

    void visit(const std::function<void(INode&)>& visitor)
    {
        // Visit current node first (pre-order traversal)
        visitor(*this);

        // Then visit all children
        for(const auto& child : _sub_nodes)
        {
            if(child)
            {
                child->visit(visitor);
            }
        }
    }

    void visit(const std::function<void(const INode&)>& visitor) const
    {
        // Visit current node first (pre-order traversal)
        visitor(*this);

        // Then visit all children
        for(const auto& child : _sub_nodes)
        {
            if(child)
            {
                // Explicitly call const version by getting const reference
                const INode& constChild = *child;
                constChild.visit(visitor);
            }
        }
    }

protected:
    std::vector<std::shared_ptr<INode>> _sub_nodes;

    Error validateSubtree()
    {
        HIPDNN_CHECK_ERROR(pre_validate_node());
        HIPDNN_CHECK_ERROR(infer_properties_node());
        for(const auto& node : _sub_nodes)
        {
            HIPDNN_CHECK_ERROR(node->validateSubtree());
        }
        HIPDNN_CHECK_ERROR(post_validate_node());
        return {};
    }

    void gatherHipdnnTensorsSubtree(
        std::unordered_set<std::shared_ptr<TensorAttributes>>& allTensors) const
    {
        gather_hipdnn_tensors(allTensors);

        for(const auto& node : _sub_nodes)
        {
            node->gatherHipdnnTensorsSubtree(allTensors);
        }
    }
};

// Any class extending BaseNode must have an attributes member with an inputs & outputs map.
// The map needs to have TensorAttributes as the value.
// BaseNode uses this to gather tensor uids, and populate unset ones.
template <typename DerivedT, NodeType Type = NodeType::UNKNOWN>
class BaseNode : public INode
{
    friend DerivedT;

private:
    DerivedT& self()
    {
        return static_cast<DerivedT&>(*this);
    }
    const DerivedT& self() const
    {
        return static_cast<const DerivedT&>(*this);
    }

public:
    NodeType getNodeType() const override
    {
        return Type;
    }

    std::string getNodeName() const override
    {
        return std::string(self().attributes.get_name());
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    void gather_hipdnn_tensors(
        std::unordered_set<std::shared_ptr<TensorAttributes>>& allTensors) const override
    {
        for(auto& [_, tensor] : self().attributes.inputs)
        {
            if(tensor)
            {
                allTensors.insert(tensor);
            }
        }

        for(auto& [_, tensor] : self().attributes.outputs)
        {
            if(tensor)
            {
                allTensors.insert(tensor);
            }
        }
    }

    Error post_validate_node() const override // NOLINT(readability-identifier-naming)
    {
        if(self().attributes.compute_data_type == DataType::NOT_SET)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "Node " + self().attributes.name + " does not have a compute_data_type set"};
        }

        for(const auto& tensorAttr : getNodeOutputTensorAttributes())
        {
            HIPDNN_CHECK_ERROR(tensorAttr->validate());
        }

        return {ErrorCode::OK, ""};
    }

    std::vector<std::shared_ptr<TensorAttributes>> getNodeInputTensorAttributes() const override
    {
        std::vector<std::shared_ptr<TensorAttributes>> inputAttributes;
        for(auto& tensorAttrPair : self().attributes.inputs)
        {
            if(tensorAttrPair.second)
            {
                inputAttributes.push_back(tensorAttrPair.second);
            }
        }

        return inputAttributes;
    }

    std::vector<std::shared_ptr<TensorAttributes>> getNodeOutputTensorAttributes() const override
    {
        std::vector<std::shared_ptr<TensorAttributes>> outputAttributes;
        for(auto& tensorAttrPair : self().attributes.outputs)
        {
            if(tensorAttrPair.second)
            {
                outputAttributes.push_back(tensorAttrPair.second);
            }
        }

        return outputAttributes;
    }

private:
    BaseNode() = default;

protected:
    using INode::INode;
};

template <typename DerivedT, NodeType Type = NodeType::UNKNOWN>
using NodeCRTP = BaseNode<DerivedT, Type>; // NOLINT
} // namespace hipdnn_frontend::graph
