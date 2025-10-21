// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hipdnn_frontend
{
namespace graph
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
    virtual Error populate_hipdnn_tensor_ids( // NOLINT(readability-identifier-naming)
        [[maybe_unused]] std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>&
            tensorLookup,
        [[maybe_unused]] int64_t& currentTensorId,
        [[maybe_unused]] std::unordered_set<int64_t>& usedIds) const
    {
        return {};
    }
    virtual void
        // NOLINTNEXTLINE(readability-identifier-naming)
        gather_hipdnn_tensor_ids([[maybe_unused]] std::unordered_set<int64_t>& usedIds,
                                 [[maybe_unused]] std::unordered_set<int64_t>& duplicateIds) const {
        };

    virtual flatbuffers::Offset<hipdnn_sdk::data_objects::Node>
        pack_node([[maybe_unused]] flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        return {};
    }

    virtual std::vector<std::shared_ptr<TensorAttributes>> getNodeInputTensorAttributes() const
    {
        return {};
    }

    virtual std::vector<std::shared_ptr<TensorAttributes>> getNodeOutputTensorAttributes() const
    {
        return {};
    }

    const std::vector<std::shared_ptr<INode>>& getSubNodes() const
    {
        return _sub_nodes;
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

    void gatherHipdnnTensorIdsSubtree(std::unordered_set<int64_t>& usedIds,
                                      std::unordered_set<int64_t>& duplicateIds) const
    {
        gather_hipdnn_tensor_ids(usedIds, duplicateIds);

        for(const auto& node : _sub_nodes)
        {
            node->gatherHipdnnTensorIdsSubtree(usedIds, duplicateIds);
        }
    }

    Error populateHipdnnTensorIdsSubtree(
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorLookup,
        int64_t& currentTensorId,
        std::unordered_set<int64_t>& usedIds)
    {
        HIPDNN_CHECK_ERROR(populate_hipdnn_tensor_ids(tensorLookup, currentTensorId, usedIds));
        for(const auto& node : _sub_nodes)
        {
            HIPDNN_CHECK_ERROR(
                node->populate_hipdnn_tensor_ids(tensorLookup, currentTensorId, usedIds));
        }
        return {};
    }

    static void processTensorUid(const std::shared_ptr<TensorAttributes>& tensor,
                                 std::unordered_set<int64_t>& usedIds,
                                 std::unordered_set<int64_t>& duplicateIds)
    {
        if(tensor && tensor->has_uid())
        {
            if(usedIds.find(tensor->get_uid()) != usedIds.end())
            {
                duplicateIds.insert(tensor->get_uid());
            }
            usedIds.insert(tensor->get_uid());
        }
    }
};

// Any class extending BaseNode must have an attributes member with an inputs & outputs map.
// The map needs to have TensorAttributes as the value.
// BaseNode uses this to gather tensor uids, and populate unset ones.
template <typename DerivedT>
class BaseNode : public INode
{
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
    // NOLINTNEXTLINE(readability-identifier-naming)
    void gather_hipdnn_tensor_ids(
        [[maybe_unused]] std::unordered_set<int64_t>& usedIds,
        [[maybe_unused]] std::unordered_set<int64_t>& duplicateIds) const override
    {
        for(auto& [_, tensor] : self().attributes.inputs)
        {
            processTensorUid(tensor, usedIds, duplicateIds);
        }

        for(auto& [_, tensor] : self().attributes.outputs)
        {
            processTensorUid(tensor, usedIds, duplicateIds);
        }
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    static int64_t get_unused_tensor_uid(int64_t& currentTensorId,
                                         std::unordered_set<int64_t>& usedIds)
    {
        while(usedIds.find(currentTensorId) != usedIds.end())
        {
            ++currentTensorId;
        }
        usedIds.insert(currentTensorId);
        return currentTensorId++;
    }

    // NOLINT(readability-identifier-naming)
    Error populate_hipdnn_tensor_ids(
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorLookup,
        int64_t& currentTensorId,
        std::unordered_set<int64_t>& usedIds) const override
    {
        for(auto& [_, tensor] : self().attributes.inputs)
        {
            if(tensor && !tensor->has_uid())
            {
                tensor->set_uid(get_unused_tensor_uid(currentTensorId, usedIds));
            }

            tensorLookup[tensor->get_uid()] = tensor;
        }

        for(auto& [_, tensor] : self().attributes.outputs)
        {
            if(!tensor)
            {
                continue;
            }

            if(!tensor->has_uid())
            {
                tensor->set_uid(get_unused_tensor_uid(currentTensorId, usedIds));
            }
            tensorLookup[tensor->get_uid()] = tensor;
        }

        return {};
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

protected:
    using INode::INode;
};

template <typename DerivedT>
using NodeCRTP = BaseNode<DerivedT>; // NOLINT
}
}
