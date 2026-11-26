// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace hipdnn_frontend::detail
{

// Collects graph tensors without duplicates and iterates them in insertion order.
// Everything derived from that order, such as the UIDs populateHipdnnTensorIds()
// assigns, follows the order in which the nodes insert their tensors. Within a node,
// inputs and outputs follow the iteration order of the attributes' std::unordered_map,
// which depends on the standard library and on the order in which the attribute setters
// were called. Identical graphs get the same UIDs only when they are built with the same
// setter order and the same standard library.
class OrderedTensorSet
{
public:
    using value_type = std::shared_ptr<graph::TensorAttributes>;
    using const_iterator = std::vector<value_type>::const_iterator;

    void insert(const value_type& tensor)
    {
        if(_members.insert(tensor.get()).second)
        {
            _tensors.push_back(tensor);
        }
    }

    size_t count(const value_type& tensor) const
    {
        return _members.count(tensor.get());
    }

    const_iterator find(const value_type& tensor) const
    {
        return count(tensor) == 0 ? end() : std::find(begin(), end(), tensor);
    }

    size_t size() const
    {
        return _tensors.size();
    }

    const_iterator begin() const
    {
        return _tensors.begin();
    }

    const_iterator end() const
    {
        return _tensors.end();
    }

private:
    std::vector<value_type> _tensors;
    std::unordered_set<const graph::TensorAttributes*> _members;
};

inline std::unordered_set<int64_t> getUsedIds(const OrderedTensorSet& allTensors)
{
    std::unordered_set<int64_t> usedIds;
    for(const auto& tensor : allTensors)
    {
        if(tensor && tensor->has_uid())
        {
            usedIds.insert(tensor->get_uid());
        }
    }
    return usedIds;
}

inline int64_t getUnusedTensorUid(int64_t& currentTensorId, std::unordered_set<int64_t>& usedIds)
{
    while(usedIds.find(currentTensorId) != usedIds.end())
    {
        ++currentTensorId;
    }
    usedIds.insert(currentTensorId);
    return currentTensorId++;
}

inline void populateHipdnnTensorIds(const OrderedTensorSet& allTensors,
                                    std::unordered_set<int64_t>& usedIds)
{
    int64_t currentTensorId = 0;

    for(const auto& tensor : allTensors)
    {
        if(!tensor)
        {
            continue;
        }

        if(!tensor->has_uid())
        {
            tensor->set_uid(getUnusedTensorUid(currentTensorId, usedIds));
        }
    }
}

inline Error checkTensorUidsSetImpl(const OrderedTensorSet& allTensors)
{
    std::vector<std::string> missingUidTensors;

    for(const auto& tensor : allTensors)
    {
        if(tensor && !tensor->has_uid())
        {
            auto name = tensor->get_name();
            missingUidTensors.push_back(name.empty() ? "(unnamed)" : name);
        }
    }

    if(!missingUidTensors.empty())
    {
        std::string errorMsg = "Tensors without UIDs: ";
        for(const auto& name : missingUidTensors)
        {
            errorMsg += name + ", ";
        }
        errorMsg.pop_back();
        errorMsg.pop_back();
        return {ErrorCode::ATTRIBUTE_NOT_SET, errorMsg};
    }

    return {ErrorCode::OK, ""};
}

inline Error checkNoDuplicateTensorIdsImpl(const OrderedTensorSet& allTensors)
{
    std::unordered_set<int64_t> seenUids;
    std::unordered_set<int64_t> duplicateUids;

    for(const auto& tensor : allTensors)
    {
        if(tensor && tensor->has_uid())
        {
            auto uid = tensor->get_uid();
            if(!seenUids.insert(uid).second)
            {
                duplicateUids.insert(uid);
            }
        }
    }

    if(!duplicateUids.empty())
    {
        std::string errorMsg = "Duplicate tensor UIDs found in the graph: ";
        for(const auto& uid : duplicateUids)
        {
            errorMsg += std::to_string(uid) + ", ";
        }
        errorMsg.erase(errorMsg.length() - 2);
        return {ErrorCode::INVALID_VALUE, errorMsg};
    }

    return {ErrorCode::OK, ""};
}

} // namespace hipdnn_frontend::detail
