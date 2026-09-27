// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace hipdnn_frontend::detail
{

inline std::unordered_set<int64_t>
    getUsedIds(const std::unordered_set<std::shared_ptr<graph::TensorAttributes>>& allTensors)
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

inline void populateHipdnnTensorIds(
    const std::unordered_set<std::shared_ptr<graph::TensorAttributes>>& allTensors,
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

inline Error checkTensorUidsSetImpl(
    const std::unordered_set<std::shared_ptr<graph::TensorAttributes>>& allTensors)
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

inline Error checkNoDuplicateTensorIdsImpl(
    const std::unordered_set<std::shared_ptr<graph::TensorAttributes>>& allTensors)
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
