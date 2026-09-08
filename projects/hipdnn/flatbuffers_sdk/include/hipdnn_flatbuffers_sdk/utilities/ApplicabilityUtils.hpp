// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

namespace hipdnn_flatbuffers_sdk::utilities
{

inline std::vector<int64_t> listUnsupportedRaggedTensorIds(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    const std::unordered_set<int64_t>& supportedRaggedIds = {})
{
    std::vector<int64_t> unsupportedRaggedIds;

    for(auto& [id, attrs] : tensorMap)
    {
        if(attrs->ragged_offset_tensor_uid().has_value() && supportedRaggedIds.count(id) == 0)
        {
            unsupportedRaggedIds.push_back(id);
        }
    }

    return unsupportedRaggedIds;
}

inline bool hasNoRaggedTensorIds(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    return listUnsupportedRaggedTensorIds(tensorMap, {}).empty();
}

}
