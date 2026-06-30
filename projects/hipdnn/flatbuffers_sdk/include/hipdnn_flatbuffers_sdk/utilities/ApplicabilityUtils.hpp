// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

namespace hipdnn_flatbuffers_sdk::utilities
{

inline std::vector<int64_t> listUnsupportedRaggedTensorIds(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    const std::vector<int64_t>& supportedRaggedIds = {})
{
    std::vector<int64_t> unsupportedRaggedIds;

    auto isSupportedRaggedTensor = [&](int64_t id) {
        return std::find(supportedRaggedIds.begin(), supportedRaggedIds.end(), id)
               != supportedRaggedIds.end();
    };

    for(auto& [id, attrs] : tensorMap)
    {
        if(attrs->ragged_offset_tensor_uid().has_value() && isSupportedRaggedTensor(id))
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
