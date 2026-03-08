// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "HipdnnException.hpp"
#include "TensorDescriptor.hpp"
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace hipdnn_backend
{

// Builds a tensor lookup map from a vector of FlatBuffer TensorAttributesT.
// Each tensor is deserialized via TensorDescriptor::fromFlatBuffer() and indexed by UID.
// Throws on null tensors or duplicate UIDs.
//
// Related utility: findTensorInMap() in DescriptorAttributeUtils.hpp retrieves a tensor
// from the map by UID with error checking.
inline std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> buildTensorMap(
    const std::vector<std::unique_ptr<hipdnn_data_sdk::data_objects::TensorAttributesT>>& tensors)
{
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> tensorMap;
    for(const auto& tensorT : tensors)
    {
        THROW_IF_NULL(
            tensorT, HIPDNN_STATUS_INTERNAL_ERROR, "buildTensorMap: null tensor in graph");

        THROW_IF_TRUE(tensorMap.count(tensorT->uid) > 0,
                      HIPDNN_STATUS_INTERNAL_ERROR,
                      "buildTensorMap: duplicate tensor UID " + std::to_string(tensorT->uid)
                          + " in graph");

        tensorMap[tensorT->uid] = TensorDescriptor::fromFlatBuffer(*tensorT);
    }
    return tensorMap;
}

} // namespace hipdnn_backend
