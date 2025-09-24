// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>

namespace hipdnn_sdk
{
namespace test_utilities
{

struct BatchnormSignatureRegistryKey
{
    bool operator==(const BatchnormSignatureRegistryKey& other) const
    {
        return inputDataType == other.inputDataType && scaleBiasDataType == other.scaleBiasDataType
               && meanVarianceDataType == other.meanVarianceDataType;
    }

    bool operator!=(const BatchnormSignatureRegistryKey& other) const
    {
        return !(*this == other);
    }

    //fix these names
    hipdnn_sdk::data_objects::DataType inputDataType;
    hipdnn_sdk::data_objects::DataType scaleBiasDataType;
    hipdnn_sdk::data_objects::DataType meanVarianceDataType;
};

struct BatchnormSignatureRegistryKeyHash
{
    std::size_t operator()(const BatchnormSignatureRegistryKey& k) const
    {
        return std::hash<int>()(static_cast<int>(k.inputDataType))
               ^ (std::hash<int>()(static_cast<int>(k.scaleBiasDataType)) << 1)
               ^ (std::hash<int>()(static_cast<int>(k.meanVarianceDataType)) << 2);
    }
};

} // namespace test_utilities
} // namespace hipdnn_sdk
