// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BaseSignatureKey.hpp>

namespace hipdnn_sdk::test_utilities
{

struct BatchnormSignatureRegistryKey
{
    hipdnn_sdk::data_objects::NodeAttributes nodeType;
    hipdnn_sdk::data_objects::DataType inputDataType;
    hipdnn_sdk::data_objects::DataType scaleBiasDataType;
    hipdnn_sdk::data_objects::DataType meanVarianceDataType;

    constexpr BatchnormSignatureRegistryKey(hipdnn_sdk::data_objects::DataType input,
                                            hipdnn_sdk::data_objects::DataType scaleBias,
                                            hipdnn_sdk::data_objects::DataType meanVariance)
        : nodeType(hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes)
        , inputDataType(input)
        , scaleBiasDataType(scaleBias)
        , meanVarianceDataType(meanVariance)
    {
    }

    constexpr std::size_t hash_self() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(inputDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(scaleBiasDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(meanVarianceDataType)) << 12);
    }

    constexpr bool equal(const BatchnormSignatureRegistryKey& other) const
    {
        return nodeType == other.nodeType && inputDataType == other.inputDataType
               && scaleBiasDataType == other.scaleBiasDataType
               && meanVarianceDataType == other.meanVarianceDataType;
    }
};

constexpr std::array<BatchnormSignatureRegistryKey, 2> ALL_SUPPORTED_BATCHNORM_SIGNATURES
    = {BatchnormSignatureRegistryKey(hipdnn_sdk::data_objects::DataType::FLOAT,
                                     hipdnn_sdk::data_objects::DataType::FLOAT,
                                     hipdnn_sdk::data_objects::DataType::FLOAT),
       BatchnormSignatureRegistryKey(hipdnn_sdk::data_objects::DataType::HALF,
                                     hipdnn_sdk::data_objects::DataType::HALF,
                                     hipdnn_sdk::data_objects::DataType::HALF)};

}
