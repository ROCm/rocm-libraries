// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormBwdPlan.hpp>

namespace hipdnn_sdk::test_utilities
{

struct BatchnormBwdSignatureKey
{
    const hipdnn_sdk::data_objects::NodeAttributes nodeType
        = hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes;
    hipdnn_sdk::data_objects::DataType inputDataType;
    hipdnn_sdk::data_objects::DataType scaleBiasDataType;
    hipdnn_sdk::data_objects::DataType meanVarianceDataType;

    BatchnormBwdSignatureKey() = default;
    constexpr BatchnormBwdSignatureKey(hipdnn_sdk::data_objects::DataType input,
                                       hipdnn_sdk::data_objects::DataType scaleBias,
                                       hipdnn_sdk::data_objects::DataType meanVariance)
        : inputDataType(input)
        , scaleBiasDataType(scaleBias)
        , meanVarianceDataType(meanVariance)
    {
    }

    BatchnormBwdSignatureKey(
        const hipdnn_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        const auto* nodeAttributes = node.attributes_as_BatchnormBackwardAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes could not be cast to BatchnormBackwardAttributes");
        }

        auto xTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
        auto meanTensorAttr = tensorMap.at(nodeAttributes->mean_tensor_uid().value());
        auto scaleTensorAttr = tensorMap.at(nodeAttributes->scale_tensor_uid());

        if(xTensorAttr == nullptr || meanTensorAttr == nullptr || scaleTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        inputDataType = xTensorAttr->data_type();
        scaleBiasDataType = scaleTensorAttr->data_type();
        meanVarianceDataType = meanTensorAttr->data_type();
    }

    std::size_t operator()(const BatchnormBwdSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(inputDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(scaleBiasDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(meanVarianceDataType)) << 12);
    }

    bool operator==(const BatchnormBwdSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && inputDataType == other.inputDataType
               && scaleBiasDataType == other.scaleBiasDataType
               && meanVarianceDataType == other.meanVarianceDataType;
    }

    static std::unordered_map<BatchnormBwdSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              BatchnormBwdSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<BatchnormBwdSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           BatchnormBwdSignatureKey>
            map;

        addPlanBuilder<hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::HALF>(map);
        addPlanBuilder<hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::BFLOAT16>(map);

        return map;
    }

    template <hipdnn_sdk::data_objects::DataType InputDataTypeEnum,
              hipdnn_sdk::data_objects::DataType ScaleBiasDataTypeEnum,
              hipdnn_sdk::data_objects::DataType MeanVarianceDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<BatchnormBwdSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  BatchnormBwdSignatureKey>& map)
    {
        map[BatchnormBwdSignatureKey(
            InputDataTypeEnum, ScaleBiasDataTypeEnum, MeanVarianceDataTypeEnum)]
            = std::make_unique<BatchnormBwdPlanBuilder<InputDataTypeEnum,
                                                       ScaleBiasDataTypeEnum,
                                                       MeanVarianceDataTypeEnum>>();
    }
};

}
