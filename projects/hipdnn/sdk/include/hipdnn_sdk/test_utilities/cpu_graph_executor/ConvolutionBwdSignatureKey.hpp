// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/ConvolutionBwdPlan.hpp>

namespace hipdnn_sdk::test_utilities
{

struct ConvolutionBwdSignatureKey
{
    const hipdnn_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes};
    hipdnn_sdk::data_objects::DataType inputDataType;
    hipdnn_sdk::data_objects::DataType accumulatorDataType;

    ConvolutionBwdSignatureKey() = default;

    constexpr ConvolutionBwdSignatureKey(hipdnn_sdk::data_objects::DataType input,
                                         hipdnn_sdk::data_objects::DataType accumulator)
        : inputDataType(input)
        , accumulatorDataType(accumulator)
    {
    }

    ConvolutionBwdSignatureKey(
        const hipdnn_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_sdk::data_objects::DataType computeType)
    {
        const auto* nodeAttributes = node.attributes_as_ConvolutionBwdAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes could not be cast to ConvolutionBwdAttributes");
        }

        auto xTensorAttr = tensorMap.at(nodeAttributes->dx_tensor_uid());

        if(xTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        inputDataType = xTensorAttr->data_type();
        accumulatorDataType = computeType;
    }

    std::size_t operator()(const ConvolutionBwdSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(inputDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(accumulatorDataType)) << 8);
    }

    bool operator==(const ConvolutionBwdSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && inputDataType == other.inputDataType
               && accumulatorDataType == other.accumulatorDataType;
    }

    static std::unordered_map<ConvolutionBwdSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              ConvolutionBwdSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<ConvolutionBwdSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           ConvolutionBwdSignatureKey>
            map;

        addPlanBuilder<hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::FLOAT>(map);

        return map;
    }

    template <hipdnn_sdk::data_objects::DataType InputDataTypeEnum,
              hipdnn_sdk::data_objects::DataType AccumulatorDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<ConvolutionBwdSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  ConvolutionBwdSignatureKey>& map)
    {
        map[ConvolutionBwdSignatureKey(InputDataTypeEnum, AccumulatorDataTypeEnum)]
            = std::make_unique<
                ConvolutionBwdPlanBuilder<InputDataTypeEnum, AccumulatorDataTypeEnum>>();
    }
};

}
