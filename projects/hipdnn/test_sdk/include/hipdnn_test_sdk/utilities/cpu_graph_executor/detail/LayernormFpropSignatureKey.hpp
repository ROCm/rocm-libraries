// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/LayernormFpropPlan.hpp>
#include <ostream>

namespace hipdnn_test_sdk::detail
{

struct LayernormFpropSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType
        = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::LayernormAttributes;
    hipdnn_flatbuffers_sdk::data_objects::DataType xDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType scaleBiasDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType meanInvVarianceDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType outputDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType;

    LayernormFpropSignatureKey() = default;
    constexpr LayernormFpropSignatureKey(
        hipdnn_flatbuffers_sdk::data_objects::DataType x,
        hipdnn_flatbuffers_sdk::data_objects::DataType scaleBias,
        hipdnn_flatbuffers_sdk::data_objects::DataType meanInvVariance,
        hipdnn_flatbuffers_sdk::data_objects::DataType output,
        hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : xDataType(x)
        , scaleBiasDataType(scaleBias)
        , meanInvVarianceDataType(meanInvVariance)
        , outputDataType(output)
        , computeDataType(compute)
    {
    }

    LayernormFpropSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        const auto* nodeAttributes = node.attributes_as_LayernormAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes could not be cast to LayernormAttributes");
        }

        auto xTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
        auto yTensorAttr = tensorMap.at(nodeAttributes->y_tensor_uid());

        if(xTensorAttr == nullptr || yTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        xDataType = xTensorAttr->data_type();
        outputDataType = yTensorAttr->data_type();
        computeDataType = node.compute_data_type();

        // Scale/bias type: use scale tensor type
        auto scaleTensorAttr = tensorMap.at(nodeAttributes->scale_tensor_uid());
        scaleBiasDataType = scaleTensorAttr->data_type();

        // Mean/inv_variance type: use mean if present, otherwise default to IO type (x type)
        if(nodeAttributes->mean_tensor_uid().has_value())
        {
            auto meanTensorAttr = tensorMap.at(nodeAttributes->mean_tensor_uid().value());
            meanInvVarianceDataType = meanTensorAttr->data_type();
        }
        else
        {
            meanInvVarianceDataType = xDataType;
        }
    }

    std::size_t operator()(const LayernormFpropSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(xDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(scaleBiasDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(meanInvVarianceDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(outputDataType)) << 16)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 20);
    }

    bool operator==(const LayernormFpropSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && xDataType == other.xDataType
               && scaleBiasDataType == other.scaleBiasDataType
               && meanInvVarianceDataType == other.meanInvVarianceDataType
               && outputDataType == other.outputDataType
               && computeDataType == other.computeDataType;
    }

    static std::unordered_map<LayernormFpropSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              LayernormFpropSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<LayernormFpropSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           LayernormFpropSignatureKey>
            map;

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16>(map);
        // MIOpen-compatible: all tensors same type, compute in float
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ScaleBiasDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType MeanInvVarianceDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<LayernormFpropSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  LayernormFpropSignatureKey>& map)
    {
        map[LayernormFpropSignatureKey(XDataTypeEnum,
                                       ScaleBiasDataTypeEnum,
                                       MeanInvVarianceDataTypeEnum,
                                       OutputDataTypeEnum,
                                       ComputeDataTypeEnum)]
            = std::make_unique<LayernormFpropPlanBuilder<XDataTypeEnum,
                                                         ScaleBiasDataTypeEnum,
                                                         MeanInvVarianceDataTypeEnum,
                                                         OutputDataTypeEnum,
                                                         ComputeDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const LayernormFpropSignatureKey& key)
{
    os << "Layernorm(x=" << key.xDataType << ", scale=" << key.scaleBiasDataType
       << ", meanInvVar=" << key.meanInvVarianceDataType << ", y=" << key.outputDataType
       << ", compute=" << key.computeDataType << ")";
    return os;
}

} // namespace hipdnn_test_sdk::detail
