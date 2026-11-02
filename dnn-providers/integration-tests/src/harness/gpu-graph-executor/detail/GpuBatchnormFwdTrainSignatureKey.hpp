// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "GpuBatchnormFwdTrainPlan.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

struct GpuBatchnormFwdTrainSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes};
    hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType scaleBiasDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType meanVarianceDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType outputDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};

    GpuBatchnormFwdTrainSignatureKey() = default;
    constexpr GpuBatchnormFwdTrainSignatureKey(
        hipdnn_flatbuffers_sdk::data_objects::DataType input,
        hipdnn_flatbuffers_sdk::data_objects::DataType scaleBias,
        hipdnn_flatbuffers_sdk::data_objects::DataType meanVariance,
        hipdnn_flatbuffers_sdk::data_objects::DataType output,
        hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : inputDataType(input)
        , scaleBiasDataType(scaleBias)
        , meanVarianceDataType(meanVariance)
        , outputDataType(output)
        , computeDataType(compute)
    {
    }

    GpuBatchnormFwdTrainSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_flatbuffers_sdk::data_objects::DataType computeType)
    {
        const auto* nodeAttributes = node.attributes_as_BatchnormAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes could not be cast to BatchnormAttributes");
        }

        auto inputTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
        auto scaleTensorAttr = tensorMap.at(nodeAttributes->scale_tensor_uid());
        auto outputTensorAttr = tensorMap.at(nodeAttributes->y_tensor_uid());
        if(inputTensorAttr == nullptr || scaleTensorAttr == nullptr || outputTensorAttr == nullptr)
        {
            throw std::runtime_error(
                "One or more required tensor attributes could not be found in the map, "
                "failed to construct key");
        }

        inputDataType = inputTensorAttr->data_type();
        scaleBiasDataType = scaleTensorAttr->data_type();
        if(nodeAttributes->mean_tensor_uid().has_value())
        {
            auto meanTensorAttr = tensorMap.at(nodeAttributes->mean_tensor_uid().value());
            meanVarianceDataType = meanTensorAttr->data_type();
        }
        else
        {
            meanVarianceDataType = scaleBiasDataType;
        }
        outputDataType = outputTensorAttr->data_type();
        computeDataType = computeType;
    }

    std::size_t operator()(const GpuBatchnormFwdTrainSignatureKey& key) const noexcept
    {
        return key.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(inputDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(scaleBiasDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(meanVarianceDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(outputDataType)) << 16)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 20);
    }

    bool operator==(const GpuBatchnormFwdTrainSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && inputDataType == other.inputDataType
               && scaleBiasDataType == other.scaleBiasDataType
               && meanVarianceDataType == other.meanVarianceDataType
               && outputDataType == other.outputDataType
               && computeDataType == other.computeDataType;
    }

    static std::unordered_map<GpuBatchnormFwdTrainSignatureKey,
                              std::unique_ptr<IGpuGraphNodePlanBuilder>,
                              GpuBatchnormFwdTrainSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<GpuBatchnormFwdTrainSignatureKey,
                           std::unique_ptr<IGpuGraphNodePlanBuilder>,
                           GpuBatchnormFwdTrainSignatureKey>
            map;

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
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

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType InputDataType,
              hipdnn_flatbuffers_sdk::data_objects::DataType ScaleBiasDataType,
              hipdnn_flatbuffers_sdk::data_objects::DataType MeanVarianceDataType,
              hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataType,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataType>
    static void addPlanBuilder(std::unordered_map<GpuBatchnormFwdTrainSignatureKey,
                                                  std::unique_ptr<IGpuGraphNodePlanBuilder>,
                                                  GpuBatchnormFwdTrainSignatureKey>& map)
    {
        map[GpuBatchnormFwdTrainSignatureKey(InputDataType,
                                             ScaleBiasDataType,
                                             MeanVarianceDataType,
                                             OutputDataType,
                                             ComputeDataType)]
            = std::make_unique<GpuBatchnormFwdTrainPlanBuilder<InputDataType,
                                                               ScaleBiasDataType,
                                                               MeanVarianceDataType,
                                                               OutputDataType,
                                                               ComputeDataType>>();

        if constexpr(ScaleBiasDataType != MeanVarianceDataType)
        {
            // Without optional mean/inv_variance tensors, set meanVarianceDataType to ScaleBiasDataType
            map[GpuBatchnormFwdTrainSignatureKey(InputDataType,
                                                 ScaleBiasDataType,
                                                 ScaleBiasDataType,
                                                 OutputDataType,
                                                 ComputeDataType)]
                = std::make_unique<GpuBatchnormFwdTrainPlanBuilder<InputDataType,
                                                                   ScaleBiasDataType,
                                                                   ScaleBiasDataType,
                                                                   OutputDataType,
                                                                   ComputeDataType>>();
        }
    }
};

inline std::ostream& operator<<(std::ostream& os, const GpuBatchnormFwdTrainSignatureKey& key)
{
    os << "GpuBatchnormFwdTrain(x=" << key.inputDataType << ", scaleBias=" << key.scaleBiasDataType
       << ", meanVariance=" << key.meanVarianceDataType << ", y=" << key.outputDataType
       << ", compute=" << key.computeDataType << ")";
    return os;
}

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
