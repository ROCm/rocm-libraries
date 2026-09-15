// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "GpuBatchnormFwdInfVariancePlan.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

struct GpuBatchnormFwdInfVarianceSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
            BatchnormInferenceAttributesVarianceExt};
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

    GpuBatchnormFwdInfVarianceSignatureKey() = default;
    constexpr GpuBatchnormFwdInfVarianceSignatureKey(
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

    GpuBatchnormFwdInfVarianceSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_flatbuffers_sdk::data_objects::DataType computeType)
    {
        const auto* nodeAttributes = node.attributes_as_BatchnormInferenceAttributesVarianceExt();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes could not be cast to BatchnormInferenceAttributesVarianceExt");
        }

        auto inputTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
        auto scaleTensorAttr = tensorMap.at(nodeAttributes->scale_tensor_uid());
        auto meanTensorAttr = tensorMap.at(nodeAttributes->mean_tensor_uid());
        auto outputTensorAttr = tensorMap.at(nodeAttributes->y_tensor_uid());
        if(inputTensorAttr == nullptr || scaleTensorAttr == nullptr || meanTensorAttr == nullptr
           || outputTensorAttr == nullptr)
        {
            throw std::runtime_error(
                "One or more required tensor attributes could not be found in the map, "
                "failed to construct key");
        }

        inputDataType = inputTensorAttr->data_type();
        scaleBiasDataType = scaleTensorAttr->data_type();
        meanVarianceDataType = meanTensorAttr->data_type();
        outputDataType = outputTensorAttr->data_type();
        computeDataType = computeType;
    }

    std::size_t operator()(const GpuBatchnormFwdInfVarianceSignatureKey& k) const noexcept
    {
        return k.hashSelf();
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

    bool operator==(const GpuBatchnormFwdInfVarianceSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && inputDataType == other.inputDataType
               && scaleBiasDataType == other.scaleBiasDataType
               && meanVarianceDataType == other.meanVarianceDataType
               && outputDataType == other.outputDataType
               && computeDataType == other.computeDataType;
    }

    static std::unordered_map<GpuBatchnormFwdInfVarianceSignatureKey,
                              std::unique_ptr<IGpuGraphNodePlanBuilder>,
                              GpuBatchnormFwdInfVarianceSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<GpuBatchnormFwdInfVarianceSignatureKey,
                           std::unique_ptr<IGpuGraphNodePlanBuilder>,
                           GpuBatchnormFwdInfVarianceSignatureKey>
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
              hipdnn_flatbuffers_sdk::data_objects::DataType MeanVarianceDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType YDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<GpuBatchnormFwdInfVarianceSignatureKey,
                                                  std::unique_ptr<IGpuGraphNodePlanBuilder>,
                                                  GpuBatchnormFwdInfVarianceSignatureKey>& map)
    {
        map[GpuBatchnormFwdInfVarianceSignatureKey(XDataTypeEnum,
                                                   ScaleBiasDataTypeEnum,
                                                   MeanVarianceDataTypeEnum,
                                                   YDataTypeEnum,
                                                   ComputeDataTypeEnum)]
            = std::make_unique<GpuBatchnormFwdInfVariancePlanBuilder<XDataTypeEnum,
                                                                     ScaleBiasDataTypeEnum,
                                                                     MeanVarianceDataTypeEnum,
                                                                     YDataTypeEnum,
                                                                     ComputeDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const GpuBatchnormFwdInfVarianceSignatureKey& key)
{
    os << "GpuBatchnormFwdInfVarianceSignatureKey(inputDataType=" << key.inputDataType
       << ", scaleBiasDataType=" << key.scaleBiasDataType
       << ", meanVarianceDataType=" << key.meanVarianceDataType
       << ", outputDataType=" << key.outputDataType << ", computeDataType=" << key.computeDataType
       << ")";
    return os;
}

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
