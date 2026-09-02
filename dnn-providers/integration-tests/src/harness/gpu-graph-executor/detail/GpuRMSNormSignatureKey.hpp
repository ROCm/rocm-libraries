// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "GpuRMSNormPlan.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

struct GpuRMSNormFwdSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormAttributes};
    hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType scaleDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType outputDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};

    GpuRMSNormFwdSignatureKey() = default;
    constexpr GpuRMSNormFwdSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType input,
                                        hipdnn_flatbuffers_sdk::data_objects::DataType scale,
                                        hipdnn_flatbuffers_sdk::data_objects::DataType output,
                                        hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : inputDataType(input)
        , scaleDataType(scale)
        , outputDataType(output)
        , computeDataType(compute)
    {
    }

    GpuRMSNormFwdSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_flatbuffers_sdk::data_objects::DataType computeType)
    {
        const auto* nodeAttributes = node.attributes_as_RMSNormAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes could not be cast to RMSNormAttributes");
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
        scaleDataType = scaleTensorAttr->data_type();
        outputDataType = outputTensorAttr->data_type();
        computeDataType = computeType;
    }

    std::size_t operator()(const GpuRMSNormFwdSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(inputDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(scaleDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(outputDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 16);
    }

    bool operator==(const GpuRMSNormFwdSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && inputDataType == other.inputDataType
               && scaleDataType == other.scaleDataType && outputDataType == other.outputDataType
               && computeDataType == other.computeDataType;
    }

    static std::unordered_map<GpuRMSNormFwdSignatureKey,
                              std::unique_ptr<IGpuGraphNodePlanBuilder>,
                              GpuRMSNormFwdSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<GpuRMSNormFwdSignatureKey,
                           std::unique_ptr<IGpuGraphNodePlanBuilder>,
                           GpuRMSNormFwdSignatureKey>
            map;

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType InputDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ScaleDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<GpuRMSNormFwdSignatureKey,
                                                  std::unique_ptr<IGpuGraphNodePlanBuilder>,
                                                  GpuRMSNormFwdSignatureKey>& map)
    {
        map[GpuRMSNormFwdSignatureKey(
            InputDataTypeEnum, ScaleDataTypeEnum, OutputDataTypeEnum, ComputeDataTypeEnum)]
            = std::make_unique<GpuRMSNormFwdPlanBuilder<InputDataTypeEnum,
                                                        ScaleDataTypeEnum,
                                                        OutputDataTypeEnum,
                                                        ComputeDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const GpuRMSNormFwdSignatureKey& key)
{
    os << "GpuRMSNormFwdSignatureKey(inputDataType=" << key.inputDataType
       << ", scaleDataType=" << key.scaleDataType << ", outputDataType=" << key.outputDataType
       << ", computeDataType=" << key.computeDataType << ")";
    return os;
}

struct GpuRMSNormBwdSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormBackwardAttributes};
    hipdnn_flatbuffers_sdk::data_objects::DataType gradOutputDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType scaleDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType gradInputDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};

    GpuRMSNormBwdSignatureKey() = default;
    constexpr GpuRMSNormBwdSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType gradOutput,
                                        hipdnn_flatbuffers_sdk::data_objects::DataType input,
                                        hipdnn_flatbuffers_sdk::data_objects::DataType scale,
                                        hipdnn_flatbuffers_sdk::data_objects::DataType gradInput,
                                        hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : gradOutputDataType(gradOutput)
        , inputDataType(input)
        , scaleDataType(scale)
        , gradInputDataType(gradInput)
        , computeDataType(compute)
    {
    }

    GpuRMSNormBwdSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_flatbuffers_sdk::data_objects::DataType computeType)
    {
        const auto* nodeAttributes = node.attributes_as_RMSNormBackwardAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes could not be cast to RMSNormBackwardAttributes");
        }

        auto gradOutputTensorAttr = tensorMap.at(nodeAttributes->dy_tensor_uid());
        auto inputTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
        auto scaleTensorAttr = tensorMap.at(nodeAttributes->scale_tensor_uid());
        auto gradInputTensorAttr = tensorMap.at(nodeAttributes->dx_tensor_uid());
        if(gradOutputTensorAttr == nullptr || inputTensorAttr == nullptr
           || scaleTensorAttr == nullptr || gradInputTensorAttr == nullptr)
        {
            throw std::runtime_error(
                "One or more required tensor attributes could not be found in the map, "
                "failed to construct key");
        }

        gradOutputDataType = gradOutputTensorAttr->data_type();
        inputDataType = inputTensorAttr->data_type();
        scaleDataType = scaleTensorAttr->data_type();
        gradInputDataType = gradInputTensorAttr->data_type();
        computeDataType = computeType;
    }

    std::size_t operator()(const GpuRMSNormBwdSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(gradOutputDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(inputDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(scaleDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(gradInputDataType)) << 16)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 20);
    }

    bool operator==(const GpuRMSNormBwdSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && gradOutputDataType == other.gradOutputDataType
               && inputDataType == other.inputDataType && scaleDataType == other.scaleDataType
               && gradInputDataType == other.gradInputDataType
               && computeDataType == other.computeDataType;
    }

    static std::unordered_map<GpuRMSNormBwdSignatureKey,
                              std::unique_ptr<IGpuGraphNodePlanBuilder>,
                              GpuRMSNormBwdSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<GpuRMSNormBwdSignatureKey,
                           std::unique_ptr<IGpuGraphNodePlanBuilder>,
                           GpuRMSNormBwdSignatureKey>
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
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
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
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType GradOutputDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType InputDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ScaleDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType GradInputDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<GpuRMSNormBwdSignatureKey,
                                                  std::unique_ptr<IGpuGraphNodePlanBuilder>,
                                                  GpuRMSNormBwdSignatureKey>& map)
    {
        map[GpuRMSNormBwdSignatureKey(GradOutputDataTypeEnum,
                                      InputDataTypeEnum,
                                      ScaleDataTypeEnum,
                                      GradInputDataTypeEnum,
                                      ComputeDataTypeEnum)]
            = std::make_unique<GpuRMSNormBwdPlanBuilder<GradOutputDataTypeEnum,
                                                        InputDataTypeEnum,
                                                        ScaleDataTypeEnum,
                                                        GradInputDataTypeEnum,
                                                        ComputeDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const GpuRMSNormBwdSignatureKey& key)
{
    os << "GpuRMSNormBwdSignatureKey(gradOutputDataType=" << key.gradOutputDataType
       << ", inputDataType=" << key.inputDataType << ", scaleDataType=" << key.scaleDataType
       << ", gradInputDataType=" << key.gradInputDataType
       << ", computeDataType=" << key.computeDataType << ")";
    return os;
}

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
