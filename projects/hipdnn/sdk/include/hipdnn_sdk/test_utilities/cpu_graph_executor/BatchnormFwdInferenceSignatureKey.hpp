// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/PluginFlatbufferTypeHelpers.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormFwdInferencePlan.hpp>

namespace hipdnn_sdk::test_utilities
{

struct BatchnormFwdInferenceSignatureKey
{
    const hipdnn_sdk::data_objects::NodeAttributes nodeType
        = hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes;
    hipdnn_sdk::data_objects::DataType inputDataType;
    hipdnn_sdk::data_objects::DataType scaleBiasDataType;
    hipdnn_sdk::data_objects::DataType meanVarianceDataType;
    hipdnn_sdk::data_objects::DataType computeDataType;
    hipdnn_sdk::data_objects::DataType outputDataType;

    BatchnormFwdInferenceSignatureKey() = default;
    constexpr BatchnormFwdInferenceSignatureKey(hipdnn_sdk::data_objects::DataType input,
                                                hipdnn_sdk::data_objects::DataType scaleBias,
                                                hipdnn_sdk::data_objects::DataType meanVariance,
                                                hipdnn_sdk::data_objects::DataType compute,
                                                hipdnn_sdk::data_objects::DataType output)
        : inputDataType(input)
        , scaleBiasDataType(scaleBias)
        , meanVarianceDataType(meanVariance)
        , computeDataType(compute)
        , outputDataType(output)
    {
    }

    BatchnormFwdInferenceSignatureKey(
        const hipdnn_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        const auto* nodeAttributes = node.attributes_as_BatchnormInferenceAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes could not be cast to BatchnormInferenceAttributes");
        }

        auto xTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
        auto scaleTensorAttr = tensorMap.at(nodeAttributes->scale_tensor_uid());
        auto meanTensorAttr = tensorMap.at(nodeAttributes->mean_tensor_uid());
        auto yTensorAttr = tensorMap.at(nodeAttributes->y_tensor_uid());

        if(xTensorAttr == nullptr || scaleTensorAttr == nullptr || meanTensorAttr == nullptr
           || yTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        inputDataType = xTensorAttr->data_type();
        scaleBiasDataType = scaleTensorAttr->data_type();
        meanVarianceDataType = meanTensorAttr->data_type();
        computeDataType = node.compute_data_type();
        outputDataType = yTensorAttr->data_type();
    }

    std::size_t operator()(const BatchnormFwdInferenceSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(inputDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(scaleBiasDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(meanVarianceDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 16)
               ^ (static_cast<std::size_t>(static_cast<int>(outputDataType)) << 20);
    }

    bool operator==(const BatchnormFwdInferenceSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && inputDataType == other.inputDataType
               && scaleBiasDataType == other.scaleBiasDataType
               && meanVarianceDataType == other.meanVarianceDataType
               && computeDataType == other.computeDataType
               && outputDataType == other.outputDataType;
    }

    static std::unordered_map<BatchnormFwdInferenceSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              BatchnormFwdInferenceSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<BatchnormFwdInferenceSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           BatchnormFwdInferenceSignatureKey>
            map;

        addPlanBuilder<hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::HALF>(map);
        addPlanBuilder<hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::BFLOAT16>(map);
        addPlanBuilder<hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::HALF>(map);
        addPlanBuilder<hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::BFLOAT16>(map);

        return map;
    }

    template <hipdnn_sdk::data_objects::DataType InputDataTypeEnum,
              hipdnn_sdk::data_objects::DataType ScaleBiasDataTypeEnum,
              hipdnn_sdk::data_objects::DataType MeanVarianceDataTypeEnum,
              hipdnn_sdk::data_objects::DataType ComputeDataTypeEnum,
              hipdnn_sdk::data_objects::DataType OutputDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<BatchnormFwdInferenceSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  BatchnormFwdInferenceSignatureKey>& map)
    {
        map[BatchnormFwdInferenceSignatureKey(InputDataTypeEnum,
                                              ScaleBiasDataTypeEnum,
                                              MeanVarianceDataTypeEnum,
                                              ComputeDataTypeEnum,
                                              OutputDataTypeEnum)]
            = std::make_unique<BatchnormFwdInferencePlanBuilder<InputDataTypeEnum,
                                                                ScaleBiasDataTypeEnum,
                                                                MeanVarianceDataTypeEnum,
                                                                ComputeDataTypeEnum,
                                                                OutputDataTypeEnum>>();
    }
};

}

template <>
struct fmt::formatter<hipdnn_sdk::test_utilities::BatchnormFwdInferenceSignatureKey>
{
    static constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const hipdnn_sdk::test_utilities::BatchnormFwdInferenceSignatureKey& key,
                FormatContext& ctx) const
    {
        return fmt::format_to(ctx.out(),
                              "BatchnormFwdInference(in={}, scale={}, mean={}, compute={}, out={})",
                              key.inputDataType,
                              key.scaleBiasDataType,
                              key.meanVarianceDataType,
                              key.computeDataType,
                              key.outputDataType);
    }
};
