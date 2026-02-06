// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <hipdnn_data_sdk/data_objects/data_types_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/BatchnormFwdInferenceWithVariancePlan.hpp>

namespace hipdnn_test_sdk::utilities
{

struct BatchnormFwdInferenceWithVarianceSignatureKey
{
    const hipdnn_data_sdk::data_objects::NodeAttributes nodeType
        = hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributesVarianceExt;
    hipdnn_data_sdk::data_objects::DataType xDataType;
    hipdnn_data_sdk::data_objects::DataType scaleBiasDataType;
    hipdnn_data_sdk::data_objects::DataType meanVarianceDataType;
    hipdnn_data_sdk::data_objects::DataType outputDataType;
    hipdnn_data_sdk::data_objects::DataType computeDataType;

    BatchnormFwdInferenceWithVarianceSignatureKey() = default;
    constexpr BatchnormFwdInferenceWithVarianceSignatureKey(
        hipdnn_data_sdk::data_objects::DataType x,
        hipdnn_data_sdk::data_objects::DataType scaleBias,
        hipdnn_data_sdk::data_objects::DataType meanVariance,
        hipdnn_data_sdk::data_objects::DataType output,
        hipdnn_data_sdk::data_objects::DataType compute)
        : xDataType(x)
        , scaleBiasDataType(scaleBias)
        , meanVarianceDataType(meanVariance)
        , outputDataType(output)
        , computeDataType(compute)
    {
    }

    BatchnormFwdInferenceWithVarianceSignatureKey(
        const hipdnn_data_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        const auto* nodeAttributes = node.attributes_as_BatchnormInferenceAttributesVarianceExt();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes could not be cast to BatchnormInferenceAttributesVarianceExt");
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

        xDataType = xTensorAttr->data_type();
        scaleBiasDataType = scaleTensorAttr->data_type();
        meanVarianceDataType = meanTensorAttr->data_type();
        computeDataType = node.compute_data_type();
        outputDataType = yTensorAttr->data_type();
    }

    std::size_t operator()(const BatchnormFwdInferenceWithVarianceSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(xDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(scaleBiasDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(meanVarianceDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(outputDataType)) << 16)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 20);
    }

    bool operator==(const BatchnormFwdInferenceWithVarianceSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && xDataType == other.xDataType
               && scaleBiasDataType == other.scaleBiasDataType
               && meanVarianceDataType == other.meanVarianceDataType
               && outputDataType == other.outputDataType
               && computeDataType == other.computeDataType;
    }

    static std::unordered_map<BatchnormFwdInferenceWithVarianceSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              BatchnormFwdInferenceWithVarianceSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<BatchnormFwdInferenceWithVarianceSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           BatchnormFwdInferenceWithVarianceSignatureKey>
            map;

        addPlanBuilder<hipdnn_data_sdk::data_objects::DataType::FLOAT,
                       hipdnn_data_sdk::data_objects::DataType::FLOAT,
                       hipdnn_data_sdk::data_objects::DataType::FLOAT,
                       hipdnn_data_sdk::data_objects::DataType::FLOAT,
                       hipdnn_data_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_data_sdk::data_objects::DataType::HALF,
                       hipdnn_data_sdk::data_objects::DataType::FLOAT,
                       hipdnn_data_sdk::data_objects::DataType::FLOAT,
                       hipdnn_data_sdk::data_objects::DataType::HALF,
                       hipdnn_data_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_data_sdk::data_objects::DataType::FLOAT,
                       hipdnn_data_sdk::data_objects::DataType::FLOAT,
                       hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_data_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_data_sdk::data_objects::DataType::HALF,
                       hipdnn_data_sdk::data_objects::DataType::FLOAT,
                       hipdnn_data_sdk::data_objects::DataType::FLOAT,
                       hipdnn_data_sdk::data_objects::DataType::FLOAT,
                       hipdnn_data_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_data_sdk::data_objects::DataType::FLOAT,
                       hipdnn_data_sdk::data_objects::DataType::FLOAT,
                       hipdnn_data_sdk::data_objects::DataType::FLOAT,
                       hipdnn_data_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_data_sdk::data_objects::DataType::HALF,
                       hipdnn_data_sdk::data_objects::DataType::HALF,
                       hipdnn_data_sdk::data_objects::DataType::HALF,
                       hipdnn_data_sdk::data_objects::DataType::HALF,
                       hipdnn_data_sdk::data_objects::DataType::HALF>(map);
        addPlanBuilder<hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_data_sdk::data_objects::DataType::BFLOAT16>(map);

        return map;
    }

    template <hipdnn_data_sdk::data_objects::DataType XDataTypeEnum,
              hipdnn_data_sdk::data_objects::DataType ScaleBiasDataTypeEnum,
              hipdnn_data_sdk::data_objects::DataType MeanVarianceDataTypeEnum,
              hipdnn_data_sdk::data_objects::DataType OutputDataTypeEnum,
              hipdnn_data_sdk::data_objects::DataType ComputeDataTypeEnum>
    static void
        addPlanBuilder(std::unordered_map<BatchnormFwdInferenceWithVarianceSignatureKey,
                                          std::unique_ptr<IGraphNodePlanBuilder>,
                                          BatchnormFwdInferenceWithVarianceSignatureKey>& map)
    {
        map[BatchnormFwdInferenceWithVarianceSignatureKey(XDataTypeEnum,
                                                          ScaleBiasDataTypeEnum,
                                                          MeanVarianceDataTypeEnum,
                                                          OutputDataTypeEnum,
                                                          ComputeDataTypeEnum)]
            = std::make_unique<
                BatchnormFwdInferenceWithVariancePlanBuilder<XDataTypeEnum,
                                                             ScaleBiasDataTypeEnum,
                                                             MeanVarianceDataTypeEnum,
                                                             OutputDataTypeEnum,
                                                             ComputeDataTypeEnum>>();
    }
};

}

template <>
struct fmt::formatter<hipdnn_test_sdk::utilities::BatchnormFwdInferenceWithVarianceSignatureKey>
{
    static constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto
        format(const hipdnn_test_sdk::utilities::BatchnormFwdInferenceWithVarianceSignatureKey& key,
               FormatContext& ctx) const
    {
        return fmt::format_to(
            ctx.out(),
            "BatchnormFwdInferenceWithVariance(x={}, scale={}, mean={}, y={}, compute={})",
            key.xDataType,
            key.scaleBiasDataType,
            key.meanVarianceDataType,
            key.outputDataType,
            key.computeDataType);
    }
};
