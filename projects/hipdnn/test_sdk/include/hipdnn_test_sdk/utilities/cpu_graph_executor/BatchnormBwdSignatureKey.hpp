// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/PluginFlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/BatchnormBwdPlan.hpp>

namespace hipdnn_test_sdk::utilities
{

struct BatchnormBwdSignatureKey
{
    const hipdnn_sdk::data_objects::NodeAttributes nodeType
        = hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes;
    hipdnn_sdk::data_objects::DataType dyDataType;
    hipdnn_sdk::data_objects::DataType xDataType;
    hipdnn_sdk::data_objects::DataType scaleBiasDataType;
    hipdnn_sdk::data_objects::DataType meanVarianceDataType;
    hipdnn_sdk::data_objects::DataType outputDataType;
    hipdnn_sdk::data_objects::DataType computeDataType;

    BatchnormBwdSignatureKey() = default;
    constexpr BatchnormBwdSignatureKey(hipdnn_sdk::data_objects::DataType dy,
                                       hipdnn_sdk::data_objects::DataType x,
                                       hipdnn_sdk::data_objects::DataType scaleBias,
                                       hipdnn_sdk::data_objects::DataType meanVariance,
                                       hipdnn_sdk::data_objects::DataType output,
                                       hipdnn_sdk::data_objects::DataType compute)
        : dyDataType(dy)
        , xDataType(x)
        , scaleBiasDataType(scaleBias)
        , meanVarianceDataType(meanVariance)
        , outputDataType(output)
        , computeDataType(compute)
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

        auto dyTensorAttr = tensorMap.at(nodeAttributes->dy_tensor_uid());
        auto xTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
        auto scaleTensorAttr = tensorMap.at(nodeAttributes->scale_tensor_uid());
        auto dxTensorAttr = tensorMap.at(nodeAttributes->dx_tensor_uid());

        if(dyTensorAttr == nullptr || xTensorAttr == nullptr || scaleTensorAttr == nullptr
           || dxTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        dyDataType = dyTensorAttr->data_type();
        xDataType = xTensorAttr->data_type();
        scaleBiasDataType = scaleTensorAttr->data_type();
        computeDataType = node.compute_data_type();
        outputDataType = dxTensorAttr->data_type();

        if(nodeAttributes->mean_tensor_uid().has_value()
           && nodeAttributes->inv_variance_tensor_uid().has_value())
        {
            auto meanTensorAttr = tensorMap.at(nodeAttributes->mean_tensor_uid().value());
            auto invVarianceTensorAttr
                = tensorMap.at(nodeAttributes->inv_variance_tensor_uid().value());

            if(meanTensorAttr->data_type() != invVarianceTensorAttr->data_type())
            {
                throw std::runtime_error(
                    "BatchnormBwdSignatureKey requires mean and inv_variance tensors "
                    "to have the same data type");
            }

            meanVarianceDataType = meanTensorAttr->data_type();
        }
        else
        {
            meanVarianceDataType = scaleBiasDataType;
        }
    }

    std::size_t operator()(const BatchnormBwdSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(dyDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(xDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(scaleBiasDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(meanVarianceDataType)) << 16)
               ^ (static_cast<std::size_t>(static_cast<int>(outputDataType)) << 20)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 24);
    }

    bool operator==(const BatchnormBwdSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && dyDataType == other.dyDataType
               && xDataType == other.xDataType && scaleBiasDataType == other.scaleBiasDataType
               && meanVarianceDataType == other.meanVarianceDataType
               && outputDataType == other.outputDataType
               && computeDataType == other.computeDataType;
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
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::HALF>(map);
        addPlanBuilder<hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::BFLOAT16>(map);
        addPlanBuilder<hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::FLOAT>(map);

        return map;
    }

    template <hipdnn_sdk::data_objects::DataType DyDataTypeEnum,
              hipdnn_sdk::data_objects::DataType InputDataTypeEnum,
              hipdnn_sdk::data_objects::DataType ScaleBiasDataTypeEnum,
              hipdnn_sdk::data_objects::DataType MeanVarianceDataTypeEnum,
              hipdnn_sdk::data_objects::DataType OutputDataTypeEnum,
              hipdnn_sdk::data_objects::DataType ComputeDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<BatchnormBwdSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  BatchnormBwdSignatureKey>& map)
    {
        map[BatchnormBwdSignatureKey(DyDataTypeEnum,
                                     InputDataTypeEnum,
                                     ScaleBiasDataTypeEnum,
                                     MeanVarianceDataTypeEnum,
                                     OutputDataTypeEnum,
                                     ComputeDataTypeEnum)]
            = std::make_unique<BatchnormBwdPlanBuilder<DyDataTypeEnum,
                                                       InputDataTypeEnum,
                                                       ScaleBiasDataTypeEnum,
                                                       MeanVarianceDataTypeEnum,
                                                       OutputDataTypeEnum,
                                                       ComputeDataTypeEnum>>();
    }
};

}

template <>
struct fmt::formatter<hipdnn_test_sdk::utilities::BatchnormBwdSignatureKey>
{
    static constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const hipdnn_test_sdk::utilities::BatchnormBwdSignatureKey& key,
                FormatContext& ctx) const
    {
        return fmt::format_to(ctx.out(),
                              "BatchnormBwd(dy={}, x={}, scale={}, mean={}, dx={}, compute={})",
                              key.dyDataType,
                              key.xDataType,
                              key.scaleBiasDataType,
                              key.meanVarianceDataType,
                              key.outputDataType,
                              key.computeDataType);
    }
};
