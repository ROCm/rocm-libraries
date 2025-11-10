// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/PluginFlatbufferTypeHelpers.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/ConvolutionWrwPlan.hpp>

namespace hipdnn_sdk::test_utilities
{

struct ConvolutionWrwSignatureKey
{
    const hipdnn_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes};
    hipdnn_sdk::data_objects::DataType xDataType;
    hipdnn_sdk::data_objects::DataType dyDataType;
    hipdnn_sdk::data_objects::DataType computeDataType;
    hipdnn_sdk::data_objects::DataType outputDataType;

    ConvolutionWrwSignatureKey() = default;

    constexpr ConvolutionWrwSignatureKey(hipdnn_sdk::data_objects::DataType x,
                                         hipdnn_sdk::data_objects::DataType dy,
                                         hipdnn_sdk::data_objects::DataType compute,
                                         hipdnn_sdk::data_objects::DataType output)
        : xDataType(x)
        , dyDataType(dy)
        , computeDataType(compute)
        , outputDataType(output)
    {
    }

    ConvolutionWrwSignatureKey(
        const hipdnn_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_sdk::data_objects::DataType computeType)
    {
        const auto* nodeAttributes = node.attributes_as_ConvolutionWrwAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes could not be cast to ConvolutionWrwAttributes");
        }

        auto xTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
        auto dyTensorAttr = tensorMap.at(nodeAttributes->dy_tensor_uid());
        auto dwTensorAttr = tensorMap.at(nodeAttributes->dw_tensor_uid());

        if(xTensorAttr == nullptr || dyTensorAttr == nullptr || dwTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        xDataType = xTensorAttr->data_type();
        dyDataType = dyTensorAttr->data_type();
        computeDataType = computeType;
        outputDataType = dwTensorAttr->data_type();
    }

    std::size_t operator()(const ConvolutionWrwSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(xDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(dyDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(outputDataType)) << 16);
    }

    bool operator==(const ConvolutionWrwSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && xDataType == other.xDataType
               && dyDataType == other.dyDataType && computeDataType == other.computeDataType
               && outputDataType == other.outputDataType;
    }

    static std::unordered_map<ConvolutionWrwSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              ConvolutionWrwSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<ConvolutionWrwSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           ConvolutionWrwSignatureKey>
            map;

        addPlanBuilder<hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::HALF>(map);
        addPlanBuilder<hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::FLOAT,
                       hipdnn_sdk::data_objects::DataType::BFLOAT16>(map);
        addPlanBuilder<hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::HALF,
                       hipdnn_sdk::data_objects::DataType::HALF>(map);
        addPlanBuilder<hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_sdk::data_objects::DataType::BFLOAT16>(map);

        return map;
    }

    template <hipdnn_sdk::data_objects::DataType XDataTypeEnum,
              hipdnn_sdk::data_objects::DataType DyDataTypeEnum,
              hipdnn_sdk::data_objects::DataType ComputeDataTypeEnum,
              hipdnn_sdk::data_objects::DataType OutputDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<ConvolutionWrwSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  ConvolutionWrwSignatureKey>& map)
    {
        map[ConvolutionWrwSignatureKey(
            XDataTypeEnum, DyDataTypeEnum, ComputeDataTypeEnum, OutputDataTypeEnum)]
            = std::make_unique<ConvolutionWrwPlanBuilder<XDataTypeEnum,
                                                         DyDataTypeEnum,
                                                         ComputeDataTypeEnum,
                                                         OutputDataTypeEnum>>();
    }
};

}

template <>
struct fmt::formatter<hipdnn_sdk::test_utilities::ConvolutionWrwSignatureKey>
{
    static constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const hipdnn_sdk::test_utilities::ConvolutionWrwSignatureKey& key,
                FormatContext& ctx) const
    {
        return fmt::format_to(ctx.out(),
                              "ConvolutionWrw(x={}, dy={}, compute={}, dw={})",
                              key.xDataType,
                              key.dyDataType,
                              key.computeDataType,
                              key.outputDataType);
    }
};
