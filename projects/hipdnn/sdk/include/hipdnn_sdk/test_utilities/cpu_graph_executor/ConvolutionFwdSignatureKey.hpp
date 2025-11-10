// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/PluginFlatbufferTypeHelpers.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/ConvolutionFwdPlan.hpp>

namespace hipdnn_sdk::test_utilities
{

struct ConvolutionFwdSignatureKey
{
    const hipdnn_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes};
    hipdnn_sdk::data_objects::DataType xDataType;
    hipdnn_sdk::data_objects::DataType wDataType;
    hipdnn_sdk::data_objects::DataType computeDataType;
    hipdnn_sdk::data_objects::DataType outputDataType;

    ConvolutionFwdSignatureKey() = default;
    constexpr ConvolutionFwdSignatureKey(hipdnn_sdk::data_objects::DataType x,
                                         hipdnn_sdk::data_objects::DataType w,
                                         hipdnn_sdk::data_objects::DataType compute,
                                         hipdnn_sdk::data_objects::DataType output)
        : xDataType(x)
        , wDataType(w)
        , computeDataType(compute)
        , outputDataType(output)
    {
    }

    ConvolutionFwdSignatureKey(
        const hipdnn_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_sdk::data_objects::DataType computeType)
    {
        const auto* nodeAttributes = node.attributes_as_ConvolutionFwdAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes could not be cast to ConvolutionFwdAttributes");
        }

        auto xTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
        auto wTensorAttr = tensorMap.at(nodeAttributes->w_tensor_uid());
        auto yTensorAttr = tensorMap.at(nodeAttributes->y_tensor_uid());
        if(xTensorAttr == nullptr || wTensorAttr == nullptr || yTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        xDataType = xTensorAttr->data_type();
        wDataType = wTensorAttr->data_type();
        computeDataType = computeType;
        outputDataType = yTensorAttr->data_type();
    }

    std::size_t operator()(const ConvolutionFwdSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(xDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(wDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(outputDataType)) << 16);
    }

    bool operator==(const ConvolutionFwdSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && xDataType == other.xDataType
               && wDataType == other.wDataType && computeDataType == other.computeDataType
               && outputDataType == other.outputDataType;
    }

    static std::unordered_map<ConvolutionFwdSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              ConvolutionFwdSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<ConvolutionFwdSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           ConvolutionFwdSignatureKey>
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

        return map;
    }

    template <hipdnn_sdk::data_objects::DataType XDataTypeEnum,
              hipdnn_sdk::data_objects::DataType WDataTypeEnum,
              hipdnn_sdk::data_objects::DataType ComputeDataTypeEnum,
              hipdnn_sdk::data_objects::DataType OutputDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<ConvolutionFwdSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  ConvolutionFwdSignatureKey>& map)
    {
        map[ConvolutionFwdSignatureKey(
            XDataTypeEnum, WDataTypeEnum, ComputeDataTypeEnum, OutputDataTypeEnum)]
            = std::make_unique<ConvolutionFwdPlanBuilder<XDataTypeEnum,
                                                         WDataTypeEnum,
                                                         ComputeDataTypeEnum,
                                                         OutputDataTypeEnum>>();
    }
};

}

template <>
struct fmt::formatter<hipdnn_sdk::test_utilities::ConvolutionFwdSignatureKey>
{
    static constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const hipdnn_sdk::test_utilities::ConvolutionFwdSignatureKey& key,
                FormatContext& ctx) const
    {
        return fmt::format_to(ctx.out(),
                              "ConvolutionFwd(x={}, w={}, compute={}, y={})",
                              key.xDataType,
                              key.wDataType,
                              key.computeDataType,
                              key.outputDataType);
    }
};
