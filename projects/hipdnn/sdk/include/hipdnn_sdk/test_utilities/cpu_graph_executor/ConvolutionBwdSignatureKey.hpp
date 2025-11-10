// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/PluginFlatbufferTypeHelpers.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/ConvolutionBwdPlan.hpp>

namespace hipdnn_sdk::test_utilities
{

struct ConvolutionBwdSignatureKey
{
    const hipdnn_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes};
    hipdnn_sdk::data_objects::DataType input0DataType;
    hipdnn_sdk::data_objects::DataType input1DataType;
    hipdnn_sdk::data_objects::DataType computeDataType;
    hipdnn_sdk::data_objects::DataType outputDataType;

    ConvolutionBwdSignatureKey() = default;

    constexpr ConvolutionBwdSignatureKey(hipdnn_sdk::data_objects::DataType input0,
                                         hipdnn_sdk::data_objects::DataType input1,
                                         hipdnn_sdk::data_objects::DataType compute,
                                         hipdnn_sdk::data_objects::DataType output)
        : input0DataType(input0)
        , input1DataType(input1)
        , computeDataType(compute)
        , outputDataType(output)
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

        auto dyTensorAttr = tensorMap.at(nodeAttributes->dy_tensor_uid());
        auto wTensorAttr = tensorMap.at(nodeAttributes->w_tensor_uid());
        auto dxTensorAttr = tensorMap.at(nodeAttributes->dx_tensor_uid());

        if(dyTensorAttr == nullptr || wTensorAttr == nullptr || dxTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        input0DataType = dyTensorAttr->data_type();
        input1DataType = wTensorAttr->data_type();
        computeDataType = computeType;
        outputDataType = dxTensorAttr->data_type();
    }

    std::size_t operator()(const ConvolutionBwdSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(input0DataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(input1DataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(outputDataType)) << 16);
    }

    bool operator==(const ConvolutionBwdSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && input0DataType == other.input0DataType
               && input1DataType == other.input1DataType && computeDataType == other.computeDataType
               && outputDataType == other.outputDataType;
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

    template <hipdnn_sdk::data_objects::DataType Input0DataTypeEnum,
              hipdnn_sdk::data_objects::DataType Input1DataTypeEnum,
              hipdnn_sdk::data_objects::DataType ComputeDataTypeEnum,
              hipdnn_sdk::data_objects::DataType OutputDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<ConvolutionBwdSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  ConvolutionBwdSignatureKey>& map)
    {
        map[ConvolutionBwdSignatureKey(
            Input0DataTypeEnum, Input1DataTypeEnum, ComputeDataTypeEnum, OutputDataTypeEnum)]
            = std::make_unique<ConvolutionBwdPlanBuilder<Input0DataTypeEnum,
                                                         Input1DataTypeEnum,
                                                         ComputeDataTypeEnum,
                                                         OutputDataTypeEnum>>();
    }
};

}

template <>
struct fmt::formatter<hipdnn_sdk::test_utilities::ConvolutionBwdSignatureKey>
{
    static constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const hipdnn_sdk::test_utilities::ConvolutionBwdSignatureKey& key,
                FormatContext& ctx) const
    {
        return fmt::format_to(ctx.out(),
                              "ConvolutionBwd(dy={}, w={}, compute={}, dx={})",
                              key.input0DataType,
                              key.input1DataType,
                              key.computeDataType,
                              key.outputDataType);
    }
};
