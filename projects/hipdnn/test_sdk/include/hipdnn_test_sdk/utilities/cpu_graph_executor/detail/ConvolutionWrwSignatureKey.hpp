// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ConvolutionWrwPlan.hpp>
#include <ostream>

namespace hipdnn_test_sdk::detail
{

struct ConvolutionWrwSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes};
    hipdnn_flatbuffers_sdk::data_objects::DataType xDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType dyDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType outputDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType;

    ConvolutionWrwSignatureKey() = default;

    constexpr ConvolutionWrwSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType x,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType dy,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType output,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : xDataType(x)
        , dyDataType(dy)
        , outputDataType(output)
        , computeDataType(compute)
    {
    }

    ConvolutionWrwSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_flatbuffers_sdk::data_objects::DataType computeType)
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
               ^ (static_cast<std::size_t>(static_cast<int>(outputDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 16);
    }

    bool operator==(const ConvolutionWrwSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && xDataType == other.xDataType
               && dyDataType == other.dyDataType && outputDataType == other.outputDataType
               && computeDataType == other.computeDataType;
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

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
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
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType DyDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<ConvolutionWrwSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  ConvolutionWrwSignatureKey>& map)
    {
        map[ConvolutionWrwSignatureKey(
            XDataTypeEnum, DyDataTypeEnum, OutputDataTypeEnum, ComputeDataTypeEnum)]
            = std::make_unique<ConvolutionWrwPlanBuilder<XDataTypeEnum,
                                                         DyDataTypeEnum,
                                                         OutputDataTypeEnum,
                                                         ComputeDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const ConvolutionWrwSignatureKey& key)
{
    os << "ConvolutionWrw(x=" << key.xDataType << ", dy=" << key.dyDataType
       << ", dw=" << key.outputDataType << ", compute=" << key.computeDataType << ")";
    return os;
}

} // namespace hipdnn_test_sdk::detail
