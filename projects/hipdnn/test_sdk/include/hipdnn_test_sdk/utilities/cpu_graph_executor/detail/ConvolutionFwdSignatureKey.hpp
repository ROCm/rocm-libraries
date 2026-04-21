// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ConvolutionFwdPlan.hpp>
#include <ostream>

namespace hipdnn_test_sdk::detail
{

struct ConvolutionFwdSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes};
    hipdnn_flatbuffers_sdk::data_objects::DataType xDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType wDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType outputDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType;

    ConvolutionFwdSignatureKey() = default;
    constexpr ConvolutionFwdSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType x,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType w,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType output,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : xDataType(x)
        , wDataType(w)
        , outputDataType(output)
        , computeDataType(compute)
    {
    }

    ConvolutionFwdSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_flatbuffers_sdk::data_objects::DataType computeType)
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
               ^ (static_cast<std::size_t>(static_cast<int>(outputDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 16);
    }

    bool operator==(const ConvolutionFwdSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && xDataType == other.xDataType
               && wDataType == other.wDataType && outputDataType == other.outputDataType
               && computeDataType == other.computeDataType;
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

        // X, W, Y, Compute
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
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType WDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<ConvolutionFwdSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  ConvolutionFwdSignatureKey>& map)
    {
        map[ConvolutionFwdSignatureKey(
            XDataTypeEnum, WDataTypeEnum, OutputDataTypeEnum, ComputeDataTypeEnum)]
            = std::make_unique<ConvolutionFwdPlanBuilder<XDataTypeEnum,
                                                         WDataTypeEnum,
                                                         OutputDataTypeEnum,
                                                         ComputeDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const ConvolutionFwdSignatureKey& key)
{
    os << "ConvolutionFwd(x=" << key.xDataType << ", w=" << key.wDataType
       << ", y=" << key.outputDataType << ", compute=" << key.computeDataType << ")";
    return os;
}

} // namespace hipdnn_test_sdk::detail
