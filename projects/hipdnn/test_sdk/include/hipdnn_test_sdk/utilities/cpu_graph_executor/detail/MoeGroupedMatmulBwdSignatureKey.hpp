// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <ostream>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/MoeGroupedMatmulBwdPlan.hpp>

namespace hipdnn_test_sdk::detail
{

struct MoeGroupedMatmulBwdSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::MoeGroupedMatmulBwdAttributes};

    hipdnn_flatbuffers_sdk::data_objects::DataType doutputDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType tokenDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType dweightDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType;

    MoeGroupedMatmulBwdSignatureKey() = default;

    constexpr MoeGroupedMatmulBwdSignatureKey(
        hipdnn_flatbuffers_sdk::data_objects::DataType doutput,
        hipdnn_flatbuffers_sdk::data_objects::DataType token,
        hipdnn_flatbuffers_sdk::data_objects::DataType dweight,
        hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : doutputDataType(doutput)
        , tokenDataType(token)
        , dweightDataType(dweight)
        , computeDataType(compute)
    {
    }

    MoeGroupedMatmulBwdSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_flatbuffers_sdk::data_objects::DataType computeType)
    {
        const auto* nodeAttributes = node.attributes_as_MoeGroupedMatmulBwdAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes could not be cast to MoeGroupedMatmulBwdAttributes");
        }

        auto doutputTensorAttr = tensorMap.at(nodeAttributes->doutput_tensor_uid());
        auto tokenTensorAttr = tensorMap.at(nodeAttributes->token_tensor_uid());
        auto dweightTensorAttr = tensorMap.at(nodeAttributes->dweight_tensor_uid());
        if(doutputTensorAttr == nullptr || tokenTensorAttr == nullptr
           || dweightTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        doutputDataType = doutputTensorAttr->data_type();
        tokenDataType = tokenTensorAttr->data_type();
        dweightDataType = dweightTensorAttr->data_type();
        computeDataType = computeType;
    }

    std::size_t operator()(const MoeGroupedMatmulBwdSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(doutputDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(tokenDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(dweightDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 16);
    }

    bool operator==(const MoeGroupedMatmulBwdSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && doutputDataType == other.doutputDataType
               && tokenDataType == other.tokenDataType && dweightDataType == other.dweightDataType
               && computeDataType == other.computeDataType;
    }

    static std::unordered_map<MoeGroupedMatmulBwdSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              MoeGroupedMatmulBwdSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<MoeGroupedMatmulBwdSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           MoeGroupedMatmulBwdSignatureKey>
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
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType DoutputDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType TokenDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType DweightDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<MoeGroupedMatmulBwdSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  MoeGroupedMatmulBwdSignatureKey>& map)
    {
        map[MoeGroupedMatmulBwdSignatureKey(
            DoutputDataTypeEnum, TokenDataTypeEnum, DweightDataTypeEnum, ComputeDataTypeEnum)]
            = std::make_unique<MoeGroupedMatmulBwdPlanBuilder<DoutputDataTypeEnum,
                                                              TokenDataTypeEnum,
                                                              DweightDataTypeEnum,
                                                              ComputeDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const MoeGroupedMatmulBwdSignatureKey& key)
{
    os << "MoeGroupedMatmulBwd(doutput=" << key.doutputDataType << ", token=" << key.tokenDataType
       << ", dweight=" << key.dweightDataType << ", compute=" << key.computeDataType << ")";
    return os;
}

} // namespace hipdnn_test_sdk::detail
