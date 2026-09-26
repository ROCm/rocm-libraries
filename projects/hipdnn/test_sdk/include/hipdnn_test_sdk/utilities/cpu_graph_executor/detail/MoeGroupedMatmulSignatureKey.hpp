// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <ostream>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/MoeGroupedMatmulPlan.hpp>

namespace hipdnn_test_sdk::detail
{

// Routing tensors are always INT32 and are not key dimensions, and neither are
// mode/top_k: one builder handles all three routing modes (NONE/GATHER/SCATTER).
struct MoeGroupedMatmulSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::MoeGroupedMatmulAttributes};

    hipdnn_flatbuffers_sdk::data_objects::DataType tokenDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType weightDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType outputDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType;

    MoeGroupedMatmulSignatureKey() = default;

    constexpr MoeGroupedMatmulSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType token,
                                           hipdnn_flatbuffers_sdk::data_objects::DataType weight,
                                           hipdnn_flatbuffers_sdk::data_objects::DataType output,
                                           hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : tokenDataType(token)
        , weightDataType(weight)
        , outputDataType(output)
        , computeDataType(compute)
    {
    }

    MoeGroupedMatmulSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_flatbuffers_sdk::data_objects::DataType computeType)
    {
        const auto* nodeAttributes = node.attributes_as_MoeGroupedMatmulAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes could not be cast to MoeGroupedMatmulAttributes");
        }

        auto tokenTensorAttr = tensorMap.at(nodeAttributes->token_tensor_uid());
        auto weightTensorAttr = tensorMap.at(nodeAttributes->weight_tensor_uid());
        auto outputTensorAttr = tensorMap.at(nodeAttributes->output_tensor_uid());
        if(tokenTensorAttr == nullptr || weightTensorAttr == nullptr || outputTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        tokenDataType = tokenTensorAttr->data_type();
        weightDataType = weightTensorAttr->data_type();
        outputDataType = outputTensorAttr->data_type();
        computeDataType = computeType;
    }

    std::size_t operator()(const MoeGroupedMatmulSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(tokenDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(weightDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(outputDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 16);
    }

    bool operator==(const MoeGroupedMatmulSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && tokenDataType == other.tokenDataType
               && weightDataType == other.weightDataType && outputDataType == other.outputDataType
               && computeDataType == other.computeDataType;
    }

    static std::unordered_map<MoeGroupedMatmulSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              MoeGroupedMatmulSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<MoeGroupedMatmulSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           MoeGroupedMatmulSignatureKey>
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

    template <hipdnn_flatbuffers_sdk::data_objects::DataType TokenDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType WeightDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<MoeGroupedMatmulSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  MoeGroupedMatmulSignatureKey>& map)
    {
        map[MoeGroupedMatmulSignatureKey(
            TokenDataTypeEnum, WeightDataTypeEnum, OutputDataTypeEnum, ComputeDataTypeEnum)]
            = std::make_unique<MoeGroupedMatmulPlanBuilder<TokenDataTypeEnum,
                                                           WeightDataTypeEnum,
                                                           OutputDataTypeEnum,
                                                           ComputeDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const MoeGroupedMatmulSignatureKey& key)
{
    os << "MoeGroupedMatmul(token=" << key.tokenDataType << ", weight=" << key.weightDataType
       << ", output=" << key.outputDataType << ", compute=" << key.computeDataType << ")";
    return os;
}

} // namespace hipdnn_test_sdk::detail
