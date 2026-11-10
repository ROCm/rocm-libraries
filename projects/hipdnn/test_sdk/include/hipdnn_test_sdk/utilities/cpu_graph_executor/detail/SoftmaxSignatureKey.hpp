// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <unordered_map>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SoftmaxPlan.hpp>

namespace hipdnn_test_sdk::detail
{

/// Keys a softmax plan by the p and s data types and the compute type. The optional stats
/// output is FLOAT for every key and is checked by SoftmaxPlanBuilder::isApplicable.
struct SoftmaxSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::SoftmaxAttributes};

    hipdnn_flatbuffers_sdk::data_objects::DataType pDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType sDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType;

    SoftmaxSignatureKey() = default;

    constexpr SoftmaxSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType p,
                                  hipdnn_flatbuffers_sdk::data_objects::DataType s,
                                  hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : pDataType(p)
        , sDataType(s)
        , computeDataType(compute)
    {
    }

    SoftmaxSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_flatbuffers_sdk::data_objects::DataType computeType)
    {
        const auto* nodeAttributes = node.attributes_as_SoftmaxAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes could not be cast to SoftmaxAttributes");
        }

        const auto* pTensorAttr = tensorMap.at(nodeAttributes->p_tensor_uid());
        const auto* sTensorAttr = tensorMap.at(nodeAttributes->s_tensor_uid());
        if(pTensorAttr == nullptr || sTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        pDataType = pTensorAttr->data_type();
        sDataType = sTensorAttr->data_type();
        computeDataType = computeType;
    }

    std::size_t operator()(const SoftmaxSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(pDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(sDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 12);
    }

    bool operator==(const SoftmaxSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && pDataType == other.pDataType
               && sDataType == other.sDataType && computeDataType == other.computeDataType;
    }

    static std::unordered_map<SoftmaxSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              SoftmaxSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<SoftmaxSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           SoftmaxSignatureKey>
            map;

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType PDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType SDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<SoftmaxSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  SoftmaxSignatureKey>& map)
    {
        map[SoftmaxSignatureKey(PDataTypeEnum, SDataTypeEnum, ComputeDataTypeEnum)]
            = std::make_unique<
                SoftmaxPlanBuilder<PDataTypeEnum, SDataTypeEnum, ComputeDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const SoftmaxSignatureKey& key)
{
    os << "Softmax(p=" << key.pDataType << ", s=" << key.sDataType
       << ", compute=" << key.computeDataType << ")";
    return os;
}

} // namespace hipdnn_test_sdk::detail
