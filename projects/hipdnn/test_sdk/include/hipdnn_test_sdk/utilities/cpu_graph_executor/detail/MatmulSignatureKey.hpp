// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <ostream>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/MatmulPlan.hpp>

namespace hipdnn_test_sdk::detail
{

struct MatmulSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::MatmulAttributes};

    hipdnn_flatbuffers_sdk::data_objects::DataType aDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType bDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType cDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType;

    MatmulSignatureKey() = default;

    constexpr MatmulSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType a,
                                 hipdnn_flatbuffers_sdk::data_objects::DataType b,
                                 hipdnn_flatbuffers_sdk::data_objects::DataType c,
                                 hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : aDataType(a)
        , bDataType(b)
        , cDataType(c)
        , computeDataType(compute)
    {
    }

    MatmulSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_flatbuffers_sdk::data_objects::DataType computeType)
    {
        const auto* nodeAttributes = node.attributes_as_MatmulAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes could not be cast to MatmulAttributes");
        }

        auto aTensorAttr = tensorMap.at(nodeAttributes->a_tensor_uid());
        auto bTensorAttr = tensorMap.at(nodeAttributes->b_tensor_uid());
        auto cTensorAttr = tensorMap.at(nodeAttributes->c_tensor_uid());
        if(aTensorAttr == nullptr || bTensorAttr == nullptr || cTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        aDataType = aTensorAttr->data_type();
        bDataType = bTensorAttr->data_type();
        cDataType = cTensorAttr->data_type();
        computeDataType = computeType;
    }

    std::size_t operator()(const MatmulSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(aDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(bDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(cDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 16);
    }

    bool operator==(const MatmulSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && aDataType == other.aDataType
               && bDataType == other.bDataType && cDataType == other.cDataType
               && computeDataType == other.computeDataType;
    }

    static std::unordered_map<MatmulSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              MatmulSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<MatmulSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           MatmulSignatureKey>
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

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType ADataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType BDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType CDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<MatmulSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  MatmulSignatureKey>& map)
    {
        map[MatmulSignatureKey(ADataTypeEnum, BDataTypeEnum, CDataTypeEnum, ComputeDataTypeEnum)]
            = std::make_unique<MatmulPlanBuilder<ADataTypeEnum,
                                                 BDataTypeEnum,
                                                 CDataTypeEnum,
                                                 ComputeDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const MatmulSignatureKey& key)
{
    os << "Matmul(a=" << key.aDataType << ", b=" << key.bDataType << ", c=" << key.cDataType
       << ", compute=" << key.computeDataType << ")";
    return os;
}

} // namespace hipdnn_test_sdk::detail
