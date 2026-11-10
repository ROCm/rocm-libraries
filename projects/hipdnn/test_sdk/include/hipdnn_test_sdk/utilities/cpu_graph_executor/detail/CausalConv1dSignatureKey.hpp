// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <ostream>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/CausalConv1dPlan.hpp>

namespace hipdnn_test_sdk::detail
{

struct CausalConv1dSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::CausalConv1dAttributes};

    /// Type of x, weight, y and the optional bias, initial_state and final_state.
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType cuSeqlensDataType;
    /// UNSET when the graph has no state_indices.
    hipdnn_flatbuffers_sdk::data_objects::DataType stateIndicesDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType;

    CausalConv1dSignatureKey() = default;

    constexpr CausalConv1dSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType data,
                                       hipdnn_flatbuffers_sdk::data_objects::DataType cuSeqlens,
                                       hipdnn_flatbuffers_sdk::data_objects::DataType stateIndices,
                                       hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : dataType(data)
        , cuSeqlensDataType(cuSeqlens)
        , stateIndicesDataType(stateIndices)
        , computeDataType(compute)
    {
    }

    CausalConv1dSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_flatbuffers_sdk::data_objects::DataType computeType)
    {
        const auto* nodeAttributes = node.attributes_as_CausalConv1dAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes could not be cast to CausalConv1dAttributes");
        }

        auto xTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
        auto cuSeqlensTensorAttr = tensorMap.at(nodeAttributes->cu_seqlens_tensor_uid());
        if(xTensorAttr == nullptr || cuSeqlensTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        dataType = xTensorAttr->data_type();
        cuSeqlensDataType = cuSeqlensTensorAttr->data_type();
        stateIndicesDataType = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
        const auto stateIndicesUid = nodeAttributes->state_indices_tensor_uid();
        if(stateIndicesUid.has_value())
        {
            auto stateIndicesTensorAttr = tensorMap.at(*stateIndicesUid);
            if(stateIndicesTensorAttr == nullptr)
            {
                throw std::runtime_error("state_indices tensor attributes could not be found in "
                                         "the map, failed to construct key");
            }
            stateIndicesDataType = stateIndicesTensorAttr->data_type();
        }
        computeDataType = computeType;
    }

    std::size_t operator()(const CausalConv1dSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(dataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(cuSeqlensDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(stateIndicesDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 16);
    }

    bool operator==(const CausalConv1dSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && dataType == other.dataType
               && cuSeqlensDataType == other.cuSeqlensDataType
               && stateIndicesDataType == other.stateIndicesDataType
               && computeDataType == other.computeDataType;
    }

    /// Data in FLOAT, HALF or BFLOAT16; INT32 or INT64 cu_seqlens; no state_indices or
    /// INT32 / INT64 state_indices; FLOAT compute.
    static std::unordered_map<CausalConv1dSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              CausalConv1dSignatureKey>
        getPlanBuilders()
    {
        using hipdnn_flatbuffers_sdk::data_objects::DataType;

        std::unordered_map<CausalConv1dSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           CausalConv1dSignatureKey>
            map;

        addIndexVariants<DataType::FLOAT>(map);
        addIndexVariants<DataType::HALF>(map);
        addIndexVariants<DataType::BFLOAT16>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType DataTypeEnum>
    static void addIndexVariants(std::unordered_map<CausalConv1dSignatureKey,
                                                    std::unique_ptr<IGraphNodePlanBuilder>,
                                                    CausalConv1dSignatureKey>& map)
    {
        using hipdnn_flatbuffers_sdk::data_objects::DataType;

        addPlanBuilder<DataTypeEnum, DataType::INT32, DataType::UNSET, DataType::FLOAT>(map);
        addPlanBuilder<DataTypeEnum, DataType::INT32, DataType::INT32, DataType::FLOAT>(map);
        addPlanBuilder<DataTypeEnum, DataType::INT32, DataType::INT64, DataType::FLOAT>(map);
        addPlanBuilder<DataTypeEnum, DataType::INT64, DataType::UNSET, DataType::FLOAT>(map);
        addPlanBuilder<DataTypeEnum, DataType::INT64, DataType::INT32, DataType::FLOAT>(map);
        addPlanBuilder<DataTypeEnum, DataType::INT64, DataType::INT64, DataType::FLOAT>(map);
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType DataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType CuSeqlensDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType StateIndicesDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<CausalConv1dSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  CausalConv1dSignatureKey>& map)
    {
        map[CausalConv1dSignatureKey(
            DataTypeEnum, CuSeqlensDataTypeEnum, StateIndicesDataTypeEnum, ComputeDataTypeEnum)]
            = std::make_unique<CausalConv1dPlanBuilder<DataTypeEnum,
                                                       CuSeqlensDataTypeEnum,
                                                       StateIndicesDataTypeEnum,
                                                       ComputeDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const CausalConv1dSignatureKey& key)
{
    os << "CausalConv1d(data=" << key.dataType << ", cu_seqlens=" << key.cuSeqlensDataType
       << ", state_indices=" << key.stateIndicesDataType << ", compute=" << key.computeDataType
       << ")";
    return os;
}

} // namespace hipdnn_test_sdk::detail
