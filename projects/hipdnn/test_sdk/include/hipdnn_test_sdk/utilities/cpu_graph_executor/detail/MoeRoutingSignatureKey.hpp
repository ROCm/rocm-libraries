// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <ostream>
#include <stdexcept>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/MoeRoutingPlan.hpp>

namespace hipdnn_test_sdk::detail
{

/// MoE routing does no arithmetic and its ids and row indices are always INT32, so the key
/// carries only the first_token_offset data type (INT32 or INT64).
struct MoeRoutingSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::MoeRoutingAttributes};

    hipdnn_flatbuffers_sdk::data_objects::DataType offsetDataType;

    MoeRoutingSignatureKey() = default;

    constexpr explicit MoeRoutingSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType offset)
        : offsetDataType(offset)
    {
    }

    MoeRoutingSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        const auto* nodeAttributes = node.attributes_as_MoeRoutingAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes could not be cast to MoeRoutingAttributes");
        }

        const auto* offsetTensorAttr
            = tensorMap.at(nodeAttributes->first_token_offset_tensor_uid());
        if(offsetTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        offsetDataType = offsetTensorAttr->data_type();
    }

    std::size_t operator()(const MoeRoutingSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(offsetDataType)) << 4);
    }

    bool operator==(const MoeRoutingSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && offsetDataType == other.offsetDataType;
    }

    static std::unordered_map<MoeRoutingSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              MoeRoutingSignatureKey>
        getPlanBuilders()
    {
        using hipdnn_flatbuffers_sdk::data_objects::DataType;

        std::unordered_map<MoeRoutingSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           MoeRoutingSignatureKey>
            map;

        addPlanBuilder<DataType::INT32>(map);
        addPlanBuilder<DataType::INT64>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType OffsetDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<MoeRoutingSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  MoeRoutingSignatureKey>& map)
    {
        map[MoeRoutingSignatureKey(OffsetDataTypeEnum)]
            = std::make_unique<MoeRoutingPlanBuilder<OffsetDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const MoeRoutingSignatureKey& key)
{
    os << "MoeRouting(first_token_offset=" << key.offsetDataType << ")";
    return os;
}

} // namespace hipdnn_test_sdk::detail
