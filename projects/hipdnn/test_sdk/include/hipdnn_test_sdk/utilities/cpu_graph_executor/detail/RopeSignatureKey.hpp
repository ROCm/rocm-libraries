// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <ostream>
#include <stdexcept>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/RopePlan.hpp>

namespace hipdnn_test_sdk::detail
{

/// RoPE always computes in float and its freqs are always FLOAT, so the key carries only the
/// data type shared by x and y.
struct RopeSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RopeAttributes};

    hipdnn_flatbuffers_sdk::data_objects::DataType dataType;

    RopeSignatureKey() = default;

    constexpr explicit RopeSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType data)
        : dataType(data)
    {
    }

    RopeSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        const auto* nodeAttributes = node.attributes_as_RopeAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes could not be cast to RopeAttributes");
        }

        const auto* xTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
        if(xTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        dataType = xTensorAttr->data_type();
    }

    std::size_t operator()(const RopeSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(dataType)) << 4);
    }

    bool operator==(const RopeSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && dataType == other.dataType;
    }

    static std::
        unordered_map<RopeSignatureKey, std::unique_ptr<IGraphNodePlanBuilder>, RopeSignatureKey>
        getPlanBuilders()
    {
        using hipdnn_flatbuffers_sdk::data_objects::DataType;

        std::unordered_map<RopeSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           RopeSignatureKey>
            map;

        addPlanBuilder<DataType::FLOAT>(map);
        addPlanBuilder<DataType::HALF>(map);
        addPlanBuilder<DataType::BFLOAT16>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType DataTypeEnum>
    static void addPlanBuilder(std::unordered_map<RopeSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  RopeSignatureKey>& map)
    {
        map[RopeSignatureKey(DataTypeEnum)] = std::make_unique<RopePlanBuilder<DataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const RopeSignatureKey& key)
{
    os << "Rope(data=" << key.dataType << ")";
    return os;
}

} // namespace hipdnn_test_sdk::detail
