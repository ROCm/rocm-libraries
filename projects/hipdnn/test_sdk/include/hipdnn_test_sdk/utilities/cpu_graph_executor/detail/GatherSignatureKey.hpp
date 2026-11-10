// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <ostream>
#include <stdexcept>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/GatherPlan.hpp>

namespace hipdnn_test_sdk::detail
{

/// Gather moves values without arithmetic, so the key carries no compute type: only the
/// data type shared by x and y and the index data type.
struct GatherSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::GatherAttributes};

    hipdnn_flatbuffers_sdk::data_objects::DataType dataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType indexDataType;

    GatherSignatureKey() = default;

    constexpr GatherSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType data,
                                 hipdnn_flatbuffers_sdk::data_objects::DataType index)
        : dataType(data)
        , indexDataType(index)
    {
    }

    GatherSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        const auto* nodeAttributes = node.attributes_as_GatherAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes could not be cast to GatherAttributes");
        }

        const auto* xTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
        const auto* indicesTensorAttr = tensorMap.at(nodeAttributes->indices_tensor_uid());
        if(xTensorAttr == nullptr || indicesTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        dataType = xTensorAttr->data_type();
        indexDataType = indicesTensorAttr->data_type();
    }

    std::size_t operator()(const GatherSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(dataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(indexDataType)) << 8);
    }

    bool operator==(const GatherSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && dataType == other.dataType
               && indexDataType == other.indexDataType;
    }

    static std::unordered_map<GatherSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              GatherSignatureKey>
        getPlanBuilders()
    {
        using hipdnn_flatbuffers_sdk::data_objects::DataType;

        std::unordered_map<GatherSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           GatherSignatureKey>
            map;

        addPlanBuilder<DataType::FLOAT, DataType::INT32>(map);
        addPlanBuilder<DataType::FLOAT, DataType::INT64>(map);
        addPlanBuilder<DataType::HALF, DataType::INT32>(map);
        addPlanBuilder<DataType::HALF, DataType::INT64>(map);
        addPlanBuilder<DataType::BFLOAT16, DataType::INT32>(map);
        addPlanBuilder<DataType::BFLOAT16, DataType::INT64>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType DataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType IndexDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<GatherSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  GatherSignatureKey>& map)
    {
        map[GatherSignatureKey(DataTypeEnum, IndexDataTypeEnum)]
            = std::make_unique<GatherPlanBuilder<DataTypeEnum, IndexDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const GatherSignatureKey& key)
{
    os << "Gather(data=" << key.dataType << ", indices=" << key.indexDataType << ")";
    return os;
}

} // namespace hipdnn_test_sdk::detail
