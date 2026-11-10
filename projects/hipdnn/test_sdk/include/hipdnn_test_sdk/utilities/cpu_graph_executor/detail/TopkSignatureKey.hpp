// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <ostream>
#include <stdexcept>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/TopkPlan.hpp>

namespace hipdnn_test_sdk::detail
{

/// Top-k compares in a fixed compute type and copies values without arithmetic, so the key
/// carries no compute type: only the data type shared by x and values and the index data type.
struct TopkSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::TopkAttributes};

    hipdnn_flatbuffers_sdk::data_objects::DataType dataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType indexDataType;

    TopkSignatureKey() = default;

    constexpr TopkSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType data,
                               hipdnn_flatbuffers_sdk::data_objects::DataType index)
        : dataType(data)
        , indexDataType(index)
    {
    }

    TopkSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        const auto* nodeAttributes = node.attributes_as_TopkAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes could not be cast to TopkAttributes");
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

    std::size_t operator()(const TopkSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(dataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(indexDataType)) << 8);
    }

    bool operator==(const TopkSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && dataType == other.dataType
               && indexDataType == other.indexDataType;
    }

    static std::
        unordered_map<TopkSignatureKey, std::unique_ptr<IGraphNodePlanBuilder>, TopkSignatureKey>
        getPlanBuilders()
    {
        using hipdnn_flatbuffers_sdk::data_objects::DataType;

        std::unordered_map<TopkSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           TopkSignatureKey>
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
    static void addPlanBuilder(std::unordered_map<TopkSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  TopkSignatureKey>& map)
    {
        map[TopkSignatureKey(DataTypeEnum, IndexDataTypeEnum)]
            = std::make_unique<TopkPlanBuilder<DataTypeEnum, IndexDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const TopkSignatureKey& key)
{
    os << "Topk(data=" << key.dataType << ", indices=" << key.indexDataType << ")";
    return os;
}

} // namespace hipdnn_test_sdk::detail
