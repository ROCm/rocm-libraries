// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "GpuResampleBwdPlan.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

struct GpuResampleBwdSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ResampleBwdAttributes};
    hipdnn_flatbuffers_sdk::data_objects::DataType dyDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType dxDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType indexDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};

    GpuResampleBwdSignatureKey() = default;
    constexpr GpuResampleBwdSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType dy,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType dx,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType compute,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType index)
        : dyDataType(dy)
        , dxDataType(dx)
        , computeDataType(compute)
        , indexDataType(index)
    {
    }

    GpuResampleBwdSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_flatbuffers_sdk::data_objects::DataType computeType)
    {
        const auto* nodeAttributes = node.attributes_as_ResampleBwdAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes could not be cast to ResampleBwdAttributes");
        }

        const auto* dyTensorAttr = tensorMap.at(nodeAttributes->dy_tensor_uid());
        const auto* dxTensorAttr = tensorMap.at(nodeAttributes->dx_tensor_uid());
        if(dyTensorAttr == nullptr || dxTensorAttr == nullptr)
        {
            throw std::runtime_error(
                "One or more required tensor attributes could not be found in the map, "
                "failed to construct key");
        }

        dyDataType = dyTensorAttr->data_type();
        dxDataType = dxTensorAttr->data_type();
        computeDataType = computeType;

        if(nodeAttributes->index_tensor_uid().has_value())
        {
            const auto* indexTensorAttr = tensorMap.at(*nodeAttributes->index_tensor_uid());
            if(indexTensorAttr == nullptr)
            {
                throw std::runtime_error("Index tensor attributes could not be found in the map, "
                                         "failed to construct key");
            }
            indexDataType = indexTensorAttr->data_type();
        }
    }

    std::size_t operator()(const GpuResampleBwdSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(dyDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(dxDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(indexDataType)) << 16);
    }

    bool operator==(const GpuResampleBwdSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && dyDataType == other.dyDataType
               && dxDataType == other.dxDataType && computeDataType == other.computeDataType
               && indexDataType == other.indexDataType;
    }

    static std::unordered_map<GpuResampleBwdSignatureKey,
                              std::unique_ptr<IGpuGraphNodePlanBuilder>,
                              GpuResampleBwdSignatureKey>
        getPlanBuilders()
    {
        using hipdnn_flatbuffers_sdk::data_objects::DataType;

        std::unordered_map<GpuResampleBwdSignatureKey,
                           std::unique_ptr<IGpuGraphNodePlanBuilder>,
                           GpuResampleBwdSignatureKey>
            map;

        // Average pooling: no index tensor
        addPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET>(map);
        addPlanBuilder<DataType::HALF, DataType::HALF, DataType::FLOAT, DataType::UNSET>(map);
        addPlanBuilder<DataType::BFLOAT16, DataType::BFLOAT16, DataType::FLOAT, DataType::UNSET>(
            map);

        // Max pooling: INT32 index tensor from the forward pass
        addPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::INT32>(map);
        addPlanBuilder<DataType::HALF, DataType::HALF, DataType::FLOAT, DataType::INT32>(map);
        addPlanBuilder<DataType::BFLOAT16, DataType::BFLOAT16, DataType::FLOAT, DataType::INT32>(
            map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType DyDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType DxDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType IndexDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<GpuResampleBwdSignatureKey,
                                                  std::unique_ptr<IGpuGraphNodePlanBuilder>,
                                                  GpuResampleBwdSignatureKey>& map)
    {
        map[GpuResampleBwdSignatureKey(
            DyDataTypeEnum, DxDataTypeEnum, ComputeDataTypeEnum, IndexDataTypeEnum)]
            = std::make_unique<GpuResampleBwdPlanBuilder<DyDataTypeEnum,
                                                         DxDataTypeEnum,
                                                         ComputeDataTypeEnum,
                                                         IndexDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const GpuResampleBwdSignatureKey& key)
{
    os << "GpuResampleBwdSignatureKey(dyDataType=" << static_cast<int>(key.dyDataType)
       << ", dxDataType=" << static_cast<int>(key.dxDataType)
       << ", computeDataType=" << static_cast<int>(key.computeDataType)
       << ", indexDataType=" << static_cast<int>(key.indexDataType) << ")";
    return os;
}

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
