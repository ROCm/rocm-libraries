// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "GpuResampleFwdPlan.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

struct GpuResampleFwdSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ResampleFwdAttributes};
    hipdnn_flatbuffers_sdk::data_objects::DataType xDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType yDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType indexDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};

    GpuResampleFwdSignatureKey() = default;
    constexpr GpuResampleFwdSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType x,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType y,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType compute,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType index)
        : xDataType(x)
        , yDataType(y)
        , computeDataType(compute)
        , indexDataType(index)
    {
    }

    GpuResampleFwdSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_flatbuffers_sdk::data_objects::DataType computeType)
    {
        const auto* nodeAttributes = node.attributes_as_ResampleFwdAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes could not be cast to ResampleFwdAttributes");
        }

        const auto* xTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
        const auto* yTensorAttr = tensorMap.at(nodeAttributes->y_tensor_uid());
        if(xTensorAttr == nullptr || yTensorAttr == nullptr)
        {
            throw std::runtime_error(
                "One or more required tensor attributes could not be found in the map, "
                "failed to construct key");
        }

        xDataType = xTensorAttr->data_type();
        yDataType = yTensorAttr->data_type();
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

    std::size_t operator()(const GpuResampleFwdSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(xDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(yDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(indexDataType)) << 16);
    }

    bool operator==(const GpuResampleFwdSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && xDataType == other.xDataType
               && yDataType == other.yDataType && computeDataType == other.computeDataType
               && indexDataType == other.indexDataType;
    }

    static std::unordered_map<GpuResampleFwdSignatureKey,
                              std::unique_ptr<IGpuGraphNodePlanBuilder>,
                              GpuResampleFwdSignatureKey>
        getPlanBuilders()
    {
        using hipdnn_flatbuffers_sdk::data_objects::DataType;

        std::unordered_map<GpuResampleFwdSignatureKey,
                           std::unique_ptr<IGpuGraphNodePlanBuilder>,
                           GpuResampleFwdSignatureKey>
            map;

        // Without index output (average pooling, or max pooling with generate_index unset)
        addPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET>(map);
        addPlanBuilder<DataType::HALF, DataType::HALF, DataType::FLOAT, DataType::UNSET>(map);
        addPlanBuilder<DataType::BFLOAT16, DataType::BFLOAT16, DataType::FLOAT, DataType::UNSET>(
            map);

        // With INT32 index output (max pooling)
        addPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::INT32>(map);
        addPlanBuilder<DataType::HALF, DataType::HALF, DataType::FLOAT, DataType::INT32>(map);
        addPlanBuilder<DataType::BFLOAT16, DataType::BFLOAT16, DataType::FLOAT, DataType::INT32>(
            map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType YDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType IndexDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<GpuResampleFwdSignatureKey,
                                                  std::unique_ptr<IGpuGraphNodePlanBuilder>,
                                                  GpuResampleFwdSignatureKey>& map)
    {
        map[GpuResampleFwdSignatureKey(
            XDataTypeEnum, YDataTypeEnum, ComputeDataTypeEnum, IndexDataTypeEnum)]
            = std::make_unique<GpuResampleFwdPlanBuilder<XDataTypeEnum,
                                                         YDataTypeEnum,
                                                         ComputeDataTypeEnum,
                                                         IndexDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const GpuResampleFwdSignatureKey& key)
{
    os << "GpuResampleFwdSignatureKey(xDataType=" << static_cast<int>(key.xDataType)
       << ", yDataType=" << static_cast<int>(key.yDataType)
       << ", computeDataType=" << static_cast<int>(key.computeDataType)
       << ", indexDataType=" << static_cast<int>(key.indexDataType) << ")";
    return os;
}

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
