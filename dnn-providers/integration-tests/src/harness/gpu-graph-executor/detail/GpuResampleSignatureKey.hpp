// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "GpuResamplePlan.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

struct GpuResampleFwdSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType
        = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ResampleFwdAttributes;
    hipdnn_flatbuffers_sdk::data_objects::DataType xDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType yDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType indexDataType;

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

        auto xTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
        auto yTensorAttr = tensorMap.at(nodeAttributes->y_tensor_uid());
        if(xTensorAttr == nullptr || yTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        xDataType = xTensorAttr->data_type();
        yDataType = yTensorAttr->data_type();
        computeDataType = computeType;
        indexDataType = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
        if(nodeAttributes->index_tensor_uid().has_value())
        {
            auto indexTensorAttr = tensorMap.at(nodeAttributes->index_tensor_uid().value());
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
        std::unordered_map<GpuResampleFwdSignatureKey,
                           std::unique_ptr<IGpuGraphNodePlanBuilder>,
                           GpuResampleFwdSignatureKey>
            map;

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT32>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT32>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT32>(map);

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
    os << "GpuResampleFwdSignatureKey(xDataType=" << key.xDataType
       << ", yDataType=" << key.yDataType << ", computeDataType=" << key.computeDataType
       << ", indexDataType=" << key.indexDataType << ")";
    return os;
}

struct GpuResampleBwdSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType
        = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ResampleBwdAttributes;
    hipdnn_flatbuffers_sdk::data_objects::DataType dyDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType dxDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType indexDataType;

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

        auto dyTensorAttr = tensorMap.at(nodeAttributes->dy_tensor_uid());
        auto dxTensorAttr = tensorMap.at(nodeAttributes->dx_tensor_uid());
        if(dyTensorAttr == nullptr || dxTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        dyDataType = dyTensorAttr->data_type();
        dxDataType = dxTensorAttr->data_type();
        computeDataType = computeType;
        indexDataType = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
        if(nodeAttributes->index_tensor_uid().has_value())
        {
            auto indexTensorAttr = tensorMap.at(nodeAttributes->index_tensor_uid().value());
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
        std::unordered_map<GpuResampleBwdSignatureKey,
                           std::unique_ptr<IGpuGraphNodePlanBuilder>,
                           GpuResampleBwdSignatureKey>
            map;

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT32>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT32>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT32>(map);

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
    os << "GpuResampleBwdSignatureKey(dyDataType=" << key.dyDataType
       << ", dxDataType=" << key.dxDataType << ", computeDataType=" << key.computeDataType
       << ", indexDataType=" << key.indexDataType << ")";
    return os;
}

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
