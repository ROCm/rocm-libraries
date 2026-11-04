// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <functional>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ResampleBwdPlan.hpp>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <unordered_map>

namespace hipdnn_test_sdk::detail
{

struct ResampleBwdSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType
        = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ResampleBwdAttributes;
    hipdnn_flatbuffers_sdk::data_objects::DataType dyDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType dxDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType indexDataType;

    ResampleBwdSignatureKey() = default;
    constexpr ResampleBwdSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType dy,
                                      hipdnn_flatbuffers_sdk::data_objects::DataType dx,
                                      hipdnn_flatbuffers_sdk::data_objects::DataType compute,
                                      hipdnn_flatbuffers_sdk::data_objects::DataType index)
        : dyDataType(dy)
        , dxDataType(dx)
        , computeDataType(compute)
        , indexDataType(index)
    {
    }

    ResampleBwdSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap)
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
        computeDataType = node.compute_data_type();
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

    std::size_t operator()(const ResampleBwdSignatureKey& k) const noexcept
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

    bool operator==(const ResampleBwdSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && dyDataType == other.dyDataType
               && dxDataType == other.dxDataType && computeDataType == other.computeDataType
               && indexDataType == other.indexDataType;
    }

    static std::unordered_map<ResampleBwdSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              ResampleBwdSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<ResampleBwdSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           ResampleBwdSignatureKey>
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
    static void addPlanBuilder(std::unordered_map<ResampleBwdSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  ResampleBwdSignatureKey>& map)
    {
        map[ResampleBwdSignatureKey(
            DyDataTypeEnum, DxDataTypeEnum, ComputeDataTypeEnum, IndexDataTypeEnum)]
            = std::make_unique<ResampleBwdPlanBuilder<DyDataTypeEnum,
                                                      DxDataTypeEnum,
                                                      ComputeDataTypeEnum,
                                                      IndexDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const ResampleBwdSignatureKey& key)
{
    os << "ResampleBwd(dy=" << key.dyDataType << ", dx=" << key.dxDataType
       << ", compute=" << key.computeDataType << ", index=" << key.indexDataType << ")";
    return os;
}

} // namespace hipdnn_test_sdk::detail
