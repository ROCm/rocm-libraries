// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/RMSNormBwdPlan.hpp>
#include <ostream>

namespace hipdnn_test_sdk::detail
{

struct RMSNormBwdSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType
        = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormBwdAttributes;

    hipdnn_flatbuffers_sdk::data_objects::DataType dyDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType xDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType scaleDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType dxDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType;

    RMSNormBwdSignatureKey() = default;
    constexpr RMSNormBwdSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType dy,
                                     hipdnn_flatbuffers_sdk::data_objects::DataType x,
                                     hipdnn_flatbuffers_sdk::data_objects::DataType scale,
                                     hipdnn_flatbuffers_sdk::data_objects::DataType dx,
                                     hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : dyDataType(dy)
        , xDataType(x)
        , scaleDataType(scale)
        , dxDataType(dx)
        , computeDataType(compute)
    {
    }

    RMSNormBwdSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        const auto* nodeAttributes = node.attributes_as_RMSNormBwdAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes could not be cast to RMSNormBwdAttributes");
        }

        auto dyTensorAttr = tensorMap.at(nodeAttributes->dy_tensor_uid());
        auto xTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
        auto scaleTensorAttr = tensorMap.at(nodeAttributes->scale_tensor_uid());
        auto dxTensorAttr = tensorMap.at(nodeAttributes->dx_tensor_uid());

        if(dyTensorAttr == nullptr || xTensorAttr == nullptr || scaleTensorAttr == nullptr
           || dxTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        dyDataType = dyTensorAttr->data_type();
        xDataType = xTensorAttr->data_type();
        scaleDataType = scaleTensorAttr->data_type();
        dxDataType = dxTensorAttr->data_type();
        computeDataType = node.compute_data_type();
    }

    std::size_t operator()(const RMSNormBwdSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(dyDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(xDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(scaleDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(dxDataType)) << 16)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 20);
    }

    bool operator==(const RMSNormBwdSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && dyDataType == other.dyDataType
               && xDataType == other.xDataType && scaleDataType == other.scaleDataType
               && dxDataType == other.dxDataType && computeDataType == other.computeDataType;
    }

    static std::unordered_map<RMSNormBwdSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              RMSNormBwdSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<RMSNormBwdSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           RMSNormBwdSignatureKey>
            map;

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType DyDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ScaleDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType DxDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<RMSNormBwdSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  RMSNormBwdSignatureKey>& map)
    {
        map[RMSNormBwdSignatureKey(
            DyDataTypeEnum, XDataTypeEnum, ScaleDataTypeEnum, DxDataTypeEnum, ComputeDataTypeEnum)]
            = std::make_unique<RMSNormBwdPlanBuilder<DyDataTypeEnum,
                                                     XDataTypeEnum,
                                                     ScaleDataTypeEnum,
                                                     DxDataTypeEnum,
                                                     ComputeDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const RMSNormBwdSignatureKey& key)
{
    os << "RMSNormBwd(dy=" << key.dyDataType << ", x=" << key.xDataType
       << ", scale=" << key.scaleDataType << ", dx=" << key.dxDataType
       << ", compute=" << key.computeDataType << ")";
    return os;
}

} // namespace hipdnn_test_sdk::detail
