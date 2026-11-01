// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "GpuConvolutionBwdPlan.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

struct GpuConvolutionBwdSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes};
    hipdnn_flatbuffers_sdk::data_objects::DataType dyDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType wDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType dxDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};

    GpuConvolutionBwdSignatureKey() = default;
    constexpr GpuConvolutionBwdSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType dy,
                                            hipdnn_flatbuffers_sdk::data_objects::DataType w,
                                            hipdnn_flatbuffers_sdk::data_objects::DataType dx,
                                            hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : dyDataType(dy)
        , wDataType(w)
        , dxDataType(dx)
        , computeDataType(compute)
    {
    }

    GpuConvolutionBwdSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_flatbuffers_sdk::data_objects::DataType computeType)
    {
        const auto* nodeAttributes = node.attributes_as_ConvolutionBwdAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes could not be cast to ConvolutionBwdAttributes");
        }

        auto dyTensorAttr = tensorMap.at(nodeAttributes->dy_tensor_uid());
        auto wTensorAttr = tensorMap.at(nodeAttributes->w_tensor_uid());
        auto dxTensorAttr = tensorMap.at(nodeAttributes->dx_tensor_uid());
        if(dyTensorAttr == nullptr || wTensorAttr == nullptr || dxTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        dyDataType = dyTensorAttr->data_type();
        wDataType = wTensorAttr->data_type();
        dxDataType = dxTensorAttr->data_type();
        computeDataType = computeType;
    }

    std::size_t operator()(const GpuConvolutionBwdSignatureKey& key) const noexcept
    {
        return key.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(dyDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(wDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(dxDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 16);
    }

    bool operator==(const GpuConvolutionBwdSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && dyDataType == other.dyDataType
               && wDataType == other.wDataType && dxDataType == other.dxDataType
               && computeDataType == other.computeDataType;
    }

    static std::unordered_map<GpuConvolutionBwdSignatureKey,
                              std::unique_ptr<IGpuGraphNodePlanBuilder>,
                              GpuConvolutionBwdSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<GpuConvolutionBwdSignatureKey,
                           std::unique_ptr<IGpuGraphNodePlanBuilder>,
                           GpuConvolutionBwdSignatureKey>
            map;

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType DyDataType,
              hipdnn_flatbuffers_sdk::data_objects::DataType WDataType,
              hipdnn_flatbuffers_sdk::data_objects::DataType DxDataType,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataType>
    static void addPlanBuilder(std::unordered_map<GpuConvolutionBwdSignatureKey,
                                                  std::unique_ptr<IGpuGraphNodePlanBuilder>,
                                                  GpuConvolutionBwdSignatureKey>& map)
    {
        map[GpuConvolutionBwdSignatureKey(DyDataType, WDataType, DxDataType, ComputeDataType)]
            = std::make_unique<
                GpuConvolutionBwdPlanBuilder<DyDataType, WDataType, DxDataType, ComputeDataType>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const GpuConvolutionBwdSignatureKey& key)
{
    os << "GpuConvolutionBwd(dy=" << key.dyDataType << ", w=" << key.wDataType
       << ", dx=" << key.dxDataType << ", compute=" << key.computeDataType << ")";
    return os;
}

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
