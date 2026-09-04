// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "GpuLayernormBwdPlan.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

struct GpuLayernormBwdSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::LayernormBackwardAttributes};
    hipdnn_flatbuffers_sdk::data_objects::DataType dyDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType scaleBiasDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType meanInvVarianceDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType dxDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};

    GpuLayernormBwdSignatureKey() = default;
    constexpr GpuLayernormBwdSignatureKey(
        hipdnn_flatbuffers_sdk::data_objects::DataType dy,
        hipdnn_flatbuffers_sdk::data_objects::DataType scaleBias,
        hipdnn_flatbuffers_sdk::data_objects::DataType meanInvVariance,
        hipdnn_flatbuffers_sdk::data_objects::DataType dx,
        hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : dyDataType(dy)
        , scaleBiasDataType(scaleBias)
        , meanInvVarianceDataType(meanInvVariance)
        , dxDataType(dx)
        , computeDataType(compute)
    {
    }

    GpuLayernormBwdSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_flatbuffers_sdk::data_objects::DataType computeType)
    {
        const auto* nodeAttributes = node.attributes_as_LayernormBackwardAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes could not be cast to LayernormBackwardAttributes");
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
        dxDataType = dxTensorAttr->data_type();
        scaleBiasDataType = scaleTensorAttr->data_type();
        if(nodeAttributes->mean_tensor_uid().has_value())
        {
            auto meanTensorAttr = tensorMap.at(nodeAttributes->mean_tensor_uid().value());
            meanInvVarianceDataType = meanTensorAttr->data_type();
        }
        else
        {
            meanInvVarianceDataType = dxDataType;
        }
        computeDataType = computeType;
    }

    std::size_t operator()(const GpuLayernormBwdSignatureKey& key) const noexcept
    {
        return key.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(nodeType) ^ (static_cast<std::size_t>(dyDataType) << 4)
               ^ (static_cast<std::size_t>(scaleBiasDataType) << 8)
               ^ (static_cast<std::size_t>(meanInvVarianceDataType) << 12)
               ^ (static_cast<std::size_t>(dxDataType) << 16)
               ^ (static_cast<std::size_t>(computeDataType) << 20);
    }

    bool operator==(const GpuLayernormBwdSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && dyDataType == other.dyDataType
               && scaleBiasDataType == other.scaleBiasDataType
               && meanInvVarianceDataType == other.meanInvVarianceDataType
               && dxDataType == other.dxDataType && computeDataType == other.computeDataType;
    }

    static std::unordered_map<GpuLayernormBwdSignatureKey,
                              std::unique_ptr<IGpuGraphNodePlanBuilder>,
                              GpuLayernormBwdSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<GpuLayernormBwdSignatureKey,
                           std::unique_ptr<IGpuGraphNodePlanBuilder>,
                           GpuLayernormBwdSignatureKey>
            map;

        // X, Scale/Bias, Mean/Inverse variance, Y, Compute
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType DyDataType,
              hipdnn_flatbuffers_sdk::data_objects::DataType ScaleBiasDataType,
              hipdnn_flatbuffers_sdk::data_objects::DataType MeanInvVarianceDataType,
              hipdnn_flatbuffers_sdk::data_objects::DataType DxDataType,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataType>
    static void addPlanBuilder(std::unordered_map<GpuLayernormBwdSignatureKey,
                                                  std::unique_ptr<IGpuGraphNodePlanBuilder>,
                                                  GpuLayernormBwdSignatureKey>& map)
    {
        // With optional mean/rstd tensors
        map[GpuLayernormBwdSignatureKey(
            DyDataType, ScaleBiasDataType, MeanInvVarianceDataType, DxDataType, ComputeDataType)]
            = std::make_unique<GpuLayernormBwdPlanBuilder<DyDataType,
                                                          ScaleBiasDataType,
                                                          MeanInvVarianceDataType,
                                                          DxDataType,
                                                          ComputeDataType>>();

        if constexpr(DxDataType != MeanInvVarianceDataType)
        {
            // Without optional mean/rstd tensors: match dx as UNSET (void) is not allowed
            map[GpuLayernormBwdSignatureKey(
                DyDataType, ScaleBiasDataType, DxDataType, DxDataType, ComputeDataType)]
                = std::make_unique<GpuLayernormBwdPlanBuilder<DyDataType,
                                                              ScaleBiasDataType,
                                                              DxDataType,
                                                              DxDataType,
                                                              ComputeDataType>>();
        }
    }
};

inline std::ostream& operator<<(std::ostream& os, const GpuLayernormBwdSignatureKey& key)
{
    os << "GpuLayernormBwd(dy=" << key.dyDataType << ", scaleBias=" << key.scaleBiasDataType
       << ", meanInvVar=" << key.meanInvVarianceDataType << ", dx=" << key.dxDataType
       << ", compute=" << key.computeDataType << ")";
    return os;
}

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
