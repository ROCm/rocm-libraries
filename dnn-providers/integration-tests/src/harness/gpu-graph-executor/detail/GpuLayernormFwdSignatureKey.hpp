// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "GpuLayernormFwdPlan.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

struct GpuLayernormFwdSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::LayernormAttributes};
    hipdnn_flatbuffers_sdk::data_objects::DataType xDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType scaleBiasDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType meanInvVarianceDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType yDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};

    GpuLayernormFwdSignatureKey() = default;
    constexpr GpuLayernormFwdSignatureKey(
        hipdnn_flatbuffers_sdk::data_objects::DataType x,
        hipdnn_flatbuffers_sdk::data_objects::DataType scaleBias,
        hipdnn_flatbuffers_sdk::data_objects::DataType meanInvVariance,
        hipdnn_flatbuffers_sdk::data_objects::DataType y,
        hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : xDataType(x)
        , scaleBiasDataType(scaleBias)
        , meanInvVarianceDataType(meanInvVariance)
        , yDataType(y)
        , computeDataType(compute)
    {
    }

    GpuLayernormFwdSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_flatbuffers_sdk::data_objects::DataType computeType)
    {
        const auto* nodeAttributes = node.attributes_as_LayernormAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes could not be cast to LayernormAttributes");
        }

        auto xTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
        auto yTensorAttr = tensorMap.at(nodeAttributes->y_tensor_uid());
        auto scaleTensorAttr = tensorMap.at(nodeAttributes->scale_tensor_uid());
        if(xTensorAttr == nullptr || scaleTensorAttr == nullptr || yTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        xDataType = xTensorAttr->data_type();
        scaleBiasDataType = scaleTensorAttr->data_type();
        if(nodeAttributes->mean_tensor_uid().has_value())
        {
            auto meanTensorAttr = tensorMap.at(nodeAttributes->mean_tensor_uid().value());
            meanInvVarianceDataType = meanTensorAttr->data_type();
        }
        else
        {
            meanInvVarianceDataType = xDataType;
        }
        yDataType = yTensorAttr->data_type();
        computeDataType = computeType;
    }

    std::size_t operator()(const GpuLayernormFwdSignatureKey& key) const noexcept
    {
        return key.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(nodeType) ^ (static_cast<std::size_t>(xDataType) << 4)
               ^ (static_cast<std::size_t>(scaleBiasDataType) << 8)
               ^ (static_cast<std::size_t>(meanInvVarianceDataType) << 12)
               ^ (static_cast<std::size_t>(yDataType) << 16)
               ^ (static_cast<std::size_t>(computeDataType) << 20);
    }

    bool operator==(const GpuLayernormFwdSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && xDataType == other.xDataType
               && scaleBiasDataType == other.scaleBiasDataType
               && meanInvVarianceDataType == other.meanInvVarianceDataType
               && yDataType == other.yDataType && computeDataType == other.computeDataType;
    }

    static std::unordered_map<GpuLayernormFwdSignatureKey,
                              std::unique_ptr<IGpuGraphNodePlanBuilder>,
                              GpuLayernormFwdSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<GpuLayernormFwdSignatureKey,
                           std::unique_ptr<IGpuGraphNodePlanBuilder>,
                           GpuLayernormFwdSignatureKey>
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

    template <hipdnn_flatbuffers_sdk::data_objects::DataType XDataType,
              hipdnn_flatbuffers_sdk::data_objects::DataType ScaleBiasDataType,
              hipdnn_flatbuffers_sdk::data_objects::DataType MeanInvVarianceDataType,
              hipdnn_flatbuffers_sdk::data_objects::DataType YDataType,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataType>
    static void addPlanBuilder(std::unordered_map<GpuLayernormFwdSignatureKey,
                                                  std::unique_ptr<IGpuGraphNodePlanBuilder>,
                                                  GpuLayernormFwdSignatureKey>& map)
    {
        // With optional mean/rstd tensors
        map[GpuLayernormFwdSignatureKey(
            XDataType, ScaleBiasDataType, MeanInvVarianceDataType, YDataType, ComputeDataType)]
            = std::make_unique<GpuLayernormFwdPlanBuilder<XDataType,
                                                          ScaleBiasDataType,
                                                          MeanInvVarianceDataType,
                                                          YDataType,
                                                          ComputeDataType>>();

        if constexpr(XDataType != MeanInvVarianceDataType)
        {
            // Without optional mean/rstd tensors: match dx as UNSET (void) is not allowed
            map[GpuLayernormFwdSignatureKey(
                XDataType, ScaleBiasDataType, XDataType, YDataType, ComputeDataType)]
                = std::make_unique<GpuLayernormFwdPlanBuilder<XDataType,
                                                              ScaleBiasDataType,
                                                              XDataType,
                                                              YDataType,
                                                              ComputeDataType>>();
        }
    }
};

inline std::ostream& operator<<(std::ostream& os, const GpuLayernormFwdSignatureKey& key)
{
    os << "GpuLayernormFwd(x=" << key.xDataType << ", scaleBias=" << key.scaleBiasDataType
       << ", meanInvVar=" << key.meanInvVarianceDataType << ", y=" << key.yDataType
       << ", compute=" << key.computeDataType << ")";
    return os;
}

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
