// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>

#include "GpuBatchnormBwdPlan.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

struct GpuBatchnormBwdSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes};
    hipdnn_flatbuffers_sdk::data_objects::DataType dyDataType{};
    hipdnn_flatbuffers_sdk::data_objects::DataType xDataType{};
    hipdnn_flatbuffers_sdk::data_objects::DataType scaleBiasDataType{};
    hipdnn_flatbuffers_sdk::data_objects::DataType meanVarianceDataType{};
    hipdnn_flatbuffers_sdk::data_objects::DataType dxDataType{};
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType{};

    GpuBatchnormBwdSignatureKey() = default;
    constexpr GpuBatchnormBwdSignatureKey(
        hipdnn_flatbuffers_sdk::data_objects::DataType dy,
        hipdnn_flatbuffers_sdk::data_objects::DataType x,
        hipdnn_flatbuffers_sdk::data_objects::DataType scaleBias,
        hipdnn_flatbuffers_sdk::data_objects::DataType meanVariance,
        hipdnn_flatbuffers_sdk::data_objects::DataType dx,
        hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : dyDataType(dy)
        , xDataType(x)
        , scaleBiasDataType(scaleBias)
        , meanVarianceDataType(meanVariance)
        , dxDataType(dx)
        , computeDataType(compute)
    {
    }

    GpuBatchnormBwdSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        const auto* attributes = node.attributes_as_BatchnormBackwardAttributes();
        if(attributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes could not be cast to BatchnormBackwardAttributes");
        }

        const auto* dy = tensorMap.at(attributes->dy_tensor_uid());
        const auto* x = tensorMap.at(attributes->x_tensor_uid());
        const auto* scale = tensorMap.at(attributes->scale_tensor_uid());
        const auto* dx = tensorMap.at(attributes->dx_tensor_uid());
        if(dy == nullptr || x == nullptr || scale == nullptr || dx == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        dyDataType = dy->data_type();
        xDataType = x->data_type();
        scaleBiasDataType = scale->data_type();
        dxDataType = dx->data_type();
        computeDataType = node.compute_data_type();

        if(attributes->mean_tensor_uid().has_value()
           && attributes->inv_variance_tensor_uid().has_value())
        {
            const auto* mean = tensorMap.at(attributes->mean_tensor_uid().value());
            const auto* invVariance = tensorMap.at(attributes->inv_variance_tensor_uid().value());
            if(mean->data_type() != invVariance->data_type())
            {
                throw std::runtime_error(
                    "GpuBatchnormBwdSignatureKey requires mean and inv_variance tensors "
                    "to have the same data type");
            }
            meanVarianceDataType = mean->data_type();
        }
        else
        {
            meanVarianceDataType = scaleBiasDataType;
        }
    }

    std::size_t operator()(const GpuBatchnormBwdSignatureKey& key) const noexcept
    {
        return key.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(dyDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(xDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(scaleBiasDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(meanVarianceDataType)) << 16)
               ^ (static_cast<std::size_t>(static_cast<int>(dxDataType)) << 20)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 24);
    }

    bool operator==(const GpuBatchnormBwdSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && dyDataType == other.dyDataType
               && xDataType == other.xDataType && scaleBiasDataType == other.scaleBiasDataType
               && meanVarianceDataType == other.meanVarianceDataType
               && dxDataType == other.dxDataType && computeDataType == other.computeDataType;
    }

    static std::unordered_map<GpuBatchnormBwdSignatureKey,
                              std::unique_ptr<IGpuGraphNodePlanBuilder>,
                              GpuBatchnormBwdSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<GpuBatchnormBwdSignatureKey,
                           std::unique_ptr<IGpuGraphNodePlanBuilder>,
                           GpuBatchnormBwdSignatureKey>
            map;

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType DyDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ScaleBiasDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType MeanVarianceDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType DxDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<GpuBatchnormBwdSignatureKey,
                                                  std::unique_ptr<IGpuGraphNodePlanBuilder>,
                                                  GpuBatchnormBwdSignatureKey>& map)
    {
        map[GpuBatchnormBwdSignatureKey(DyDataTypeEnum,
                                        XDataTypeEnum,
                                        ScaleBiasDataTypeEnum,
                                        MeanVarianceDataTypeEnum,
                                        DxDataTypeEnum,
                                        ComputeDataTypeEnum)]
            = std::make_unique<GpuBatchnormBwdPlanBuilder<DyDataTypeEnum,
                                                          XDataTypeEnum,
                                                          ScaleBiasDataTypeEnum,
                                                          MeanVarianceDataTypeEnum,
                                                          DxDataTypeEnum,
                                                          ComputeDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const GpuBatchnormBwdSignatureKey& key)
{
    os << "GpuBatchnormBwd(dy=" << key.dyDataType << ", x=" << key.xDataType
       << ", scale=" << key.scaleBiasDataType << ", mean=" << key.meanVarianceDataType
       << ", dx=" << key.dxDataType << ", compute=" << key.computeDataType << ")";
    return os;
}

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
