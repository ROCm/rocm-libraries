// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "GpuConvolutionWrwPlan.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

struct GpuConvolutionWrwSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes};
    hipdnn_flatbuffers_sdk::data_objects::DataType xDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType dyDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType dwDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};

    GpuConvolutionWrwSignatureKey() = default;
    constexpr GpuConvolutionWrwSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType x,
                                            hipdnn_flatbuffers_sdk::data_objects::DataType dy,
                                            hipdnn_flatbuffers_sdk::data_objects::DataType dw,
                                            hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : xDataType(x)
        , dyDataType(dy)
        , dwDataType(dw)
        , computeDataType(compute)
    {
    }

    GpuConvolutionWrwSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_flatbuffers_sdk::data_objects::DataType computeType)
    {
        const auto* nodeAttributes = node.attributes_as_ConvolutionWrwAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes could not be cast to ConvolutionWrwAttributes");
        }

        auto xTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
        auto dyTensorAttr = tensorMap.at(nodeAttributes->dy_tensor_uid());
        auto dwTensorAttr = tensorMap.at(nodeAttributes->dw_tensor_uid());
        if(xTensorAttr == nullptr || dyTensorAttr == nullptr || dwTensorAttr == nullptr)
        {
            throw std::runtime_error(
                "One or more tensor attributes could be found in the map, failed to construct key");
        }

        xDataType = xTensorAttr->data_type();
        dyDataType = dyTensorAttr->data_type();
        dwDataType = dwTensorAttr->data_type();
        computeDataType = computeType;
    }

    std::size_t operator()(const GpuConvolutionWrwSignatureKey& key) const noexcept
    {
        return key.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(xDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(dyDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(dwDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 16);
    }

    bool operator==(const GpuConvolutionWrwSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && xDataType == other.xDataType
               && dyDataType == other.dyDataType && dwDataType == other.dwDataType
               && computeDataType == other.computeDataType;
    }

    static std::unordered_map<GpuConvolutionWrwSignatureKey,
                              std::unique_ptr<IGpuGraphNodePlanBuilder>,
                              GpuConvolutionWrwSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<GpuConvolutionWrwSignatureKey,
                           std::unique_ptr<IGpuGraphNodePlanBuilder>,
                           GpuConvolutionWrwSignatureKey>
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

    template <hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType DyDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType DwDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<GpuConvolutionWrwSignatureKey,
                                                  std::unique_ptr<IGpuGraphNodePlanBuilder>,
                                                  GpuConvolutionWrwSignatureKey>& map)
    {
        map[GpuConvolutionWrwSignatureKey(
            XDataTypeEnum, DyDataTypeEnum, DwDataTypeEnum, ComputeDataTypeEnum)]
            = std::make_unique<GpuConvolutionWrwPlanBuilder<XDataTypeEnum,
                                                            DyDataTypeEnum,
                                                            DwDataTypeEnum,
                                                            ComputeDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const GpuConvolutionWrwSignatureKey& key)
{
    os << "GpuConvolutionWrw(x=" << key.xDataType << ", dy=" << key.dyDataType
       << ", dw=" << key.dwDataType << ", compute=" << key.computeDataType << ")";
    return os;
}

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
