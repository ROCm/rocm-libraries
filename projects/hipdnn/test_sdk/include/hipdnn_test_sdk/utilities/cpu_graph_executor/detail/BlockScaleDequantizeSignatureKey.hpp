// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BlockScaleDequantizePlan.hpp>
#include <ostream>

namespace hipdnn_test_sdk::detail
{

struct BlockScaleDequantizeSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType
        = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BlockScaleDequantizeAttributes;
    hipdnn_flatbuffers_sdk::data_objects::DataType xDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType scaleDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType outputDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType;

    BlockScaleDequantizeSignatureKey() = default;
    constexpr BlockScaleDequantizeSignatureKey(
        hipdnn_flatbuffers_sdk::data_objects::DataType x,
        hipdnn_flatbuffers_sdk::data_objects::DataType scale,
        hipdnn_flatbuffers_sdk::data_objects::DataType output,
        hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : xDataType(x)
        , scaleDataType(scale)
        , outputDataType(output)
        , computeDataType(compute)
    {
    }

    BlockScaleDequantizeSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        const auto* nodeAttributes = node.attributes_as_BlockScaleDequantizeAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes could not be cast to BlockScaleDequantizeAttributes");
        }

        auto xTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
        auto scaleTensorAttr = tensorMap.at(nodeAttributes->scale_tensor_uid());
        auto yTensorAttr = tensorMap.at(nodeAttributes->y_tensor_uid());

        if(xTensorAttr == nullptr || scaleTensorAttr == nullptr || yTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        xDataType = xTensorAttr->data_type();
        scaleDataType = scaleTensorAttr->data_type();
        computeDataType = node.compute_data_type();
        outputDataType = yTensorAttr->data_type();
    }

    std::size_t operator()(const BlockScaleDequantizeSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(xDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(scaleDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(outputDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 16);
    }

    bool operator==(const BlockScaleDequantizeSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && xDataType == other.xDataType
               && scaleDataType == other.scaleDataType && outputDataType == other.outputDataType
               && computeDataType == other.computeDataType;
    }

    static std::unordered_map<BlockScaleDequantizeSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              BlockScaleDequantizeSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<BlockScaleDequantizeSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           BlockScaleDequantizeSignatureKey>
            map;

        // Float/Float/Float/Float (matches integration test defaults)
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        // Half input with float scale/output/compute
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        // BFloat16 input with float scale/output/compute
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        // Half input with float scale, half output, float compute
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        // BFloat16 input with float scale, bfloat16 output, float compute
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        // FP8 E4M3 input with E8M0 scale, float output/compute (real MX dequantize)
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        // FP8 E5M2 input with E8M0 scale, float output/compute (real MX dequantize)
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E5M2,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        // FP8 E4M3 input with E8M0 scale, half output, float compute
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        // FP8 E5M2 input with E8M0 scale, half output, float compute
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E5M2,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        // FP4 E2M1 input with E8M0 scale, float output/compute (MX dequantize)
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FP4_E2M1,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        // FP4 E2M1 input with E8M0 scale, half output, float compute
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FP4_E2M1,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        // FP6 E2M3 input with E8M0 scale, float output/compute (MX dequantize)
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FP6_E2M3,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        // FP6 E2M3 input with E8M0 scale, half output, float compute
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FP6_E2M3,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        // FP6 E3M2 input with E8M0 scale, float output/compute (MX dequantize)
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FP6_E3M2,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        // FP6 E3M2 input with E8M0 scale, half output, float compute
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FP6_E3M2,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ScaleDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<BlockScaleDequantizeSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  BlockScaleDequantizeSignatureKey>& map)
    {
        map[BlockScaleDequantizeSignatureKey(
            XDataTypeEnum, ScaleDataTypeEnum, OutputDataTypeEnum, ComputeDataTypeEnum)]
            = std::make_unique<BlockScaleDequantizePlanBuilder<XDataTypeEnum,
                                                               ScaleDataTypeEnum,
                                                               OutputDataTypeEnum,
                                                               ComputeDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const BlockScaleDequantizeSignatureKey& key)
{
    os << "BlockScaleDequantize(x=" << key.xDataType << ", scale=" << key.scaleDataType
       << ", y=" << key.outputDataType << ", compute=" << key.computeDataType << ")";
    return os;
}

} // namespace hipdnn_test_sdk::detail
