// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BlockScaleQuantizePlan.hpp>
#include <ostream>

namespace hipdnn_test_sdk::detail
{

struct BlockScaleQuantizeSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType
        = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BlockScaleQuantizeAttributes;
    hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType outputDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType scaleDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType;

    BlockScaleQuantizeSignatureKey() = default;

    constexpr BlockScaleQuantizeSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType input,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType output,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType scale,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : inputDataType(input)
        , outputDataType(output)
        , scaleDataType(scale)
        , computeDataType(compute)
    {
    }

    BlockScaleQuantizeSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        const auto* nodeAttributes = node.attributes_as_BlockScaleQuantizeAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes could not be cast to BlockScaleQuantizeAttributes");
        }

        auto xTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
        auto scaleTensorAttr = tensorMap.at(nodeAttributes->scale_tensor_uid());
        auto yTensorAttr = tensorMap.at(nodeAttributes->y_tensor_uid());

        if(xTensorAttr == nullptr || yTensorAttr == nullptr || scaleTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        inputDataType = xTensorAttr->data_type();
        outputDataType = yTensorAttr->data_type();
        scaleDataType = scaleTensorAttr->data_type();
        computeDataType = node.compute_data_type();
    }

    std::size_t operator()(const BlockScaleQuantizeSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(inputDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(outputDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(scaleDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 16);
    }

    bool operator==(const BlockScaleQuantizeSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && inputDataType == other.inputDataType
               && outputDataType == other.outputDataType && scaleDataType == other.scaleDataType
               && computeDataType == other.computeDataType;
    }

    static std::unordered_map<BlockScaleQuantizeSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              BlockScaleQuantizeSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<BlockScaleQuantizeSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           BlockScaleQuantizeSignatureKey>
            map;

        // Float/Float/Float/Float
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        // Float input, Float scale, Half/BFloat16 output
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        // Float input, E8M0 scale, FP8 output variants
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E5M2,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        // Half input, E8M0 scale, FP8 output variants
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E5M2,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        // BFloat16 input, E8M0 scale, FP8 output variants
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E5M2,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        // Float input, E8M0 scale, FP4/FP6 output variants
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP4_E2M1,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP6_E2M3,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP6_E3M2,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        // Half input, E8M0 scale, FP4/FP6 output variants
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP4_E2M1,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP6_E2M3,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP6_E3M2,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        // BFloat16 input, E8M0 scale, FP4/FP6 output variants
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP4_E2M1,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP6_E2M3,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP6_E3M2,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType InputDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ScaleDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<BlockScaleQuantizeSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  BlockScaleQuantizeSignatureKey>& map)
    {
        map[BlockScaleQuantizeSignatureKey(
            InputDataTypeEnum, OutputDataTypeEnum, ScaleDataTypeEnum, ComputeDataTypeEnum)]
            = std::make_unique<BlockScaleQuantizePlanBuilder<InputDataTypeEnum,
                                                             OutputDataTypeEnum,
                                                             ScaleDataTypeEnum,
                                                             ComputeDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const BlockScaleQuantizeSignatureKey& key)
{
    os << "BlockScaleQuantize(x=" << key.inputDataType << ", y=" << key.outputDataType
       << ", scale=" << key.scaleDataType << ", compute=" << key.computeDataType << ")";
    return os;
}

} // namespace hipdnn_test_sdk::detail
