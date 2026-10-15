// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "GpuBlockScalePlan.hpp"

#include <ostream>

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

template <hipdnn_flatbuffers_sdk::data_objects::NodeAttributes NodeType>
struct GpuBlockScaleSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType = NodeType;
    hipdnn_flatbuffers_sdk::data_objects::DataType xDataType{};
    hipdnn_flatbuffers_sdk::data_objects::DataType yDataType{};
    hipdnn_flatbuffers_sdk::data_objects::DataType scaleDataType{};
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType{};

    GpuBlockScaleSignatureKey() = default;
    constexpr GpuBlockScaleSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType x,
                                        hipdnn_flatbuffers_sdk::data_objects::DataType y,
                                        hipdnn_flatbuffers_sdk::data_objects::DataType scale,
                                        hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : xDataType(x)
        , yDataType(y)
        , scaleDataType(scale)
        , computeDataType(compute)
    {
    }

    GpuBlockScaleSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        int64_t xUid;
        int64_t yUid;
        int64_t scaleUid;
        if constexpr(NodeType
                     == hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
                         BlockScaleQuantizeAttributes)
        {
            const auto* attributes = node.attributes_as_BlockScaleQuantizeAttributes();
            if(attributes == nullptr)
            {
                throw std::runtime_error("Invalid block-scale quantize attributes");
            }
            xUid = attributes->x_tensor_uid();
            yUid = attributes->y_tensor_uid();
            scaleUid = attributes->scale_tensor_uid();
        }
        else
        {
            const auto* attributes = node.attributes_as_BlockScaleDequantizeAttributes();
            if(attributes == nullptr)
            {
                throw std::runtime_error("Invalid block-scale dequantize attributes");
            }
            xUid = attributes->x_tensor_uid();
            yUid = attributes->y_tensor_uid();
            scaleUid = attributes->scale_tensor_uid();
        }
        xDataType = tensorMap.at(xUid)->data_type();
        yDataType = tensorMap.at(yUid)->data_type();
        scaleDataType = tensorMap.at(scaleUid)->data_type();
        computeDataType = node.compute_data_type();
    }

    std::size_t operator()(const GpuBlockScaleSignatureKey& key) const noexcept
    {
        return key.hashSelf();
    }
    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(nodeType) ^ (static_cast<std::size_t>(xDataType) << 4)
               ^ (static_cast<std::size_t>(yDataType) << 8)
               ^ (static_cast<std::size_t>(scaleDataType) << 12)
               ^ (static_cast<std::size_t>(computeDataType) << 16);
    }
    bool operator==(const GpuBlockScaleSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && xDataType == other.xDataType
               && yDataType == other.yDataType && scaleDataType == other.scaleDataType
               && computeDataType == other.computeDataType;
    }

    static std::unordered_map<GpuBlockScaleSignatureKey,
                              std::unique_ptr<IGpuGraphNodePlanBuilder>,
                              GpuBlockScaleSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<GpuBlockScaleSignatureKey,
                           std::unique_ptr<IGpuGraphNodePlanBuilder>,
                           GpuBlockScaleSignatureKey>
            map;
        addBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                   hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                   hipdnn_flatbuffers_sdk::data_objects::DataType::HALF>(map);
        addBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                   hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16>(map);
        using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
        if constexpr(NodeType
                     == hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
                         BlockScaleQuantizeAttributes)
        {
            addCompactQuantizeBuilders<DataType::FLOAT>(map);
            addCompactQuantizeBuilders<DataType::HALF>(map);
            addCompactQuantizeBuilders<DataType::BFLOAT16>(map);
        }
        else
        {
            addCompactDequantizeBuilders<DataType::FLOAT>(map);
            addCompactDequantizeBuilders<DataType::HALF>(map);
        }
        return map;
    }

private:
    template <hipdnn_flatbuffers_sdk::data_objects::DataType X,
              hipdnn_flatbuffers_sdk::data_objects::DataType Y,
              hipdnn_flatbuffers_sdk::data_objects::DataType Scale
              = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>
    static void addBuilder(std::unordered_map<GpuBlockScaleSignatureKey,
                                              std::unique_ptr<IGpuGraphNodePlanBuilder>,
                                              GpuBlockScaleSignatureKey>& map)
    {
        using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
        const GpuBlockScaleSignatureKey key(X, Y, Scale, DataType::FLOAT);
        if constexpr(NodeType
                     == hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
                         BlockScaleQuantizeAttributes)
        {
            map[key] = std::make_unique<
                GpuBlockScaleQuantizePlanBuilder<X, Y, Scale, DataType::FLOAT>>();
        }
        else
        {
            map[key] = std::make_unique<
                GpuBlockScaleDequantizePlanBuilder<X, Scale, Y, DataType::FLOAT>>();
        }
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType X>
    static void
        addCompactQuantizeBuilders(std::unordered_map<GpuBlockScaleSignatureKey,
                                                      std::unique_ptr<IGpuGraphNodePlanBuilder>,
                                                      GpuBlockScaleSignatureKey>& map)
    {
        using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
        addBuilder<X, DataType::FP4_E2M1, DataType::FP8_E8M0>(map);
        addBuilder<X, DataType::FP6_E2M3, DataType::FP8_E8M0>(map);
        addBuilder<X, DataType::FP6_E3M2, DataType::FP8_E8M0>(map);
        addBuilder<X, DataType::FP8_E4M3, DataType::FP8_E8M0>(map);
        addBuilder<X, DataType::FP8_E5M2, DataType::FP8_E8M0>(map);
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType Y>
    static void
        addCompactDequantizeBuilders(std::unordered_map<GpuBlockScaleSignatureKey,
                                                        std::unique_ptr<IGpuGraphNodePlanBuilder>,
                                                        GpuBlockScaleSignatureKey>& map)
    {
        using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
        addBuilder<DataType::FP4_E2M1, Y, DataType::FP8_E8M0>(map);
        addBuilder<DataType::FP6_E2M3, Y, DataType::FP8_E8M0>(map);
        addBuilder<DataType::FP6_E3M2, Y, DataType::FP8_E8M0>(map);
        addBuilder<DataType::FP8_E4M3, Y, DataType::FP8_E8M0>(map);
        addBuilder<DataType::FP8_E5M2, Y, DataType::FP8_E8M0>(map);
    }
};

using GpuBlockScaleQuantizeSignatureKey = GpuBlockScaleSignatureKey<
    hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BlockScaleQuantizeAttributes>;
using GpuBlockScaleDequantizeSignatureKey = GpuBlockScaleSignatureKey<
    hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BlockScaleDequantizeAttributes>;

template <hipdnn_flatbuffers_sdk::data_objects::NodeAttributes NodeType>
inline std::ostream& operator<<(std::ostream& os, const GpuBlockScaleSignatureKey<NodeType>& key)
{
    os << "GpuBlockScale(type=" << key.nodeType << ", x=" << key.xDataType
       << ", y=" << key.yDataType << ", scale=" << key.scaleDataType
       << ", compute=" << key.computeDataType << ")";
    return os;
}

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
