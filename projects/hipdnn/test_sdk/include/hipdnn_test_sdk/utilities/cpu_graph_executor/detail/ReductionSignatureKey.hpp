// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <ostream>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ReductionPlan.hpp>

namespace hipdnn_test_sdk::detail
{

struct ReductionSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ReductionAttributes};

    hipdnn_flatbuffers_sdk::data_objects::DataType xDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType yDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType;

    ReductionSignatureKey() = default;

    constexpr ReductionSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType x,
                                    hipdnn_flatbuffers_sdk::data_objects::DataType y,
                                    hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : xDataType(x)
        , yDataType(y)
        , computeDataType(compute)
    {
    }

    ReductionSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_flatbuffers_sdk::data_objects::DataType computeType)
    {
        const auto* nodeAttributes = node.attributes_as_ReductionAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes could not be cast to ReductionAttributes");
        }

        auto xTensorAttr = tensorMap.at(nodeAttributes->in_tensor_uid());
        auto yTensorAttr = tensorMap.at(nodeAttributes->out_tensor_uid());
        if(xTensorAttr == nullptr || yTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        xDataType = xTensorAttr->data_type();
        yDataType = yTensorAttr->data_type();
        computeDataType = computeType;
    }

    std::size_t operator()(const ReductionSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(xDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(yDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 12);
    }

    bool operator==(const ReductionSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && xDataType == other.xDataType
               && yDataType == other.yDataType && computeDataType == other.computeDataType;
    }

    static std::unordered_map<ReductionSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              ReductionSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<ReductionSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           ReductionSignatureKey>
            map;

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType YDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<ReductionSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  ReductionSignatureKey>& map)
    {
        map[ReductionSignatureKey(XDataTypeEnum, YDataTypeEnum, ComputeDataTypeEnum)]
            = std::make_unique<
                ReductionPlanBuilder<XDataTypeEnum, YDataTypeEnum, ComputeDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const ReductionSignatureKey& key)
{
    os << "Reduction(x=" << key.xDataType << ", y=" << key.yDataType
       << ", compute=" << key.computeDataType << ")";
    return os;
}

} // namespace hipdnn_test_sdk::detail
