// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <functional>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ResampleFwdPlan.hpp>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <unordered_map>

namespace hipdnn_test_sdk::detail
{

struct ResampleFwdSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType
        = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ResampleFwdAttributes;
    hipdnn_flatbuffers_sdk::data_objects::DataType xDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType yDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType indexDataType;

    ResampleFwdSignatureKey() = default;
    constexpr ResampleFwdSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType x,
                                      hipdnn_flatbuffers_sdk::data_objects::DataType y,
                                      hipdnn_flatbuffers_sdk::data_objects::DataType compute,
                                      hipdnn_flatbuffers_sdk::data_objects::DataType index)
        : xDataType(x)
        , yDataType(y)
        , computeDataType(compute)
        , indexDataType(index)
    {
    }

    ResampleFwdSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        const auto* nodeAttributes = node.attributes_as_ResampleFwdAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes could not be cast to ResampleFwdAttributes");
        }

        auto xTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
        auto yTensorAttr = tensorMap.at(nodeAttributes->y_tensor_uid());
        if(xTensorAttr == nullptr || yTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        xDataType = xTensorAttr->data_type();
        yDataType = yTensorAttr->data_type();
        computeDataType = node.compute_data_type();
        indexDataType = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
        if(nodeAttributes->index_tensor_uid().has_value())
        {
            auto indexTensorAttr = tensorMap.at(nodeAttributes->index_tensor_uid().value());
            if(indexTensorAttr == nullptr)
            {
                throw std::runtime_error("Index tensor attributes could not be found in the map, "
                                         "failed to construct key");
            }
            indexDataType = indexTensorAttr->data_type();
        }
    }

    std::size_t operator()(const ResampleFwdSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(xDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(yDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(indexDataType)) << 16);
    }

    bool operator==(const ResampleFwdSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && xDataType == other.xDataType
               && yDataType == other.yDataType && computeDataType == other.computeDataType
               && indexDataType == other.indexDataType;
    }

    static std::unordered_map<ResampleFwdSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              ResampleFwdSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<ResampleFwdSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           ResampleFwdSignatureKey>
            map;

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT32>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT32>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT32>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType YDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType IndexDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<ResampleFwdSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  ResampleFwdSignatureKey>& map)
    {
        map[ResampleFwdSignatureKey(
            XDataTypeEnum, YDataTypeEnum, ComputeDataTypeEnum, IndexDataTypeEnum)]
            = std::make_unique<ResampleFwdPlanBuilder<XDataTypeEnum,
                                                      YDataTypeEnum,
                                                      ComputeDataTypeEnum,
                                                      IndexDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const ResampleFwdSignatureKey& key)
{
    os << "ResampleFwd(x=" << key.xDataType << ", y=" << key.yDataType
       << ", compute=" << key.computeDataType << ", index=" << key.indexDataType << ")";
    return os;
}

} // namespace hipdnn_test_sdk::detail
