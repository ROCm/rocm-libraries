// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <ostream>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/GdnPlan.hpp>

namespace hipdnn_test_sdk::detail
{

struct GdnSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::GdnAttributes};

    /// Type of q, k, v and o. Gates, states and checkpoints are staged at run time (GdnPlan.hpp).
    hipdnn_flatbuffers_sdk::data_objects::DataType ioDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType cuSeqlensDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType;

    GdnSignatureKey() = default;

    constexpr GdnSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType io,
                              hipdnn_flatbuffers_sdk::data_objects::DataType cuSeqlens,
                              hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : ioDataType(io)
        , cuSeqlensDataType(cuSeqlens)
        , computeDataType(compute)
    {
    }

    GdnSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_flatbuffers_sdk::data_objects::DataType computeType)
    {
        const auto* nodeAttributes = node.attributes_as_GdnAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes could not be cast to GdnAttributes");
        }

        auto qTensorAttr = tensorMap.at(nodeAttributes->q_tensor_uid());
        auto cuSeqlensTensorAttr = tensorMap.at(nodeAttributes->cu_seqlens_tensor_uid());
        if(qTensorAttr == nullptr || cuSeqlensTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        ioDataType = qTensorAttr->data_type();
        cuSeqlensDataType = cuSeqlensTensorAttr->data_type();
        computeDataType = computeType;
    }

    std::size_t operator()(const GdnSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(ioDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(cuSeqlensDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 12);
    }

    bool operator==(const GdnSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && ioDataType == other.ioDataType
               && cuSeqlensDataType == other.cuSeqlensDataType
               && computeDataType == other.computeDataType;
    }

    /// io in FLOAT, HALF or BFLOAT16; INT32 or INT64 cu_seqlens; FLOAT compute (FP32 state).
    static std::
        unordered_map<GdnSignatureKey, std::unique_ptr<IGraphNodePlanBuilder>, GdnSignatureKey>
        getPlanBuilders()
    {
        using hipdnn_flatbuffers_sdk::data_objects::DataType;

        std::unordered_map<GdnSignatureKey, std::unique_ptr<IGraphNodePlanBuilder>, GdnSignatureKey>
            map;

        addPlanBuilder<DataType::FLOAT, DataType::INT32, DataType::FLOAT>(map);
        addPlanBuilder<DataType::FLOAT, DataType::INT64, DataType::FLOAT>(map);
        addPlanBuilder<DataType::HALF, DataType::INT32, DataType::FLOAT>(map);
        addPlanBuilder<DataType::HALF, DataType::INT64, DataType::FLOAT>(map);
        addPlanBuilder<DataType::BFLOAT16, DataType::INT32, DataType::FLOAT>(map);
        addPlanBuilder<DataType::BFLOAT16, DataType::INT64, DataType::FLOAT>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType IoDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType CuSeqlensDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<GdnSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  GdnSignatureKey>& map)
    {
        map[GdnSignatureKey(IoDataTypeEnum, CuSeqlensDataTypeEnum, ComputeDataTypeEnum)]
            = std::make_unique<
                GdnPlanBuilder<IoDataTypeEnum, CuSeqlensDataTypeEnum, ComputeDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const GdnSignatureKey& key)
{
    os << "Gdn(io=" << key.ioDataType << ", cu_seqlens=" << key.cuSeqlensDataType
       << ", compute=" << key.computeDataType << ")";
    return os;
}

} // namespace hipdnn_test_sdk::detail
