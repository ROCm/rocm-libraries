// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>
#include <unordered_map>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaBwdPlan.hpp>

namespace hipdnn_test_sdk::detail
{

struct SdpaBwdSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType
        = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::SdpaBackwardAttributes;
    hipdnn_flatbuffers_sdk::data_objects::DataType qDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType kDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType vDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType oDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType doDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType dqDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType dkDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType dvDataType;

    SdpaBwdSignatureKey() = default;
    constexpr SdpaBwdSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType q,
                                  hipdnn_flatbuffers_sdk::data_objects::DataType k,
                                  hipdnn_flatbuffers_sdk::data_objects::DataType v,
                                  hipdnn_flatbuffers_sdk::data_objects::DataType o,
                                  hipdnn_flatbuffers_sdk::data_objects::DataType dO,
                                  hipdnn_flatbuffers_sdk::data_objects::DataType dq,
                                  hipdnn_flatbuffers_sdk::data_objects::DataType dk,
                                  hipdnn_flatbuffers_sdk::data_objects::DataType dv)
        : qDataType(q)
        , kDataType(k)
        , vDataType(v)
        , oDataType(o)
        , doDataType(dO)
        , dqDataType(dq)
        , dkDataType(dk)
        , dvDataType(dv)
    {
    }

    SdpaBwdSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        const auto* nodeAttributes = node.attributes_as_SdpaBackwardAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes could not be cast to SdpaBackwardAttributes");
        }

        const auto* qAttr = tensorMap.at(nodeAttributes->q_tensor_uid());
        const auto* kAttr = tensorMap.at(nodeAttributes->k_tensor_uid());
        const auto* vAttr = tensorMap.at(nodeAttributes->v_tensor_uid());
        const auto* oAttr = tensorMap.at(nodeAttributes->o_tensor_uid());
        const auto* doAttr = tensorMap.at(nodeAttributes->do_tensor_uid());
        const auto* dqAttr = tensorMap.at(nodeAttributes->dq_tensor_uid());
        const auto* dkAttr = tensorMap.at(nodeAttributes->dk_tensor_uid());
        const auto* dvAttr = tensorMap.at(nodeAttributes->dv_tensor_uid());

        if(qAttr == nullptr || kAttr == nullptr || vAttr == nullptr || oAttr == nullptr
           || doAttr == nullptr || dqAttr == nullptr || dkAttr == nullptr || dvAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        qDataType = qAttr->data_type();
        kDataType = kAttr->data_type();
        vDataType = vAttr->data_type();
        oDataType = oAttr->data_type();
        doDataType = doAttr->data_type();
        dqDataType = dqAttr->data_type();
        dkDataType = dkAttr->data_type();
        dvDataType = dvAttr->data_type();
    }

    std::size_t operator()(const SdpaBwdSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(qDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(kDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(vDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(oDataType)) << 16)
               ^ (static_cast<std::size_t>(static_cast<int>(doDataType)) << 20)
               ^ (static_cast<std::size_t>(static_cast<int>(dqDataType)) << 24)
               ^ (static_cast<std::size_t>(static_cast<int>(dkDataType)) << 28)
               ^ (static_cast<std::size_t>(static_cast<int>(dvDataType)) << 32);
    }

    bool operator==(const SdpaBwdSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && qDataType == other.qDataType
               && kDataType == other.kDataType && vDataType == other.vDataType
               && oDataType == other.oDataType && doDataType == other.doDataType
               && dqDataType == other.dqDataType && dkDataType == other.dkDataType
               && dvDataType == other.dvDataType;
    }

    static std::unordered_map<SdpaBwdSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              SdpaBwdSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<SdpaBwdSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           SdpaBwdSignatureKey>
            map;

        // All-float
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        // All-bfloat16
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16>(map);
        // All-half
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType QDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType KDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType VDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ODataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType DODataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType DQDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType DKDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType DVDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<SdpaBwdSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  SdpaBwdSignatureKey>& map)
    {
        map[SdpaBwdSignatureKey(QDataTypeEnum,
                                KDataTypeEnum,
                                VDataTypeEnum,
                                ODataTypeEnum,
                                DODataTypeEnum,
                                DQDataTypeEnum,
                                DKDataTypeEnum,
                                DVDataTypeEnum)]
            = std::make_unique<SdpaBwdPlanBuilder<QDataTypeEnum,
                                                  KDataTypeEnum,
                                                  VDataTypeEnum,
                                                  ODataTypeEnum,
                                                  DODataTypeEnum,
                                                  DQDataTypeEnum,
                                                  DKDataTypeEnum,
                                                  DVDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const SdpaBwdSignatureKey& key)
{
    os << "SdpaBwd(q=" << key.qDataType << ", k=" << key.kDataType << ", v=" << key.vDataType
       << ", o=" << key.oDataType << ", dO=" << key.doDataType << ", dq=" << key.dqDataType
       << ", dk=" << key.dkDataType << ", dv=" << key.dvDataType << ")";
    return os;
}

} // namespace hipdnn_test_sdk::detail
