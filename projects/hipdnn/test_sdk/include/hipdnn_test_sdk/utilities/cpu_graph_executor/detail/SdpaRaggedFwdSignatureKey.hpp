// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <ostream>
#include <stdexcept>
#include <unordered_map>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaRaggedFwdPlan.hpp>

namespace hipdnn_test_sdk::detail
{

// Signature key for the ragged SDPA forward CPU reference (RFC-0014 packed + ragged_offset).
// It has the same fields as SdpaFwdSignatureKey but is a separate type, so dense and ragged
// plans land in separate registry buckets. CpuReferenceGraphExecutor::buildSignatureKey picks
// one based on hasRaggedSdpaPrimary.
struct SdpaRaggedFwdSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType
        = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::SdpaAttributes;
    hipdnn_flatbuffers_sdk::data_objects::DataType qDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
    hipdnn_flatbuffers_sdk::data_objects::DataType kDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
    hipdnn_flatbuffers_sdk::data_objects::DataType vDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
    hipdnn_flatbuffers_sdk::data_objects::DataType oDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;

    SdpaRaggedFwdSignatureKey() = default;
    constexpr SdpaRaggedFwdSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType q,
                                        hipdnn_flatbuffers_sdk::data_objects::DataType k,
                                        hipdnn_flatbuffers_sdk::data_objects::DataType v,
                                        hipdnn_flatbuffers_sdk::data_objects::DataType o)
        : qDataType(q)
        , kDataType(k)
        , vDataType(v)
        , oDataType(o)
    {
    }

    SdpaRaggedFwdSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        const auto* nodeAttributes = node.attributes_as_SdpaAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes could not be cast to SdpaAttributes");
        }

        const auto* qAttr = tensorMap.at(nodeAttributes->q_tensor_uid());
        const auto* kAttr = tensorMap.at(nodeAttributes->k_tensor_uid());
        const auto* vAttr = tensorMap.at(nodeAttributes->v_tensor_uid());
        const auto* oAttr = tensorMap.at(nodeAttributes->o_tensor_uid());

        if(qAttr == nullptr || kAttr == nullptr || vAttr == nullptr || oAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        qDataType = qAttr->data_type();
        kDataType = kAttr->data_type();
        vDataType = vAttr->data_type();
        oDataType = oAttr->data_type();
    }

    std::size_t operator()(const SdpaRaggedFwdSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(qDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(kDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(vDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(oDataType)) << 16);
    }

    bool operator==(const SdpaRaggedFwdSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && qDataType == other.qDataType
               && kDataType == other.kDataType && vDataType == other.vDataType
               && oDataType == other.oDataType;
    }

    static std::unordered_map<SdpaRaggedFwdSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              SdpaRaggedFwdSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<SdpaRaggedFwdSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           SdpaRaggedFwdSignatureKey>
            map;

        // The tuples the GPU ragged plan registers: all bf16, and fp8 E4M3 inputs with bf16
        // output.
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType QDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType KDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType VDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ODataTypeEnum>
    static void addPlanBuilder(std::unordered_map<SdpaRaggedFwdSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  SdpaRaggedFwdSignatureKey>& map)
    {
        map[SdpaRaggedFwdSignatureKey(QDataTypeEnum, KDataTypeEnum, VDataTypeEnum, ODataTypeEnum)]
            = std::make_unique<SdpaRaggedFwdPlanBuilder<QDataTypeEnum,
                                                        KDataTypeEnum,
                                                        VDataTypeEnum,
                                                        ODataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const SdpaRaggedFwdSignatureKey& key)
{
    os << "SdpaRaggedFwd(q=" << key.qDataType << ", k=" << key.kDataType << ", v=" << key.vDataType
       << ", o=" << key.oDataType << ")";
    return os;
}

} // namespace hipdnn_test_sdk::detail
