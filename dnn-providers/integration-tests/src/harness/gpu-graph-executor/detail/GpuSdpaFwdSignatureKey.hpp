// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <ostream>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>

#include "GpuSdpaFwdPlan.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

struct GpuSdpaFwdSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::SdpaAttributes};
    hipdnn_flatbuffers_sdk::data_objects::DataType qDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType kDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType vDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType oDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    // UNSET means the node has no attention mask.
    hipdnn_flatbuffers_sdk::data_objects::DataType maskDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};

    GpuSdpaFwdSignatureKey() = default;
    constexpr GpuSdpaFwdSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType q,
                                     hipdnn_flatbuffers_sdk::data_objects::DataType k,
                                     hipdnn_flatbuffers_sdk::data_objects::DataType v,
                                     hipdnn_flatbuffers_sdk::data_objects::DataType o,
                                     hipdnn_flatbuffers_sdk::data_objects::DataType mask
                                     = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET)
        : qDataType(q)
        , kDataType(k)
        , vDataType(v)
        , oDataType(o)
        , maskDataType(mask)
    {
    }

    GpuSdpaFwdSignatureKey(
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

        // A dangling mask uid throws std::out_of_range, as for q/k/v/o.
        if(nodeAttributes->attn_mask_tensor_uid().has_value())
        {
            const auto* maskAttr = tensorMap.at(nodeAttributes->attn_mask_tensor_uid().value());
            if(maskAttr == nullptr)
            {
                throw std::runtime_error("The mask tensor attributes could not be found in the "
                                         "map, failed to construct key");
            }
            maskDataType = maskAttr->data_type();
        }
    }

    std::size_t operator()(const GpuSdpaFwdSignatureKey& k) const noexcept
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
               ^ (static_cast<std::size_t>(static_cast<int>(maskDataType)) << 20);
    }

    bool operator==(const GpuSdpaFwdSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && qDataType == other.qDataType
               && kDataType == other.kDataType && vDataType == other.vDataType
               && oDataType == other.oDataType && maskDataType == other.maskDataType;
    }

    static std::unordered_map<GpuSdpaFwdSignatureKey,
                              std::unique_ptr<IGpuGraphNodePlanBuilder>,
                              GpuSdpaFwdSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<GpuSdpaFwdSignatureKey,
                           std::unique_ptr<IGpuGraphNodePlanBuilder>,
                           GpuSdpaFwdSignatureKey>
            map;

        // Q, K, V, O
        addPlanBuilderAllMasks<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                               hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                               hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                               hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilderAllMasks<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                               hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                               hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                               hipdnn_flatbuffers_sdk::data_objects::DataType::HALF>(map);
        addPlanBuilderAllMasks<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                               hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                               hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                               hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16>(map);
        addPlanBuilderAllMasks<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                               hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                               hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                               hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilderAllMasks<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                               hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                               hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                               hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType QDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType KDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType VDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ODataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType MaskDataTypeEnum
              = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET>
    static void addPlanBuilder(std::unordered_map<GpuSdpaFwdSignatureKey,
                                                  std::unique_ptr<IGpuGraphNodePlanBuilder>,
                                                  GpuSdpaFwdSignatureKey>& map)
    {
        map[GpuSdpaFwdSignatureKey(
            QDataTypeEnum, KDataTypeEnum, VDataTypeEnum, ODataTypeEnum, MaskDataTypeEnum)]
            = std::make_unique<GpuSdpaFwdPlanBuilder<QDataTypeEnum,
                                                     KDataTypeEnum,
                                                     VDataTypeEnum,
                                                     ODataTypeEnum,
                                                     MaskDataTypeEnum>>();
    }

    // Registers the Q/K/V/O combination for no mask and for each supported mask dtype.
    template <hipdnn_flatbuffers_sdk::data_objects::DataType QDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType KDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType VDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ODataTypeEnum>
    static void addPlanBuilderAllMasks(std::unordered_map<GpuSdpaFwdSignatureKey,
                                                          std::unique_ptr<IGpuGraphNodePlanBuilder>,
                                                          GpuSdpaFwdSignatureKey>& map)
    {
        using hipdnn_flatbuffers_sdk::data_objects::DataType;
        addPlanBuilder<QDataTypeEnum, KDataTypeEnum, VDataTypeEnum, ODataTypeEnum, DataType::UNSET>(
            map);
        addPlanBuilder<QDataTypeEnum, KDataTypeEnum, VDataTypeEnum, ODataTypeEnum, DataType::FLOAT>(
            map);
        addPlanBuilder<QDataTypeEnum, KDataTypeEnum, VDataTypeEnum, ODataTypeEnum, DataType::HALF>(
            map);
        addPlanBuilder<QDataTypeEnum,
                       KDataTypeEnum,
                       VDataTypeEnum,
                       ODataTypeEnum,
                       DataType::BFLOAT16>(map);
    }
};

inline std::ostream& operator<<(std::ostream& os, const GpuSdpaFwdSignatureKey& key)
{
    os << "GpuSdpaFwd(q=" << key.qDataType << ", k=" << key.kDataType << ", v=" << key.vDataType
       << ", o=" << key.oDataType << ", mask=" << key.maskDataType << ")";
    return os;
}

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
