// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>
#include <unordered_map>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaFwdPlan.hpp>

namespace hipdnn_test_sdk::detail
{

struct SdpaFwdSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType
        = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::SdpaAttributes;
    hipdnn_flatbuffers_sdk::data_objects::DataType qDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType kDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType vDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType oDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType maskDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;

    SdpaFwdSignatureKey() = default;
    constexpr SdpaFwdSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType q,
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

    SdpaFwdSignatureKey(
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

        if(nodeAttributes->attn_mask_tensor_uid().has_value())
        {
            const auto* maskAttr = tensorMap.at(*nodeAttributes->attn_mask_tensor_uid());
            if(maskAttr == nullptr)
            {
                throw std::runtime_error("Mask tensor attributes could not be found in the map, "
                                         "failed to construct key");
            }
            maskDataType = maskAttr->data_type();
        }
    }

    std::size_t operator()(const SdpaFwdSignatureKey& k) const noexcept
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

    bool operator==(const SdpaFwdSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && qDataType == other.qDataType
               && kDataType == other.kDataType && vDataType == other.vDataType
               && oDataType == other.oDataType && maskDataType == other.maskDataType;
    }

    static std::unordered_map<SdpaFwdSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              SdpaFwdSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<SdpaFwdSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           SdpaFwdSignatureKey>
            map;

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
        // FP8 OCP E4M3 (bias 7) inputs with BF16 output. Retained for the forthcoming
        // gfx950/MI350 OCP fp8 forward path (kernels to be added soon). Currently
        // unreachable via the ASM engine, which accepts only FP8_E4M3_FNUZ and declines
        // OCP fp8 (see SdpaFwdPlanBuilder::isApplicable); kept so the reference is ready
        // once those kernels land. Dequant math is dtype-generic, so it stays correct.
        // See the FNUZ registration just below.
        addPlanBuilderAllMasks<hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3,
                               hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3,
                               hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3,
                               hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16>(map);
        // FP8 E4M3 FNUZ inputs with BF16 output — the gfx942/MI300 hardware fp8 encoding
        // (exp bias 8, non-bit-compatible with OCP E4M3) that the vendored ASM fp8 kernels
        // consume; registered so the reference dequantizes the same bytes as the device.
        addPlanBuilderAllMasks<hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3_FNUZ,
                               hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3_FNUZ,
                               hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3_FNUZ,
                               hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType QDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType KDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType VDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ODataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType MaskDataTypeEnum
              = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET>
    static void addPlanBuilder(std::unordered_map<SdpaFwdSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  SdpaFwdSignatureKey>& map)
    {
        map[SdpaFwdSignatureKey(
            QDataTypeEnum, KDataTypeEnum, VDataTypeEnum, ODataTypeEnum, MaskDataTypeEnum)]
            = std::make_unique<SdpaFwdPlanBuilder<QDataTypeEnum,
                                                  KDataTypeEnum,
                                                  VDataTypeEnum,
                                                  ODataTypeEnum,
                                                  MaskDataTypeEnum>>();
    }

    // Registers the builder for the no-mask (UNSET) case and every supported mask dtype.
    template <hipdnn_flatbuffers_sdk::data_objects::DataType QDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType KDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType VDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ODataTypeEnum>
    static void addPlanBuilderAllMasks(std::unordered_map<SdpaFwdSignatureKey,
                                                          std::unique_ptr<IGraphNodePlanBuilder>,
                                                          SdpaFwdSignatureKey>& map)
    {
        addPlanBuilder<QDataTypeEnum,
                       KDataTypeEnum,
                       VDataTypeEnum,
                       ODataTypeEnum,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET>(map);
        addPlanBuilder<QDataTypeEnum,
                       KDataTypeEnum,
                       VDataTypeEnum,
                       ODataTypeEnum,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<QDataTypeEnum,
                       KDataTypeEnum,
                       VDataTypeEnum,
                       ODataTypeEnum,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF>(map);
        addPlanBuilder<QDataTypeEnum,
                       KDataTypeEnum,
                       VDataTypeEnum,
                       ODataTypeEnum,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16>(map);
    }
};

inline std::ostream& operator<<(std::ostream& os, const SdpaFwdSignatureKey& key)
{
    os << "SdpaFwd(q=" << key.qDataType << ", k=" << key.kDataType << ", v=" << key.vDataType
       << ", o=" << key.oDataType << ", mask=" << key.maskDataType << ")";
    return os;
}

} // namespace hipdnn_test_sdk::detail
