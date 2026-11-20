// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <unordered_map>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "GpuSdpaBwdPlan.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

// Key for the GPU SDPA backward plans: the I/O dtype (Q's, which the builder requires of every
// Q/K/V/O/dO/dQ/dK/dV tensor) and the mask dtype (UNSET = no mask).
struct GpuSdpaBwdSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::SdpaBackwardAttributes};
    hipdnn_flatbuffers_sdk::data_objects::DataType ioDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType maskDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};

    GpuSdpaBwdSignatureKey() = default;

    constexpr GpuSdpaBwdSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType io,
                                     hipdnn_flatbuffers_sdk::data_objects::DataType mask)
        : ioDataType(io)
        , maskDataType(mask)
    {
    }

    GpuSdpaBwdSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        const auto* attrs = node.attributes_as_SdpaBackwardAttributes();
        if(attrs == nullptr)
        {
            throw std::runtime_error("Node attributes could not be cast to SdpaBackwardAttributes");
        }
        ioDataType = tensorMap.at(attrs->q_tensor_uid())->data_type();
        if(attrs->attn_mask_tensor_uid().has_value())
        {
            maskDataType = tensorMap.at(attrs->attn_mask_tensor_uid().value())->data_type();
        }
    }

    std::size_t operator()(const GpuSdpaBwdSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(ioDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(maskDataType)) << 8);
    }

    bool operator==(const GpuSdpaBwdSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && ioDataType == other.ioDataType
               && maskDataType == other.maskDataType;
    }

    static std::unordered_map<GpuSdpaBwdSignatureKey,
                              std::unique_ptr<IGpuGraphNodePlanBuilder>,
                              GpuSdpaBwdSignatureKey>
        getPlanBuilders()
    {
        using hipdnn_flatbuffers_sdk::data_objects::DataType;
        std::unordered_map<GpuSdpaBwdSignatureKey,
                           std::unique_ptr<IGpuGraphNodePlanBuilder>,
                           GpuSdpaBwdSignatureKey>
            map;
        addAllMasks<DataType::FLOAT>(map);
        addAllMasks<DataType::HALF>(map);
        addAllMasks<DataType::BFLOAT16>(map);
        return map;
    }

private:
    template <hipdnn_flatbuffers_sdk::data_objects::DataType IoDataTypeEnum>
    static void addAllMasks(std::unordered_map<GpuSdpaBwdSignatureKey,
                                               std::unique_ptr<IGpuGraphNodePlanBuilder>,
                                               GpuSdpaBwdSignatureKey>& map)
    {
        using hipdnn_flatbuffers_sdk::data_objects::DataType;
        map[GpuSdpaBwdSignatureKey(IoDataTypeEnum, DataType::UNSET)]
            = std::make_unique<GpuSdpaBwdPlanBuilder<IoDataTypeEnum, DataType::UNSET>>();
        map[GpuSdpaBwdSignatureKey(IoDataTypeEnum, DataType::FLOAT)]
            = std::make_unique<GpuSdpaBwdPlanBuilder<IoDataTypeEnum, DataType::FLOAT>>();
        map[GpuSdpaBwdSignatureKey(IoDataTypeEnum, DataType::HALF)]
            = std::make_unique<GpuSdpaBwdPlanBuilder<IoDataTypeEnum, DataType::HALF>>();
        map[GpuSdpaBwdSignatureKey(IoDataTypeEnum, DataType::BFLOAT16)]
            = std::make_unique<GpuSdpaBwdPlanBuilder<IoDataTypeEnum, DataType::BFLOAT16>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const GpuSdpaBwdSignatureKey& key)
{
    os << "GpuSdpaBwd(io=" << key.ioDataType << ", mask=" << key.maskDataType << ")";
    return os;
}

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
