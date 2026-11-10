// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <ostream>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PagedCacheStorePlan.hpp>

namespace hipdnn_test_sdk::detail
{

struct PagedCacheStoreSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PagedCacheStoreAttributes};

    hipdnn_flatbuffers_sdk::data_objects::DataType xDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType slotMappingDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType containerDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType;

    PagedCacheStoreSignatureKey() = default;

    constexpr PagedCacheStoreSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType x,
                                          hipdnn_flatbuffers_sdk::data_objects::DataType slot,
                                          hipdnn_flatbuffers_sdk::data_objects::DataType container,
                                          hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : xDataType(x)
        , slotMappingDataType(slot)
        , containerDataType(container)
        , computeDataType(compute)
    {
    }

    PagedCacheStoreSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        [[maybe_unused]] const hipdnn_flatbuffers_sdk::data_objects::DataType computeType)
    {
        const auto* nodeAttributes = node.attributes_as_PagedCacheStoreAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes could not be cast to PagedCacheStoreAttributes");
        }

        auto xTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
        auto slotTensorAttr = tensorMap.at(nodeAttributes->slot_mapping_tensor_uid());
        auto containerTensorAttr = tensorMap.at(nodeAttributes->container_tensor_uid());
        if(xTensorAttr == nullptr || slotTensorAttr == nullptr || containerTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        xDataType = xTensorAttr->data_type();
        slotMappingDataType = slotTensorAttr->data_type();
        containerDataType = containerTensorAttr->data_type();
        // The schema carries no compute type (lowered nodes leave it UNSET): x * scale is
        // defined in FP32, so every node keys on FLOAT compute.
        computeDataType = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    }

    std::size_t operator()(const PagedCacheStoreSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(xDataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(slotMappingDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(containerDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 16);
    }

    bool operator==(const PagedCacheStoreSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && xDataType == other.xDataType
               && slotMappingDataType == other.slotMappingDataType
               && containerDataType == other.containerDataType
               && computeDataType == other.computeDataType;
    }

    /// x and container in FLOAT, HALF or BFLOAT16 (same type), or x in one of those stored
    /// into an FP8_E4M3 / FP8_E5M2 container; INT32 or INT64 slot mapping; FLOAT compute.
    static std::unordered_map<PagedCacheStoreSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              PagedCacheStoreSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<PagedCacheStoreSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           PagedCacheStoreSignatureKey>
            map;

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT32,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT32,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT32,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E5M2,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT64,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT64,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT64,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E5M2,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT32,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT32,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT32,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E5M2,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT64,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT64,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT64,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E5M2,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT32,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT32,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT32,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E5M2,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT64,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT64,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT64,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E5M2,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType SlotDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ContainerDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<PagedCacheStoreSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  PagedCacheStoreSignatureKey>& map)
    {
        map[PagedCacheStoreSignatureKey(
            XDataTypeEnum, SlotDataTypeEnum, ContainerDataTypeEnum, ComputeDataTypeEnum)]
            = std::make_unique<PagedCacheStorePlanBuilder<XDataTypeEnum,
                                                          SlotDataTypeEnum,
                                                          ContainerDataTypeEnum,
                                                          ComputeDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const PagedCacheStoreSignatureKey& key)
{
    os << "PagedCacheStore(x=" << key.xDataType << ", slot_mapping=" << key.slotMappingDataType
       << ", container=" << key.containerDataType << ", compute=" << key.computeDataType << ")";
    return os;
}

} // namespace hipdnn_test_sdk::detail
