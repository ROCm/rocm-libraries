// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/PagedCacheStoreAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

[[nodiscard]] inline Error unpackPagedCacheStoreOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::PagedCacheStoreAttributes& attributes)
{
    // Unpack x tensor
    std::shared_ptr<graph::TensorAttributes> xTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_XDESC_EXT,
                                               tensorMap,
                                               xTensor,
                                               "Paged cache store XDESC_EXT tensor"));
    attributes.set_x(xTensor);

    // Unpack slot_mapping tensor
    std::shared_ptr<graph::TensorAttributes> slotMappingTensor;
    HIPDNN_CHECK_ERROR(
        unpackAndRegisterTensor(opDesc,
                                HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_SLOT_MAPPING_DESC_EXT,
                                tensorMap,
                                slotMappingTensor,
                                "Paged cache store SLOT_MAPPING_DESC_EXT tensor"));
    attributes.set_slot_mapping(slotMappingTensor);

    // Unpack container tensor
    std::shared_ptr<graph::TensorAttributes> containerTensor;
    HIPDNN_CHECK_ERROR(
        unpackAndRegisterTensor(opDesc,
                                HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_CONTAINER_DESC_EXT,
                                tensorMap,
                                containerTensor,
                                "Paged cache store CONTAINER_DESC_EXT tensor"));
    attributes.set_container(containerTensor);

    // Unpack scale tensor
    std::shared_ptr<graph::TensorAttributes> scaleTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_SCALE_DESC_EXT,
                                            tensorMap,
                                            scaleTensor,
                                            "Paged cache store SCALE_DESC_EXT tensor"));
    if(scaleTensor)
    {
        attributes.set_scale(scaleTensor);
    }

    // Unpack operation name
    std::string opName;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrString(opDesc, HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    attributes.set_name(opName);

    return {};
}

} // namespace hipdnn_frontend::detail
