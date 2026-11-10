// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/TopkAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

[[nodiscard]] inline Error unpackTopkOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::TopkAttributes& attributes)
{
    // Unpack x tensor
    std::shared_ptr<graph::TensorAttributes> xTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_TOPK_XDESC_EXT,
                                               tensorMap,
                                               xTensor,
                                               "Top-k XDESC_EXT tensor"));
    attributes.set_x(xTensor);

    // Unpack indices tensor
    std::shared_ptr<graph::TensorAttributes> indicesTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_TOPK_INDICES_DESC_EXT,
                                               tensorMap,
                                               indicesTensor,
                                               "Top-k INDICES_DESC_EXT tensor"));
    attributes.set_indices(indicesTensor);

    // Unpack values tensor
    std::shared_ptr<graph::TensorAttributes> valuesTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_TOPK_VALUES_DESC_EXT,
                                            tensorMap,
                                            valuesTensor,
                                            "Top-k VALUES_DESC_EXT tensor"));
    if(valuesTensor)
    {
        attributes.set_values(valuesTensor);
    }

    // Unpack k
    int64_t k = 0;
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(
        opDesc, HIPDNN_ATTR_OPERATION_TOPK_K_EXT, HIPDNN_TYPE_INT64, k, "Top-k k"));
    attributes.set_k(k);

    // Unpack largest
    bool largest = false;
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_OPERATION_TOPK_LARGEST_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               largest,
                                               "Top-k largest"));
    attributes.set_largest(largest);

    // Unpack operation name
    std::string opName;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrString(opDesc, HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    attributes.set_name(opName);

    return {};
}

} // namespace hipdnn_frontend::detail
