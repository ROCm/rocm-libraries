// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/MoeRoutingAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

[[nodiscard]] inline Error unpackMoeRoutingOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::MoeRoutingAttributes& attributes)
{
    // Unpack topk_ids tensor
    std::shared_ptr<graph::TensorAttributes> topkIdsTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOPK_IDS_DESC_EXT,
                                               tensorMap,
                                               topkIdsTensor,
                                               "MoE routing TOPK_IDS_DESC_EXT tensor"));
    attributes.set_topk_ids(topkIdsTensor);

    // Unpack first_token_offset tensor
    std::shared_ptr<graph::TensorAttributes> firstTokenOffsetTensor;
    HIPDNN_CHECK_ERROR(
        unpackAndRegisterTensor(opDesc,
                                HIPDNN_ATTR_OPERATION_MOE_ROUTING_FIRST_TOKEN_OFFSET_DESC_EXT,
                                tensorMap,
                                firstTokenOffsetTensor,
                                "MoE routing FIRST_TOKEN_OFFSET_DESC_EXT tensor"));
    attributes.set_first_token_offset(firstTokenOffsetTensor);

    // Unpack token_index tensor
    std::shared_ptr<graph::TensorAttributes> tokenIndexTensor;
    HIPDNN_CHECK_ERROR(
        unpackAndRegisterTensor(opDesc,
                                HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOKEN_INDEX_DESC_EXT,
                                tensorMap,
                                tokenIndexTensor,
                                "MoE routing TOKEN_INDEX_DESC_EXT tensor"));
    attributes.set_token_index(tokenIndexTensor);

    // Unpack token_ks tensor
    std::shared_ptr<graph::TensorAttributes> tokenKsTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOKEN_KS_DESC_EXT,
                                               tensorMap,
                                               tokenKsTensor,
                                               "MoE routing TOKEN_KS_DESC_EXT tensor"));
    attributes.set_token_ks(tokenKsTensor);

    // Unpack num_experts
    int64_t numExperts = 0;
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_OPERATION_MOE_ROUTING_NUM_EXPERTS_EXT,
                                               HIPDNN_TYPE_INT64,
                                               numExperts,
                                               "MoE routing num_experts"));
    attributes.set_num_experts(numExperts);

    // Unpack operation name
    std::string opName;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrString(opDesc, HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    attributes.set_name(opName);

    return {};
}

} // namespace hipdnn_frontend::detail
