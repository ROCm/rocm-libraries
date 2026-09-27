// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/MoeGroupedMatmulBwdAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

[[nodiscard]] inline Error unpackMoeGroupedMatmulBwdOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::MoeGroupedMatmulBwdAttributes& attributes)
{
    // Unpack doutput tensor
    std::shared_ptr<graph::TensorAttributes> doutputTensor;
    HIPDNN_CHECK_ERROR(
        unpackAndRegisterTensor(opDesc,
                                HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC,
                                tensorMap,
                                doutputTensor,
                                "MoE grouped matmul backward DOUTPUT_DESC tensor"));
    attributes.set_doutput(doutputTensor);

    // Unpack token tensor
    std::shared_ptr<graph::TensorAttributes> tokenTensor;
    HIPDNN_CHECK_ERROR(
        unpackAndRegisterTensor(opDesc,
                                HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_TOKEN_DESC,
                                tensorMap,
                                tokenTensor,
                                "MoE grouped matmul backward TOKEN_DESC tensor"));
    attributes.set_token(tokenTensor);

    // Unpack first_token_offset tensor
    std::shared_ptr<graph::TensorAttributes> firstTokenOffsetTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc,
        HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC,
        tensorMap,
        firstTokenOffsetTensor,
        "MoE grouped matmul backward FIRST_TOKEN_OFFSET_DESC tensor"));
    attributes.set_first_token_offset(firstTokenOffsetTensor);

    // Unpack dweight tensor
    std::shared_ptr<graph::TensorAttributes> dweightTensor;
    HIPDNN_CHECK_ERROR(
        unpackAndRegisterTensor(opDesc,
                                HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DWEIGHT_DESC,
                                tensorMap,
                                dweightTensor,
                                "MoE grouped matmul backward DWEIGHT_DESC tensor"));
    attributes.set_dweight(dweightTensor);

    // Unpack compute data type
    auto [dt, dtErr] = unpackGraphDataType(opDesc,
                                           HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC,
                                           "MoE grouped matmul backward compute data type");
    if(dtErr.is_bad())
    {
        return dtErr;
    }
    attributes.set_compute_data_type(dt);

    // Unpack operation name
    std::string opName;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrString(opDesc, HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    attributes.set_name(opName);

    return {};
}

} // namespace hipdnn_frontend::detail
