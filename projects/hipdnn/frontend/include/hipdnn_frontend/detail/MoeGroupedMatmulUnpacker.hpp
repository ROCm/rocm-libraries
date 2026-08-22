// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/MoeGroupedMatmulAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

[[nodiscard]] inline Error unpackMoeGroupedMatmulOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::MoeGroupedMatmulAttributes& attributes)
{
    // Unpack token tensor
    std::shared_ptr<graph::TensorAttributes> tokenTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC,
                                               tensorMap,
                                               tokenTensor,
                                               "MoE grouped matmul TOKEN_DESC tensor"));
    attributes.set_token(tokenTensor);

    // Unpack weight tensor
    std::shared_ptr<graph::TensorAttributes> weightTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC,
                                               tensorMap,
                                               weightTensor,
                                               "MoE grouped matmul WEIGHT_DESC tensor"));
    attributes.set_weight(weightTensor);

    // Unpack first_token_offset tensor
    std::shared_ptr<graph::TensorAttributes> firstTokenOffsetTensor;
    HIPDNN_CHECK_ERROR(
        unpackAndRegisterTensor(opDesc,
                                HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC,
                                tensorMap,
                                firstTokenOffsetTensor,
                                "MoE grouped matmul FIRST_TOKEN_OFFSET_DESC tensor"));
    attributes.set_first_token_offset(firstTokenOffsetTensor);

    // Unpack output tensor
    std::shared_ptr<graph::TensorAttributes> outputTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC,
                                               tensorMap,
                                               outputTensor,
                                               "MoE grouped matmul OUTPUT_DESC tensor"));
    attributes.set_output(outputTensor);

    // The mode determines which routing attributes are present.
    hipdnnMoeGroupedMatmulMode_t mode{};
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE,
                                               HIPDNN_TYPE_MOE_GROUPED_MATMUL_MODE,
                                               mode,
                                               "MoE grouped matmul mode"));
    auto [modeResult, modeErr] = fromHipdnnMoeGroupedMatmulMode(mode);
    if(modeErr.is_bad())
    {
        return modeErr;
    }
    attributes.set_mode(modeResult);

    switch(modeResult)
    {
    case MoeGroupedMatmulMode::NONE:
        break;
    case MoeGroupedMatmulMode::GATHER:
    {
        std::shared_ptr<graph::TensorAttributes> tokenIndexTensor;
        HIPDNN_CHECK_ERROR(
            unpackAndRegisterTensor(opDesc,
                                    HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC,
                                    tensorMap,
                                    tokenIndexTensor,
                                    "MoE grouped matmul TOKEN_INDEX_DESC tensor"));
        attributes.set_token_index(tokenIndexTensor);
    }
    break;
    case MoeGroupedMatmulMode::SCATTER:
    {
        std::shared_ptr<graph::TensorAttributes> tokenIndexTensor;
        HIPDNN_CHECK_ERROR(
            unpackAndRegisterTensor(opDesc,
                                    HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC,
                                    tensorMap,
                                    tokenIndexTensor,
                                    "MoE grouped matmul TOKEN_INDEX_DESC tensor"));
        attributes.set_token_index(tokenIndexTensor);
    }
        {
            std::shared_ptr<graph::TensorAttributes> tokenKsTensor;
            HIPDNN_CHECK_ERROR(
                unpackAndRegisterTensor(opDesc,
                                        HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC,
                                        tensorMap,
                                        tokenKsTensor,
                                        "MoE grouped matmul TOKEN_KS_DESC tensor"));
            attributes.set_token_ks(tokenKsTensor);
        }
        {
            int32_t topK = 0;
            HIPDNN_CHECK_ERROR(
                getDescriptorAttrScalar(opDesc,
                                        HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K,
                                        HIPDNN_TYPE_INT32,
                                        topK,
                                        "MoE grouped matmul top_k"));
            attributes.set_top_k(topK);
        }
        break;
    default:
        return {ErrorCode::INVALID_VALUE, "MoE grouped matmul has an unknown routing mode"};
    }

    // Unpack compute data type
    auto [dt, dtErr] = unpackGraphDataType(opDesc,
                                           HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC,
                                           "MoE grouped matmul compute data type");
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
