// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/SdpaBackwardAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

/// Unpacks an SDPA backward operation descriptor and populates
/// SdpaBackwardAttributes with tensors (using tensorMap for sharing)
/// and scalar/enum parameters.
[[nodiscard]] inline Error unpackSdpaBwdOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::SdpaBackwardAttributes& attributes)
{
    // Required input tensors
    std::shared_ptr<graph::TensorAttributes> qTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_SDPA_BWD_Q_EXT, tensorMap, qTensor, "SDPA bprop Q"));
    attributes.set_q(qTensor);

    std::shared_ptr<graph::TensorAttributes> kTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_SDPA_BWD_K_EXT, tensorMap, kTensor, "SDPA bprop K"));
    attributes.set_k(kTensor);

    std::shared_ptr<graph::TensorAttributes> vTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_SDPA_BWD_V_EXT, tensorMap, vTensor, "SDPA bprop V"));
    attributes.set_v(vTensor);

    std::shared_ptr<graph::TensorAttributes> oTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_SDPA_BWD_O_EXT, tensorMap, oTensor, "SDPA bprop O"));
    attributes.set_o(oTensor);

    std::shared_ptr<graph::TensorAttributes> doTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_SDPA_BWD_DO_EXT, tensorMap, doTensor, "SDPA bprop dO"));
    attributes.set_do(doTensor);

    std::shared_ptr<graph::TensorAttributes> statsTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_SDPA_BWD_STATS_EXT,
                                               tensorMap,
                                               statsTensor,
                                               "SDPA bprop Stats"));
    attributes.set_stats(statsTensor);

    // Required output tensors
    std::shared_ptr<graph::TensorAttributes> dqTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_SDPA_BWD_DQ_EXT, tensorMap, dqTensor, "SDPA bprop dQ"));
    attributes.set_dq(dqTensor);

    std::shared_ptr<graph::TensorAttributes> dkTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_SDPA_BWD_DK_EXT, tensorMap, dkTensor, "SDPA bprop dK"));
    attributes.set_dk(dkTensor);

    std::shared_ptr<graph::TensorAttributes> dvTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_SDPA_BWD_DV_EXT, tensorMap, dvTensor, "SDPA bprop dV"));
    attributes.set_dv(dvTensor);

    // Optional input tensors
    std::shared_ptr<graph::TensorAttributes> scaleTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_BWD_SCALE_EXT,
                                            tensorMap,
                                            scaleTensor,
                                            "SDPA bprop SCALE"));
    if(scaleTensor)
    {
        attributes.set_attn_scale(scaleTensor);
    }

    std::shared_ptr<graph::TensorAttributes> attnMaskTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_BWD_ATTN_MASK_EXT,
                                            tensorMap,
                                            attnMaskTensor,
                                            "SDPA bprop ATTN_MASK"));
    if(attnMaskTensor)
    {
        // The attention mask is stored via set_bias() in SdpaBackwardAttributes
        // (same naming convention as the forward pass).
        attributes.set_bias(attnMaskTensor);
    }

    std::shared_ptr<graph::TensorAttributes> seqLenQTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_Q_EXT,
                                            tensorMap,
                                            seqLenQTensor,
                                            "SDPA bprop SEQ_LEN_Q"));
    if(seqLenQTensor)
    {
        attributes.set_seq_len_q(seqLenQTensor);
    }

    std::shared_ptr<graph::TensorAttributes> seqLenKvTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KV_EXT,
                                            tensorMap,
                                            seqLenKvTensor,
                                            "SDPA bprop SEQ_LEN_KV"));
    if(seqLenKvTensor)
    {
        attributes.set_seq_len_kv(seqLenKvTensor);
    }

    std::shared_ptr<graph::TensorAttributes> seedTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(
        opDesc, HIPDNN_ATTR_OPERATION_SDPA_BWD_SEED_EXT, tensorMap, seedTensor, "SDPA bprop SEED"));
    if(seedTensor)
    {
        attributes.set_seed(seedTensor);
    }

    std::shared_ptr<graph::TensorAttributes> offsetTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_BWD_OFFSET_EXT,
                                            tensorMap,
                                            offsetTensor,
                                            "SDPA bprop OFFSET"));
    if(offsetTensor)
    {
        attributes.set_offset(offsetTensor);
    }

    std::shared_ptr<graph::TensorAttributes> dropoutMaskTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_BWD_DROPOUT_MASK_EXT,
                                            tensorMap,
                                            dropoutMaskTensor,
                                            "SDPA bprop DROPOUT_MASK"));
    if(dropoutMaskTensor)
    {
        attributes.set_dropout_mask(dropoutMaskTensor);
    }

    std::shared_ptr<graph::TensorAttributes> dropoutScaleTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_BWD_DROPOUT_SCALE_EXT,
                                            tensorMap,
                                            dropoutScaleTensor,
                                            "SDPA bprop DROPOUT_SCALE"));
    if(dropoutScaleTensor)
    {
        attributes.set_dropout_scale(dropoutScaleTensor);
    }

    std::shared_ptr<graph::TensorAttributes> dropoutScaleInvTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_BWD_DROPOUT_SCALE_INV_EXT,
                                            tensorMap,
                                            dropoutScaleInvTensor,
                                            "SDPA bprop DROPOUT_SCALE_INV"));
    if(dropoutScaleInvTensor)
    {
        attributes.set_dropout_scale_inv(dropoutScaleInvTensor);
    }

    // Optional output tensor
    std::shared_ptr<graph::TensorAttributes> dbiasTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_BWD_DBIAS_EXT,
                                            tensorMap,
                                            dbiasTensor,
                                            "SDPA bprop DBIAS"));
    if(dbiasTensor)
    {
        attributes.set_dbias(dbiasTensor);
    }

    // Boolean scalars — plain bool members, so unpack into optional and apply default
    {
        std::optional<bool> opt;
        HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(opDesc,
                                                           HIPDNN_ATTR_SDPA_BWD_ALIBI_MASK_EXT,
                                                           HIPDNN_TYPE_BOOLEAN,
                                                           opt,
                                                           "SDPA bprop alibi_mask"));
        attributes.alibi_mask = opt.value_or(false);
    }
    {
        std::optional<bool> opt;
        HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(opDesc,
                                                           HIPDNN_ATTR_SDPA_BWD_PADDING_MASK_EXT,
                                                           HIPDNN_TYPE_BOOLEAN,
                                                           opt,
                                                           "SDPA bprop padding_mask"));
        attributes.padding_mask = opt.value_or(false);
    }
    {
        std::optional<bool> opt;
        HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(opDesc,
                                                           HIPDNN_ATTR_SDPA_BWD_CAUSAL_MASK_EXT,
                                                           HIPDNN_TYPE_BOOLEAN,
                                                           opt,
                                                           "SDPA bprop causal_mask"));
        attributes.causal_mask = opt.value_or(false);
    }
    {
        std::optional<bool> opt;
        HIPDNN_CHECK_ERROR(
            getDescriptorAttrOptionalScalar(opDesc,
                                            HIPDNN_ATTR_SDPA_BWD_CAUSAL_MASK_BOTTOM_RIGHT_EXT,
                                            HIPDNN_TYPE_BOOLEAN,
                                            opt,
                                            "SDPA bprop causal_mask_bottom_right"));
        attributes.causal_mask_bottom_right = opt.value_or(false);
    }

    // Optional float scalars
    HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(opDesc,
                                                       HIPDNN_ATTR_SDPA_BWD_DROPOUT_PROBABILITY_EXT,
                                                       HIPDNN_TYPE_FLOAT,
                                                       attributes.dropout_probability,
                                                       "SDPA bprop dropout_probability"));
    HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(opDesc,
                                                       HIPDNN_ATTR_SDPA_BWD_ATTN_SCALE_VALUE_EXT,
                                                       HIPDNN_TYPE_FLOAT,
                                                       attributes.attn_scale_value,
                                                       "SDPA bprop attn_scale_value"));

    // Optional int64 scalars
    HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(opDesc,
                                                       HIPDNN_ATTR_SDPA_BWD_LEFT_BOUND_EXT,
                                                       HIPDNN_TYPE_INT64,
                                                       attributes.left_bound,
                                                       "SDPA bprop left_bound"));
    HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(opDesc,
                                                       HIPDNN_ATTR_SDPA_BWD_RIGHT_BOUND_EXT,
                                                       HIPDNN_TYPE_INT64,
                                                       attributes.right_bound,
                                                       "SDPA bprop right_bound"));

    // Diagonal alignment
    hipdnnDiagonalAlignment_t diagAlign{};
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_SDPA_BWD_DIAGONAL_ALIGNMENT_EXT,
                                               HIPDNN_TYPE_DIAGONAL_ALIGNMENT_EXT,
                                               diagAlign,
                                               "SDPA bprop diagonal_alignment"));
    auto [alignVal, alignErr] = fromHipdnnDiagonalAlignment(diagAlign);
    if(alignErr.is_bad())
    {
        return alignErr;
    }
    attributes.set_diagonal_alignment(alignVal);

    // Compute data type
    auto [dt, dtErr]
        = unpackGraphDataType(opDesc, HIPDNN_ATTR_SDPA_BWD_COMP_TYPE_EXT, "SDPA bprop math prec");
    if(dtErr.is_bad())
    {
        return dtErr;
    }
    attributes.set_compute_data_type(dt);

    // Operation name
    std::string opName;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrString(opDesc, HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    attributes.set_name(opName);

    return {};
}

} // namespace hipdnn_frontend::detail
