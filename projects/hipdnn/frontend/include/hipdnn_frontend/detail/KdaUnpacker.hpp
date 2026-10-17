// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/KdaAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

[[nodiscard]] inline Error unpackKdaOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::KdaAttributes& attributes)
{
    // Unpack q tensor
    std::shared_ptr<graph::TensorAttributes> qTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_KDA_Q_DESC_EXT, tensorMap, qTensor, "KDA Q_DESC_EXT tensor"));
    attributes.set_q(qTensor);

    // Unpack k tensor
    std::shared_ptr<graph::TensorAttributes> kTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_KDA_K_DESC_EXT, tensorMap, kTensor, "KDA K_DESC_EXT tensor"));
    attributes.set_k(kTensor);

    // Unpack v tensor
    std::shared_ptr<graph::TensorAttributes> vTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_KDA_V_DESC_EXT, tensorMap, vTensor, "KDA V_DESC_EXT tensor"));
    attributes.set_v(vTensor);

    // Unpack g tensor
    std::shared_ptr<graph::TensorAttributes> gTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_KDA_G_DESC_EXT, tensorMap, gTensor, "KDA G_DESC_EXT tensor"));
    attributes.set_g(gTensor);

    // Unpack beta tensor
    std::shared_ptr<graph::TensorAttributes> betaTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_KDA_BETA_DESC_EXT,
                                               tensorMap,
                                               betaTensor,
                                               "KDA BETA_DESC_EXT tensor"));
    attributes.set_beta(betaTensor);

    // Unpack cu_seqlens tensor
    std::shared_ptr<graph::TensorAttributes> cuSeqlensTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_KDA_CU_SEQLENS_DESC_EXT,
                                               tensorMap,
                                               cuSeqlensTensor,
                                               "KDA CU_SEQLENS_DESC_EXT tensor"));
    attributes.set_cu_seqlens(cuSeqlensTensor);

    // Unpack initial_state tensor
    std::shared_ptr<graph::TensorAttributes> initialStateTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_KDA_INITIAL_STATE_DESC_EXT,
                                            tensorMap,
                                            initialStateTensor,
                                            "KDA INITIAL_STATE_DESC_EXT tensor"));
    if(initialStateTensor)
    {
        attributes.set_initial_state(initialStateTensor);
    }

    // Unpack a_log tensor
    std::shared_ptr<graph::TensorAttributes> aLogTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_KDA_A_LOG_DESC_EXT,
                                            tensorMap,
                                            aLogTensor,
                                            "KDA A_LOG_DESC_EXT tensor"));
    if(aLogTensor)
    {
        attributes.set_a_log(aLogTensor);
    }

    // Unpack dt_bias tensor
    std::shared_ptr<graph::TensorAttributes> dtBiasTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_KDA_DT_BIAS_DESC_EXT,
                                            tensorMap,
                                            dtBiasTensor,
                                            "KDA DT_BIAS_DESC_EXT tensor"));
    if(dtBiasTensor)
    {
        attributes.set_dt_bias(dtBiasTensor);
    }

    // Unpack state_indices tensor
    std::shared_ptr<graph::TensorAttributes> stateIndicesTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_KDA_STATE_INDICES_DESC_EXT,
                                            tensorMap,
                                            stateIndicesTensor,
                                            "KDA STATE_INDICES_DESC_EXT tensor"));
    if(stateIndicesTensor)
    {
        attributes.set_state_indices(stateIndicesTensor);
    }

    // Unpack o tensor
    std::shared_ptr<graph::TensorAttributes> oTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_KDA_O_DESC_EXT, tensorMap, oTensor, "KDA O_DESC_EXT tensor"));
    attributes.set_o(oTensor);

    // Unpack final_state tensor
    std::shared_ptr<graph::TensorAttributes> finalStateTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_KDA_FINAL_STATE_DESC_EXT,
                                            tensorMap,
                                            finalStateTensor,
                                            "KDA FINAL_STATE_DESC_EXT tensor"));
    if(finalStateTensor)
    {
        attributes.set_final_state(finalStateTensor);
    }

    // Unpack state_checkpoints tensor
    std::shared_ptr<graph::TensorAttributes> stateCheckpointsTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_KDA_STATE_CHECKPOINTS_DESC_EXT,
                                            tensorMap,
                                            stateCheckpointsTensor,
                                            "KDA STATE_CHECKPOINTS_DESC_EXT tensor"));
    if(stateCheckpointsTensor)
    {
        attributes.set_state_checkpoints(stateCheckpointsTensor);
    }

    // Unpack scale (optional)
    {
        std::optional<float> scale;
        HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(
            opDesc, HIPDNN_ATTR_OPERATION_KDA_SCALE_EXT, HIPDNN_TYPE_FLOAT, scale, "KDA scale"));
        if(scale.has_value())
        {
            attributes.set_scale(*scale);
        }
    }

    // Unpack output_final_state
    bool outputFinalState = false;
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_OPERATION_KDA_OUTPUT_FINAL_STATE_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               outputFinalState,
                                               "KDA output_final_state"));
    attributes.set_output_final_state(outputFinalState);

    // Unpack use_qk_l2norm
    bool useQkL2norm = false;
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_OPERATION_KDA_USE_QK_L2NORM_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               useQkL2norm,
                                               "KDA use_qk_l2norm"));
    attributes.set_use_qk_l2norm(useQkL2norm);

    // Unpack checkpoint_every_n_tokens
    int64_t checkpointEveryNTokens = 0;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrScalar(opDesc,
                                HIPDNN_ATTR_OPERATION_KDA_CHECKPOINT_EVERY_N_TOKENS_EXT,
                                HIPDNN_TYPE_INT64,
                                checkpointEveryNTokens,
                                "KDA checkpoint_every_n_tokens"));
    attributes.set_checkpoint_every_n_tokens(checkpointEveryNTokens);

    // Unpack use_beta_sigmoid
    bool useBetaSigmoid = false;
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_OPERATION_KDA_USE_BETA_SIGMOID_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               useBetaSigmoid,
                                               "KDA use_beta_sigmoid"));
    attributes.set_use_beta_sigmoid(useBetaSigmoid);

    // Unpack allow_neg_eigval
    bool allowNegEigval = false;
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_OPERATION_KDA_ALLOW_NEG_EIGVAL_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               allowNegEigval,
                                               "KDA allow_neg_eigval"));
    attributes.set_allow_neg_eigval(allowNegEigval);

    // Unpack safe_gate
    bool safeGate = false;
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_OPERATION_KDA_SAFE_GATE_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               safeGate,
                                               "KDA safe_gate"));
    attributes.set_safe_gate(safeGate);

    // Unpack gate_domain
    hipdnnGateDomain_t gateDomain{};
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_OPERATION_KDA_GATE_DOMAIN_EXT,
                                               HIPDNN_TYPE_GATE_DOMAIN_EXT,
                                               gateDomain,
                                               "KDA gate_domain"));
    auto [gateDomainResult, gateDomainErr] = fromHipdnnGateDomain(gateDomain);
    if(gateDomainErr.is_bad())
    {
        return gateDomainErr;
    }
    attributes.set_gate_domain(gateDomainResult);

    // Unpack gate_lower_bound (optional)
    {
        std::optional<float> gateLowerBound;
        HIPDNN_CHECK_ERROR(
            getDescriptorAttrOptionalScalar(opDesc,
                                            HIPDNN_ATTR_OPERATION_KDA_GATE_LOWER_BOUND_EXT,
                                            HIPDNN_TYPE_FLOAT,
                                            gateLowerBound,
                                            "KDA gate_lower_bound"));
        if(gateLowerBound.has_value())
        {
            attributes.set_gate_lower_bound(*gateLowerBound);
        }
    }

    // Unpack batch_invariant
    bool batchInvariant = false;
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_OPERATION_KDA_BATCH_INVARIANT_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               batchInvariant,
                                               "KDA batch_invariant"));
    attributes.set_batch_invariant(batchInvariant);

    // Unpack overwrite_initial_state
    bool overwriteInitialState = false;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrScalar(opDesc,
                                HIPDNN_ATTR_OPERATION_KDA_OVERWRITE_INITIAL_STATE_EXT,
                                HIPDNN_TYPE_BOOLEAN,
                                overwriteInitialState,
                                "KDA overwrite_initial_state"));
    attributes.set_overwrite_initial_state(overwriteInitialState);

    // Unpack compute data type
    auto [dt, dtErr] = unpackGraphDataType(
        opDesc, HIPDNN_ATTR_OPERATION_KDA_COMP_TYPE_EXT, "KDA compute data type");
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
