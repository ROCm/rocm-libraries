// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/CausalConv1dAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

[[nodiscard]] inline Error unpackCausalConv1dOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::CausalConv1dAttributes& attributes)
{
    // Unpack x tensor
    std::shared_ptr<graph::TensorAttributes> xTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_X_EXT,
                                               tensorMap,
                                               xTensor,
                                               "causal conv1d X_EXT tensor"));
    attributes.set_x(xTensor);

    // Unpack weight tensor
    std::shared_ptr<graph::TensorAttributes> weightTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_WEIGHT_EXT,
                                               tensorMap,
                                               weightTensor,
                                               "causal conv1d WEIGHT_EXT tensor"));
    attributes.set_weight(weightTensor);

    // Unpack cu_seqlens tensor
    std::shared_ptr<graph::TensorAttributes> cuSeqlensTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_CU_SEQLENS_EXT,
                                               tensorMap,
                                               cuSeqlensTensor,
                                               "causal conv1d CU_SEQLENS_EXT tensor"));
    attributes.set_cu_seqlens(cuSeqlensTensor);

    // Unpack y tensor
    std::shared_ptr<graph::TensorAttributes> yTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_Y_EXT,
                                               tensorMap,
                                               yTensor,
                                               "causal conv1d Y_EXT tensor"));
    attributes.set_y(yTensor);

    // Unpack bias tensor
    std::shared_ptr<graph::TensorAttributes> biasTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_BIAS_EXT,
                                            tensorMap,
                                            biasTensor,
                                            "causal conv1d BIAS_EXT tensor"));
    if(biasTensor)
    {
        attributes.set_bias(biasTensor);
    }

    // Unpack initial_state tensor
    std::shared_ptr<graph::TensorAttributes> initialStateTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_INITIAL_STATE_EXT,
                                            tensorMap,
                                            initialStateTensor,
                                            "causal conv1d INITIAL_STATE_EXT tensor"));
    if(initialStateTensor)
    {
        attributes.set_initial_state(initialStateTensor);
    }

    // Unpack state_indices tensor
    std::shared_ptr<graph::TensorAttributes> stateIndicesTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_STATE_INDICES_EXT,
                                            tensorMap,
                                            stateIndicesTensor,
                                            "causal conv1d STATE_INDICES_EXT tensor"));
    if(stateIndicesTensor)
    {
        attributes.set_state_indices(stateIndicesTensor);
    }

    // Unpack final_state tensor
    std::shared_ptr<graph::TensorAttributes> finalStateTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_FINAL_STATE_EXT,
                                            tensorMap,
                                            finalStateTensor,
                                            "causal conv1d FINAL_STATE_EXT tensor"));
    if(finalStateTensor)
    {
        attributes.set_final_state(finalStateTensor);
        attributes.set_output_final_state(true);
    }

    // Unpack activation
    hipdnnCausalConv1dActivation_t activation{};
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_ACTIVATION_EXT,
                                               HIPDNN_TYPE_CAUSAL_CONV1D_ACTIVATION_EXT,
                                               activation,
                                               "causal conv1d activation"));
    auto [activationResult, activationErr] = fromHipdnnCausalConv1dActivation(activation);
    if(activationErr.is_bad())
    {
        return activationErr;
    }
    attributes.set_activation(activationResult);

    // Unpack compute data type
    auto [dt, dtErr] = unpackGraphDataType(
        opDesc, HIPDNN_ATTR_CAUSAL_CONV1D_COMP_TYPE_EXT, "causal conv1d compute data type");
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
