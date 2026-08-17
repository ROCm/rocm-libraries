// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/RMSNormAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

[[nodiscard]] inline Error unpackRMSNormOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::RMSNormAttributes& attributes)
{
    // Unpack x tensor
    std::shared_ptr<graph::TensorAttributes> xTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_RMSNORM_X_EXT, tensorMap, xTensor, "rmsnorm X_EXT tensor"));
    attributes.set_x(xTensor);

    // Unpack scale tensor
    std::shared_ptr<graph::TensorAttributes> scaleTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_RMSNORM_SCALE_EXT,
                                               tensorMap,
                                               scaleTensor,
                                               "rmsnorm SCALE_EXT tensor"));
    attributes.set_scale(scaleTensor);

    // Unpack epsilon tensor
    std::shared_ptr<graph::TensorAttributes> epsilonTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_RMSNORM_EPSILON_EXT,
                                               tensorMap,
                                               epsilonTensor,
                                               "rmsnorm EPSILON_EXT tensor"));
    attributes.set_epsilon(epsilonTensor);

    // Unpack y tensor
    std::shared_ptr<graph::TensorAttributes> yTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_RMSNORM_Y_EXT, tensorMap, yTensor, "rmsnorm Y_EXT tensor"));
    attributes.set_y(yTensor);

    // Unpack bias tensor
    std::shared_ptr<graph::TensorAttributes> biasTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_RMSNORM_BIAS_EXT,
                                            tensorMap,
                                            biasTensor,
                                            "rmsnorm BIAS_EXT tensor"));
    if(biasTensor)
    {
        attributes.set_bias(biasTensor);
    }

    // Unpack inv_rms tensor
    std::shared_ptr<graph::TensorAttributes> invRmsTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_RMSNORM_INV_RMS_EXT,
                                            tensorMap,
                                            invRmsTensor,
                                            "rmsnorm INV_RMS_EXT tensor"));
    if(invRmsTensor)
    {
        attributes.set_inv_rms(invRmsTensor);
    }

    // Unpack forward_phase
    hipdnnNormFwdPhase_t forwardPhase{};
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_OPERATION_RMSNORM_FWD_PHASE_EXT,
                                               HIPDNN_TYPE_NORM_FWD_PHASE,
                                               forwardPhase,
                                               "rmsnorm forward_phase"));
    auto [forwardPhaseResult, forwardPhaseErr] = fromHipdnnNormFwdPhase(forwardPhase);
    if(forwardPhaseErr.is_bad())
    {
        return forwardPhaseErr;
    }
    attributes.set_forward_phase(forwardPhaseResult);

    // Unpack compute data type
    auto [dt, dtErr] = unpackGraphDataType(
        opDesc, HIPDNN_ATTR_RMSNORM_COMP_TYPE_EXT, "rmsnorm compute data type");
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
