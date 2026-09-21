// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/LayernormAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

/// Unpacks a layernorm operation from a backend operation descriptor.
/// Populates the LayernormAttributes with tensors (using tensorMap for sharing)
/// and layernorm parameters.
[[nodiscard]] inline Error unpackLayernormOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::LayernormAttributes& attributes)
{
    // Unpack X (input) tensor
    std::shared_ptr<graph::TensorAttributes> xTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_LAYERNORM_X_EXT, tensorMap, xTensor, "layernorm X tensor"));
    attributes.set_x(xTensor);

    // Unpack Scale tensor
    std::shared_ptr<graph::TensorAttributes> scaleTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_LAYERNORM_SCALE_EXT,
                                               tensorMap,
                                               scaleTensor,
                                               "layernorm Scale tensor"));
    attributes.set_scale(scaleTensor);

    // Unpack Bias tensor
    std::shared_ptr<graph::TensorAttributes> biasTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_LAYERNORM_BIAS_EXT,
                                               tensorMap,
                                               biasTensor,
                                               "layernorm Bias tensor"));
    attributes.set_bias(biasTensor);

    // Unpack Epsilon tensor
    std::shared_ptr<graph::TensorAttributes> epsilonTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_LAYERNORM_EPSILON_EXT,
                                               tensorMap,
                                               epsilonTensor,
                                               "layernorm Epsilon tensor"));
    attributes.set_epsilon(epsilonTensor);

    // Unpack Y (output) tensor
    std::shared_ptr<graph::TensorAttributes> yTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_LAYERNORM_Y_EXT, tensorMap, yTensor, "layernorm Y tensor"));
    attributes.set_y(yTensor);

    // Unpack optional Mean tensor
    std::shared_ptr<graph::TensorAttributes> meanTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_LAYERNORM_MEAN_EXT,
                                            tensorMap,
                                            meanTensor,
                                            "layernorm Mean tensor"));
    if(meanTensor)
    {
        attributes.set_mean(meanTensor);
    }

    // Unpack optional InvVariance tensor
    std::shared_ptr<graph::TensorAttributes> invVarianceTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_LAYERNORM_INV_VARIANCE_EXT,
                                            tensorMap,
                                            invVarianceTensor,
                                            "layernorm InvVariance tensor"));
    if(invVarianceTensor)
    {
        attributes.set_inv_variance(invVarianceTensor);
    }

    // Unpack forward phase
    hipdnnNormFwdPhase_t fwdPhase{};
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_OPERATION_LAYERNORM_FWD_PHASE_EXT,
                                               HIPDNN_TYPE_NORM_FWD_PHASE,
                                               fwdPhase,
                                               "layernorm forward phase"));
    auto [phase, phaseErr] = fromHipdnnNormFwdPhase(fwdPhase);
    if(phaseErr.is_bad())
    {
        return phaseErr;
    }
    attributes.set_forward_phase(phase);

    // Unpack normalized dim count
    int64_t normalizedDimCount = 0;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrScalar(opDesc,
                                HIPDNN_ATTR_OPERATION_LAYERNORM_NORMALIZED_DIM_COUNT_EXT,
                                HIPDNN_TYPE_INT64,
                                normalizedDimCount,
                                "layernorm normalized dim count"));
    attributes.set_normalized_dim_count(normalizedDimCount);

    // Unpack compute data type
    auto [dt, dtErr] = unpackGraphDataType(
        opDesc, HIPDNN_ATTR_LAYERNORM_COMP_TYPE_EXT, "layernorm compute data type");
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
