// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributesVarianceExt.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

[[nodiscard]] inline Error unpackBatchnormInferenceVarianceExtOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::BatchnormInferenceAttributesVarianceExt& attributes)
{
    // Unpack x tensor
    std::shared_ptr<graph::TensorAttributes> xTensor;
    HIPDNN_CHECK_ERROR(
        unpackAndRegisterTensor(opDesc,
                                HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_X_EXT,
                                tensorMap,
                                xTensor,
                                "batchnorminferencevarianceext X tensor"));
    attributes.set_x(xTensor);

    // Unpack mean tensor
    std::shared_ptr<graph::TensorAttributes> meanTensor;
    HIPDNN_CHECK_ERROR(
        unpackAndRegisterTensor(opDesc,
                                HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_MEAN_EXT,
                                tensorMap,
                                meanTensor,
                                "batchnorminferencevarianceext MEAN tensor"));
    attributes.set_mean(meanTensor);

    // Unpack variance tensor
    std::shared_ptr<graph::TensorAttributes> varianceTensor;
    HIPDNN_CHECK_ERROR(
        unpackAndRegisterTensor(opDesc,
                                HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_VARIANCE_EXT,
                                tensorMap,
                                varianceTensor,
                                "batchnorminferencevarianceext VARIANCE tensor"));
    attributes.set_variance(varianceTensor);

    // Unpack scale tensor
    std::shared_ptr<graph::TensorAttributes> scaleTensor;
    HIPDNN_CHECK_ERROR(
        unpackAndRegisterTensor(opDesc,
                                HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_SCALE_EXT,
                                tensorMap,
                                scaleTensor,
                                "batchnorminferencevarianceext SCALE tensor"));
    attributes.set_scale(scaleTensor);

    // Unpack bias tensor
    std::shared_ptr<graph::TensorAttributes> biasTensor;
    HIPDNN_CHECK_ERROR(
        unpackAndRegisterTensor(opDesc,
                                HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_BIAS_EXT,
                                tensorMap,
                                biasTensor,
                                "batchnorminferencevarianceext BIAS tensor"));
    attributes.set_bias(biasTensor);

    // Unpack y tensor
    std::shared_ptr<graph::TensorAttributes> yTensor;
    HIPDNN_CHECK_ERROR(
        unpackAndRegisterTensor(opDesc,
                                HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_Y_EXT,
                                tensorMap,
                                yTensor,
                                "batchnorminferencevarianceext Y tensor"));
    attributes.set_y(yTensor);

    // Unpack epsilon tensor
    std::shared_ptr<graph::TensorAttributes> epsilonTensor;
    HIPDNN_CHECK_ERROR(
        unpackAndRegisterTensor(opDesc,
                                HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_EPSILON_EXT,
                                tensorMap,
                                epsilonTensor,
                                "batchnorminferencevarianceext EPSILON tensor"));
    attributes.set_epsilon(epsilonTensor);

    // Unpack compute data type
    auto [dt, dtErr] = unpackGraphDataType(opDesc,
                                           HIPDNN_ATTR_BATCHNORM_INF_VAR_COMP_TYPE_EXT,
                                           "batchnorminferencevarianceext compute data type");
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
