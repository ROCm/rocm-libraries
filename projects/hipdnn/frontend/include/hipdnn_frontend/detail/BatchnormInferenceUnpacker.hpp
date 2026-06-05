// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

[[nodiscard]] inline Error unpackBatchnormInferenceOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::BatchnormInferenceAttributes& attributes)
{
    // Unpack x tensor
    std::shared_ptr<graph::TensorAttributes> xTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT,
                                               tensorMap,
                                               xTensor,
                                               "batchnorminference X_EXT tensor"));
    attributes.set_x(xTensor);

    // Unpack mean tensor
    std::shared_ptr<graph::TensorAttributes> meanTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN_EXT,
                                               tensorMap,
                                               meanTensor,
                                               "batchnorminference MEAN_EXT tensor"));
    attributes.set_mean(meanTensor);

    // Unpack inv_variance tensor
    std::shared_ptr<graph::TensorAttributes> invVarianceTensor;
    HIPDNN_CHECK_ERROR(
        unpackAndRegisterTensor(opDesc,
                                HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE_EXT,
                                tensorMap,
                                invVarianceTensor,
                                "batchnorminference INV_VARIANCE_EXT tensor"));
    attributes.set_inv_variance(invVarianceTensor);

    // Unpack scale tensor
    std::shared_ptr<graph::TensorAttributes> scaleTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE_EXT,
                                               tensorMap,
                                               scaleTensor,
                                               "batchnorminference SCALE_EXT tensor"));
    attributes.set_scale(scaleTensor);

    // Unpack bias tensor
    std::shared_ptr<graph::TensorAttributes> biasTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS_EXT,
                                               tensorMap,
                                               biasTensor,
                                               "batchnorminference BIAS_EXT tensor"));
    attributes.set_bias(biasTensor);

    // Unpack y tensor
    std::shared_ptr<graph::TensorAttributes> yTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y_EXT,
                                               tensorMap,
                                               yTensor,
                                               "batchnorminference Y_EXT tensor"));
    attributes.set_y(yTensor);

    // Unpack compute data type
    auto [dt, dtErr] = unpackGraphDataType(
        opDesc, HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE_EXT, "batchnorminference compute data type");
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
