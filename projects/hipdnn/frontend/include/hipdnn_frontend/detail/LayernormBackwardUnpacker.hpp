// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/LayernormBackwardAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

[[nodiscard]] inline Error unpackLayernormBackwardOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::LayernormBackwardAttributes& attributes)
{
    // Unpack dy tensor
    std::shared_ptr<graph::TensorAttributes> dyTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY,
                                               tensorMap,
                                               dyTensor,
                                               "layernormbackward DY tensor"));
    attributes.set_dy(dyTensor);

    // Unpack x tensor
    std::shared_ptr<graph::TensorAttributes> xTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_X,
                                               tensorMap,
                                               xTensor,
                                               "layernormbackward X tensor"));
    attributes.set_x(xTensor);

    // Unpack scale tensor
    std::shared_ptr<graph::TensorAttributes> scaleTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_SCALE,
                                               tensorMap,
                                               scaleTensor,
                                               "layernormbackward SCALE tensor"));
    attributes.set_scale(scaleTensor);

    // Unpack mean tensor
    std::shared_ptr<graph::TensorAttributes> meanTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_MEAN,
                                            tensorMap,
                                            meanTensor,
                                            "layernormbackward MEAN tensor"));
    if(meanTensor)
    {
        attributes.set_mean(meanTensor);
    }

    // Unpack inv_variance tensor
    std::shared_ptr<graph::TensorAttributes> invVarianceTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_INV_VARIANCE,
                                            tensorMap,
                                            invVarianceTensor,
                                            "layernormbackward INV_VARIANCE tensor"));
    if(invVarianceTensor)
    {
        attributes.set_inv_variance(invVarianceTensor);
    }

    // Unpack epsilon tensor
    std::shared_ptr<graph::TensorAttributes> epsilonTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_EPSILON,
                                            tensorMap,
                                            epsilonTensor,
                                            "layernormbackward EPSILON tensor"));
    if(epsilonTensor)
    {
        attributes.set_epsilon(epsilonTensor);
    }

    // Unpack dx tensor
    std::shared_ptr<graph::TensorAttributes> dxTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DX,
                                               tensorMap,
                                               dxTensor,
                                               "layernormbackward DX tensor"));
    attributes.set_dx(dxTensor);

    // Unpack dscale tensor
    std::shared_ptr<graph::TensorAttributes> dscaleTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DSCALE,
                                               tensorMap,
                                               dscaleTensor,
                                               "layernormbackward DSCALE tensor"));
    attributes.set_dscale(dscaleTensor);

    // Unpack dbias tensor
    std::shared_ptr<graph::TensorAttributes> dbiasTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DBIAS,
                                               tensorMap,
                                               dbiasTensor,
                                               "layernormbackward DBIAS tensor"));
    attributes.set_dbias(dbiasTensor);

    // Unpack normalized_dim_count
    int64_t normalizedDimCount = 0;
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_LAYERNORM_BACKWARD_NORMALIZED_DIM_COUNT,
                                               HIPDNN_TYPE_INT64,
                                               normalizedDimCount,
                                               "layernormbackward normalized_dim_count"));
    attributes.set_normalized_dim_count(normalizedDimCount);

    // Unpack compute data type
    auto [dt, dtErr] = unpackGraphDataType(opDesc,
                                           HIPDNN_ATTR_LAYERNORM_BACKWARD_COMP_TYPE_EXT,
                                           "layernormbackward compute data type");
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
