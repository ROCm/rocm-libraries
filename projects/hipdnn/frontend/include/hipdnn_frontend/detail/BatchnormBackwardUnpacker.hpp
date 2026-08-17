// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/BatchnormBackwardAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <unordered_map>
#include <vector>

namespace hipdnn_frontend::detail
{

/// Unpacks a batchnorm backward operation from a backend operation descriptor.
/// Populates the BatchnormBackwardAttributes with tensors (using tensorMap for sharing)
/// and compute data type.
[[nodiscard]] inline Error unpackBatchnormBackwardOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::BatchnormBackwardAttributes& attributes)
{
    // Unpack required tensors
    std::shared_ptr<graph::TensorAttributes> dyTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT,
                                               tensorMap,
                                               dyTensor,
                                               "batchnorm backward DY tensor"));
    attributes.set_dy(dyTensor);

    std::shared_ptr<graph::TensorAttributes> xTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_X_EXT,
                                               tensorMap,
                                               xTensor,
                                               "batchnorm backward X tensor"));
    attributes.set_x(xTensor);

    std::shared_ptr<graph::TensorAttributes> scaleTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_SCALE_EXT,
                                               tensorMap,
                                               scaleTensor,
                                               "batchnorm backward Scale tensor"));
    attributes.set_scale(scaleTensor);

    std::shared_ptr<graph::TensorAttributes> dxTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DX_EXT,
                                               tensorMap,
                                               dxTensor,
                                               "batchnorm backward DX tensor"));
    attributes.set_dx(dxTensor);

    std::shared_ptr<graph::TensorAttributes> dscaleTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DSCALE_EXT,
                                               tensorMap,
                                               dscaleTensor,
                                               "batchnorm backward DScale tensor"));
    attributes.set_dscale(dscaleTensor);

    std::shared_ptr<graph::TensorAttributes> dbiasTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DBIAS_EXT,
                                               tensorMap,
                                               dbiasTensor,
                                               "batchnorm backward DBias tensor"));
    attributes.set_dbias(dbiasTensor);

    // Unpack optional tensors
    std::shared_ptr<graph::TensorAttributes> meanTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_MEAN_EXT,
                                            tensorMap,
                                            meanTensor,
                                            "batchnorm backward Mean tensor"));
    if(meanTensor)
    {
        attributes.set_mean(meanTensor);
    }

    std::shared_ptr<graph::TensorAttributes> invVarianceTensor;
    HIPDNN_CHECK_ERROR(
        unpackOptionalTensor(opDesc,
                             HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_INV_VARIANCE_EXT,
                             tensorMap,
                             invVarianceTensor,
                             "batchnorm backward InvVariance tensor"));
    if(invVarianceTensor)
    {
        attributes.set_inv_variance(invVarianceTensor);
    }

    // Unpack peer_stats tensor array
    std::vector<std::shared_ptr<graph::TensorAttributes>> peerStatsTensors;
    HIPDNN_CHECK_ERROR(
        unpackAndRegisterTensorArray(opDesc,
                                     HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_PEER_STATS_EXT,
                                     tensorMap,
                                     peerStatsTensors,
                                     "batchnorm backward peer_stats tensors"));
    if(!peerStatsTensors.empty())
    {
        attributes.set_peer_stats(std::move(peerStatsTensors));
    }

    // Unpack compute data type
    auto [dt, dtErr] = unpackGraphDataType(opDesc,
                                           HIPDNN_ATTR_BATCHNORM_BACKWARD_COMP_TYPE_EXT,
                                           "batchnorm backward compute data type");
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
