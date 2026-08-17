// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/BatchnormAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <unordered_map>
#include <vector>

namespace hipdnn_frontend::detail
{

/// Unpacks a batchnorm forward training operation from a backend operation descriptor.
/// Populates the BatchnormAttributes with tensors (using tensorMap for sharing)
/// and compute data type.
[[nodiscard]] inline Error unpackBatchnormOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::BatchnormAttributes& attributes)
{
    // Unpack required tensors
    std::shared_ptr<graph::TensorAttributes> xTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT, tensorMap, xTensor, "batchnorm X tensor"));
    attributes.set_x(xTensor);

    std::shared_ptr<graph::TensorAttributes> scaleTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_BATCHNORM_SCALE_EXT,
                                               tensorMap,
                                               scaleTensor,
                                               "batchnorm Scale tensor"));
    attributes.set_scale(scaleTensor);

    std::shared_ptr<graph::TensorAttributes> biasTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_BATCHNORM_BIAS_EXT,
                                               tensorMap,
                                               biasTensor,
                                               "batchnorm Bias tensor"));
    attributes.set_bias(biasTensor);

    std::shared_ptr<graph::TensorAttributes> epsilonTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_BATCHNORM_EPSILON_EXT,
                                               tensorMap,
                                               epsilonTensor,
                                               "batchnorm Epsilon tensor"));
    attributes.set_epsilon(epsilonTensor);

    std::shared_ptr<graph::TensorAttributes> yTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_BATCHNORM_Y_EXT, tensorMap, yTensor, "batchnorm Y tensor"));
    attributes.set_y(yTensor);

    // Unpack optional tensors
    std::shared_ptr<graph::TensorAttributes> prevRunningMeanTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_MEAN_EXT,
                                            tensorMap,
                                            prevRunningMeanTensor,
                                            "batchnorm PrevRunningMean tensor"));
    if(prevRunningMeanTensor)
    {
        attributes.set_prev_running_mean(prevRunningMeanTensor);
    }

    std::shared_ptr<graph::TensorAttributes> prevRunningVarianceTensor;
    HIPDNN_CHECK_ERROR(
        unpackOptionalTensor(opDesc,
                             HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_VARIANCE_EXT,
                             tensorMap,
                             prevRunningVarianceTensor,
                             "batchnorm PrevRunningVariance tensor"));
    if(prevRunningVarianceTensor)
    {
        attributes.set_prev_running_variance(prevRunningVarianceTensor);
    }

    std::shared_ptr<graph::TensorAttributes> momentumTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_BATCHNORM_MOMENTUM_EXT,
                                            tensorMap,
                                            momentumTensor,
                                            "batchnorm Momentum tensor"));
    if(momentumTensor)
    {
        attributes.set_momentum(momentumTensor);
    }

    std::shared_ptr<graph::TensorAttributes> meanTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_BATCHNORM_MEAN_EXT,
                                            tensorMap,
                                            meanTensor,
                                            "batchnorm Mean tensor"));
    if(meanTensor)
    {
        attributes.set_mean(meanTensor);
    }

    std::shared_ptr<graph::TensorAttributes> invVarianceTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_BATCHNORM_INV_VARIANCE_EXT,
                                            tensorMap,
                                            invVarianceTensor,
                                            "batchnorm InvVariance tensor"));
    if(invVarianceTensor)
    {
        attributes.set_inv_variance(invVarianceTensor);
    }

    std::shared_ptr<graph::TensorAttributes> nextRunningMeanTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_MEAN_EXT,
                                            tensorMap,
                                            nextRunningMeanTensor,
                                            "batchnorm NextRunningMean tensor"));
    if(nextRunningMeanTensor)
    {
        attributes.set_next_running_mean(nextRunningMeanTensor);
    }

    std::shared_ptr<graph::TensorAttributes> nextRunningVarianceTensor;
    HIPDNN_CHECK_ERROR(
        unpackOptionalTensor(opDesc,
                             HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_VARIANCE_EXT,
                             tensorMap,
                             nextRunningVarianceTensor,
                             "batchnorm NextRunningVariance tensor"));
    if(nextRunningVarianceTensor)
    {
        attributes.set_next_running_variance(nextRunningVarianceTensor);
    }

    // Unpack peer_stats tensor array
    std::vector<std::shared_ptr<graph::TensorAttributes>> peerStatsTensors;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensorArray(opDesc,
                                                    HIPDNN_ATTR_OPERATION_BATCHNORM_PEER_STATS_EXT,
                                                    tensorMap,
                                                    peerStatsTensors,
                                                    "batchnorm peer_stats tensors"));
    if(!peerStatsTensors.empty())
    {
        attributes.set_peer_stats(std::move(peerStatsTensors));
    }

    // Unpack compute data type
    auto [dt, dtErr] = unpackGraphDataType(
        opDesc, HIPDNN_ATTR_BATCHNORM_COMP_TYPE_EXT, "batchnorm compute data type");
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
