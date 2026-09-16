// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "engines/hip_mlops_engine/plans/ApplicabilityChecks.hpp"
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_backward_attributes_generated.h>

namespace hip_kernel_provider::rmsnorm
{

class RMSnormValidator : public IValidator
{
private:
    void checkTensorLayoutsAndDimsSupported(const std::vector<int64_t>& tensorIds) override;

    void checkTensorDataTypesSupported(const std::vector<int64_t>& ioTensorIds,
                                       const std::vector<int64_t>& affineTensorIds,
                                       const std::vector<int64_t>& statTensorIds,
                                       const std::vector<int64_t>& intermediateTensorIds);

    void checkTensorShapesSupported(const std::vector<int64_t>& ioTensorIds,
                                    const std::vector<int64_t>& affineTensorIds,
                                    const std::vector<int64_t>& statTensorIds,
                                    const std::vector<int64_t>& intermediateTensorIds);

    static void checkAffineNormalizedShape(const std::vector<int64_t>& affineDims,
                                           const std::vector<int64_t>& ioDims);

    static void checkActivationModeSupported(
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwiseAttr);

public:
    RMSnormValidator(
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMapLocal)
        : IValidator(tensorMapLocal) {};

    // --- High-Level Configuration Validators ---
    void checkFwdTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::RMSNormAttributes& rmsNormFwdAttr);

    void checkFwdActivationTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::RMSNormAttributes& rmsNormFwdAttr,
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwiseAttr);

    void checkBwdTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::RMSNormBackwardAttributes& rmsNormBwdAttr);

    void checkBwdActivationTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwiseAttr,
        const hipdnn_flatbuffers_sdk::data_objects::RMSNormBackwardAttributes& rmsNormBwdAttr);
};

} // namespace hip_kernel_provider::rmsnorm
