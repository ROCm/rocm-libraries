// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "engines/plans/ApplicabilityChecks.hpp"
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_backward_attributes_generated.h>

namespace hip_kernel_provider::rmsnorm
{

class RMSnormValidator : public IValidator
{
private:
    void validateSupportedLayout(const std::vector<int64_t>& strideOrder, size_t numDims) override;

    void checkTensorLayoutsAndDimsSupported() override;

    void checkTensorDataTypesSupported(const std::vector<int64_t>& ioTensorIds,
                                       const std::vector<int64_t>& affineTensorIds,
                                       const std::vector<int64_t>& statTensorIds);

    void checkTensorShapesSupported(const std::vector<int64_t>& ioTensorIds,
                                    const std::vector<int64_t>& affineTensorIds,
                                    const std::vector<int64_t>& statTensorIds);

    static void checkAffineNormalizedShape(const std::vector<int64_t>& affineDims,
                                           const std::vector<int64_t>& ioDims);

public:
    RMSnormValidator(
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMapLocal)
        : IValidator(tensorMapLocal) {};

    // --- High-Level Configuration Validators ---
    void checkTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::RMSNormAttributes& rmsNormAttr);

    void checkBwdTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::RMSNormBackwardAttributes& rmsNormBwdAttr);
};

} // namespace hip_kernel_provider::rmsnorm
