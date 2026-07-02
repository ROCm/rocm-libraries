// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "engines/hip_mlops_engine/plans/ApplicabilityChecks.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/resample_fwd_attributes_generated.h>

namespace hip_kernel_provider::resample
{

class ResampleValidator : public IValidator
{
public:
    explicit ResampleValidator(
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMapLocal)
        : IValidator(tensorMapLocal) {};

    void checkTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::ResampleFwdAttributes& resampleAttr);

private:
    void checkTensorLayoutsAndDimsSupported(const std::vector<int64_t>& tensorIds) override;

    void checkTensorDataTypesSupported(
        const hipdnn_flatbuffers_sdk::data_objects::ResampleFwdAttributes& resampleAttr);

    void checkTensorShapesSupported(
        const hipdnn_flatbuffers_sdk::data_objects::ResampleFwdAttributes& resampleAttr);
};

} // namespace hip_kernel_provider::resample
