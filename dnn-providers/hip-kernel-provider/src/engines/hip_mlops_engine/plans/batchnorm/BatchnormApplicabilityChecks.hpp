// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "engines/hip_mlops_engine/plans/ApplicabilityChecks.hpp"
#include <array>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_variance_ext_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>

namespace hip_kernel_provider
{

class BatchnormValidator : public IValidator
{
    // --- Activation Mode Validators ---
private:
    static void checkFwdActivationModeSupported(
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& activAttr);

    static void checkBwdActivationModeSupported(
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& activAttr);

    // --- Validation Utilities ---

    static void validateSpatialDimensions(const std::vector<int64_t>& ioDims);

    // --- Component Validators ---

    void checkTensorLayoutsAndDimsSupported(const std::vector<int64_t>& tensorIds) override;

    void checkTensorDataTypesSupported(const std::vector<int64_t>& ioTensorIds,
                                       const std::vector<int64_t>& affineTensorIds,
                                       const std::vector<int64_t>& statTensorIds,
                                       const std::vector<int64_t>& intermediateTensorIds);

    void checkTensorShapesSupported(const std::vector<int64_t>& ioTensorIds,
                                    const std::vector<int64_t>& affineTensorIds,
                                    const std::vector<int64_t>& statTensorIds,
                                    bool isTraining);

    void checkTensorConfigSupported(const std::vector<int64_t>& ioTensorIds,
                                    const std::vector<int64_t>& affineTensorIds,
                                    const std::vector<int64_t>& statTensorIds,
                                    const std::vector<int64_t>& intermediateTensorIds,
                                    bool isTraining);

public:
    BatchnormValidator(
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMapLocal)
        : IValidator(tensorMapLocal) {};

    // --- High-Level Configuration Validators ---

    void checkInferenceTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr);

    void checkInferenceVarianceExtTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt&
            bnInfAttr);

    void checkInferenceActivationTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr);

    void checkInferenceVarianceExtActivationTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt&
            bnInfAttr,
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr);

    void checkFwdTrainingTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& bnAttr);

    void checkFwdTrainingActivationTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& bnAttr,
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr);

    void checkBwdTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormBackwardAttributes& bnBwdAttr);

    void checkBwdActivationTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr,
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormBackwardAttributes& bnBwdAttr);
};

} // namespace hip_kernel_provider
