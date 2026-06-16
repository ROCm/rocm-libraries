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

    void checkTensorLayoutsAndDimsSupported() override;

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

// --- Batchnorm Type Configuration ---

// hip-kernel-provider batchnorm requirements (based on underlying kernel constraints):
// - IO tensors: same type (FLOAT, HALF, or BFLOAT16)
// - Affine/Stat/Intermediate tensors: FLOAT only
struct BnTensorTypes
{
    hipdnn_flatbuffers_sdk::data_objects::DataType io;
    hipdnn_flatbuffers_sdk::data_objects::DataType affine;
    hipdnn_flatbuffers_sdk::data_objects::DataType stat;
    hipdnn_flatbuffers_sdk::data_objects::DataType intermediate;
};

namespace bn_type_configs
{
using DT = hipdnn_flatbuffers_sdk::data_objects::DataType;

inline constexpr BnTensorTypes ALL_FLOAT = {DT::FLOAT, DT::FLOAT, DT::FLOAT, DT::FLOAT};
inline constexpr BnTensorTypes HALF_IO = {DT::HALF, DT::FLOAT, DT::FLOAT, DT::FLOAT};
inline constexpr BnTensorTypes BFLOAT16_IO = {DT::BFLOAT16, DT::FLOAT, DT::FLOAT, DT::FLOAT};

inline constexpr std::array<BnTensorTypes, 3> VALID = {ALL_FLOAT, HALF_IO, BFLOAT16_IO};

std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> getAllowedIoTypes();
std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> getAllowedAffineTypes();
std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> getAllowedStatTypes();
std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> getAllowedIntermediateTypes();

} // namespace bn_type_configs

} // namespace hip_kernel_provider
