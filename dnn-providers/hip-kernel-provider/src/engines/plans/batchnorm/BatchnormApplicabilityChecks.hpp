// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_data_sdk/data_objects/batchnorm_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/batchnorm_inference_attributes_variance_ext_generated.h>
#include <hipdnn_data_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>

namespace hip_kernel_provider::batchnorm
{

// --- Tensor Descriptor Value Object ---

struct BatchnormTensorDescriptor
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    std::vector<int64_t> strideOrder;

    explicit BatchnormTensorDescriptor(const hipdnn_data_sdk::data_objects::TensorAttributes* attr);

    size_t numDims() const
    {
        return dims.size();
    }
    bool isPacked() const;
};

// --- Validation Utilities ---

namespace validators
{
void validateDimensionCount(size_t numDims);

void validateConsistentDimensions(const std::vector<BatchnormTensorDescriptor>& tensors);

void validatePackedTensors(const std::vector<BatchnormTensorDescriptor>& tensors);

void validateSupportedLayout(const std::vector<int64_t>& strideOrder, size_t numDims);

void validateConsistentLayouts(const std::vector<BatchnormTensorDescriptor>& tensors);

void validateDataTypeIsSupported(
    hipdnn_data_sdk::data_objects::DataType dataType,
    const std::unordered_set<hipdnn_data_sdk::data_objects::DataType>& allowedTypes,
    const std::string& errorMessage);

void validateConsistentDataTypes(
    const std::vector<int64_t>& tensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    const std::unordered_set<hipdnn_data_sdk::data_objects::DataType>& allowedTypes,
    const std::string& typeErrorMessage,
    const std::string& consistencyErrorMessage);

void validateFixedDataType(
    const std::vector<int64_t>& tensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    hipdnn_data_sdk::data_objects::DataType expectedType,
    const std::string& errorMessage);

void validateConsistentShapes(
    const std::vector<int64_t>& tensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    const std::vector<int64_t>& referenceShape,
    const std::string& errorMessage);

void validateSpatialDimensions(const std::vector<int64_t>& ioDims);

} // namespace validators

// --- Component Validators ---

void checkTensorLayoutsAndDimsSupported(
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap);

void checkTensorDataTypesSupported(
    const std::vector<int64_t>& ioTensorIds,
    const std::vector<int64_t>& affineTensorIds,
    const std::vector<int64_t>& statTensorIds,
    const std::vector<int64_t>& intermediateTensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap);

void checkTensorShapesSupported(
    const std::vector<int64_t>& ioTensorIds,
    const std::vector<int64_t>& affineTensorIds,
    const std::vector<int64_t>& statTensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    bool isTraining);

// --- High-Level Configuration Validators ---

void checkBatchnormInferenceTensorConfigSupported(
    const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap);

void checkBatchnormInferenceVarianceExtTensorConfigSupported(
    const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributesVarianceExt& bnInfAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap);

void checkBatchnormInferenceActivationTensorConfigSupported(
    const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& actAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap);

void checkBatchnormInferenceVarianceExtActivationTensorConfigSupported(
    const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributesVarianceExt& bnInfAttr,
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& actAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap);

void checkBatchnormFwdActivationModeSupported(
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& activAttr);

void checkBatchnormBwdActivationModeSupported(
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& activAttr);

void checkBatchnormFwdTrainingTensorConfigSupported(
    const hipdnn_data_sdk::data_objects::BatchnormAttributes& bnAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap);

// --- Batchnorm Type Configuration ---

// hip-kernel-provider batchnorm requirements (based on underlying kernel constraints):
// - IO tensors: same type (FLOAT, HALF, or BFLOAT16)
// - Affine/Stat/Intermediate tensors: FLOAT only
struct TensorTypes
{
    hipdnn_data_sdk::data_objects::DataType io;
    hipdnn_data_sdk::data_objects::DataType affine;
    hipdnn_data_sdk::data_objects::DataType stat;
    hipdnn_data_sdk::data_objects::DataType intermediate;
};

namespace type_configs
{
using DT = hipdnn_data_sdk::data_objects::DataType;

inline constexpr TensorTypes ALL_FLOAT = {DT::FLOAT, DT::FLOAT, DT::FLOAT, DT::FLOAT};
inline constexpr TensorTypes HALF_IO = {DT::HALF, DT::FLOAT, DT::FLOAT, DT::FLOAT};
inline constexpr TensorTypes BFLOAT16_IO = {DT::BFLOAT16, DT::FLOAT, DT::FLOAT, DT::FLOAT};

inline constexpr std::array<TensorTypes, 3> VALID = {ALL_FLOAT, HALF_IO, BFLOAT16_IO};

std::unordered_set<hipdnn_data_sdk::data_objects::DataType> getAllowedIoTypes();
std::unordered_set<hipdnn_data_sdk::data_objects::DataType> getAllowedAffineTypes();
std::unordered_set<hipdnn_data_sdk::data_objects::DataType> getAllowedStatTypes();
std::unordered_set<hipdnn_data_sdk::data_objects::DataType> getAllowedIntermediateTypes();

} // namespace type_configs

} // namespace hip_kernel_provider::batchnorm
