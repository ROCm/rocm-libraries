// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_sdk/data_objects/batchnorm_attributes_generated.h>
#include <hipdnn_sdk/data_objects/batchnorm_backward_attributes_generated.h>
#include <hipdnn_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>

namespace miopen_legacy_plugin
{

// ============================================================================
// Tensor Descriptor Value Object
// ============================================================================

struct BatchnormTensorDescriptor
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    std::vector<int64_t> strideOrder;

    explicit BatchnormTensorDescriptor(const hipdnn_sdk::data_objects::TensorAttributes* attr);

    size_t numDims() const
    {
        return dims.size();
    }
    bool isPacked() const;
};

// ============================================================================
// Validation Utilities Namespace
// ============================================================================

namespace validators
{

// Layout and Dimension Validators
void validateDimensionCount(size_t numDims);

void validateConsistentDimensions(const std::vector<BatchnormTensorDescriptor>& tensors);

void validatePackedTensors(const std::vector<BatchnormTensorDescriptor>& tensors);

void validateSupportedLayout(const std::vector<int64_t>& strideOrder, size_t numDims);

void validateConsistentLayouts(const std::vector<BatchnormTensorDescriptor>& tensors);

// Data Type Validators
void validateDataTypeIsSupported(
    hipdnn_sdk::data_objects::DataType dataType,
    const std::vector<hipdnn_sdk::data_objects::DataType>& allowedTypes,
    const std::string& errorMessage);

void validateConsistentDataTypes(
    const std::vector<int64_t>& tensorIds,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap,
    const std::vector<hipdnn_sdk::data_objects::DataType>& allowedTypes,
    const std::string& typeErrorMessage,
    const std::string& consistencyErrorMessage);

void validateFixedDataType(
    const std::vector<int64_t>& tensorIds,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap,
    hipdnn_sdk::data_objects::DataType expectedType,
    const std::string& errorMessage);

// Shape Validators
void validateConsistentShapes(
    const std::vector<int64_t>& tensorIds,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap,
    const std::vector<int64_t>& referenceShape,
    const std::string& errorMessage);

void validateSpatialDimensions(const std::vector<int64_t>& ioDims);

} // namespace validators

// ============================================================================
// Component Validators (Orchestrate Atomic Validators)
// ============================================================================

void checkTensorLayoutsAndDimsSupported(
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
        tensorMap);

void checkTensorDataTypesSupported(
    const std::vector<int64_t>& ioTensorIds,
    const std::vector<int64_t>& affineTensorIds,
    const std::vector<int64_t>& statTensorIds,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
        tensorMap);

void checkTensorShapesSupported(
    const std::vector<int64_t>& ioTensorIds,
    const std::vector<int64_t>& affineTensorIds,
    const std::vector<int64_t>& statTensorIds,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap,
    bool isTraining);

// ============================================================================
// High-Level Configuration Validators
// ============================================================================

void checkBatchnormTensorConfigSupported(
    const hipdnn_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
        tensorMap);

void checkBatchnormTensorConfigSupported(
    const hipdnn_sdk::data_objects::BatchnormAttributes& bnAttr,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
        tensorMap);

void checkBatchnormTensorConfigSupported(
    const hipdnn_sdk::data_objects::BatchnormBackwardAttributes& bnBwdAttr,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
        tensorMap);

void checkBatchnormTensorConfigSupported(
    const hipdnn_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const hipdnn_sdk::data_objects::PointwiseAttributes& actAttr,
    const hipdnn_sdk::data_objects::BatchnormBackwardAttributes& bnBwdAttr,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
        tensorMap);

void checkBatchnormFwdActivationModeSupported(
    const hipdnn_sdk::data_objects::PointwiseAttributes& activAttr);

void checkBatchnormBwdActivationModeSupported(
    const hipdnn_sdk::data_objects::PointwiseAttributes& activAttr);

} // namespace miopen_legacy_plugin
