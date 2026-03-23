// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_data_sdk/data_objects/layernorm_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>

namespace hip_kernel_provider::layernorm
{

// --- Tensor Descriptor Value Object ---

struct LayernormTensorDescriptor
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    std::vector<int64_t> strideOrder;

    explicit LayernormTensorDescriptor(const hipdnn_data_sdk::data_objects::TensorAttributes* attr);

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

void validateConsistentDimensions(const std::vector<LayernormTensorDescriptor>& tensors);

void validatePackedTensors(const std::vector<LayernormTensorDescriptor>& tensors);

void validateSupportedLayout(const std::vector<int64_t>& strideOrder, size_t numDims);

void validateConsistentLayouts(const std::vector<LayernormTensorDescriptor>& tensors);

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

void validateConsistentDataTypes(
    const std::vector<std::optional<int64_t>>& tensorIds,
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

void validateConsistentShapes(
    const std::vector<std::optional<int64_t>>& tensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    const std::vector<int64_t>& referenceShape,
    const std::string& errorMessage);

void validateNormalizedDim(
    const std::vector<int64_t>& ioTensorIds,
    const std::vector<int64_t>& affineTensorIds,
    const std::vector<std::optional<int64_t>>& statTensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap);

} // namespace validators

// --- Component Validators ---

void checkTensorLayoutsAndDimsSupported(
    const std::vector<int64_t>& tensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap);

void checkTensorLayoutsAndDimsSupported(
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap);

void checkTensorDataTypesSupported(
    const std::vector<int64_t>& ioTensorIds,
    const std::vector<int64_t>& affineTensorIds,
    const std::vector<std::optional<int64_t>>& statTensorIds,
    const std::vector<int64_t>& epsilonTensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap);

void checkTensorShapesSupported(
    const std::vector<int64_t>& ioTensorIds,
    const std::vector<int64_t>& affineTensorIds,
    const std::vector<std::optional<int64_t>>& statTensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap);

// --- High-level Configuration Validators ---

void checkLayernormTensorConfigSupported(
    const hipdnn_data_sdk::data_objects::LayernormAttributes& lnAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap);

// Layernorm Type Configuration ---

// hip-kernel-provider layernorm requirements (based on underlying kernel constraints):
// - IO/Affine/Stat tensors: same type (FLOAT, HALF or BFLOAT16)
// - Epsilon tensors: FLOAT only
struct TensorTypes
{
    hipdnn_data_sdk::data_objects::DataType io;
    hipdnn_data_sdk::data_objects::DataType affine;
    hipdnn_data_sdk::data_objects::DataType stat;
    hipdnn_data_sdk::data_objects::DataType epsilon;
};

namespace type_configs
{
using DT = hipdnn_data_sdk::data_objects::DataType;

inline constexpr TensorTypes FLOAT = {DT::FLOAT, DT::FLOAT, DT::FLOAT, DT::FLOAT};
inline constexpr TensorTypes HALF = {DT::HALF, DT::HALF, DT::HALF, DT::FLOAT};
inline constexpr TensorTypes BFLOAT16 = {DT::BFLOAT16, DT::BFLOAT16, DT::BFLOAT16, DT::FLOAT};

inline constexpr std::array<TensorTypes, 3> VALID = {FLOAT, HALF, BFLOAT16};

std::unordered_set<hipdnn_data_sdk::data_objects::DataType> getAllowedIoTypes();
std::unordered_set<hipdnn_data_sdk::data_objects::DataType> getAllowedAffineTypes();
std::unordered_set<hipdnn_data_sdk::data_objects::DataType> getAllowedStatTypes();
std::unordered_set<hipdnn_data_sdk::data_objects::DataType> getAllowedEpsilonTypes();

} // namespace type_configs

} // namespace hip_kernel_provider::layernorm
