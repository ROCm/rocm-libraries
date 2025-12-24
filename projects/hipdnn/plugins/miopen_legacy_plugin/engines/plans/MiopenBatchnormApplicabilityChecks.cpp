// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <numeric>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "MiopenBatchnormApplicabilityChecks.hpp"
#include "MiopenUtils.hpp"

namespace miopen_legacy_plugin
{

// ============================================
// Tensor Descriptor Implementation
// ============================================

BatchnormTensorDescriptor::BatchnormTensorDescriptor(
    const hipdnn_data_sdk::data_objects::TensorAttributes* attr)
    : dims(attr->dims()->begin(), attr->dims()->end())
    , strides(attr->strides()->begin(), attr->strides()->end())
    , strideOrder(hipdnn_data_sdk::utilities::extractStrideOrder(strides))
{
}

bool BatchnormTensorDescriptor::isPacked() const
{
    return hipdnn_data_sdk::utilities::isTensorPacked(dims, strides);
}

// ============================================
// Validation Utilities Implementation
// ============================================

namespace validators
{

// Layout and Dimension Validators

void validateDimensionCount(size_t numDims)
{
    constexpr size_t MIN_SUPPORTED_DIMS = 4;
    constexpr size_t MAX_SUPPORTED_DIMS = 5;

    if(numDims < MIN_SUPPORTED_DIMS || numDims > MAX_SUPPORTED_DIMS)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm implementation supports only 4D or 5D tensors.");
    }
}

void validateConsistentDimensions(const std::vector<BatchnormTensorDescriptor>& tensors)
{
    if(tensors.empty())
    {
        return;
    }

    const size_t expectedDims = tensors[0].numDims();
    validateDimensionCount(expectedDims);

    for(size_t i = 1; i < tensors.size(); ++i)
    {
        if(tensors[i].numDims() != expectedDims)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "All tensors for batchnorm must have the same number of dimensions.");
        }
    }
}

void validatePackedTensors(const std::vector<BatchnormTensorDescriptor>& tensors)
{
    for(const auto& tensor : tensors)
    {
        if(!tensor.isPacked())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm implementation supports only packed tensors.");
        }
    }
}

void validateSupportedLayout(const std::vector<int64_t>& strideOrder, size_t numDims)
{
    if(numDims == 4)
    {
        const auto layoutNchw = hipdnn_data_sdk::utilities::TensorLayout::NCHW;
        const auto layoutNhwc = hipdnn_data_sdk::utilities::TensorLayout::NHWC;

        if(strideOrder != layoutNchw.strideOrder && strideOrder != layoutNhwc.strideOrder)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm implementation supports only NCHW and NHWC layouts for 4D tensors.");
        }
    }
    else // numDims == 5
    {
        const auto layoutNcdhw = hipdnn_data_sdk::utilities::TensorLayout::NCDHW;
        const auto layoutNdhwc = hipdnn_data_sdk::utilities::TensorLayout::NDHWC;

        if(strideOrder != layoutNcdhw.strideOrder && strideOrder != layoutNdhwc.strideOrder)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm implementation supports only NCDHW and NDHWC layouts for 5D tensors.");
        }
    }
}

void validateConsistentLayouts(const std::vector<BatchnormTensorDescriptor>& tensors)
{
    if(tensors.empty())
    {
        return;
    }

    // Helper lambda to check if a tensor has degenerate dimensions (all dims are 1).
    // Degenerate tensors like [1,1,1,1] with strides [1,1,1,1] are layout-agnostic
    // because memory layout is irrelevant when there's only one element.
    auto isDegenerate = [](const BatchnormTensorDescriptor& tensor) {
        return std::all_of(
            tensor.dims.begin(), tensor.dims.end(), [](int64_t d) { return d == 1; });
    };

    // Find the first non-degenerate tensor to use as the layout reference.
    // We cannot use degenerate tensors as reference because their stride order
    // (e.g., [3,2,1,0] from strides [1,1,1,1]) is ambiguous and defaults to NCHW
    // regardless of the actual intended layout.
    // Note: Tensor order is unpredictable (from unordered_map iteration), so we
    // search for ANY non-degenerate tensor - they all must have the same layout anyway.
    size_t referenceIndex = 0;
    for(size_t i = 0; i < tensors.size(); ++i)
    {
        if(!isDegenerate(tensors[i]))
        {
            referenceIndex = i;
            break;
        }
    }

    // If all tensors are degenerate, no layout validation is needed
    if(isDegenerate(tensors[referenceIndex]))
    {
        return;
    }

    const auto& referenceStrideOrder = tensors[referenceIndex].strideOrder;
    const size_t numDims = tensors[referenceIndex].numDims();

    // Validate reference tensor's layout is supported
    validateSupportedLayout(referenceStrideOrder, numDims);

    // Validate all other non-degenerate tensors match the reference layout
    for(size_t i = 0; i < tensors.size(); ++i)
    {
        if(i == referenceIndex)
        {
            continue; // Skip the reference tensor itself
        }

        // Degenerate tensors are layout-agnostic, skip validation
        if(isDegenerate(tensors[i]))
        {
            continue;
        }

        // Non-degenerate tensors must have the same layout as reference
        if(tensors[i].strideOrder != referenceStrideOrder)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "All tensors for batchnorm must have the same layout.");
        }
    }
}

// Data Type Validators

void validateDataTypeIsSupported(
    hipdnn_data_sdk::data_objects::DataType dataType,
    const std::vector<hipdnn_data_sdk::data_objects::DataType>& allowedTypes,
    const std::string& errorMessage)
{
    for(const auto& allowedType : allowedTypes)
    {
        if(dataType == allowedType)
        {
            return;
        }
    }
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM, errorMessage);
}

void validateConsistentDataTypes(
    const std::vector<int64_t>& tensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    const std::vector<hipdnn_data_sdk::data_objects::DataType>& allowedTypes,
    const std::string& typeErrorMessage,
    const std::string& consistencyErrorMessage)
{
    if(tensorIds.empty())
    {
        return;
    }

    const auto& firstTensor = miopen_utils::findTensorAttributes(tensorMap, tensorIds[0]);
    const auto referenceType = firstTensor.data_type();

    validateDataTypeIsSupported(referenceType, allowedTypes, typeErrorMessage);

    for(size_t i = 1; i < tensorIds.size(); ++i)
    {
        const auto& tensor = miopen_utils::findTensorAttributes(tensorMap, tensorIds[i]);
        if(tensor.data_type() != referenceType)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           consistencyErrorMessage);
        }
    }
}

void validateFixedDataType(
    const std::vector<int64_t>& tensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    hipdnn_data_sdk::data_objects::DataType expectedType,
    const std::string& errorMessage)
{
    for(const auto tensorId : tensorIds)
    {
        const auto& tensor = miopen_utils::findTensorAttributes(tensorMap, tensorId);
        if(tensor.data_type() != expectedType)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           errorMessage);
        }
    }
}

// Shape Validators

void validateConsistentShapes(
    const std::vector<int64_t>& tensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    const std::vector<int64_t>& referenceShape,
    const std::string& errorMessage)
{
    for(const auto tensorId : tensorIds)
    {
        const auto& tensorAttr = miopen_utils::findTensorAttributes(tensorMap, tensorId);
        const std::vector<int64_t> dims(tensorAttr.dims()->begin(), tensorAttr.dims()->end());
        if(dims != referenceShape)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           errorMessage);
        }
    }
}

void validateSpatialDimensions(const std::vector<int64_t>& ioDims)
{
    if(ioDims.size() < 3)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "IO tensor must have at least 3 dimensions for batchnorm.");
    }

    const auto spatialSize
        = std::accumulate(ioDims.begin() + 2, ioDims.end(), int64_t{1}, std::multiplies<>());

    if(ioDims[0] * spatialSize <= 1)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "The product of the batch size and spatial dimensions must be greater than 1 for "
            "batchnorm.");
    }
}

} // namespace validators

// ============================================
// Component Validators (Orchestrate Atomic Validators)
// ============================================

void checkTensorLayoutsAndDimsSupported(
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    // Convert Flatbuffer attributes to lightweight descriptors (single access per tensor)
    // Skip pass-by-value tensors (scalars like epsilon, momentum) since:
    // 1. They have no meaningful layout/dimensions to validate
    // 2. Their data type is implicitly validated by MIOpen (always FP32)
    // 3. value_type != NONE indicates a scalar with embedded value
    std::vector<BatchnormTensorDescriptor> tensors;
    tensors.reserve(tensorMap.size());

    for(const auto& [id, attr] : tensorMap)
    {
        // Skip pass-by-value tensors (epsilon, momentum, etc.)
        if(attr->value_type() != hipdnn_data_sdk::data_objects::TensorValue::NONE)
        {
            continue;
        }
        tensors.emplace_back(attr);
    }

    // Run validations in logical order
    validators::validateConsistentDimensions(tensors); // Must all be 4D or all 5D
    validators::validatePackedTensors(tensors); // Must be contiguous
    validators::validateConsistentLayouts(tensors); // Must have same layout (NCHW, NHWC, etc.)
}

void checkTensorDataTypesSupported(
    const std::vector<int64_t>& ioTensorIds,
    const std::vector<int64_t>& affineTensorIds,
    const std::vector<int64_t>& statTensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    using DataType = hipdnn_data_sdk::data_objects::DataType;

    // Validate IO tensors (FLOAT, HALF, or BFLOAT16 - all must match)
    validators::validateConsistentDataTypes(
        ioTensorIds,
        tensorMap,
        {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16},
        "Batchnorm implementation supports only FLOAT, HALF, and BFLOAT16 data types for x, y, "
        "dy, and dx tensors.",
        "All IO tensors for batchnorm must have the same data type.");

    // Validate affine tensors (FLOAT only)
    validators::validateFixedDataType(affineTensorIds,
                                      tensorMap,
                                      DataType::FLOAT,
                                      "Batchnorm implementation supports only FLOAT data type for "
                                      "scale and bias tensors.");

    // Validate stat tensors (FLOAT only)
    validators::validateFixedDataType(statTensorIds,
                                      tensorMap,
                                      DataType::FLOAT,
                                      "Batchnorm implementation supports only FLOAT data type for "
                                      "mean and variance tensors.");
}

void checkTensorShapesSupported(
    const std::vector<int64_t>& ioTensorIds,
    const std::vector<int64_t>& affineTensorIds,
    const std::vector<int64_t>& statTensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    bool isTraining)
{
    if(ioTensorIds.empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "At least one IO tensor must be provided for batchnorm.");
    }

    // Cache first IO tensor's dims
    const auto& ioTensorAttr = miopen_utils::findTensorAttributes(tensorMap, ioTensorIds[0]);
    const std::vector<int64_t> ioDims(ioTensorAttr.dims()->begin(), ioTensorAttr.dims()->end());

    // Validate all IO tensors have same shape
    validators::validateConsistentShapes(
        ioTensorIds, tensorMap, ioDims, "All IO tensors for batchnorm must have the same shape.");

    // Validate derived shapes
    const auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(ioDims);
    validators::validateConsistentShapes(affineTensorIds,
                                         tensorMap,
                                         derivedDims,
                                         "Scale and bias tensors for batchnorm must have shape "
                                         "derived from IO tensor shape.");
    validators::validateConsistentShapes(statTensorIds,
                                         tensorMap,
                                         derivedDims,
                                         "Mean and variance tensors for batchnorm must have shape "
                                         "derived from IO tensor shape.");

    // Training-specific validation
    if(isTraining)
    {
        validators::validateSpatialDimensions(ioDims);
    }
}

// ============================================
// High-Level Configuration Validators
// ============================================

namespace
{

void checkBatchnormTensorConfigSupported(
    const std::vector<int64_t>& ioTensorIds,
    const std::vector<int64_t>& affineTensorIds,
    const std::vector<int64_t>& statTensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    bool isTraining)
{
    checkTensorLayoutsAndDimsSupported(tensorMap);
    checkTensorDataTypesSupported(ioTensorIds, affineTensorIds, statTensorIds, tensorMap);
    checkTensorShapesSupported(ioTensorIds, affineTensorIds, statTensorIds, tensorMap, isTraining);
}

} // namespace

void checkBatchnormTensorConfigSupported(
    const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    std::vector<int64_t> ioTensorIds = {bnInfAttr.x_tensor_uid(), bnInfAttr.y_tensor_uid()};
    std::vector<int64_t> affineTensorIds
        = {bnInfAttr.scale_tensor_uid(), bnInfAttr.bias_tensor_uid()};
    std::vector<int64_t> statTensorIds
        = {bnInfAttr.mean_tensor_uid(), bnInfAttr.inv_variance_tensor_uid()};

    checkBatchnormTensorConfigSupported(
        ioTensorIds, affineTensorIds, statTensorIds, tensorMap, false);
}

void checkBatchnormTensorConfigSupported(
    const hipdnn_data_sdk::data_objects::BatchnormAttributes& bnAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    std::vector<int64_t> ioTensorIds = {bnAttr.x_tensor_uid(), bnAttr.y_tensor_uid()};
    std::vector<int64_t> affineTensorIds = {bnAttr.scale_tensor_uid(), bnAttr.bias_tensor_uid()};
    std::vector<int64_t> statTensorIds;
    if(bnAttr.mean_tensor_uid().has_value())
    {
        statTensorIds.push_back(bnAttr.mean_tensor_uid().value());
    }
    if(bnAttr.inv_variance_tensor_uid().has_value())
    {
        statTensorIds.push_back(bnAttr.inv_variance_tensor_uid().value());
    }

    checkBatchnormTensorConfigSupported(
        ioTensorIds, affineTensorIds, statTensorIds, tensorMap, true);
}

void checkBatchnormTensorConfigSupported(
    const hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes& bnBwdAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    std::vector<int64_t> ioTensorIds
        = {bnBwdAttr.x_tensor_uid(), bnBwdAttr.dy_tensor_uid(), bnBwdAttr.dx_tensor_uid()};
    std::vector<int64_t> affineTensorIds = {
        bnBwdAttr.scale_tensor_uid(), bnBwdAttr.dscale_tensor_uid(), bnBwdAttr.dbias_tensor_uid()};
    std::vector<int64_t> statTensorIds;
    if(bnBwdAttr.mean_tensor_uid().has_value())
    {
        statTensorIds.push_back(bnBwdAttr.mean_tensor_uid().value());
    }
    if(bnBwdAttr.inv_variance_tensor_uid().has_value())
    {
        statTensorIds.push_back(bnBwdAttr.inv_variance_tensor_uid().value());
    }

    checkBatchnormTensorConfigSupported(
        ioTensorIds, affineTensorIds, statTensorIds, tensorMap, true);
}

void checkBatchnormTensorConfigSupported(
    const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& actAttr,
    const hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes& bnBwdAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    std::vector<int64_t> ioTensorIds = {bnBwdAttr.x_tensor_uid(),
                                        actAttr.in_1_tensor_uid().value(), // dy
                                        bnBwdAttr.dx_tensor_uid()};
    std::vector<int64_t> affineTensorIds = {bnBwdAttr.scale_tensor_uid(),
                                            bnBwdAttr.dscale_tensor_uid(),
                                            bnBwdAttr.dbias_tensor_uid(),
                                            bnInfAttr.bias_tensor_uid()};
    std::vector<int64_t> statTensorIds;
    if(bnBwdAttr.mean_tensor_uid().has_value())
    {
        statTensorIds.push_back(bnBwdAttr.mean_tensor_uid().value());
    }
    if(bnBwdAttr.inv_variance_tensor_uid().has_value())
    {
        statTensorIds.push_back(bnBwdAttr.inv_variance_tensor_uid().value());
    }

    checkBatchnormTensorConfigSupported(
        ioTensorIds, affineTensorIds, statTensorIds, tensorMap, true);
}

// ============================================
// Activation Mode Validators
// ============================================

namespace
{

void checkBatchnormActivationModeSupported(
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& activAttr, bool isBwd)
{
    // MIOpen currently only supports miopenActivationPASTHRU, miopenActivationRELU,
    // miopenActivationCLIPPEDRELU and miopenActivationCLAMP for batchnorm fusions

    if(activAttr.operation() == hipdnn_data_sdk::data_objects::PointwiseMode::IDENTITY)
    {
        // miopenActivationPASTHRU
        return;
    }

    if(activAttr.operation()
       == (isBwd ? hipdnn_data_sdk::data_objects::PointwiseMode::RELU_BWD
                 : hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD))
    {
        // miopenActivationRELU - Standard ReLU (no parameters)
        // miopenActivationCLIPPEDRELU - Clipped ReLU (relu_upper_clip only)
        // miopenActivationCLAMP - CLAMP (relu_lower_clip + relu_upper_clip)
        // miopenActivationLEAKYRELU - Leaky ReLU is not supported!
        if(!activAttr.relu_lower_clip_slope())
        {
            return;
        }
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm fused activation does not support Leaky ReLU.");
    }

    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Unsupported activation mode for batchnorm fusion.");
}

} // namespace

void checkBatchnormFwdActivationModeSupported(
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& activAttr)
{
    checkBatchnormActivationModeSupported(activAttr, false);
}

void checkBatchnormBwdActivationModeSupported(
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& activAttr)
{
    checkBatchnormActivationModeSupported(activAttr, true);
}

} // namespace miopen_legacy_plugin
