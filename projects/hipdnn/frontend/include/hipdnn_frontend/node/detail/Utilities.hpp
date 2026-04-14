// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <algorithm>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <vector>

namespace hipdnn_frontend::detail
{

// Find common shape from inputs.
// Takes the max in each dim, and if any dim is not 1, or equal, then it's incompatible.
// For example:
// input_shapes = {{1, 2}, {1, 2}, {1, 2, 5}} -> common_shape = {1, 2, 5}
// input_shapes = {{1, 2, 3}, {1, 2, 4}, {1, 2}} -> error
inline Error findCommonShape(const std::vector<std::vector<int64_t>>& inputShapes,
                             std::vector<int64_t>& commonShape)
{
    if(inputShapes.empty())
    {
        return {ErrorCode::INVALID_VALUE, "Input shapes cannot be empty"};
    }

    const size_t dims
        = std::max_element(inputShapes.begin(),
                           inputShapes.end(),
                           [](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
                               return a.size() < b.size();
                           })
              ->size();

    commonShape.resize(dims, 1);

    for(auto& current : inputShapes)
    {
        for(size_t j = current.size(); j-- > 0;)
        {
            if(commonShape[j] != current[j] && commonShape[j] != 1 && current[j] != 1)
            {
                return {ErrorCode::INVALID_VALUE, "Incompatible shapes"};
            }

            commonShape[j] = std::max(commonShape[j], current[j]);
        }
    }

    return {};
}

// Helper to check if tensor dimensions are set (not null, has dimensions)
// Returns true if dimensions are set by user, false if they will be inferred in infer_properties_node()
inline bool areTensorDimensionsSet(const std::shared_ptr<graph::TensorAttributes>& tensor)
{
    return tensor && !tensor->get_dim().empty();
}

// Helper to get tensor name for error messages (uses tensor's name if set, otherwise fallback)
inline std::string getTensorNameForError(const std::shared_ptr<graph::TensorAttributes>& tensor,
                                         const std::string& fallbackName)
{
    if(tensor && !tensor->get_name().empty())
    {
        return tensor->get_name();
    }
    return fallbackName;
}

// Determines if batch normalization is in spatial mode based on scale tensor shape
// Following MIOpen's DeriveBNTensorDescriptor convention:
// Spatial mode: scale has shape [1, C, 1, 1, ...] - batch and spatial dims are 1
// Per-activation mode: scale has shape [1, C, H, W, ...] - spatial dims match input
// Note: Scale/bias tensors always use channel-first convention (C at index 1)
inline bool isBatchNormSpatialMode(const std::shared_ptr<graph::TensorAttributes>& scale)
{
    if(!scale || scale->get_dim().empty() || scale->get_dim().size() < 2)
    {
        return true; // Default to spatial if not fully initialized
    }

    const auto& scaleDims = scale->get_dim();

    // Check if all spatial dimensions (indices 2+) are 1
    for(size_t i = 2; i < scaleDims.size(); ++i)
    {
        if(scaleDims[i] != 1)
        {
            return false; // per-activation mode
        }
    }

    return true; // spatial mode
}

// Validates batch normalization training spatial dimension constraints
// Uses tensor names if set, otherwise uses fallback names for error messages
// Returns an Error indicating if the input tensor dimensions are valid for batch norm training
inline Error validateBatchNormTrainingSpatialDimensions(
    const std::shared_ptr<graph::TensorAttributes>& x,
    const std::shared_ptr<graph::TensorAttributes>& scale,
    const std::string& operation = "Batch normalization training",
    const std::string& xFallback = "Input tensor",
    const std::string& scaleFallback = "Scale tensor")
{
    if(!x)
    {
        return {ErrorCode::ATTRIBUTE_NOT_SET, getTensorNameForError(x, xFallback) + " is not set"};
    }

    if(!scale)
    {
        return {ErrorCode::ATTRIBUTE_NOT_SET,
                getTensorNameForError(scale, scaleFallback) + " is not set"};
    }

    const auto& dims = x->get_dim();

    if(dims.size() < 2)
    {
        return {ErrorCode::INVALID_VALUE,
                getTensorNameForError(x, xFallback) + " must have at least 2 dimensions, but has "
                    + std::to_string(dims.size())};
    }

    if(isBatchNormSpatialMode(scale))
    {
        // Spatial mode: normalizes over N*spatial_dims per channel
        // Requires N*H*W > 1 (or N*D*H*W > 1 for 3D)

        // Batchnorm dims follow NCHW & NCDHW order
        int64_t spatialElements = dims[0]; // Start with N
        for(size_t i = 2; i < dims.size(); ++i)
        {
            spatialElements *= dims[i]; // Multiply by spatial dimensions
        }

        if(spatialElements <= 1)
        {
            return {ErrorCode::INVALID_VALUE,
                    operation
                        + " (spatial mode) requires more than 1 value per channel. "
                          "N * spatial_dimensions must be > 1. Got N="
                        + std::to_string(dims[0])};
        }
    }
    else
    {
        // TODO: Add per-activation mode support (validate N > 1)
        return {ErrorCode::INVALID_VALUE,
                "Batch normalization per-activation mode is not currently supported. "
                "Use spatial mode by ensuring scale/bias tensors have shape [1, C, 1, 1, ...]"};
    }

    return {ErrorCode::OK, ""};
}

// Validates tensor has minimum required dimensions
// Uses tensor's name if set, otherwise uses fallbackName for error messages
inline Error validateMinimumTensorDimensions(const std::shared_ptr<graph::TensorAttributes>& tensor,
                                             size_t minDims,
                                             const std::string& fallbackName = "Tensor")
{
    if(!tensor)
    {
        return {ErrorCode::ATTRIBUTE_NOT_SET,
                getTensorNameForError(tensor, fallbackName) + " is not set"};
    }

    const auto& dims = tensor->get_dim();

    HIPDNN_RETURN_IF_LT(dims.size(),
                        minDims,
                        ErrorCode::INVALID_VALUE,
                        getTensorNameForError(tensor, fallbackName) + " must have at least "
                            + std::to_string(minDims) + " dimensions, but has "
                            + std::to_string(dims.size()));

    return {ErrorCode::OK, ""};
}

// Validates two tensors have matching shapes
// Uses tensor names if set, otherwise uses fallback names for error messages
// NOTE: This function expects both tensors to have dimensions set - it will fail if not set
inline Error validateTensorShapesMatch(const std::shared_ptr<graph::TensorAttributes>& tensor1,
                                       const std::shared_ptr<graph::TensorAttributes>& tensor2,
                                       const std::string& fallbackName1 = "Tensor1",
                                       const std::string& fallbackName2 = "Tensor2")
{
    if(!tensor1)
    {
        return {ErrorCode::ATTRIBUTE_NOT_SET,
                getTensorNameForError(tensor1, fallbackName1) + " is not set"};
    }

    if(!tensor2)
    {
        return {ErrorCode::ATTRIBUTE_NOT_SET,
                getTensorNameForError(tensor2, fallbackName2) + " is not set"};
    }

    const auto& dims1 = tensor1->get_dim();
    const auto& dims2 = tensor2->get_dim();

    if(dims1.size() != dims2.size())
    {
        return {ErrorCode::INVALID_VALUE,
                getTensorNameForError(tensor1, fallbackName1) + " and "
                    + getTensorNameForError(tensor2, fallbackName2)
                    + " must have the same number of dimensions: " + std::to_string(dims1.size())
                    + " vs " + std::to_string(dims2.size())};
    }

    // Find first mismatch location (or end if all match)
    auto [it1, it2] = std::mismatch(dims1.begin(), dims1.end(), dims2.begin());

    if(it1 != dims1.end())
    {
        auto index = static_cast<size_t>(std::distance(dims1.begin(), it1));
        return {ErrorCode::INVALID_VALUE,
                getTensorNameForError(tensor1, fallbackName1) + " and "
                    + getTensorNameForError(tensor2, fallbackName2)
                    + " dimension mismatch at index " + std::to_string(index) + ": "
                    + std::to_string(*it1) + " vs " + std::to_string(*it2)};
    }

    return {ErrorCode::OK, ""};
}

// Validates two tensors have matching shapes (only validates if second tensor has dimensions set)
// Uses tensor names if set, otherwise uses fallback names for error messages
// Returns OK if tensor2 dimensions not yet set (will be inferred in infer_properties_node)
// Use this for validating input vs output tensor consistency when output may not be set yet
inline Error validateTensorShapesMatchIfSet(const std::shared_ptr<graph::TensorAttributes>& tensor1,
                                            const std::shared_ptr<graph::TensorAttributes>& tensor2,
                                            const std::string& fallbackName1 = "Tensor1",
                                            const std::string& fallbackName2 = "Tensor2")
{
    if(!areTensorDimensionsSet(tensor2))
    {
        return {ErrorCode::OK, ""}; // tensor2 dimensions not set yet, will be inferred
    }

    return validateTensorShapesMatch(tensor1, tensor2, fallbackName1, fallbackName2);
}

// Validates tensor has channel-only shape [1, C, 1, 1, ...] for batch normalization parameters
// Uses tensor's name if set, otherwise uses fallbackName for error messages
// NOTE: This function expects tensor dimensions to be set - it will fail if not set
inline Error validateChannelOnlyTensorShape(const std::shared_ptr<graph::TensorAttributes>& tensor,
                                            int64_t expectedChannels,
                                            const std::string& fallbackName = "Tensor")
{
    if(!tensor)
    {
        return {ErrorCode::ATTRIBUTE_NOT_SET,
                getTensorNameForError(tensor, fallbackName) + " is not set"};
    }

    const auto& dims = tensor->get_dim();

    HIPDNN_RETURN_IF_LT(dims.size(),
                        2,
                        ErrorCode::INVALID_VALUE,
                        getTensorNameForError(tensor, fallbackName)
                            + " must have at least 2 dimensions for batch normalization");

    // Check batch dimension is 1
    HIPDNN_RETURN_IF_NE(dims[0],
                        1,
                        ErrorCode::INVALID_VALUE,
                        getTensorNameForError(tensor, fallbackName)
                            + " batch dimension (index 0) must be 1, got "
                            + std::to_string(dims[0]));

    // Check channel dimension matches expected
    HIPDNN_RETURN_IF_NE(
        dims[1],
        expectedChannels,
        ErrorCode::INVALID_VALUE,
        getTensorNameForError(tensor, fallbackName) + " channel dimension (index 1) must be "
            + std::to_string(expectedChannels) + ", got " + std::to_string(dims[1]));

    // Check all spatial dimensions are 1
    for(size_t i = 2; i < dims.size(); ++i)
    {
        HIPDNN_RETURN_IF_NE(dims[i],
                            1,
                            ErrorCode::INVALID_VALUE,
                            getTensorNameForError(tensor, fallbackName)
                                + " spatial dimension at index " + std::to_string(i)
                                + " must be 1 for spatial batch normalization, got "
                                + std::to_string(dims[i]));
    }

    return {ErrorCode::OK, ""};
}

// Validates tensor has non-batch [1, C, D, H, ...] for rms norm parmeters
// Uses tensor's name if set, otherwise uses fallbackName for error messages
// NOTE: This function expects tensor dimensions to be set - it will fail if not set
inline Error validateNonBatchShapeMatch(const std::shared_ptr<graph::TensorAttributes>& scale,
                                        const std::shared_ptr<graph::TensorAttributes>& input,
                                        const std::string& fallbackName = "Tensor")
{
    if(!scale)
    {
        return {ErrorCode::ATTRIBUTE_NOT_SET,
                getTensorNameForError(scale, fallbackName) + " is not set"};
    }

    if(!input)
    {
        return {ErrorCode::ATTRIBUTE_NOT_SET,
                getTensorNameForError(input, fallbackName) + " is not set"};
    }

    const auto& scaleDims = scale->get_dim();
    const auto& inputDims = input->get_dim();

    HIPDNN_RETURN_IF_LT(scaleDims.size(),
                        inputDims.size(),
                        ErrorCode::INVALID_VALUE,
                        getTensorNameForError(scale, fallbackName) + " must match input rank");

    // Check batch dimension is 1
    HIPDNN_RETURN_IF_NE(scaleDims[0],
                        1,
                        ErrorCode::INVALID_VALUE,
                        getTensorNameForError(scale, fallbackName)
                            + " batch dimension (index 0) must be 1, got "
                            + std::to_string(scaleDims[0]));

    for(size_t i = 1, s = scaleDims.size(); i < s; ++i)
    {
        if(scaleDims[i] != 1)
        {
            // Check channel dimension matches expected
            HIPDNN_RETURN_IF_NE(scaleDims[i],
                                inputDims[i],
                                ErrorCode::INVALID_VALUE,
                                getTensorNameForError(scale, fallbackName) + " dimension at index "
                                    + std::to_string(i) + " must match input dimension at index "
                                    + std::to_string(i) + ", got " + std::to_string(scaleDims[i])
                                    + " vs " + std::to_string(inputDims[i]));
        }
    }

    return {ErrorCode::OK, ""};
}

// Validates channel-only shape for optional tensors (only validates if dimensions are set)
// Uses tensor's name if set, otherwise uses fallbackName for error messages
// Returns OK if tensor dimensions not yet set (will be inferred in infer_properties_node)
// Use this for optional output tensors that may not have dimensions set during pre_validate_node
inline Error validateChannelOnlyShapeIfSet(const std::shared_ptr<graph::TensorAttributes>& tensor,
                                           int64_t expectedChannels,
                                           const std::string& fallbackName = "Tensor")
{
    if(!areTensorDimensionsSet(tensor))
    {
        return {ErrorCode::OK, ""}; // Dimensions not set yet, will be inferred
    }

    // Dimensions are set, validate strictly
    return validateChannelOnlyTensorShape(tensor, expectedChannels, fallbackName);
}

// Validates normalization statistics tensor shape (e.g., inv_rms) against input and scale tensors.
// Where scale has a non-1 dim (normalized axis), stats must be 1;
// where scale has dim 1 (non-normalized axis), stats must match input.
// For typical channel-norm with scale [1,C,1,1], this yields stats shape [N,1,H,W].
// Only validates if tensor dimensions are already set (same pattern as validateChannelOnlyShapeIfSet)
inline Error validateNormStatsShapeIfSet(const std::shared_ptr<graph::TensorAttributes>& tensor,
                                         const std::shared_ptr<graph::TensorAttributes>& input,
                                         const std::shared_ptr<graph::TensorAttributes>& scale,
                                         const std::string& fallbackName = "Tensor")
{
    if(!areTensorDimensionsSet(tensor))
    {
        return {ErrorCode::OK, ""}; // Dimensions not set yet, will be inferred
    }

    if(!input)
    {
        return {ErrorCode::ATTRIBUTE_NOT_SET, "Input tensor is not set"};
    }

    if(!scale || scale->get_dim().empty())
    {
        return {ErrorCode::ATTRIBUTE_NOT_SET, "Scale tensor dimensions are not set"};
    }

    const auto& dims = tensor->get_dim();
    const auto& inputDims = input->get_dim();
    const auto& scaleDims = scale->get_dim();

    HIPDNN_RETURN_IF_NE(
        dims.size(),
        inputDims.size(),
        ErrorCode::INVALID_VALUE,
        getTensorNameForError(tensor, fallbackName) + " must have the same rank as input, expected "
            + std::to_string(inputDims.size()) + " but got " + std::to_string(dims.size()));

    for(size_t i = 0; i < dims.size(); ++i)
    {
        if(scaleDims[i] != 1)
        {
            // Normalized axis: stats dim must be 1
            HIPDNN_RETURN_IF_NE(dims[i],
                                1,
                                ErrorCode::INVALID_VALUE,
                                getTensorNameForError(tensor, fallbackName) + " dimension at index "
                                    + std::to_string(i)
                                    + " must be 1 (normalized axis, scale is non-1), got "
                                    + std::to_string(dims[i]));
        }
        else
        {
            // Non-normalized axis: stats dim must match input
            HIPDNN_RETURN_IF_NE(dims[i],
                                inputDims[i],
                                ErrorCode::INVALID_VALUE,
                                getTensorNameForError(tensor, fallbackName) + " dimension at index "
                                    + std::to_string(i) + " must match input ("
                                    + std::to_string(inputDims[i]) + "), got "
                                    + std::to_string(dims[i]));
        }
    }

    return {ErrorCode::OK, ""};
}

// Validates scalar parameter tensor is properly configured
// Uses param's name if set, otherwise uses fallbackName for error messages
// Used for required scalar parameters (e.g., epsilon) that must have dimensions set
inline Error validateScalarParameter(const std::shared_ptr<graph::TensorAttributes>& param,
                                     const std::string& fallbackName = "Parameter")
{
    if(!param)
    {
        return {ErrorCode::ATTRIBUTE_NOT_SET,
                getTensorNameForError(param, fallbackName) + " parameter is not set"};
    }

    const auto& dims = param->get_dim();
    if(dims.empty())
    {
        return {ErrorCode::ATTRIBUTE_NOT_SET,
                getTensorNameForError(param, fallbackName) + " dimensions are not set"};
    }

    // Scalar parameters should be single-element tensors
    // Typically [1] or [1, 1, 1, 1] depending on how they're created
    int64_t totalElements = 1;
    for(auto dim : dims)
    {
        totalElements *= dim;
    }

    HIPDNN_RETURN_IF_NE(totalElements,
                        1,
                        ErrorCode::INVALID_VALUE,
                        getTensorNameForError(param, fallbackName)
                            + " must be a scalar (single element), but has "
                            + std::to_string(totalElements) + " elements");

    HIPDNN_RETURN_IF_FALSE(param->get_pass_by_value(),
                           ErrorCode::INVALID_VALUE,
                           getTensorNameForError(param, fallbackName)
                               + " must be a pass-by-value tensor");

    // Note: We can't validate the actual value (e.g., epsilon > 0) at pre-validation time
    // since the data isn't available yet. This validation is structural only.

    return {ErrorCode::OK, ""};
}

} // namespace hipdnn_frontend::detail
