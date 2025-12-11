// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Error.hpp"
#include "attributes/TensorAttributes.hpp"
#include "node/Node.hpp"
#include <algorithm>
#include <hipdnn_backend.h>
#include <hipdnn_sdk/logging/CallbackTypes.h>
#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/logging/LoggingUtils.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <numeric>
#include <ranges>
#include <vector>

namespace hipdnn_frontend
{
namespace graph
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

    size_t dims
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

// Utility function to create Tensor_attributes from a Tensor
template <class T,
          class HostAlloc = hipdnn_sdk::utilities::HostAllocator<T>,
          class DeviceAlloc = hipdnn_sdk::utilities::DeviceAllocator<T>>
inline TensorAttributes
    makeTensorAttributes(const std::string& name,
                         DataType dataType,
                         const hipdnn_sdk::utilities::Tensor<T, HostAlloc, DeviceAlloc>& tensor)
{
    return TensorAttributes()
        .set_name(name)
        .set_data_type(dataType)
        .set_dim(tensor.dims())
        .set_stride(tensor.strides());
}

inline TensorAttributes makeTensorAttributes(const std::string& name,
                                             DataType dataType,
                                             const std::vector<int64_t>& dims,
                                             const std::vector<int64_t>& strides)
{
    return TensorAttributes().set_name(name).set_data_type(dataType).set_dim(dims).set_stride(
        strides);
}

inline TensorAttributes makeTensorAttributes(const std::string& name,
                                             const std::vector<int64_t>& dims,
                                             const std::vector<int64_t>& strides)
{
    return TensorAttributes().set_name(name).set_dim(dims).set_stride(strides);
}

inline std::unique_ptr<hipdnn_sdk::utilities::ITensor>
    createTensorFromAttribute(const TensorAttributes& attribute)
{
    return hipdnn_sdk::utilities::createTensor(
        toSdkType(attribute.get_data_type()), attribute.get_dim(), attribute.get_stride());
}

// Determines if batch normalization is in spatial mode based on scale tensor shape
// Following MIOpen's DeriveBNTensorDescriptor convention:
// Spatial mode: scale has shape [1, C, 1, 1, ...] - batch and spatial dims are 1
// Per-activation mode: scale has shape [1, C, H, W, ...] - spatial dims match input
// Note: Scale/bias tensors always use channel-first convention (C at index 1)
inline bool isBatchNormSpatialMode(const std::shared_ptr<TensorAttributes>& scale)
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
// Returns an Error indicating if the input tensor dimensions are valid for batch norm training
inline Error
    validateBatchNormTrainingSpatialDimensions(const std::shared_ptr<TensorAttributes>& x,
                                               const std::shared_ptr<TensorAttributes>& scale,
                                               const std::string& operation
                                               = "Batch normalization training")
{
    if(!x || !scale || x->get_dim().size() < 2)
    {
        return {ErrorCode::OK, ""}; // Skip validation if dimensions not set yet
    }

    const auto& dims = x->get_dim();

    if(isBatchNormSpatialMode(scale))
    {
        // Spatial mode: normalizes over N*spatial_dims per channel
        // Requires N*H*W > 1 (or N*D*H*W > 1 for 3D)

        // dims are always declared in NCHW & NCDHW order
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
}

inline int32_t initializeFrontendLogging(hipdnnCallback_t fn = hipdnnLoggingCallback_ext)
{
    if(fn == nullptr)
    {
        return -1;
    }

    static bool s_loggingInitialized = false;
    static bool s_loggingEnabled = hipdnn_sdk::logging::isLoggingEnabled();

    if(s_loggingInitialized || !s_loggingEnabled)
    {
        return 0;
    }

#ifdef COMPONENT_NAME
    hipdnn::logging::initializeCallbackLogging(COMPONENT_NAME, fn);
#else
    return -1;
#endif

    s_loggingInitialized = true;
    HIPDNN_LOG_INFO("Frontend logging initialized via callback.");

    return 0;
}

#define HIPDNN_FE_LOG_INFO(...)                       \
    do                                                \
    {                                                 \
        hipdnn_frontend::initializeFrontendLogging(); \
        HIPDNN_LOG_INFO(__VA_ARGS__);                 \
    } while(0)

#define HIPDNN_FE_LOG_WARN(...)                       \
    do                                                \
    {                                                 \
        hipdnn_frontend::initializeFrontendLogging(); \
        HIPDNN_LOG_WARN(__VA_ARGS__);                 \
    } while(0)

#define HIPDNN_FE_LOG_ERROR(...)                      \
    do                                                \
    {                                                 \
        hipdnn_frontend::initializeFrontendLogging(); \
        HIPDNN_LOG_ERROR(__VA_ARGS__);                \
    } while(0)

#define HIPDNN_FE_LOG_FATAL(...)                      \
    do                                                \
    {                                                 \
        hipdnn_frontend::initializeFrontendLogging(); \
        HIPDNN_LOG_FATAL(__VA_ARGS__);                \
    } while(0)

}
