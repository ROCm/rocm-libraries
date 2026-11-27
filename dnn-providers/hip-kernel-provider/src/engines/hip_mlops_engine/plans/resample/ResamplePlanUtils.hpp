// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "compilation/KernelCompileOptions.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace hip_kernel_provider::resample
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// The performance choice of a resample launch: its block size. The kernels index by
/// blockDim.x and return past the last element, so any block HIP launches runs them. The
/// default is the hand-written plans' configuration.
struct ResampleConfig
{
    unsigned int blockSize = 256;
};

inline bool operator==(const ResampleConfig& lhs, const ResampleConfig& rhs)
{
    return lhs.blockSize == rhs.blockSize;
}

inline bool operator!=(const ResampleConfig& lhs, const ResampleConfig& rhs)
{
    return !(lhs == rhs);
}

/// Whether @p config's block is one HIP launches: 1 to 1024 threads.
inline bool isBlockSizeSupported(const ResampleConfig& config)
{
    return config.blockSize >= 1 && config.blockSize <= 1024;
}

/// What a resample launch compiles and how wide it runs: the compile options that bake the
/// graph into the kernel, and the grid of blockSize-thread blocks covering one thread per
/// output element. Built by buildResampleFwdKernel()/buildResampleBwdKernel(), which the HIP
/// MLOps engine's plans and the kernel ingestor's HipMlops packs both call, so the two engines
/// compile the same program for the same graph.
struct ResampleKernelBuild
{
    compilation::KernelCompileOptions options;
    unsigned int blockSize;
    unsigned int gridSize;
};

inline std::vector<int64_t> toStdVector(const flatbuffers::Vector<int64_t>* values)
{
    return hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(values);
}

inline std::vector<int64_t> tensorDims(const data_objects::TensorAttributes& tensor)
{
    return toStdVector(tensor.dims());
}

inline int64_t dimAt(const data_objects::TensorAttributes* tensor, size_t index)
{
    return tensor->dims()->Get(static_cast<flatbuffers::uoffset_t>(index));
}

inline int64_t strideAt(const data_objects::TensorAttributes* tensor, size_t index)
{
    return tensor->strides()->Get(static_cast<flatbuffers::uoffset_t>(index));
}

inline const char* getIndexTypeString(const data_objects::TensorAttributes* index)
{
    if(index == nullptr || index->data_type() == data_objects::DataType::INT32)
    {
        return "int32_t";
    }

    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Resample index tensor must have INT32 data type.");
}

inline void validateSpatialVector(const std::vector<int64_t>& values,
                                  size_t spatialDims,
                                  const std::string& operationName,
                                  const std::string& parameterName,
                                  bool allowZero)
{
    if(values.size() != spatialDims)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            operationName + " " + parameterName
                + " rank must match the number of spatial dimensions.");
    }

    for(const auto value : values)
    {
        if(allowZero ? value < 0 : value <= 0)
        {
            std::string message = operationName;
            message.append(" ")
                .append(parameterName)
                .append(" values must be ")
                .append(allowZero ? "non-negative." : "positive.");
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM, message);
        }
    }
}

inline void validateResampleSpatialParameters(const std::vector<int64_t>& prePadding,
                                              const std::vector<int64_t>& postPadding,
                                              const std::vector<int64_t>& stride,
                                              const std::vector<int64_t>& window,
                                              size_t spatialDims,
                                              const std::string& operationName)
{
    validateSpatialVector(prePadding, spatialDims, operationName, "pre_padding", true);
    validateSpatialVector(postPadding, spatialDims, operationName, "post_padding", true);
    validateSpatialVector(stride, spatialDims, operationName, "stride", false);
    validateSpatialVector(window, spatialDims, operationName, "window", false);

    // The kernels form the window product in int64_t.
    int64_t windowProduct = 1;
    for(const auto value : window)
    {
        if(__builtin_mul_overflow(windowProduct, value, &windowProduct))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                operationName + " window product does not fit in int64_t.");
        }
    }
}

// The output extent along one spatial dimension. The padded input extent must fit in int64_t:
// the kernels evaluate out * stride + k - pre_padding and in + pre_padding - k, both bounded by
// it, in int64_t.
inline int64_t resampleOutputExtent(int64_t input,
                                    int64_t prePadding,
                                    int64_t postPadding,
                                    int64_t stride,
                                    int64_t window,
                                    const std::string& operationName)
{
    int64_t padded = 0;
    if(__builtin_add_overflow(input, prePadding, &padded)
       || __builtin_add_overflow(padded, postPadding, &padded))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            operationName + " padded spatial extent does not fit in int64_t.");
    }
    return (padded - window) / stride + 1;
}

inline void validateResampleTensorRanks(const std::vector<int64_t>& inputDims,
                                        const std::vector<int64_t>& outputDims,
                                        const std::string& operationName)
{
    if(inputDims.size() != outputDims.size())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            operationName + " requires input and output tensors to have the same rank.");
    }

    if(inputDims.size() < 4 || inputDims.size() > 5)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, operationName + " supports 4D and 5D tensors.");
    }
}

inline void validateResampleOutputShape(const std::vector<int64_t>& xDims,
                                        const std::vector<int64_t>& yDims,
                                        const std::vector<int64_t>& prePadding,
                                        const std::vector<int64_t>& postPadding,
                                        const std::vector<int64_t>& stride,
                                        const std::vector<int64_t>& window)
{
    validateResampleTensorRanks(xDims, yDims, "ResampleFwd");

    const auto spatialDims = xDims.size() - 2;
    validateResampleSpatialParameters(
        prePadding, postPadding, stride, window, spatialDims, "ResampleFwd");

    if(xDims[0] != yDims[0] || xDims[1] != yDims[1])
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "ResampleFwd preserves the batch and channel dimensions.");
    }

    for(size_t i = 0; i < spatialDims; ++i)
    {
        const auto expected = resampleOutputExtent(
            xDims[i + 2], prePadding[i], postPadding[i], stride[i], window[i], "ResampleFwd");
        if(expected <= 0 || yDims[i + 2] != expected)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "ResampleFwd y spatial dimensions must match the resample parameters.");
        }
    }
}

inline void validateResampleBwdOutputShape(const std::vector<int64_t>& dyDims,
                                           const std::vector<int64_t>& dxDims,
                                           const std::vector<int64_t>& prePadding,
                                           const std::vector<int64_t>& postPadding,
                                           const std::vector<int64_t>& stride,
                                           const std::vector<int64_t>& window)
{
    validateResampleTensorRanks(dyDims, dxDims, "ResampleBwd");

    const auto spatialDims = dyDims.size() - 2;
    validateResampleSpatialParameters(
        prePadding, postPadding, stride, window, spatialDims, "ResampleBwd");

    if(dyDims[0] != dxDims[0] || dyDims[1] != dxDims[1])
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "ResampleBwd preserves the batch and channel dimensions.");
    }

    for(size_t i = 0; i < spatialDims; ++i)
    {
        const auto expectedDy = resampleOutputExtent(
            dxDims[i + 2], prePadding[i], postPadding[i], stride[i], window[i], "ResampleBwd");
        if(expectedDy <= 0 || dyDims[i + 2] != expectedDy)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "ResampleBwd dx and dy spatial dimensions must be compatible with the resample "
                "parameters.");
        }
    }
}

inline void validateResampleIndexShape(const data_objects::TensorAttributes& indexTensor,
                                       const std::vector<int64_t>& yDims,
                                       const std::string& operationName)
{
    if(tensorDims(indexTensor) != yDims)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            operationName + " index tensor must have the same shape as y.");
    }
}

} // namespace hip_kernel_provider::resample
