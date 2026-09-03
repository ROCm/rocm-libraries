// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace hip_kernel_provider::resample
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

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
        const auto expected
            = (xDims[i + 2] + prePadding[i] + postPadding[i] - window[i]) / stride[i] + 1;
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
        const auto expectedDy
            = (dxDims[i + 2] + prePadding[i] + postPadding[i] - window[i]) / stride[i] + 1;
        if(expectedDy <= 0 || dyDims[i + 2] != expectedDy)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "ResampleBwd dy spatial dimensions must match the resample parameters.");
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
