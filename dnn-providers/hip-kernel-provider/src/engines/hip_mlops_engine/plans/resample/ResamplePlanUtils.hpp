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

template <typename T>
std::vector<T> toStdVector(const flatbuffers::Vector<T>* vector)
{
    return hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(vector);
}

inline std::vector<int64_t>
    tensorDims(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tensor)
{
    return toStdVector(tensor.dims());
}

inline void validateSpatialVector(const std::vector<int64_t>& values,
                                  size_t spatialDims,
                                  const std::string& name,
                                  const std::string& operationName)
{
    if(values.size() != spatialDims)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            operationName + " requires " + name + " to match the number of spatial dimensions.");
    }
}

inline void validateResampleOutputShape(const std::vector<int64_t>& xDims,
                                        const std::vector<int64_t>& yDims,
                                        const std::vector<int64_t>& prePadding,
                                        const std::vector<int64_t>& postPadding,
                                        const std::vector<int64_t>& stride,
                                        const std::vector<int64_t>& window,
                                        const std::string& operationName)
{
    if(xDims.size() < 4 || xDims.size() > 5 || yDims.size() != xDims.size())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, operationName + " supports only matching 4D or 5D tensors.");
    }

    if(yDims[0] != xDims[0] || yDims[1] != xDims[1])
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, operationName + " requires matching N and C dimensions.");
    }

    const size_t spatialDims = xDims.size() - 2;
    validateSpatialVector(prePadding, spatialDims, "pre_padding", operationName);
    validateSpatialVector(postPadding, spatialDims, "post_padding", operationName);
    validateSpatialVector(stride, spatialDims, "stride", operationName);
    validateSpatialVector(window, spatialDims, "window", operationName);

    for(size_t i = 0; i < spatialDims; ++i)
    {
        if(stride[i] <= 0 || window[i] <= 0)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                operationName + " requires positive stride and window values.");
        }

        const auto paddedInput = xDims[i + 2] + prePadding[i] + postPadding[i];
        const auto expectedDim = (paddedInput - window[i]) / stride[i] + 1;
        if(expectedDim <= 0 || yDims[i + 2] != expectedDim)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                operationName + " output tensor dimensions do not match resample parameters.");
        }
    }
}

inline void validateResampleIndexShape(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& indexTensor,
    const std::vector<int64_t>& yDims,
    const std::string& operationName)
{
    if(indexTensor.data_type() != hipdnn_flatbuffers_sdk::data_objects::DataType::INT32)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, operationName + " index tensor must have INT32 data type.");
    }

    const auto indexDims = tensorDims(indexTensor);
    if(indexDims != yDims)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            operationName + " index tensor dimensions must match the output tensor.");
    }
}

} // namespace hip_kernel_provider::resample