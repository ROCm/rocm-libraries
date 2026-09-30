// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "ApplicabilityChecks.hpp"
#include "core/Utils.hpp"

using namespace hip_kernel_provider::core::utils;

namespace hip_kernel_provider
{

// --- Tensor Descriptor Implementation ---

TensorDescriptor::TensorDescriptor(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* attr)
    : dims(attr->dims()->begin(), attr->dims()->end())
    , strides(attr->strides()->begin(), attr->strides()->end())
    , strideOrder(hipdnn_data_sdk::utilities::extractStrideOrder(strides))
{
}

bool TensorDescriptor::isPacked() const
{
    return hipdnn_data_sdk::utilities::isTensorPacked(dims, strides);
}

void IValidator::validateDimensionCount(size_t numDims)
{
    constexpr size_t MIN_SUPPORTED_DIMS = 4;
    constexpr size_t MAX_SUPPORTED_DIMS = 5;

    if(numDims < MIN_SUPPORTED_DIMS || numDims > MAX_SUPPORTED_DIMS)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Only 4D or 5D tensors supported.");
    }
}

void IValidator::validateConsistentDimensions(const std::vector<TensorDescriptor>& tensors)
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
                "All tensors must have the same number of dimensions.");
        }
    }
}

void IValidator::validatePackedTensors(const std::vector<TensorDescriptor>& tensors)
{
    for(const auto& tensor : tensors)
    {
        if(!tensor.isPacked())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           "Only packed tensors supported.");
        }
    }
}

void IValidator::validateSupportedLayout(const std::vector<int64_t>& strideOrder, size_t numDims)
{
    if(numDims == 4)
    {
        const auto layoutNchw = hipdnn_data_sdk::utilities::TensorLayout::NCHW;
        const auto layoutNhwc = hipdnn_data_sdk::utilities::TensorLayout::NHWC;

        if(strideOrder != layoutNchw.strideOrder && strideOrder != layoutNhwc.strideOrder)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Only NCHW and NHWC layouts supported for 4D tensors.");
        }
    }
    else
    {
        const auto layoutNcdhw = hipdnn_data_sdk::utilities::TensorLayout::NCDHW;
        const auto layoutNdhwc = hipdnn_data_sdk::utilities::TensorLayout::NDHWC;

        if(strideOrder != layoutNcdhw.strideOrder && strideOrder != layoutNdhwc.strideOrder)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Only NCDHW and NDHWC layouts supported for 5D tensors.");
        }
    }
}

void IValidator::validateConsistentLayouts(const std::vector<TensorDescriptor>& tensors)
{
    if(tensors.empty())
    {
        return;
    }

    // Use first tensor with meaningful layout as reference
    int64_t referenceIndex = -1;
    for(size_t i = 0; i < tensors.size(); ++i)
    {
        if(hipdnn_data_sdk::utilities::isLayoutAgnostic(tensors[i].dims))
        {
            continue;
        }

        if(referenceIndex == -1)
        {
            referenceIndex = static_cast<int64_t>(i);
            validateSupportedLayout(tensors[static_cast<size_t>(referenceIndex)].strideOrder,
                                    tensors[static_cast<size_t>(referenceIndex)].numDims());
        }
        else
        {
            if(tensors[i].strideOrder != tensors[static_cast<size_t>(referenceIndex)].strideOrder)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM, "All tensors must have the same layout.");
            }
        }
    }
}

void IValidator::validateDataTypeIsSupported(
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
    const std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType>& allowedTypes,
    const std::string& errorMessage)
{
    if(allowedTypes.count(dataType) > 0)
    {
        return;
    }
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM, errorMessage);
}

void IValidator::validateConsistentDataTypes(
    const std::vector<int64_t>& tensorIds,
    const std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType>& allowedTypes,
    const std::string& typeErrorMessage,
    const std::string& consistencyErrorMessage)
{
    if(tensorIds.empty())
    {
        return;
    }

    const auto& firstTensor = findTensorAttributes(_tensorMap, tensorIds[0]);
    const auto referenceType = firstTensor.data_type();

    validateDataTypeIsSupported(referenceType, allowedTypes, typeErrorMessage);

    for(size_t i = 1; i < tensorIds.size(); ++i)
    {
        const auto& tensor = findTensorAttributes(_tensorMap, tensorIds[i]);
        if(tensor.data_type() != referenceType)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           consistencyErrorMessage);
        }
    }
}

void IValidator::validateFixedDataType(const std::vector<int64_t>& tensorIds,
                                       hipdnn_flatbuffers_sdk::data_objects::DataType expectedType,
                                       const std::string& errorMessage)
{
    for(const auto tensorId : tensorIds)
    {
        const auto& tensor = findTensorAttributes(_tensorMap, tensorId);
        if(tensor.data_type() != expectedType)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           errorMessage);
        }
    }
}

void IValidator::validateConsistentShapes(const std::vector<int64_t>& tensorIds,
                                          const std::vector<int64_t>& referenceShape,
                                          const std::string& errorMessage)
{
    for(const auto tensorId : tensorIds)
    {
        const auto& tensorAttr = findTensorAttributes(_tensorMap, tensorId);
        const std::vector<int64_t> dims(tensorAttr.dims()->begin(), tensorAttr.dims()->end());
        if(dims != referenceShape)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           errorMessage);
        }
    }
}

} // namespace hip_kernel_provider
