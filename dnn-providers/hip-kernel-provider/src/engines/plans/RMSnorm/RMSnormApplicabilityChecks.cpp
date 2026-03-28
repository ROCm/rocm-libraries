// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <unordered_set>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "HipKernelUtils.hpp"
#include "RMSnormApplicabilityChecks.hpp"

namespace hip_kernel_provider::rmsnorm
{

// --- Tensor Descriptor Implementation ---
RMSnormTensorDescriptor::RMSnormTensorDescriptor(
    const hipdnn_data_sdk::data_objects::TensorAttributes* attr)
    : dims(attr->dims()->begin(), attr->dims()->end())
    , strides(attr->strides()->begin(), attr->strides()->end())
    , strideOrder(hipdnn_data_sdk::utilities::extractStrideOrder(strides))
{
}

bool RMSnormTensorDescriptor::isPacked() const
{
    return hipdnn_data_sdk::utilities::isTensorPacked(dims, strides);
}

namespace
{

// --- Validation Utilities ---
void validateDimensionCount(size_t numDims)
{
    constexpr size_t MIN_SUPPORTED_DIMS = 4;
    constexpr size_t MAX_SUPPORTED_DIMS = 5;

    if(numDims < MIN_SUPPORTED_DIMS || numDims > MAX_SUPPORTED_DIMS)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "RMSnorm implementation supports only 4D or 5D tensors.");
    }
}

void validateConsistentDimensions(const std::vector<RMSnormTensorDescriptor>& tensors)
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
                "All tensors for RMSnorm must have the same number of dimensions.");
        }
    }
}

void validatePackedTensors(const std::vector<RMSnormTensorDescriptor>& tensors)
{
    for(const auto& tensor : tensors)
    {
        if(!tensor.isPacked())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "RMSnorm implementation supports only packed tensors.");
        }
    }
}

void validateSupportedLayout(const std::vector<int64_t>& strideOrder, size_t numDims)
{
    if(numDims == 4)
    {
        const auto layoutNchw = hipdnn_data_sdk::utilities::TensorLayout::NCHW;

        if(strideOrder != layoutNchw.strideOrder)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "RMSnorm implementation supports only NCHW layouts for 4D tensors.");
        }
    }
    else
    {
        const auto layoutNcdhw = hipdnn_data_sdk::utilities::TensorLayout::NCDHW;

        if(strideOrder != layoutNcdhw.strideOrder)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "RMSnorm implementation supports only NCDHW layouts for 5D tensors.");
        }
    }
}

void validateConsistentLayouts(const std::vector<RMSnormTensorDescriptor>& tensors)
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
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "All tensors for RMSnorm must have the same layout.");
            }
        }
    }
}

void validateDataTypeIsSupported(
    hipdnn_data_sdk::data_objects::DataType dataType,
    const std::unordered_set<hipdnn_data_sdk::data_objects::DataType>& allowedTypes,
    const std::string& errorMessage)
{
    if(allowedTypes.count(dataType) > 0)
    {
        return;
    }
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM, errorMessage);
}

void validateConsistentDataTypes(
    const std::vector<int64_t>& tensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    const std::unordered_set<hipdnn_data_sdk::data_objects::DataType>& allowedTypes,
    const std::string& typeErrorMessage,
    const std::string& consistencyErrorMessage)
{
    if(tensorIds.empty())
    {
        return;
    }

    const auto& firstTensor = hip_kernel_utils::findTensorAttributes(tensorMap, tensorIds[0]);
    const auto referenceType = firstTensor.data_type();

    validateDataTypeIsSupported(referenceType, allowedTypes, typeErrorMessage);

    for(size_t i = 1; i < tensorIds.size(); ++i)
    {
        const auto& tensor = hip_kernel_utils::findTensorAttributes(tensorMap, tensorIds[i]);
        if(tensor.data_type() != referenceType)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           consistencyErrorMessage);
        }
    }
}

void validateConsistentShapes(
    const std::vector<int64_t>& tensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    const std::vector<int64_t>& referenceShape,
    const std::string& errorMessage)
{
    for(const auto tensorId : tensorIds)
    {
        const auto& tensorAttr = hip_kernel_utils::findTensorAttributes(tensorMap, tensorId);
        const std::vector<int64_t> dims(tensorAttr.dims()->begin(), tensorAttr.dims()->end());
        if(dims != referenceShape)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           errorMessage);
        }
    }
}

// --- Component Validators ---

void checkTensorLayoutsAndDimsSupported(
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    // Skip tensors with embedded scalar values (epsilon) - they don't have layouts or dimensions to validate
    std::vector<RMSnormTensorDescriptor> tensors;
    tensors.reserve(tensorMap.size());

    for(const auto& [id, attr] : tensorMap)
    {
        if(attr->value_type() != hipdnn_data_sdk::data_objects::TensorValue::NONE)
        {
            continue;
        }
        tensors.emplace_back(attr);
    }

    validateConsistentDimensions(tensors);
    validatePackedTensors(tensors);
    validateConsistentLayouts(tensors);
}

void checkTensorDataTypesSupported(
    const std::vector<int64_t>& ioTensorIds,
    const std::vector<int64_t>& affineTensorIds,
    const std::vector<int64_t>& statTensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    std::unordered_set<hipdnn_data_sdk::data_objects::DataType> allowedIOTypes{
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
        hipdnn_data_sdk::data_objects::DataType::HALF};

    validateConsistentDataTypes(ioTensorIds,
                                tensorMap,
                                allowedIOTypes,
                                "RMSnorm implementation supports only FLOAT, HALF, and BFLOAT16 "
                                "data types for x & y tensors.",
                                "All IO tensors for RMSnorm must have the same data type.");

    // Only fp32 compute type is supported for now
    std::unordered_set<hipdnn_data_sdk::data_objects::DataType> allowedComputeTypes{
        hipdnn_data_sdk::data_objects::DataType::FLOAT

    };
    validateConsistentDataTypes(affineTensorIds,
                                tensorMap,
                                allowedComputeTypes,
                                "RMSnorm affine tensors use unsupported data type.",
                                "All affine tensors for RMSnorm must have the same data type.");

    validateConsistentDataTypes(statTensorIds,
                                tensorMap,
                                allowedComputeTypes,
                                "RMSnorm stat tensors use unsupported data type.",
                                "All stat tensors for RMSnorm must have the same data type.");
}

void checkTensorShapesSupported(
    const std::vector<int64_t>& ioTensorIds,
    const std::vector<int64_t>& affineTensorIds,
    const std::vector<int64_t>& statTensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    if(ioTensorIds.empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "At least one IO tensor must be provided for RMSnorm.");
    }

    const auto& ioTensorAttr = hip_kernel_utils::findTensorAttributes(tensorMap, ioTensorIds[0]);
    const std::vector<int64_t> ioDims(ioTensorAttr.dims()->begin(), ioTensorAttr.dims()->end());

    validateConsistentShapes(
        ioTensorIds, tensorMap, ioDims, "All IO tensors for RMSnorm must have the same shape.");

    const std::vector<int64_t> affineDims = hipdnn_data_sdk::utilities::getDerivedShape(ioDims);
    validateConsistentShapes(affineTensorIds,
                             tensorMap,
                             affineDims,
                             "Scale and bias tensors for RMSnorm must have channel-only shape "
                             "derived from IO tensor shape.");

    // inv_rms should get norm stats shape [N, 1, H, W]
    std::vector<int64_t> invRMSDims = ioDims;
    invRMSDims[1] = 1;
    validateConsistentShapes(statTensorIds,
                             tensorMap,
                             invRMSDims,
                             "RMS variance tensor for RMSnorm must have single channel shape "
                             "derived from IO tensor shape.");
}
} // anonymous namespace

// --- High-Level Configuration Validators ---
void checkRMSnormTensorConfigSupported(
    const hipdnn_data_sdk::data_objects::RMSNormAttributes& rmsNormAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    std::vector<int64_t> ioTensorIds = {rmsNormAttr.x_tensor_uid(), rmsNormAttr.y_tensor_uid()};
    std::vector<int64_t> affineTensorIds = {rmsNormAttr.scale_tensor_uid()};
    if(rmsNormAttr.bias_tensor_uid().has_value())
    {
        affineTensorIds.push_back(rmsNormAttr.bias_tensor_uid().value());
    }

    std::vector<int64_t> statTensorIds;
    if(rmsNormAttr.inv_rms_tensor_uid().has_value())
    {
        statTensorIds.push_back(rmsNormAttr.inv_rms_tensor_uid().value());
    }

    checkTensorLayoutsAndDimsSupported(tensorMap);
    checkTensorDataTypesSupported(ioTensorIds, affineTensorIds, statTensorIds, tensorMap);
    checkTensorShapesSupported(ioTensorIds, affineTensorIds, statTensorIds, tensorMap);
}

} // namespace hip_kernel_provider
