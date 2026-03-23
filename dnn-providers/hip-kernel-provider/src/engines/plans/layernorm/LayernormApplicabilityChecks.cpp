// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/plans/layernorm/LayernormUtilities.hpp"
#include "hipdnn_data_sdk/data_objects/data_types_generated.h"
#include "hipdnn_data_sdk/data_objects/tensor_attributes_generated.h"
#include "hipdnn_plugin_sdk/PluginApiDataTypes.h"
#include "hipdnn_plugin_sdk/PluginException.hpp"
#include <cstdint>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <unordered_map>
#include <unordered_set>

#include "HipKernelUtils.hpp"
#include "LayernormApplicabilityChecks.hpp"

namespace hip_kernel_provider::layernorm
{

// --- Type Configuration Helpers ---

std::unordered_set<hipdnn_data_sdk::data_objects::DataType> type_configs::getAllowedIoTypes()
{
    std::unordered_set<hipdnn_data_sdk::data_objects::DataType> types;
    for(const auto& config : VALID)
    {
        types.insert(config.io);
    }
    return types;
}

std::unordered_set<hipdnn_data_sdk::data_objects::DataType> type_configs::getAllowedAffineTypes()
{
    std::unordered_set<hipdnn_data_sdk::data_objects::DataType> types;
    for(const auto& config : VALID)
    {
        types.insert(config.affine);
    }
    return types;
}

std::unordered_set<hipdnn_data_sdk::data_objects::DataType> type_configs::getAllowedStatTypes()
{
    std::unordered_set<hipdnn_data_sdk::data_objects::DataType> types;
    for(const auto& config : VALID)
    {
        types.insert(config.stat);
    }
    return types;
}

std::unordered_set<hipdnn_data_sdk::data_objects::DataType> type_configs::getAllowedEpsilonTypes()
{
    std::unordered_set<hipdnn_data_sdk::data_objects::DataType> types;
    for(const auto& config : VALID)
    {
        types.insert(config.epsilon);
    }
    return types;
}

// --- Tensor Descriptor Implementation ---

LayernormTensorDescriptor::LayernormTensorDescriptor(
    const hipdnn_data_sdk::data_objects::TensorAttributes* attr)
    : dims(attr->dims()->begin(), attr->dims()->end())
    , strides(attr->strides()->begin(), attr->strides()->end())
    , strideOrder(hipdnn_data_sdk::utilities::extractStrideOrder(strides))
{
}

bool LayernormTensorDescriptor::isPacked() const
{
    return hipdnn_data_sdk::utilities::isTensorPacked(dims, strides);
}

// --- Validation Utilities ---

namespace validators
{

void validateDimensionCount(size_t numDims)
{
    constexpr size_t MIN_SUPPORTED_DIMS = 4;
    constexpr size_t MAX_SUPPORTED_DIMS = 5;

    if(numDims < MIN_SUPPORTED_DIMS || numDims > MAX_SUPPORTED_DIMS)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Layernorm implementation supports only 4D or 5D tensors.");
    }
}

void validateConsistentDimensions(const std::vector<LayernormTensorDescriptor>& tensors)
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
                "All tensors for layernorm must have the same number of dimensions.");
        }
    }
}

void validatePackedTensors(const std::vector<LayernormTensorDescriptor>& tensors)
{
    for(const auto& tensor : tensors)
    {
        if(!tensor.isPacked())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Layernorm implementation only supports all packed tensors.");
        }
    }
}

void validateSupportedLayout(const std::vector<int64_t>& strideOrder, size_t numDims)
{
    if(numDims == 4)
    {
        const auto layoutNCHW = hipdnn_data_sdk::utilities::TensorLayout::NCHW;
        const auto layoutNHWC = hipdnn_data_sdk::utilities::TensorLayout::NHWC;

        if(strideOrder != layoutNCHW.strideOrder && strideOrder != layoutNHWC.strideOrder)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Layernorm implementation only supports NCHW and NHWC layouts for 4D tensors.");
        }
    }
    else if(numDims == 5)
    {
        const auto layoutNCDHW = hipdnn_data_sdk::utilities::TensorLayout::NCDHW;
        const auto layoutNDHWC = hipdnn_data_sdk::utilities::TensorLayout::NDHWC;

        if(strideOrder != layoutNCDHW.strideOrder && strideOrder != layoutNDHWC.strideOrder)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Layernorm implementation only supports NCDHW and NDHWC layouts for 5D tensors.");
        }
    }
}

void validateConsistentLayouts(const std::vector<LayernormTensorDescriptor>& tensors)
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
                    "All tensors for layernorm must have the same layout.");
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

void validateConsistentDataTypes(
    const std::vector<std::optional<int64_t>>& tensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    const std::unordered_set<hipdnn_data_sdk::data_objects::DataType>& allowedTypes,
    const std::string& typeErrorMessage,
    const std::string& consistencyErrorMessage)
{
    std::vector<int64_t> actualTensorIds;
    for(auto i : tensorIds)
    {
        if(i.has_value())
        {
            actualTensorIds.emplace_back(i.value());
        }
    }

    validateConsistentDataTypes(
        actualTensorIds, tensorMap, allowedTypes, typeErrorMessage, consistencyErrorMessage);
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
        const auto& tensor = hip_kernel_utils::findTensorAttributes(tensorMap, tensorId);
        if(tensor.data_type() != expectedType)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           errorMessage);
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

void validateConsistentShapes(
    const std::vector<std::optional<int64_t>>& tensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    const std::vector<int64_t>& referenceShape,
    const std::string& errorMessage)
{
    std::vector<int64_t> actualTensorIds;
    for(auto i : tensorIds)
    {
        if(i.has_value())
        {
            actualTensorIds.emplace_back(i.value());
        }
    }

    validateConsistentShapes(actualTensorIds, tensorMap, referenceShape, errorMessage);
}

void validateNormalizedDim(
    const std::vector<int64_t>& ioTensorIds,
    const std::vector<int64_t>& affineTensorIds,
    const std::vector<std::optional<int64_t>>& statTensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    if(ioTensorIds.empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Normalized dimension detection for layernorm requires IO tensors.");
    }

    if(affineTensorIds.empty() && statTensorIds.empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Normalized dimension detection for layernorm requires at least one affine tensor or "
            "one stat tensor.");
    }

    const auto& ioAttr = hip_kernel_utils::findTensorAttributes(tensorMap, ioTensorIds[0]);
    const std::vector<int64_t> ioDims(ioAttr.dims()->begin(), ioAttr.dims()->end());

    size_t affineNormalizedDimMin
        = getMinNormalizedDimFromAffine(ioTensorIds[0], affineTensorIds[0], tensorMap);
    size_t affineNormalizedDimMax
        = getMaxNormalizedDimFromAffine(ioTensorIds[0], affineTensorIds[0], tensorMap);

    size_t statNormalizedDimMin = 0;
    size_t statNormalizedDimMax = ioDims.size();
    for(auto statTensorId : statTensorIds)
    {
        if(statTensorId.has_value())
        {
            statNormalizedDimMin
                = getMinNormalizedDimFromStat(ioTensorIds[0], statTensorId.value(), tensorMap);
            statNormalizedDimMax
                = getMaxNormalizedDimFromStat(ioTensorIds[0], statTensorId.value(), tensorMap);
            break;
        }
    }

    if(std::max(affineNormalizedDimMin, statNormalizedDimMin)
       > std::min(affineNormalizedDimMax, statNormalizedDimMax))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Affine tensors and stat tensors produce conflicting normalized dimensions for "
            "layernorm.");
    }
}

} // namespace validators

// --- Component Validators

void checkTensorLayoutsAndDimsSupported(
    const std::vector<int64_t>& tensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    std::vector<LayernormTensorDescriptor> tensors;
    tensors.reserve(tensorIds.size());

    for(const auto& id : tensorIds)
    {
        auto attr = tensorMap.at(id);
        if(attr->value_type() == hipdnn_data_sdk::data_objects::TensorValue::NONE)
        {
            tensors.emplace_back(attr);
        }
    }

    validators::validateConsistentDimensions(tensors);
    validators::validatePackedTensors(tensors);
    validators::validateConsistentLayouts(tensors);
}

void checkTensorLayoutsAndDimsSupported(
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    // Skip tensors with embedded scalar values (epsilon, momentum) - they don't have layouts or dimensions to validate
    std::vector<int64_t> tensorIds;
    tensorIds.reserve(tensorMap.size());
    for(const auto& [id, attr] : tensorMap)
    {
        if(attr->value_type() != hipdnn_data_sdk::data_objects::TensorValue::NONE)
        {
            continue;
        }
        tensorIds.emplace_back(id);
    }

    checkTensorLayoutsAndDimsSupported(tensorIds, tensorMap);
}

void checkTensorDataTypesSupported(
    const std::vector<int64_t>& ioTensorIds,
    const std::vector<int64_t>& affineTensorIds,
    const std::vector<std::optional<int64_t>>& statTensorIds,
    const std::vector<int64_t>& epsilonTensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    const auto allowedIoTypes = type_configs::getAllowedIoTypes();
    validators::validateConsistentDataTypes(
        ioTensorIds,
        tensorMap,
        allowedIoTypes,
        "Layernorm implementation supports only FLOAT, HALF and BFLOAT16 data types for x and y "
        "tensors.",
        "All IO tensors for layernorm must have the same data type.");

    const auto allowedAffineTypes = type_configs::getAllowedAffineTypes();
    validators::validateConsistentDataTypes(
        affineTensorIds,
        tensorMap,
        allowedAffineTypes,
        "Layernorm implementation supports only FLOAT, HALF and BFLOAT16 data types for scale and "
        "bias tensors.",
        "All affine tensors for layernorm must have the same data type.");

    const auto allowedStatTypes = type_configs::getAllowedStatTypes();
    validators::validateConsistentDataTypes(
        statTensorIds,
        tensorMap,
        allowedStatTypes,
        "Layernorm implementation supports only FLOAT, HALF and BFLOAT16 data types for mean and "
        "inverse variance tensors.",
        "All stat tensors for layernorm must have the same data type.");

    std::vector<int64_t> allTensorIds;
    allTensorIds.insert(allTensorIds.end(), ioTensorIds.begin(), ioTensorIds.end());
    allTensorIds.insert(allTensorIds.end(), affineTensorIds.begin(), affineTensorIds.end());
    for(auto tensorId : statTensorIds)
    {
        if(tensorId.has_value())
        {
            allTensorIds.emplace_back(tensorId.value());
        }
    }
    std::unordered_set<hipdnn_data_sdk::data_objects::DataType> allAllowedTypes;
    allAllowedTypes.insert(allowedIoTypes.begin(), allowedIoTypes.end());
    allAllowedTypes.insert(allowedAffineTypes.begin(), allowedAffineTypes.end());
    allAllowedTypes.insert(allowedStatTypes.begin(), allowedStatTypes.end());
    validators::validateConsistentDataTypes(
        allTensorIds,
        tensorMap,
        allAllowedTypes,
        "Layernorm implementation only supports FLOAT, HALF and BFLOAT16 data types.",
        "All IO, affine and stat tensors for layernorm must have the same data type.");

    const auto allowedEpsilonTypes = type_configs::getAllowedEpsilonTypes();
    if(allowedEpsilonTypes.size() == 1)
    {
        validators::validateFixedDataType(
            epsilonTensorIds,
            tensorMap,
            *allowedEpsilonTypes.begin(),
            "Layernorm implementation supports only FLOAT data type for epsilon tensors.");
    }
    else
    {
        validators::validateConsistentDataTypes(
            epsilonTensorIds,
            tensorMap,
            allowedEpsilonTypes,
            "Layernorm epsilon tensors use unsupported data type.",
            "All epsilon tensors for layernorm must have the same data type.");
    }
}

void checkTensorShapesSupported(
    const std::vector<int64_t>& ioTensorIds,
    const std::vector<int64_t>& affineTensorIds,
    const std::vector<std::optional<int64_t>>& statTensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    if(ioTensorIds.empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "At least one IO tensor must be provided for layernorm.");
    }

    const auto& ioTensorAttr = hip_kernel_utils::findTensorAttributes(tensorMap, ioTensorIds[0]);
    const std::vector<int64_t> ioDims(ioTensorAttr.dims()->begin(), ioTensorAttr.dims()->end());

    validators::validateConsistentShapes(
        ioTensorIds, tensorMap, ioDims, "All IO tensors for layernorm must have the same shape.");

    const auto& affineTensorAttr
        = hip_kernel_utils::findTensorAttributes(tensorMap, affineTensorIds[0]);
    const std::vector<int64_t> affineDims(affineTensorAttr.dims()->begin(),
                                          affineTensorAttr.dims()->end());
    validators::validateConsistentShapes(
        affineTensorIds,
        tensorMap,
        affineDims,
        "All affine tensors for layernorm must have the same shape.");

    if(!statTensorIds.empty())
    {
        for(const auto& statTensorId : statTensorIds)
        {
            if(statTensorId.has_value())
            {
                const auto& statTensorAttr
                    = hip_kernel_utils::findTensorAttributes(tensorMap, statTensorId.value());
                const std::vector<int64_t> statDims(statTensorAttr.dims()->begin(),
                                                    statTensorAttr.dims()->end());
                validators::validateConsistentShapes(
                    statTensorIds,
                    tensorMap,
                    statDims,
                    "All stat tensors for layernorm must have the same shape.");
                break;
            }
        }
    }
}

// --- High-level Configuration Validators ---

void checkLayernormTensorConfigSupported(
    const hipdnn_data_sdk::data_objects::LayernormAttributes& lnAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    std::vector<int64_t> ioTensorIds = {lnAttr.x_tensor_uid(), lnAttr.y_tensor_uid()};
    std::vector<int64_t> affineTensorIds = {lnAttr.scale_tensor_uid(), lnAttr.bias_tensor_uid()};
    std::vector<std::optional<int64_t>> statTensorIds
        = {lnAttr.mean_tensor_uid(), lnAttr.inv_variance_tensor_uid()};
    std::vector<int64_t> epsilonTensorIds = {lnAttr.epsilon_tensor_uid()};
    std::vector<int64_t> ioAndStatTensorIds
        = std::vector<int64_t>(ioTensorIds.begin(), ioTensorIds.end());
    for(auto statTensorId : statTensorIds)
    {
        if(statTensorId.has_value())
        {
            ioAndStatTensorIds.emplace_back(statTensorId.value());
        }
    }

    checkTensorLayoutsAndDimsSupported(ioAndStatTensorIds, tensorMap);
    checkTensorDataTypesSupported(
        ioTensorIds, affineTensorIds, statTensorIds, epsilonTensorIds, tensorMap);
    validators::validateNormalizedDim(ioTensorIds, affineTensorIds, statTensorIds, tensorMap);
    checkTensorShapesSupported(ioTensorIds, affineTensorIds, statTensorIds, tensorMap);
}

} // namespace hip_kernel_provider::layernorm
