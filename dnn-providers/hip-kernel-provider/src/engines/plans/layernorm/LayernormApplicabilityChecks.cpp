// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/plans/layernorm/LayernormUtilities.hpp"
#include "hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h"
#include "hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h"
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

std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> type_configs::getAllowedIoTypes()
{
    std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> types;
    for(const auto& config : VALID)
    {
        types.insert(config.io);
    }
    return types;
}

std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType>
    type_configs::getAllowedAffineTypes()
{
    std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> types;
    for(const auto& config : VALID)
    {
        types.insert(config.affine);
    }
    return types;
}

std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType>
    type_configs::getAllowedStatTypes()
{
    std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> types;
    for(const auto& config : VALID)
    {
        types.insert(config.stat);
    }
    return types;
}

std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType>
    type_configs::getAllowedEpsilonTypes()
{
    std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> types;
    for(const auto& config : VALID)
    {
        types.insert(config.epsilon);
    }
    return types;
}

// --- Validation Utilities ---

void LayernormValidator::validateNormalizedDim(const std::vector<int64_t>& ioTensorIds,
                                               const std::vector<int64_t>& affineTensorIds,
                                               const std::vector<int64_t>& statTensorIds)
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

    const auto& ioAttr = hip_kernel_utils::findTensorAttributes(_tensorMap, ioTensorIds[0]);
    const std::vector<int64_t> ioDims(ioAttr.dims()->begin(), ioAttr.dims()->end());

    size_t affineNormalizedDimMin
        = getMinNormalizedDimFromAffine(ioTensorIds[0], affineTensorIds[0], _tensorMap);
    size_t affineNormalizedDimMax
        = getMaxNormalizedDimFromAffine(ioTensorIds[0], affineTensorIds[0], _tensorMap);

    size_t statNormalizedDimMin
        = statTensorIds.empty()
              ? 0
              : getMinNormalizedDimFromStat(ioTensorIds[0], statTensorIds[0], _tensorMap);
    size_t statNormalizedDimMax
        = statTensorIds.empty()
              ? ioDims.size()
              : getMaxNormalizedDimFromStat(ioTensorIds[0], statTensorIds[0], _tensorMap);

    if(std::max(affineNormalizedDimMin, statNormalizedDimMin)
       > std::min(affineNormalizedDimMax, statNormalizedDimMax))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Affine tensors and stat tensors produce conflicting normalized dimensions for "
            "layernorm.");
    }
}

// --- Component Validators

void LayernormValidator::checkTensorIDLayoutsAndDimsSupported(const std::vector<int64_t>& tensorIds)
{
    // Skip tensors with embedded scalar values (epsilon, momentum) - they don't have layouts or dimensions to validate
    std::vector<TensorDescriptor> tensors;
    tensors.reserve(tensorIds.size());

    for(const auto& id : tensorIds)
    {
        auto attr = _tensorMap.at(id);
        if(attr->value_type() == hipdnn_flatbuffers_sdk::data_objects::TensorValue::NONE)
        {
            tensors.emplace_back(attr);
        }
    }

    validateConsistentDimensions(tensors);
    validatePackedTensors(tensors);
    validateConsistentLayouts(tensors);
}

void LayernormValidator::checkTensorLayoutsAndDimsSupported()
{
    // Skip tensors with embedded scalar values (epsilon, momentum) - they don't have layouts or dimensions to validate
    std::vector<int64_t> tensorIds;
    tensorIds.reserve(_tensorMap.size());
    for(const auto& [id, attr] : _tensorMap)
    {
        if(attr->value_type() == hipdnn_flatbuffers_sdk::data_objects::TensorValue::NONE)
        {
            tensorIds.emplace_back(id);
        }
    }

    checkTensorIDLayoutsAndDimsSupported(tensorIds);
}

void LayernormValidator::checkTensorDataTypesSupported(const std::vector<int64_t>& ioTensorIds,
                                                       const std::vector<int64_t>& affineTensorIds,
                                                       const std::vector<int64_t>& statTensorIds,
                                                       const std::vector<int64_t>& epsilonTensorIds)
{
    const auto allowedIoTypes = type_configs::getAllowedIoTypes();
    validateConsistentDataTypes(
        ioTensorIds,
        allowedIoTypes,
        "Layernorm implementation supports only FLOAT, HALF and BFLOAT16 data types for x and y "
        "tensors.",
        "All IO tensors for layernorm must have the same data type.");

    const auto allowedAffineTypes = type_configs::getAllowedAffineTypes();
    validateConsistentDataTypes(
        affineTensorIds,
        allowedAffineTypes,
        "Layernorm implementation supports only FLOAT, HALF and BFLOAT16 data types for scale and "
        "bias tensors.",
        "All affine tensors for layernorm must have the same data type.");

    const auto allowedStatTypes = type_configs::getAllowedStatTypes();
    validateConsistentDataTypes(
        statTensorIds,
        allowedStatTypes,
        "Layernorm implementation supports only FLOAT, HALF and BFLOAT16 data types for mean and "
        "inverse variance tensors.",
        "All stat tensors for layernorm must have the same data type.");

    std::vector<int64_t> allTensorIds;
    allTensorIds.insert(allTensorIds.end(), ioTensorIds.begin(), ioTensorIds.end());
    allTensorIds.insert(allTensorIds.end(), affineTensorIds.begin(), affineTensorIds.end());
    allTensorIds.insert(allTensorIds.end(), statTensorIds.begin(), statTensorIds.end());

    std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> allAllowedTypes;
    allAllowedTypes.insert(allowedIoTypes.begin(), allowedIoTypes.end());
    allAllowedTypes.insert(allowedAffineTypes.begin(), allowedAffineTypes.end());
    allAllowedTypes.insert(allowedStatTypes.begin(), allowedStatTypes.end());
    validateConsistentDataTypes(
        allTensorIds,
        allAllowedTypes,
        "Layernorm implementation only supports FLOAT, HALF and BFLOAT16 data types.",
        "All IO, affine and stat tensors for layernorm must have the same data type.");

    const auto allowedEpsilonTypes = type_configs::getAllowedEpsilonTypes();
    if(allowedEpsilonTypes.size() == 1)
    {
        validateFixedDataType(
            epsilonTensorIds,
            *allowedEpsilonTypes.begin(),
            "Layernorm implementation supports only FLOAT data type for epsilon tensors.");
    }
    else
    {
        validateConsistentDataTypes(
            epsilonTensorIds,
            allowedEpsilonTypes,
            "Layernorm epsilon tensors use unsupported data type.",
            "All epsilon tensors for layernorm must have the same data type.");
    }
}

void LayernormValidator::checkTensorShapesSupported(const std::vector<int64_t>& ioTensorIds,
                                                    const std::vector<int64_t>& affineTensorIds,
                                                    const std::vector<int64_t>& statTensorIds)
{
    if(ioTensorIds.empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "At least one IO tensor must be provided for layernorm.");
    }

    const auto& ioTensorAttr = hip_kernel_utils::findTensorAttributes(_tensorMap, ioTensorIds[0]);
    const std::vector<int64_t> ioDims(ioTensorAttr.dims()->begin(), ioTensorAttr.dims()->end());

    validateConsistentShapes(
        ioTensorIds, ioDims, "All IO tensors for layernorm must have the same shape.");

    const auto& affineTensorAttr
        = hip_kernel_utils::findTensorAttributes(_tensorMap, affineTensorIds[0]);
    const std::vector<int64_t> affineDims(affineTensorAttr.dims()->begin(),
                                          affineTensorAttr.dims()->end());
    validateConsistentShapes(
        affineTensorIds, affineDims, "All affine tensors for layernorm must have the same shape.");

    if(!statTensorIds.empty())
    {
        const auto& statTensorAttr
            = hip_kernel_utils::findTensorAttributes(_tensorMap, statTensorIds[0]);
        const std::vector<int64_t> statDims(statTensorAttr.dims()->begin(),
                                            statTensorAttr.dims()->end());
        validateConsistentShapes(statTensorIds,
                                 statDims,
                                 "Mean and variance tensors for batchnorm must have shape "
                                 "derived from IO tensor shape.");
    }
}

// --- High-level Configuration Validators ---

void LayernormValidator::checkTensorConfigSupported(
    const hipdnn_flatbuffers_sdk::data_objects::LayernormAttributes& lnAttr)
{
    std::vector<int64_t> ioTensorIds = {lnAttr.x_tensor_uid(), lnAttr.y_tensor_uid()};
    std::vector<int64_t> affineTensorIds = {lnAttr.scale_tensor_uid(), lnAttr.bias_tensor_uid()};
    std::vector<int64_t> statTensorIds;
    if(lnAttr.mean_tensor_uid().has_value())
    {
        statTensorIds.push_back(lnAttr.mean_tensor_uid().value());
    }
    if(lnAttr.inv_variance_tensor_uid().has_value())
    {
        statTensorIds.push_back(lnAttr.inv_variance_tensor_uid().value());
    }
    std::vector<int64_t> epsilonTensorIds = {lnAttr.epsilon_tensor_uid()};

    std::vector<int64_t> ioAndStatTensorIds
        = std::vector<int64_t>(ioTensorIds.begin(), ioTensorIds.end());
    ioAndStatTensorIds.insert(ioAndStatTensorIds.end(), statTensorIds.begin(), statTensorIds.end());

    checkTensorIDLayoutsAndDimsSupported(ioAndStatTensorIds);
    checkTensorDataTypesSupported(ioTensorIds, affineTensorIds, statTensorIds, epsilonTensorIds);
    validateNormalizedDim(ioTensorIds, affineTensorIds, statTensorIds);
    checkTensorShapesSupported(ioTensorIds, affineTensorIds, statTensorIds);
}

} // namespace hip_kernel_provider::layernorm
