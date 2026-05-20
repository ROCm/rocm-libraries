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
// --- Validation Utilities ---

void RMSnormValidator::validateSupportedLayout(const std::vector<int64_t>& strideOrder,
                                               size_t numDims)
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

// --- Component Validators ---

void RMSnormValidator::checkTensorLayoutsAndDimsSupported()
{
    // Skip tensors with embedded scalar values (epsilon) - they don't have layouts or dimensions to validate
    std::vector<TensorDescriptor> tensors;
    tensors.reserve(_tensorMap.size());

    for(const auto& [id, attr] : _tensorMap)
    {
        if(attr->value_type() != hipdnn_flatbuffers_sdk::data_objects::TensorValue::NONE)
        {
            continue;
        }
        tensors.emplace_back(attr);
    }

    validateConsistentDimensions(tensors);
    validatePackedTensors(tensors);
    validateConsistentLayouts(tensors);
}

void RMSnormValidator::checkTensorDataTypesSupported(const std::vector<int64_t>& ioTensorIds,
                                                     const std::vector<int64_t>& affineTensorIds,
                                                     const std::vector<int64_t>& statTensorIds)
{
    const std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> allowedIOTypes{
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF};

    for(const auto ioTensorId : ioTensorIds)
    {
        const auto& tensorAttr = hip_kernel_utils::findTensorAttributes(_tensorMap, ioTensorId);
        validateDataTypeIsSupported(tensorAttr.data_type(),
                                    allowedIOTypes,
                                    "RMSnorm implementation supports only FLOAT, HALF, and "
                                    "BFLOAT16 data types for x and y tensors.");
    }

    // Only fp32 compute type is supported for now
    const std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> allowedComputeTypes{
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT

    };
    validateConsistentDataTypes(affineTensorIds,
                                allowedComputeTypes,
                                "RMSnorm affine tensors use unsupported data type.",
                                "All affine tensors for RMSnorm must have the same data type.");

    const std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> allowedStatTypes{
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF};

    validateConsistentDataTypes(statTensorIds,
                                allowedStatTypes,
                                "RMSnorm stat tensors use unsupported data type.",
                                "All stat tensors for RMSnorm must have the same data type.");
}

void RMSnormValidator::checkTensorShapesSupported(const std::vector<int64_t>& ioTensorIds,
                                                  const std::vector<int64_t>& affineTensorIds,
                                                  const std::vector<int64_t>& statTensorIds)
{
    if(ioTensorIds.empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "At least one IO tensor must be provided for RMSnorm.");
    }

    const auto& ioTensorAttr = hip_kernel_utils::findTensorAttributes(_tensorMap, ioTensorIds[0]);
    const std::vector<int64_t> ioDims(ioTensorAttr.dims()->begin(), ioTensorAttr.dims()->end());

    validateConsistentShapes(
        ioTensorIds, ioDims, "All IO tensors for RMSnorm must have the same shape.");

    const auto& affineTensorAttr
        = hip_kernel_utils::findTensorAttributes(_tensorMap, affineTensorIds[0]);
    const std::vector<int64_t> affineDims(affineTensorAttr.dims()->begin(),
                                          affineTensorAttr.dims()->end());
    validateConsistentShapes(affineTensorIds,
                             affineDims,
                             "Scale and bias tensors for RMSnorm must have the same shape.");

    checkAffineNormalizedShape(affineDims, ioDims);

    // inv_rms shapes is derived from scale and input:
    // Where scale has a non-1 dim, inv_rms gets 1 (normalized dimension collapses).
    // Where scale has dim 1, inv_rms keeps the input dim.
    std::vector<int64_t> invRMSDims = ioDims;
    for(size_t i = 0; i < invRMSDims.size(); ++i)
    {
        if(affineDims[i] != 1)
        {
            invRMSDims[i] = 1;
        }
    }
    validateConsistentShapes(
        statTensorIds,
        invRMSDims,
        "RMS variance tensor for RMSnorm must be derived from scale and IO shape.");
}

void RMSnormValidator::checkAffineNormalizedShape(const std::vector<int64_t>& affineDims,
                                                  const std::vector<int64_t>& ioDims)
{
    const auto [scaleMismatch, _]
        = std::mismatch(affineDims.rbegin(), affineDims.rend(), ioDims.rbegin(), ioDims.rend());
    const auto matchCount = static_cast<size_t>(std::distance(affineDims.rbegin(), scaleMismatch));
    const size_t normalizeDim
        = (matchCount == affineDims.size()) ? 1 : affineDims.size() - matchCount;

    for(unsigned i = 0; i < normalizeDim; ++i)
    {
        if(affineDims[i] != 1)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "Affine tensors not correctly normalized");
        }
    }
}

void RMSnormValidator::checkTensorConfigSupported(
    const hipdnn_flatbuffers_sdk::data_objects::RMSNormAttributes& rmsNormAttr)
{
    const std::vector<int64_t> ioTensorIds
        = {rmsNormAttr.x_tensor_uid(), rmsNormAttr.y_tensor_uid()};
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

    checkTensorLayoutsAndDimsSupported();
    checkTensorDataTypesSupported(ioTensorIds, affineTensorIds, statTensorIds);
    checkTensorShapesSupported(ioTensorIds, affineTensorIds, statTensorIds);
}

void RMSnormValidator::checkBwdTensorConfigSupported(
    const hipdnn_flatbuffers_sdk::data_objects::RMSNormBackwardAttributes& rmsNormBwdAttr)
{
    const std::vector<int64_t> ioTensorIds = {
        rmsNormBwdAttr.dy_tensor_uid(),
        rmsNormBwdAttr.x_tensor_uid(),
        rmsNormBwdAttr.dx_tensor_uid(),
    };
    const std::vector<int64_t> statTensorIds = {rmsNormBwdAttr.inv_rms_tensor_uid()};

    std::vector<int64_t> affineTensorIds
        = {rmsNormBwdAttr.scale_tensor_uid(), rmsNormBwdAttr.dscale_tensor_uid()};
    if(rmsNormBwdAttr.dbias_tensor_uid().has_value())
    {
        affineTensorIds.push_back(rmsNormBwdAttr.dbias_tensor_uid().value());
    }

    checkTensorLayoutsAndDimsSupported();
    checkTensorDataTypesSupported(ioTensorIds, affineTensorIds, statTensorIds);
    checkTensorShapesSupported(ioTensorIds, affineTensorIds, statTensorIds);
}

} // namespace hip_kernel_provider::rmsnorm
