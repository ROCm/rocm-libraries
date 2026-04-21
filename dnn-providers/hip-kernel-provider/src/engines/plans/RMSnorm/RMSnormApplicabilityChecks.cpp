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
    std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> allowedIOTypes{
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF};

    validateConsistentDataTypes(ioTensorIds,
                                allowedIOTypes,
                                "RMSnorm implementation supports only FLOAT, HALF, and BFLOAT16 "
                                "data types for x & y tensors.",
                                "All IO tensors for RMSnorm must have the same data type.");

    // Only fp32 compute type is supported for now
    std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> allowedComputeTypes{
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT

    };
    validateConsistentDataTypes(affineTensorIds,
                                allowedComputeTypes,
                                "RMSnorm affine tensors use unsupported data type.",
                                "All affine tensors for RMSnorm must have the same data type.");

    validateConsistentDataTypes(statTensorIds,
                                allowedComputeTypes,
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

    const std::vector<int64_t> affineDims = hipdnn_data_sdk::utilities::getDerivedShape(ioDims);
    validateConsistentShapes(affineTensorIds,
                             affineDims,
                             "Scale and bias tensors for RMSnorm must have channel-only shape "
                             "derived from IO tensor shape.");

    // inv_rms should get norm stats shape [N, 1, H, W]
    std::vector<int64_t> invRMSDims = ioDims;
    invRMSDims[1] = 1;
    validateConsistentShapes(statTensorIds,
                             invRMSDims,
                             "RMS variance tensor for RMSnorm must have single channel shape "
                             "derived from IO tensor shape.");
}

void RMSnormValidator::checkTensorConfigSupported(
    const hipdnn_flatbuffers_sdk::data_objects::RMSNormAttributes& rmsNormAttr)
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

    checkTensorLayoutsAndDimsSupported();
    checkTensorDataTypesSupported(ioTensorIds, affineTensorIds, statTensorIds);
    checkTensorShapesSupported(ioTensorIds, affineTensorIds, statTensorIds);
}

void RMSnormValidator::checkBwdTensorConfigSupported(
    const hipdnn_flatbuffers_sdk::data_objects::RMSNormBackwardAttributes& rmsNormBwdAttr)
{
    std::vector<int64_t> ioTensorIds = {rmsNormBwdAttr.dy_tensor_uid(),
                                        rmsNormBwdAttr.x_tensor_uid(),
                                        rmsNormBwdAttr.dx_tensor_uid()};

    std::vector<int64_t> affineTensorIds
        = {rmsNormBwdAttr.scale_tensor_uid(), rmsNormBwdAttr.dscale_tensor_uid()};
    if(rmsNormBwdAttr.dbias_tensor_uid().has_value())
    {
        affineTensorIds.push_back(rmsNormBwdAttr.dbias_tensor_uid().value());
    }

    std::vector<int64_t> statTensorIds;
    if(rmsNormBwdAttr.inv_rms_tensor_uid().has_value())
    {
        statTensorIds.push_back(rmsNormBwdAttr.inv_rms_tensor_uid().value());
    }

    checkTensorLayoutsAndDimsSupported();
    checkTensorDataTypesSupported(ioTensorIds, affineTensorIds, statTensorIds);
    checkTensorShapesSupported(ioTensorIds, affineTensorIds, statTensorIds);
}

} // namespace hip_kernel_provider::rmsnorm
