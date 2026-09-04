// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <unordered_set>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "RMSnormApplicabilityChecks.hpp"
#include "core/Utils.hpp"

using namespace hip_kernel_provider::core::utils;

namespace hip_kernel_provider::rmsnorm
{
// --- Component Validators ---

void RMSnormValidator::checkTensorLayoutsAndDimsSupported(const std::vector<int64_t>& tensorIds)
{
    // Skip tensors with embedded scalar values (epsilon) - they don't have layouts or dimensions to validate
    std::vector<TensorDescriptor> tensors;
    tensors.reserve(tensorIds.size());

    for(const auto& id : tensorIds)
    {
        auto attr = _tensorMap.at(id);
        if(!hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(attr))
        {
            tensors.emplace_back(attr);
        }
    }

    validateConsistentDimensions(tensors);
    validatePackedTensors(tensors);
    validateConsistentLayouts(tensors);
}

void RMSnormValidator::checkTensorDataTypesSupported(
    const std::vector<int64_t>& ioTensorIds,
    const std::vector<int64_t>& affineTensorIds,
    const std::vector<int64_t>& statTensorIds,
    const std::vector<int64_t>& intermediateTensorIds)
{
    const std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> allowedIOTypes{
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF};

    for(const auto ioTensorId : ioTensorIds)
    {
        const auto& tensorAttr = findTensorAttributes(_tensorMap, ioTensorId);
        validateDataTypeIsSupported(tensorAttr.data_type(),
                                    allowedIOTypes,
                                    "RMSnorm implementation supports only FLOAT, HALF, and "
                                    "BFLOAT16 data types for x and y tensors.");
    }

    const std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> allowedAffineTypes{
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF};

    validateConsistentDataTypes(affineTensorIds,
                                allowedAffineTypes,
                                "RMSnorm affine tensors use unsupported data type.",
                                "All affine tensors for RMSnorm must have the same data type.");

    // Only fp32 compute type is supported for now
    const std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> allowedComputeTypes{
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT};

    validateConsistentDataTypes(statTensorIds,
                                allowedComputeTypes,
                                "RMSnorm stat tensors use unsupported data type.",
                                "All stat tensors for RMSnorm must have the same data type.");

    const std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType>
        allowedIntermediateTypes{hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT};

    validateConsistentDataTypes(
        intermediateTensorIds,
        allowedIntermediateTypes,
        "RMSnorm intermediate tensors use unsupported data type.",
        "All intermediate tensors for RMSnorm must have the same data type.");
}

void RMSnormValidator::checkTensorShapesSupported(const std::vector<int64_t>& ioTensorIds,
                                                  const std::vector<int64_t>& affineTensorIds,
                                                  const std::vector<int64_t>& statTensorIds,
                                                  const std::vector<int64_t>& intermediateTensorIds)
{
    if(ioTensorIds.empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "At least one IO tensor must be provided for RMSnorm.");
    }

    const auto& ioTensorAttr = findTensorAttributes(_tensorMap, ioTensorIds[0]);
    const std::vector<int64_t> ioDims(ioTensorAttr.dims()->begin(), ioTensorAttr.dims()->end());

    validateConsistentShapes(
        ioTensorIds, ioDims, "All IO tensors for RMSnorm must have the same shape.");

    const auto& affineTensorAttr = findTensorAttributes(_tensorMap, affineTensorIds[0]);
    const std::vector<int64_t> affineDims(affineTensorAttr.dims()->begin(),
                                          affineTensorAttr.dims()->end());
    validateConsistentShapes(affineTensorIds,
                             affineDims,
                             "Scale and bias tensors for RMSnorm must have the same shape.");

    if(!intermediateTensorIds.empty())
    {
        const auto& intermediateTensorAttr
            = findTensorAttributes(_tensorMap, intermediateTensorIds[0]);
        const std::vector<int64_t> intermediateDims(intermediateTensorAttr.dims()->begin(),
                                                    intermediateTensorAttr.dims()->end());
        validateConsistentShapes(intermediateTensorIds,
                                 intermediateDims,
                                 "Intermediate tensors for RMSnorm must have the same shape.");
    }

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

void RMSnormValidator::checkFwdTensorConfigSupported(
    const hipdnn_flatbuffers_sdk::data_objects::RMSNormAttributes& rmsNormFwdAttr)
{
    const std::vector<int64_t> ioTensorIds
        = {rmsNormFwdAttr.x_tensor_uid(), rmsNormFwdAttr.y_tensor_uid()};
    std::vector<int64_t> affineTensorIds = {rmsNormFwdAttr.scale_tensor_uid()};
    if(rmsNormFwdAttr.bias_tensor_uid().has_value())
    {
        affineTensorIds.push_back(rmsNormFwdAttr.bias_tensor_uid().value());
    }
    std::vector<int64_t> statTensorIds;
    if(rmsNormFwdAttr.inv_rms_tensor_uid().has_value())
    {
        statTensorIds.push_back(rmsNormFwdAttr.inv_rms_tensor_uid().value());
    }

    std::vector<int64_t> allTensors = std::vector<int64_t>(ioTensorIds.begin(), ioTensorIds.end());
    allTensors.insert(allTensors.end(), affineTensorIds.begin(), affineTensorIds.end());
    allTensors.insert(allTensors.end(), statTensorIds.begin(), statTensorIds.end());

    checkTensorLayoutsAndDimsSupported(allTensors);
    checkTensorDataTypesSupported(ioTensorIds, affineTensorIds, statTensorIds, {});
    checkTensorShapesSupported(ioTensorIds, affineTensorIds, statTensorIds, {});
}

void RMSnormValidator::checkFwdActivationTensorConfigSupported(
    const hipdnn_flatbuffers_sdk::data_objects::RMSNormAttributes& rmsNormFwdAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwiseAttr)
{
    checkActivationModeSupported(pointwiseAttr);

    const std::vector<int64_t> ioTensorIds
        = {rmsNormFwdAttr.x_tensor_uid(), pointwiseAttr.out_0_tensor_uid()};
    std::vector<int64_t> affineTensorIds = {rmsNormFwdAttr.scale_tensor_uid()};
    if(rmsNormFwdAttr.bias_tensor_uid().has_value())
    {
        affineTensorIds.push_back(rmsNormFwdAttr.bias_tensor_uid().value());
    }
    std::vector<int64_t> statTensorIds;
    if(rmsNormFwdAttr.inv_rms_tensor_uid().has_value())
    {
        statTensorIds.push_back(rmsNormFwdAttr.inv_rms_tensor_uid().value());
    }
    const std::vector<int64_t> intermediateTensorIds
        = {rmsNormFwdAttr.y_tensor_uid(), pointwiseAttr.in_0_tensor_uid()};

    std::vector<int64_t> allTensors = std::vector<int64_t>(ioTensorIds.begin(), ioTensorIds.end());
    allTensors.insert(allTensors.end(), affineTensorIds.begin(), affineTensorIds.end());
    allTensors.insert(allTensors.end(), statTensorIds.begin(), statTensorIds.end());
    allTensors.insert(allTensors.end(), intermediateTensorIds.begin(), intermediateTensorIds.end());

    checkTensorLayoutsAndDimsSupported(allTensors);
    checkTensorDataTypesSupported(
        ioTensorIds, affineTensorIds, statTensorIds, intermediateTensorIds);
    checkTensorShapesSupported(ioTensorIds, affineTensorIds, statTensorIds, intermediateTensorIds);
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

    std::vector<int64_t> allTensors = std::vector<int64_t>(ioTensorIds.begin(), ioTensorIds.end());
    allTensors.insert(allTensors.end(), affineTensorIds.begin(), affineTensorIds.end());
    allTensors.insert(allTensors.end(), statTensorIds.begin(), statTensorIds.end());

    checkTensorLayoutsAndDimsSupported(allTensors);
    checkTensorDataTypesSupported(ioTensorIds, affineTensorIds, statTensorIds, {});
    checkTensorShapesSupported(ioTensorIds, affineTensorIds, statTensorIds, {});
}

void RMSnormValidator::checkBwdActivationTensorConfigSupported(
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwiseAttr,
    const hipdnn_flatbuffers_sdk::data_objects::RMSNormBackwardAttributes& rmsNormBwdAttr)
{
    checkActivationModeSupported(pointwiseAttr);

    const auto activationIn1Uid = pointwiseAttr.in_1_tensor_uid();
    if(!activationIn1Uid.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation backward node must have a second input tensor (in_1)");
    }

    const std::vector<int64_t> ioTensorIds = {
        pointwiseAttr.in_0_tensor_uid(),
        rmsNormBwdAttr.x_tensor_uid(),
        rmsNormBwdAttr.dx_tensor_uid(),
        activationIn1Uid.value(),
    };
    const std::vector<int64_t> statTensorIds = {rmsNormBwdAttr.inv_rms_tensor_uid()};
    std::vector<int64_t> affineTensorIds
        = {rmsNormBwdAttr.scale_tensor_uid(), rmsNormBwdAttr.dscale_tensor_uid()};
    if(rmsNormBwdAttr.dbias_tensor_uid().has_value())
    {
        affineTensorIds.push_back(rmsNormBwdAttr.dbias_tensor_uid().value());
    }
    const std::vector<int64_t> intermediateTensorIds
        = {pointwiseAttr.out_0_tensor_uid(), rmsNormBwdAttr.dy_tensor_uid()};

    std::vector<int64_t> allTensors = std::vector<int64_t>(ioTensorIds.begin(), ioTensorIds.end());
    allTensors.insert(allTensors.end(), affineTensorIds.begin(), affineTensorIds.end());
    allTensors.insert(allTensors.end(), statTensorIds.begin(), statTensorIds.end());
    allTensors.insert(allTensors.end(), intermediateTensorIds.begin(), intermediateTensorIds.end());

    checkTensorLayoutsAndDimsSupported(allTensors);
    checkTensorDataTypesSupported(
        ioTensorIds, affineTensorIds, statTensorIds, intermediateTensorIds);
    checkTensorShapesSupported(ioTensorIds, affineTensorIds, statTensorIds, intermediateTensorIds);
}

void RMSnormValidator::checkActivationModeSupported(
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwiseAttr)
{
    // hip-kernel-provider rmsnorm supports: PASSTHRU, RELU, CLIPPEDRELU, CLAMP (no Leaky ReLU)

    switch(pointwiseAttr.operation())
    {
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::IDENTITY:
        return;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD:
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD:
        if(!pointwiseAttr.relu_lower_clip_slope().has_value())
        {
            return;
        }
        if(pointwiseAttr.relu_lower_clip().has_value()
           && pointwiseAttr.relu_lower_clip().value() != 0.0f)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Rmsnorm fused activation does not support standard ReLU with a non-zero "
                "lower_clip.");
        }
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Rmsnorm fused activation does not support Leaky ReLU.");
    default:
        const std::string activationModeName(EnumNamePointwiseMode(pointwiseAttr.operation()));
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported activation mode for rmsnorm fusion: " + activationModeName + ".");
    }
}

} // namespace hip_kernel_provider::rmsnorm
