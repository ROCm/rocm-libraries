// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <numeric>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "BatchnormApplicabilityChecks.hpp"
#include "core/Utils.hpp"

namespace hip_kernel_provider
{

// --- Type Configuration Helpers ---

std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType>
    bn_type_configs::getAllowedIoTypes()
{
    std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> types;
    for(const auto& config : VALID)
    {
        types.insert(config.io);
    }
    return types;
}

std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType>
    bn_type_configs::getAllowedAffineTypes()
{
    std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> types;
    for(const auto& config : VALID)
    {
        types.insert(config.affine);
    }
    return types;
}

std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType>
    bn_type_configs::getAllowedStatTypes()
{
    std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> types;
    for(const auto& config : VALID)
    {
        types.insert(config.stat);
    }
    return types;
}

std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType>
    bn_type_configs::getAllowedIntermediateTypes()
{
    std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> types;
    for(const auto& config : VALID)
    {
        types.insert(config.intermediate);
    }
    return types;
}

void BatchnormValidator::validateSpatialDimensions(const std::vector<int64_t>& ioDims)
{
    if(ioDims.size() < 3)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "IO tensor must have at least 3 dimensions for batchnorm.");
    }

    const auto spatialSize
        = std::accumulate(ioDims.begin() + 2, ioDims.end(), int64_t{1}, std::multiplies<>());

    if(ioDims[0] * spatialSize <= 1)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "The product of the batch size and spatial dimensions must be greater than 1 for "
            "batchnorm.");
    }
}

// --- Component Validators ---

void BatchnormValidator::checkTensorLayoutsAndDimsSupported()
{
    // Skip tensors with embedded scalar values (epsilon, momentum) - they don't have layouts or dimensions to validate
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

void BatchnormValidator::checkTensorDataTypesSupported(
    const std::vector<int64_t>& ioTensorIds,
    const std::vector<int64_t>& affineTensorIds,
    const std::vector<int64_t>& statTensorIds,
    const std::vector<int64_t>& intermediateTensorIds)
{
    validateConsistentDataTypes(
        ioTensorIds,
        bn_type_configs::getAllowedIoTypes(),
        "Batchnorm implementation supports only FLOAT, HALF, and BFLOAT16 data types for x, y, "
        "dy, and dx tensors.",
        "All IO tensors for batchnorm must have the same data type.");

    const auto allowedAffineTypes = bn_type_configs::getAllowedAffineTypes();
    if(allowedAffineTypes.size() == 1)
    {
        validateFixedDataType(affineTensorIds,
                              *allowedAffineTypes.begin(),
                              "Batchnorm implementation supports only FLOAT data type "
                              "for scale and bias tensors.");
    }
    else
    {
        validateConsistentDataTypes(
            affineTensorIds,
            allowedAffineTypes,
            "Batchnorm affine tensors use unsupported data type.",
            "All affine tensors for batchnorm must have the same data type.");
    }

    const auto allowedStatTypes = bn_type_configs::getAllowedStatTypes();
    if(allowedStatTypes.size() == 1)
    {
        validateFixedDataType(statTensorIds,
                              *allowedStatTypes.begin(),
                              "Batchnorm implementation supports only FLOAT data type "
                              "for mean and variance tensors.");
    }
    else
    {
        validateConsistentDataTypes(statTensorIds,
                                    allowedStatTypes,
                                    "Batchnorm stat tensors use unsupported data type.",
                                    "All stat tensors for batchnorm must have the same data type.");
    }

    const auto allowedIntermediateTypes = bn_type_configs::getAllowedIntermediateTypes();
    if(allowedIntermediateTypes.size() == 1)
    {
        validateFixedDataType(
            intermediateTensorIds,
            *allowedIntermediateTypes.begin(),
            "Batchnorm implementation supports only FLOAT data type for intermediate tensors.");
    }
    else
    {
        validateConsistentDataTypes(
            intermediateTensorIds,
            allowedIntermediateTypes,
            "Batchnorm intermediate tensors use unsupported data type.",
            "All intermediate tensors for batchnorm must have the same data type.");
    }
}

void BatchnormValidator::checkTensorShapesSupported(const std::vector<int64_t>& ioTensorIds,
                                                    const std::vector<int64_t>& affineTensorIds,
                                                    const std::vector<int64_t>& statTensorIds,

                                                    bool isTraining)
{
    if(ioTensorIds.empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "At least one IO tensor must be provided for batchnorm.");
    }

    const auto& ioTensorAttr = core::utils::findTensorAttributes(_tensorMap, ioTensorIds[0]);
    const std::vector<int64_t> ioDims(ioTensorAttr.dims()->begin(), ioTensorAttr.dims()->end());

    validateConsistentShapes(
        ioTensorIds, ioDims, "All IO tensors for batchnorm must have the same shape.");

    const auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(ioDims);
    validateConsistentShapes(affineTensorIds,
                             derivedDims,
                             "Scale and bias tensors for batchnorm must have shape "
                             "derived from IO tensor shape.");
    validateConsistentShapes(statTensorIds,
                             derivedDims,
                             "Mean and variance tensors for batchnorm must have shape "
                             "derived from IO tensor shape.");

    if(isTraining)
    {
        validateSpatialDimensions(ioDims);
    }
}

// --- High-Level Configuration Validators ---

void BatchnormValidator::checkTensorConfigSupported(
    const std::vector<int64_t>& ioTensorIds,
    const std::vector<int64_t>& affineTensorIds,
    const std::vector<int64_t>& statTensorIds,
    const std::vector<int64_t>& intermediateTensorIds,
    bool isTraining)
{
    checkTensorLayoutsAndDimsSupported();
    checkTensorDataTypesSupported(
        ioTensorIds, affineTensorIds, statTensorIds, intermediateTensorIds);
    checkTensorShapesSupported(ioTensorIds, affineTensorIds, statTensorIds, isTraining);
}

void BatchnormValidator::checkInferenceTensorConfigSupported(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr)
{
    const std::vector<int64_t> ioTensorIds = {bnInfAttr.x_tensor_uid(), bnInfAttr.y_tensor_uid()};
    const std::vector<int64_t> affineTensorIds
        = {bnInfAttr.scale_tensor_uid(), bnInfAttr.bias_tensor_uid()};
    const std::vector<int64_t> statTensorIds
        = {bnInfAttr.mean_tensor_uid(), bnInfAttr.inv_variance_tensor_uid()};

    checkTensorConfigSupported(ioTensorIds, affineTensorIds, statTensorIds, {}, false);
}

void BatchnormValidator::checkInferenceVarianceExtTensorConfigSupported(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt& bnInfAttr)
{
    const std::vector<int64_t> ioTensorIds = {bnInfAttr.x_tensor_uid(), bnInfAttr.y_tensor_uid()};
    const std::vector<int64_t> affineTensorIds
        = {bnInfAttr.scale_tensor_uid(), bnInfAttr.bias_tensor_uid()};
    const std::vector<int64_t> statTensorIds
        = {bnInfAttr.mean_tensor_uid(), bnInfAttr.variance_tensor_uid()};

    checkTensorConfigSupported(ioTensorIds, affineTensorIds, statTensorIds, {}, false);
}

void BatchnormValidator::checkInferenceActivationTensorConfigSupported(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr)
{
    checkFwdActivationModeSupported(actAttr);

    const std::vector<int64_t> ioTensorIds = {bnInfAttr.x_tensor_uid(), actAttr.out_0_tensor_uid()};
    const std::vector<int64_t> affineTensorIds
        = {bnInfAttr.scale_tensor_uid(), bnInfAttr.bias_tensor_uid()};
    const std::vector<int64_t> statTensorIds
        = {bnInfAttr.mean_tensor_uid(), bnInfAttr.inv_variance_tensor_uid()};
    const std::vector<int64_t> intermediateTensorIds
        = {bnInfAttr.y_tensor_uid(), actAttr.in_0_tensor_uid()};

    checkTensorConfigSupported(
        ioTensorIds, affineTensorIds, statTensorIds, intermediateTensorIds, false);
}

void BatchnormValidator::checkInferenceVarianceExtActivationTensorConfigSupported(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt& bnInfAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr)
{
    checkFwdActivationModeSupported(actAttr);

    const std::vector<int64_t> ioTensorIds = {bnInfAttr.x_tensor_uid(), actAttr.out_0_tensor_uid()};
    const std::vector<int64_t> affineTensorIds
        = {bnInfAttr.scale_tensor_uid(), bnInfAttr.bias_tensor_uid()};
    const std::vector<int64_t> statTensorIds
        = {bnInfAttr.mean_tensor_uid(), bnInfAttr.variance_tensor_uid()};
    const std::vector<int64_t> intermediateTensorIds
        = {bnInfAttr.y_tensor_uid(), actAttr.in_0_tensor_uid()};

    checkTensorConfigSupported(
        ioTensorIds, affineTensorIds, statTensorIds, intermediateTensorIds, false);
}

void BatchnormValidator::checkFwdTrainingTensorConfigSupported(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& bnAttr)
{
    if(bnAttr.peer_stats_tensor_uid() != nullptr && !bnAttr.peer_stats_tensor_uid()->empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm forward training does not support peer statistics");
    }

    const std::vector<int64_t> ioTensorIds = {bnAttr.x_tensor_uid(), bnAttr.y_tensor_uid()};
    const std::vector<int64_t> affineTensorIds
        = {bnAttr.scale_tensor_uid(), bnAttr.bias_tensor_uid()};
    std::vector<int64_t> statTensorIds;
    if(bnAttr.mean_tensor_uid().has_value())
    {
        statTensorIds.push_back(bnAttr.mean_tensor_uid().value());
    }
    if(bnAttr.inv_variance_tensor_uid().has_value())
    {
        statTensorIds.push_back(bnAttr.inv_variance_tensor_uid().value());
    }

    checkTensorConfigSupported(ioTensorIds, affineTensorIds, statTensorIds, {}, true);
}

void BatchnormValidator::checkFwdTrainingActivationTensorConfigSupported(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& bnAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr)
{
    checkFwdActivationModeSupported(actAttr);

    if(bnAttr.peer_stats_tensor_uid() != nullptr && !bnAttr.peer_stats_tensor_uid()->empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm forward training does not support peer statistics");
    }

    const std::vector<int64_t> ioTensorIds = {bnAttr.x_tensor_uid(), actAttr.out_0_tensor_uid()};
    const std::vector<int64_t> affineTensorIds
        = {bnAttr.scale_tensor_uid(), bnAttr.bias_tensor_uid()};
    std::vector<int64_t> statTensorIds;
    if(bnAttr.mean_tensor_uid().has_value())
    {
        statTensorIds.push_back(bnAttr.mean_tensor_uid().value());
    }
    if(bnAttr.inv_variance_tensor_uid().has_value())
    {
        statTensorIds.push_back(bnAttr.inv_variance_tensor_uid().value());
    }
    const std::vector<int64_t> intermediateTensorIds
        = {bnAttr.y_tensor_uid(), actAttr.in_0_tensor_uid()};

    checkTensorConfigSupported(
        ioTensorIds, affineTensorIds, statTensorIds, intermediateTensorIds, true);
}

void BatchnormValidator::checkBwdTensorConfigSupported(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormBackwardAttributes& bnBwdAttr)
{
    if(bnBwdAttr.peer_stats_tensor_uid() != nullptr && !bnBwdAttr.peer_stats_tensor_uid()->empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Batchnorm backward does not support peer statistics");
    }

    const std::vector<int64_t> ioTensorIds
        = {bnBwdAttr.x_tensor_uid(), bnBwdAttr.dy_tensor_uid(), bnBwdAttr.dx_tensor_uid()};
    const std::vector<int64_t> affineTensorIds = {
        bnBwdAttr.scale_tensor_uid(), bnBwdAttr.dscale_tensor_uid(), bnBwdAttr.dbias_tensor_uid()};
    std::vector<int64_t> statTensorIds;
    if(bnBwdAttr.mean_tensor_uid().has_value())
    {
        statTensorIds.push_back(bnBwdAttr.mean_tensor_uid().value());
    }
    if(bnBwdAttr.inv_variance_tensor_uid().has_value())
    {
        statTensorIds.push_back(bnBwdAttr.inv_variance_tensor_uid().value());
    }

    checkTensorConfigSupported(ioTensorIds, affineTensorIds, statTensorIds, {}, true);
}

void BatchnormValidator::checkBwdActivationTensorConfigSupported(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr,
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormBackwardAttributes& bnBwdAttr)
{
    checkBwdActivationModeSupported(actAttr);

    if(bnBwdAttr.peer_stats_tensor_uid() != nullptr && !bnBwdAttr.peer_stats_tensor_uid()->empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward fusion does not support peer statistics");
    }

    const auto actIn1Uid = actAttr.in_1_tensor_uid();
    if(!actIn1Uid.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation backward node must have a second input tensor (in_1)");
    }

    const std::vector<int64_t> ioTensorIds
        = {bnBwdAttr.x_tensor_uid(), actAttr.in_0_tensor_uid(), bnBwdAttr.dx_tensor_uid()};
    const std::vector<int64_t> affineTensorIds = {bnBwdAttr.scale_tensor_uid(),
                                                  bnBwdAttr.dscale_tensor_uid(),
                                                  bnBwdAttr.dbias_tensor_uid(),
                                                  bnInfAttr.bias_tensor_uid()};
    std::vector<int64_t> statTensorIds;
    if(bnBwdAttr.mean_tensor_uid().has_value())
    {
        statTensorIds.push_back(bnBwdAttr.mean_tensor_uid().value());
    }
    if(bnBwdAttr.inv_variance_tensor_uid().has_value())
    {
        statTensorIds.push_back(bnBwdAttr.inv_variance_tensor_uid().value());
    }
    const std::vector<int64_t> intermediateTensorIds = {bnInfAttr.y_tensor_uid(),
                                                        *actIn1Uid,
                                                        actAttr.out_0_tensor_uid(),
                                                        bnBwdAttr.dy_tensor_uid()};

    checkTensorConfigSupported(
        ioTensorIds, affineTensorIds, statTensorIds, intermediateTensorIds, true);
}

// --- Activation Mode Validators ---

namespace
{

void checkActivationModeSupported(
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& activAttr, bool isBwd)
{
    // hip-kernel-provider batchnorm supports: PASSTHRU, RELU, CLIPPEDRELU, CLAMP (no Leaky ReLU)

    if(activAttr.operation() == hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::IDENTITY)
    {
        return;
    }

    if(activAttr.operation()
       == (isBwd ? hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD
                 : hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD))
    {
        if(!activAttr.relu_lower_clip_slope())
        {
            return;
        }
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm fused activation does not support Leaky ReLU.");
    }

    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Unsupported activation mode for batchnorm fusion.");
}

} // namespace

void BatchnormValidator::checkFwdActivationModeSupported(
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& activAttr)
{
    checkActivationModeSupported(activAttr, false);
}

void BatchnormValidator::checkBwdActivationModeSupported(
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& activAttr)
{
    checkActivationModeSupported(activAttr, true);
}

} // namespace hip_kernel_provider
