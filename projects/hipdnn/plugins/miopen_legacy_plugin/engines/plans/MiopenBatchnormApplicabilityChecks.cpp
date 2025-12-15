// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <numeric>
#include <vector>

#include <hipdnn_sdk/plugin/PluginException.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

#include "MiopenBatchnormApplicabilityChecks.hpp"
#include "MiopenUtils.hpp"

namespace miopen_legacy_plugin
{

namespace
{

void checkTensorLayoutsAndDimsSupported(
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap)
{
    constexpr size_t MIN_SUPPORTED_DIMS = 4;
    constexpr size_t MAX_SUPPORTED_DIMS = 5;
    size_t numDims = 0; // Not set
    std::vector<int64_t> strideOrder;

    for(const auto& tensorPair : tensorMap)
    {
        const hipdnn_sdk::data_objects::TensorAttributes* tensorAttr = tensorPair.second;

        // All tensors must have the same number of dimensions
        if(numDims == 0)
        {
            numDims = tensorAttr->dims()->size();
            if(numDims < MIN_SUPPORTED_DIMS || numDims > MAX_SUPPORTED_DIMS)
            {
                throw hipdnn_plugin::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "Batchnorm implementation supports only 4D or 5D tensors.");
            }
        }
        else
        {
            if(tensorAttr->dims()->size() != numDims)
            {
                throw hipdnn_plugin::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "All tensors for batchnorm must have the same number of dimensions.");
            }
        }

        const std::vector<int64_t> dims(tensorAttr->dims()->begin(), tensorAttr->dims()->end());
        const std::vector<int64_t> strides(tensorAttr->strides()->begin(),
                                           tensorAttr->strides()->end());

        // MIOpen only supports packed tensors for batch normalization
        if(!hipdnn_sdk::utilities::isTensorPacked(dims, strides))
        {
            throw hipdnn_plugin::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm implementation supports only packed tensors.");
        }

        // MIOpen only supports NCHW, NCDHW, NHWC, and NDHWC layouts for batch normalization
        const auto currentStrideOrder = hipdnn_sdk::utilities::extractStrideOrder(strides);
        if(strideOrder.empty())
        {
            if(numDims == 4)
            {
                const auto layoutNchw = hipdnn_sdk::utilities::TensorLayout::NCHW;
                const auto layoutNhwc = hipdnn_sdk::utilities::TensorLayout::NHWC;
                if(!(currentStrideOrder == layoutNchw.strideOrder
                     || currentStrideOrder == layoutNhwc.strideOrder))
                {
                    throw hipdnn_plugin::HipdnnPluginException(
                        HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                        "Batchnorm implementation supports only NCHW and NHWC layouts for 4D "
                        "tensors.");
                }
            }
            else // numDims == 5
            {
                const auto layoutNcdhw = hipdnn_sdk::utilities::TensorLayout::NCDHW;
                const auto layoutNdhwc = hipdnn_sdk::utilities::TensorLayout::NDHWC;
                if(!(currentStrideOrder == layoutNcdhw.strideOrder
                     || currentStrideOrder == layoutNdhwc.strideOrder))
                {
                    throw hipdnn_plugin::HipdnnPluginException(
                        HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                        "Batchnorm implementation supports only NCDHW and NDHWC layouts for 5D "
                        "tensors.");
                }
            }
            strideOrder = currentStrideOrder;
        }
        else
        {
            if(currentStrideOrder != strideOrder)
            {
                throw hipdnn_plugin::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "All tensors for batchnorm must have the same layout.");
            }
        }
    }
}

void checkTensorDataTypesSupported(
    const std::vector<int64_t>& ioTensorIds,
    const std::vector<int64_t>& affineTensorIds,
    const std::vector<int64_t>& statTensorIds,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap)
{
    // MIOpen only supports FLOAT, HALF, and BFLOAT16 data types for x, y, dy, and dx tensors
    // All tensors must have the same data type
    hipdnn_sdk::data_objects::DataType ioDataType = hipdnn_sdk::data_objects::DataType::UNSET;
    for(const auto tensorId : ioTensorIds)
    {
        const auto& tensorAttr = miopen_utils::findTensorAttributes(tensorMap, tensorId);
        if(ioDataType == hipdnn_sdk::data_objects::DataType::UNSET)
        {
            ioDataType = tensorAttr.data_type();
            if(ioDataType != hipdnn_sdk::data_objects::DataType::FLOAT
               && ioDataType != hipdnn_sdk::data_objects::DataType::HALF
               && ioDataType != hipdnn_sdk::data_objects::DataType::BFLOAT16)
            {
                throw hipdnn_plugin::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "Batchnorm implementation supports only FLOAT, HALF, and BFLOAT16 data types "
                    "for x, y, dy, and dx tensors.");
            }
        }
        else
        {
            if(tensorAttr.data_type() != ioDataType)
            {
                throw hipdnn_plugin::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "All IO tensors for batchnorm must have the same data type.");
            }
        }
    }

    // MIOpen only supports FLOAT data type for scale and bias tensors
    for(const auto tensorId : affineTensorIds)
    {
        const auto& tensorAttr = miopen_utils::findTensorAttributes(tensorMap, tensorId);
        if(tensorAttr.data_type() != hipdnn_sdk::data_objects::DataType::FLOAT)
        {
            throw hipdnn_plugin::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm implementation supports only FLOAT data type for scale and bias "
                "tensors.");
        }
    }

    // MIOpen only supports FLOAT data type for mean and variance tensors
    for(const auto tensorId : statTensorIds)
    {
        const auto& tensorAttr = miopen_utils::findTensorAttributes(tensorMap, tensorId);
        if(tensorAttr.data_type() != hipdnn_sdk::data_objects::DataType::FLOAT)
        {
            throw hipdnn_plugin::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm implementation supports only FLOAT data type for mean and variance "
                "tensors.");
        }
    }
}

void checkTensorShapesSupported(
    const std::vector<int64_t>& ioTensorIds,
    const std::vector<int64_t>& affineTensorIds,
    const std::vector<int64_t>& statTensorIds,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap,
    bool isTraining)
{
    if(ioTensorIds.empty())
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "At least one IO tensor must be provided for batchnorm.");
    }

    // All IO tensors must have the same shape
    const auto& ioTensorAttr = miopen_utils::findTensorAttributes(tensorMap, ioTensorIds[0]);
    const auto ioDims = std::vector(ioTensorAttr.dims()->begin(), ioTensorAttr.dims()->end());
    for(size_t i = 1; i < ioTensorIds.size(); ++i)
    {
        const auto& tensorAttr = miopen_utils::findTensorAttributes(tensorMap, ioTensorIds[i]);
        const auto dims = std::vector(tensorAttr.dims()->begin(), tensorAttr.dims()->end());
        if(dims != ioDims)
        {
            throw hipdnn_plugin::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "All IO tensors for batchnorm must have the same shape.");
        }
    }

    const auto derivedDims = hipdnn_sdk::utilities::getDerivedShape(ioDims);
    // Scale and bias tensors must have shape derived from IO tensor shape
    for(const auto tensorId : affineTensorIds)
    {
        const auto& tensorAttr = miopen_utils::findTensorAttributes(tensorMap, tensorId);
        const auto dims = std::vector(tensorAttr.dims()->begin(), tensorAttr.dims()->end());
        if(dims != derivedDims)
        {
            throw hipdnn_plugin::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Scale and bias tensors for batchnorm must "
                                                       "have shape derived from IO tensor shape.");
        }
    }

    // Mean and variance tensors must have shape derived from IO tensor shape
    for(const auto tensorId : statTensorIds)
    {
        const auto& tensorAttr = miopen_utils::findTensorAttributes(tensorMap, tensorId);
        const auto dims = std::vector(tensorAttr.dims()->begin(), tensorAttr.dims()->end());
        if(dims != derivedDims)
        {
            throw hipdnn_plugin::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Mean and variance tensors for batchnorm must have shape derived from IO tensor "
                "shape.");
        }
    }

    if(isTraining)
    {
        // For Spatial BN: need N*spatial_size > 1 to compute variance
        if(ioDims.size() < 3)
        {
            throw hipdnn_plugin::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "IO tensor must have at least 3 dimensions for batchnorm.");
        }
        const auto spatialSize
            = std::accumulate(ioDims.begin() + 2, ioDims.end(), int64_t{1}, std::multiplies<>());
        if(ioDims[0] * spatialSize <= 1)
        {
            throw hipdnn_plugin::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "The product of the batch size and spatial dimensions must be greater than 1 for "
                "batchnorm.");
        }
    }
}

void checkBatchnormTensorConfigSupported(
    const std::vector<int64_t>& ioTensorIds,
    const std::vector<int64_t>& affineTensorIds,
    const std::vector<int64_t>& statTensorIds,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap,
    bool isTraining)
{
    checkTensorLayoutsAndDimsSupported(tensorMap);
    checkTensorDataTypesSupported(ioTensorIds, affineTensorIds, statTensorIds, tensorMap);
    checkTensorShapesSupported(ioTensorIds, affineTensorIds, statTensorIds, tensorMap, isTraining);
}

} // namespace

void checkBatchnormTensorConfigSupported(
    const hipdnn_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap)
{
    std::vector<int64_t> ioTensorIds = {bnInfAttr.x_tensor_uid(), bnInfAttr.y_tensor_uid()};
    std::vector<int64_t> affineTensorIds
        = {bnInfAttr.scale_tensor_uid(), bnInfAttr.bias_tensor_uid()};
    std::vector<int64_t> statTensorIds
        = {bnInfAttr.mean_tensor_uid(), bnInfAttr.inv_variance_tensor_uid()};

    checkBatchnormTensorConfigSupported(
        ioTensorIds, affineTensorIds, statTensorIds, tensorMap, false);
}

void checkBatchnormTensorConfigSupported(
    const hipdnn_sdk::data_objects::BatchnormAttributes& bnAttr,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap)
{
    std::vector<int64_t> ioTensorIds = {bnAttr.x_tensor_uid(), bnAttr.y_tensor_uid()};
    std::vector<int64_t> affineTensorIds = {bnAttr.scale_tensor_uid(), bnAttr.bias_tensor_uid()};
    std::vector<int64_t> statTensorIds;
    if(bnAttr.mean_tensor_uid().has_value())
    {
        statTensorIds.push_back(bnAttr.mean_tensor_uid().value());
    }
    if(bnAttr.inv_variance_tensor_uid().has_value())
    {
        statTensorIds.push_back(bnAttr.inv_variance_tensor_uid().value());
    }

    checkBatchnormTensorConfigSupported(
        ioTensorIds, affineTensorIds, statTensorIds, tensorMap, true);
}

void checkBatchnormTensorConfigSupported(
    const hipdnn_sdk::data_objects::BatchnormBackwardAttributes& bnBwdAttr,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap)
{
    std::vector<int64_t> ioTensorIds
        = {bnBwdAttr.x_tensor_uid(), bnBwdAttr.dy_tensor_uid(), bnBwdAttr.dx_tensor_uid()};
    std::vector<int64_t> affineTensorIds = {
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

    checkBatchnormTensorConfigSupported(
        ioTensorIds, affineTensorIds, statTensorIds, tensorMap, true);
}

void checkBatchnormTensorConfigSupported(
    const hipdnn_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const hipdnn_sdk::data_objects::PointwiseAttributes& actAttr,
    const hipdnn_sdk::data_objects::BatchnormBackwardAttributes& bnBwdAttr,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap)
{
    std::vector<int64_t> ioTensorIds = {bnBwdAttr.x_tensor_uid(),
                                        actAttr.in_1_tensor_uid().value(), // dy
                                        bnBwdAttr.dx_tensor_uid()};
    std::vector<int64_t> affineTensorIds = {bnBwdAttr.scale_tensor_uid(),
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

    checkBatchnormTensorConfigSupported(
        ioTensorIds, affineTensorIds, statTensorIds, tensorMap, true);
}

void checkBatchnormFwdActivationModeSupported(
    const hipdnn_sdk::data_objects::PointwiseAttributes& activAttr, bool isBwd)
{
    // MIOpen currently only supports miopenActivationPASTHRU, miopenActivationRELU,
    // miopenActivationCLIPPEDRELU and miopenActivationCLAMP for batchnorm fusions

    if(activAttr.operation() == hipdnn_sdk::data_objects::PointwiseMode::IDENTITY)
    {
        // miopenActivationPASTHRU
        return;
    }

    if(activAttr.operation()
       == (isBwd ? hipdnn_sdk::data_objects::PointwiseMode::RELU_BWD
                 : hipdnn_sdk::data_objects::PointwiseMode::RELU_FWD))
    {
        // miopenActivationRELU - Standard ReLU (no parameters)
        // miopenActivationCLIPPEDRELU - Clipped ReLU (relu_upper_clip only)
        // miopenActivationCLAMP - CLAMP (relu_lower_clip + relu_upper_clip)
        // miopenActivationLEAKYRELU - Leaky ReLU is not supported!
        if(!activAttr.relu_lower_clip_slope())
        {
            return;
        }
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm fused activation does not support Leaky ReLU.");
    }

    throw hipdnn_plugin::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                               "Unsupported activation mode for batchnorm fusion.");
}

void checkBatchnormFwdActivationModeSupported(
    const hipdnn_sdk::data_objects::PointwiseAttributes& activAttr)
{
    checkBatchnormFwdActivationModeSupported(activAttr, false);
}

void checkBatchnormBwdActivationModeSupported(
    const hipdnn_sdk::data_objects::PointwiseAttributes& activAttr)
{
    checkBatchnormFwdActivationModeSupported(activAttr, true);
}

} // namespace miopen_legacy_plugin
