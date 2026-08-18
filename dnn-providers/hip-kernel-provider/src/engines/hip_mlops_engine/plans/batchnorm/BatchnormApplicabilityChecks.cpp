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

void BatchnormValidator::checkTensorLayoutsAndDimsSupported(const std::vector<int64_t>& tensorIds)
{
    // Skip tensors with embedded scalar values (epsilon, momentum) - they don't have layouts or dimensions to validate
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

    validatePackedTensors(tensors);
    validateConsistentLayouts(tensors);
}

void BatchnormValidator::checkTensorDataTypesSupported(
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
        const auto& ioTensorAttr = core::utils::findTensorAttributes(_tensorMap, ioTensorId);
        validateDataTypeIsSupported(ioTensorAttr.data_type(),
                                    allowedIOTypes,
                                    "Batchnorm implementation supports only FLOAT, HALF, and "
                                    "BFLOAT16 data types for x, y, tensors");
    }

    const std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> allowedAffineTypes{
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT};
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

    const std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> allowedStatTypes{
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT};
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

    const std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType>
        allowedIntermediateTypes{hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT};
    if(allowedStatTypes.size() == 1)
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

    // Validate consistent dimensions for IO and stat tensors (affine tensors can have fewer dimensions due to broadcasting)
    std::vector<TensorDescriptor> ioStatTensors;
    ioStatTensors.reserve(ioTensorIds.size() + statTensorIds.size());
    for(const auto& tensorId : ioTensorIds)
    {
        const auto& tensorAttr = _tensorMap.at(tensorId);
        ioStatTensors.emplace_back(tensorAttr);
    }
    for(const auto& tensorId : statTensorIds)
    {
        const auto& tensorAttr = _tensorMap.at(tensorId);
        ioStatTensors.emplace_back(tensorAttr);
    }
    validateConsistentDimensions(ioStatTensors);

    const auto& ioTensorAttr = core::utils::findTensorAttributes(_tensorMap, ioTensorIds[0]);
    const std::vector<int64_t> ioDims(ioTensorAttr.dims()->begin(), ioTensorAttr.dims()->end());

    validateConsistentShapes(
        ioTensorIds, ioDims, "All IO tensors for batchnorm must have the same shape.");

    // Check that scale and bias tensors must have the same shape
    if(!affineTensorIds.empty())
    {
        const auto& referenceAffineAttr
            = core::utils::findTensorAttributes(_tensorMap, affineTensorIds[0]);
        const std::vector<int64_t> referenceAffineDims(referenceAffineAttr.dims()->begin(),
                                                       referenceAffineAttr.dims()->end());

        validateConsistentShapes(affineTensorIds,
                                 referenceAffineDims,
                                 "Scale and bias tensors for batchnorm must have the same shape.");
    }

    // Checks if the affine tensor shape is valid for broadcasting to the IO tensor shape as per
    // NumPy broadcasting rules with the constraint that the channel dimension must match.
    const auto isValidAffineShapeForIo
        = [&](const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& affineTensorAttr) {
              const std::vector<int64_t> affineDims(affineTensorAttr.dims()->begin(),
                                                    affineTensorAttr.dims()->end());

              const size_t affineRank = affineDims.size();
              const size_t ioRank = ioDims.size();

              if(affineRank < 1 || affineRank > ioRank)
              {
                  return false;
              }

              const int64_t numChannels = ioDims[1];
              const bool ioIsChannelLast = core::utils::isChannelLastLayout(&ioTensorAttr);

              // Checking against the IO tensor's layout implicitly rejects affine tensors
              // whose dimensions don't satisfy the IO layout's expected channel position.
              if(ioIsChannelLast)
              {
                  // For channel-last, C is at index 0 for reduced rank or index 1 for full rank
                  const size_t channelDimIndex = affineRank == ioRank ? 1 : 0;

                  if(affineDims[channelDimIndex] != numChannels)
                  {
                      return false;
                  }

                  for(size_t i = 0; i < affineRank; ++i)
                  {
                      if(i != channelDimIndex && affineDims[i] != 1)
                      {
                          return false;
                      }
                  }
              }
              else
              {
                  // For channel-first, only full rank or reduced rank are allowed
                  if(affineRank != ioRank && affineRank != ioRank - 1)
                  {
                      return false;
                  }

                  // C is at index 1 for full rank and index 0 for reduced rank
                  const size_t channelDimIndex = affineRank == ioRank ? 1 : 0;

                  if(affineDims[channelDimIndex] != numChannels)
                  {
                      return false;
                  }

                  for(size_t i = 0; i < affineRank; ++i)
                  {
                      if(i != channelDimIndex && affineDims[i] != 1)
                      {
                          return false;
                      }
                  }
              }

              return true;
          };

    // Check that scale and bias tensors have any broadcastable shape with same number of channels as IO tensors
    for(const auto& tensorId : affineTensorIds)
    {
        const auto& tensorAttr = core::utils::findTensorAttributes(_tensorMap, tensorId);

        if(!isValidAffineShapeForIo(tensorAttr))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "BatchNorm affine tensor shape "
                    + hipdnn_data_sdk::utilities::vecToString(
                        std::vector<int64_t>(tensorAttr.dims()->begin(), tensorAttr.dims()->end()))
                    + " is incompatible with the IO tensor shape "
                    + hipdnn_data_sdk::utilities::vecToString(ioDims)
                    + ". The affine tensor shape must be broadcastable to the IO tensor shape with "
                      "matching channel dimension.");
        }
    }

    const auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(ioDims);
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
    std::vector<int64_t> allTensors = std::vector<int64_t>(ioTensorIds.begin(), ioTensorIds.end());
    allTensors.insert(allTensors.end(), affineTensorIds.begin(), affineTensorIds.end());
    allTensors.insert(allTensors.end(), statTensorIds.begin(), statTensorIds.end());
    allTensors.insert(allTensors.end(), intermediateTensorIds.begin(), intermediateTensorIds.end());

    checkTensorLayoutsAndDimsSupported(allTensors);
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
