// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <numeric>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>

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

void BatchnormValidator::validateDimensionCount(size_t numDims)
{
    constexpr size_t MIN_SUPPORTED_DIMS = 3;
    constexpr size_t MAX_SUPPORTED_DIMS = 5;

    if(numDims < MIN_SUPPORTED_DIMS || numDims > MAX_SUPPORTED_DIMS)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Only 3D, 4D or 5D tensors supported.");
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
    validateElementCountFitsUint32(
        ioDims,
        "Batchnorm kernels size and index tensors with 32-bit unsigned arithmetic; the IO "
        "tensor has too many elements.");

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

    auto xDataType
        = core::utils::findTensorAttributes(_tensorMap, bnInfAttr.x_tensor_uid()).data_type();
    auto yDataType
        = core::utils::findTensorAttributes(_tensorMap, bnInfAttr.y_tensor_uid()).data_type();
    if(xDataType != yDataType && yDataType != hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm inference requires that x and y are the same type or that y is of type "
            "float");
    }
}

void BatchnormValidator::checkInferenceVarianceExtTensorConfigSupported(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt& bnInfAttr)
{
    const std::vector<int64_t> ioTensorIds = {bnInfAttr.x_tensor_uid(), bnInfAttr.y_tensor_uid()};
    const std::vector<int64_t> affineTensorIds
        = {bnInfAttr.scale_tensor_uid(), bnInfAttr.bias_tensor_uid()};
    const std::vector<int64_t> statTensorIds
        = {bnInfAttr.mean_tensor_uid(), bnInfAttr.variance_tensor_uid()};

    checkEpsilonSupported(bnInfAttr.epsilon_tensor_uid());

    checkTensorConfigSupported(ioTensorIds, affineTensorIds, statTensorIds, {}, false);

    auto xDataType
        = core::utils::findTensorAttributes(_tensorMap, bnInfAttr.x_tensor_uid()).data_type();
    auto yDataType
        = core::utils::findTensorAttributes(_tensorMap, bnInfAttr.y_tensor_uid()).data_type();
    if(xDataType != yDataType && yDataType != hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm inference requires that x and y are the same type or that y is of type "
            "float");
    }
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

    auto xDataType
        = core::utils::findTensorAttributes(_tensorMap, bnInfAttr.x_tensor_uid()).data_type();
    auto yDataType
        = core::utils::findTensorAttributes(_tensorMap, actAttr.out_0_tensor_uid()).data_type();
    if(xDataType != yDataType && yDataType != hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm inference requires that x and y are the same type or that y is of type "
            "float");
    }
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

    checkEpsilonSupported(bnInfAttr.epsilon_tensor_uid());
    const std::vector<int64_t> intermediateTensorIds
        = {bnInfAttr.y_tensor_uid(), actAttr.in_0_tensor_uid()};

    checkTensorConfigSupported(
        ioTensorIds, affineTensorIds, statTensorIds, intermediateTensorIds, false);

    auto xDataType
        = core::utils::findTensorAttributes(_tensorMap, bnInfAttr.x_tensor_uid()).data_type();
    auto yDataType
        = core::utils::findTensorAttributes(_tensorMap, actAttr.out_0_tensor_uid()).data_type();
    if(xDataType != yDataType && yDataType != hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm inference requires that x and y are the same type or that y is of type "
            "float");
    }
}

namespace
{
using BatchnormAttrs = hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes;

// The training plan updates running statistics only when all five operands are
// present; any other combination would drop them silently.
void checkRunningStatsComplete(const BatchnormAttrs& bnAttr)
{
    const int present = static_cast<int>(bnAttr.prev_running_mean_tensor_uid().has_value())
                        + static_cast<int>(bnAttr.prev_running_variance_tensor_uid().has_value())
                        + static_cast<int>(bnAttr.momentum_tensor_uid().has_value())
                        + static_cast<int>(bnAttr.next_running_mean_tensor_uid().has_value())
                        + static_cast<int>(bnAttr.next_running_variance_tensor_uid().has_value());
    if(present != 0 && present != 5)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm forward training requires prev_running_mean, prev_running_variance, "
            "momentum, next_running_mean and next_running_variance together or not at all.");
    }
}

// When the plan updates running statistics it reads prev and writes next with the saved-statistic
// layout, so the four statistic operands are validated like the saved ones (FLOAT, packed, the
// shape derived from the IO tensor); momentum is resolved as a scalar operand at plan build.
void addRunningStats(
    const BatchnormAttrs& bnAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    std::vector<int64_t>& statTensorIds)
{
    if(!bnAttr.prev_running_mean_tensor_uid().has_value())
    {
        return;
    }
    statTensorIds.push_back(bnAttr.prev_running_mean_tensor_uid().value());
    statTensorIds.push_back(bnAttr.prev_running_variance_tensor_uid().value());
    statTensorIds.push_back(bnAttr.next_running_mean_tensor_uid().value());
    statTensorIds.push_back(bnAttr.next_running_variance_tensor_uid().value());
    hipdnn_plugin_sdk::makeScalarOperand(
        tensorMap, bnAttr.momentum_tensor_uid().value(), "Momentum");
}

// The plan writes saved statistics only when both saved mean and saved inv_variance
// are present, so exactly one of them would never be written.
void checkSavedStatsPair(const BatchnormAttrs& bnAttr)
{
    if(bnAttr.mean_tensor_uid().has_value() != bnAttr.inv_variance_tensor_uid().has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm forward training requires saved mean and saved inv_variance together or "
            "not at all.");
    }
}
} // namespace

namespace batchnorm
{

bool hasUnrequestedSavedPair(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& bnAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    if(!bnAttr.mean_tensor_uid().has_value() || !bnAttr.inv_variance_tensor_uid().has_value())
    {
        return false;
    }

    if(bnAttr.prev_running_mean_tensor_uid().has_value()
       || bnAttr.prev_running_variance_tensor_uid().has_value()
       || bnAttr.next_running_mean_tensor_uid().has_value()
       || bnAttr.next_running_variance_tensor_uid().has_value()
       || bnAttr.momentum_tensor_uid().has_value())
    {
        return false;
    }

    return core::utils::findTensorAttributes(tensorMap, bnAttr.mean_tensor_uid().value()).virtual_()
           && core::utils::findTensorAttributes(tensorMap, bnAttr.inv_variance_tensor_uid().value())
                  .virtual_();
}

} // namespace batchnorm

void BatchnormValidator::checkFwdTrainingTensorConfigSupported(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& bnAttr)
{
    if(bnAttr.peer_stats_tensor_uid() != nullptr && !bnAttr.peer_stats_tensor_uid()->empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm forward training does not support peer statistics");
    }

    checkRunningStatsComplete(bnAttr);
    checkSavedStatsPair(bnAttr);
    checkEpsilonSupported(bnAttr.epsilon_tensor_uid());

    const std::vector<int64_t> ioTensorIds = {bnAttr.x_tensor_uid(), bnAttr.y_tensor_uid()};
    const std::vector<int64_t> affineTensorIds
        = {bnAttr.scale_tensor_uid(), bnAttr.bias_tensor_uid()};
    // A saved pair that is virtual in both tensors is not requested: the plan never reads or
    // writes it, so the role-specific checks skip it.
    const bool unrequestedSavedPair = batchnorm::hasUnrequestedSavedPair(bnAttr, _tensorMap);
    std::vector<int64_t> statTensorIds;
    if(!unrequestedSavedPair)
    {
        if(bnAttr.mean_tensor_uid().has_value())
        {
            statTensorIds.push_back(bnAttr.mean_tensor_uid().value());
        }
        if(bnAttr.inv_variance_tensor_uid().has_value())
        {
            statTensorIds.push_back(bnAttr.inv_variance_tensor_uid().value());
        }
    }

    addRunningStats(bnAttr, _tensorMap, statTensorIds);

    checkTensorConfigSupported(ioTensorIds, affineTensorIds, statTensorIds, {}, true);

    auto xDataType
        = core::utils::findTensorAttributes(_tensorMap, bnAttr.x_tensor_uid()).data_type();
    auto yDataType
        = core::utils::findTensorAttributes(_tensorMap, bnAttr.y_tensor_uid()).data_type();
    if(xDataType != yDataType && yDataType != hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm forward training requires that x and y are the same type or that y is of "
            "type float");
    }
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

    checkRunningStatsComplete(bnAttr);
    checkSavedStatsPair(bnAttr);
    checkEpsilonSupported(bnAttr.epsilon_tensor_uid());

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

    addRunningStats(bnAttr, _tensorMap, statTensorIds);

    checkTensorConfigSupported(
        ioTensorIds, affineTensorIds, statTensorIds, intermediateTensorIds, true);

    auto xDataType
        = core::utils::findTensorAttributes(_tensorMap, bnAttr.x_tensor_uid()).data_type();
    auto yDataType
        = core::utils::findTensorAttributes(_tensorMap, actAttr.out_0_tensor_uid()).data_type();
    if(xDataType != yDataType && yDataType != hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm forward training requires that x and y are the same type or that y is of "
            "type float");
    }
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

    auto xDataType
        = core::utils::findTensorAttributes(_tensorMap, bnBwdAttr.x_tensor_uid()).data_type();
    auto dyDataType
        = core::utils::findTensorAttributes(_tensorMap, bnBwdAttr.dy_tensor_uid()).data_type();
    auto dxDataType
        = core::utils::findTensorAttributes(_tensorMap, bnBwdAttr.dx_tensor_uid()).data_type();
    if(xDataType != dxDataType)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward requires that x and dx are the same type");
    }
    if(xDataType != dyDataType
       && dyDataType != hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward requires that x and dy are the same type or that dy is of type "
            "float");
    }
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

    auto xDataType
        = core::utils::findTensorAttributes(_tensorMap, bnBwdAttr.x_tensor_uid()).data_type();
    auto dyDataType
        = core::utils::findTensorAttributes(_tensorMap, actAttr.in_0_tensor_uid()).data_type();
    auto dxDataType
        = core::utils::findTensorAttributes(_tensorMap, bnBwdAttr.dx_tensor_uid()).data_type();
    if(xDataType != dxDataType)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward requires that x and dx are the same type");
    }
    if(xDataType != dyDataType
       && dyDataType != hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward requires that x and dy are the same type or that dy is of type "
            "float");
    }
}

// --- Activation Mode Validators ---

void BatchnormValidator::checkFwdActivationModeSupported(
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& activAttr)
{
    checkNormalizationActivationModeSupported(activAttr, false, "Batchnorm");
}

void BatchnormValidator::checkBwdActivationModeSupported(
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& activAttr)
{
    checkNormalizationActivationModeSupported(activAttr, true, "Batchnorm");
}

namespace batchnorm
{

void checkOneNodeOperandsNotVirtual(
    const hipdnn_flatbuffers_sdk::data_objects::Node& node,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    using hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;

    std::vector<int64_t> uids;
    switch(node.attributes_type())
    {
    case NodeAttributes::BatchnormAttributes:
    {
        const auto& attr = *node.attributes_as_BatchnormAttributes();
        uids = {attr.x_tensor_uid(),
                attr.y_tensor_uid(),
                attr.scale_tensor_uid(),
                attr.bias_tensor_uid()};
        // A saved pair that is virtual in both tensors is not requested and is left out.
        if(!hasUnrequestedSavedPair(attr, tensorMap))
        {
            for(const auto& savedUid : {attr.mean_tensor_uid(), attr.inv_variance_tensor_uid()})
            {
                if(savedUid.has_value())
                {
                    uids.push_back(savedUid.value());
                }
            }
        }
        for(const auto& optionalUid : {attr.prev_running_mean_tensor_uid(),
                                       attr.prev_running_variance_tensor_uid(),
                                       attr.next_running_mean_tensor_uid(),
                                       attr.next_running_variance_tensor_uid()})
        {
            if(optionalUid.has_value())
            {
                uids.push_back(optionalUid.value());
            }
        }
        break;
    }
    case NodeAttributes::BatchnormInferenceAttributes:
    {
        const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();
        uids = {attr.x_tensor_uid(),
                attr.mean_tensor_uid(),
                attr.inv_variance_tensor_uid(),
                attr.scale_tensor_uid(),
                attr.bias_tensor_uid(),
                attr.y_tensor_uid()};
        break;
    }
    case NodeAttributes::BatchnormInferenceAttributesVarianceExt:
    {
        const auto& attr = *node.attributes_as_BatchnormInferenceAttributesVarianceExt();
        uids = {attr.x_tensor_uid(),
                attr.mean_tensor_uid(),
                attr.variance_tensor_uid(),
                attr.scale_tensor_uid(),
                attr.bias_tensor_uid(),
                attr.y_tensor_uid()};
        break;
    }
    case NodeAttributes::BatchnormBackwardAttributes:
    {
        const auto& attr = *node.attributes_as_BatchnormBackwardAttributes();
        uids = {attr.dy_tensor_uid(),
                attr.x_tensor_uid(),
                attr.scale_tensor_uid(),
                attr.dx_tensor_uid(),
                attr.dscale_tensor_uid(),
                attr.dbias_tensor_uid()};
        for(const auto& optionalUid : {attr.mean_tensor_uid(), attr.inv_variance_tensor_uid()})
        {
            if(optionalUid.has_value())
            {
                uids.push_back(optionalUid.value());
            }
        }
        break;
    }
    default:
        return;
    }

    for(const auto uid : uids)
    {
        if(core::utils::findTensorAttributes(tensorMap, uid).virtual_())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm tensors must be non-virtual for a one-node graph");
        }
    }
}

void checkInferenceActivationTensors(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    // in_0 must be the batchnorm inference output (forward path)
    if(actAttr.in_0_tensor_uid() != bnInfAttr.y_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation in_0 must be the batchnorm inference output tensor (y)");
    }

    // Check for virtual tensors
    const auto& bnInfTensorX
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.x_tensor_uid());
    const auto& bnInfTensorMean
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.mean_tensor_uid());
    const auto& bnInfTensorInvVar
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.inv_variance_tensor_uid());
    const auto& bnInfTensorScale
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.scale_tensor_uid());
    const auto& bnInfTensorBias
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.bias_tensor_uid());
    const auto& bnInfTensorY
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.y_tensor_uid());

    if(bnInfTensorX.virtual_() || bnInfTensorMean.virtual_() || bnInfTensorInvVar.virtual_()
       || bnInfTensorScale.virtual_() || bnInfTensorBias.virtual_() || !bnInfTensorY.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm inference input tensors must be non-virtual, output tensor must be virtual");
    }

    const auto& actTensorIn0
        = core::utils::findTensorAttributes(tensorMap, actAttr.in_0_tensor_uid());
    const auto& actTensorOut
        = core::utils::findTensorAttributes(tensorMap, actAttr.out_0_tensor_uid());

    if(!actTensorIn0.virtual_() || actTensorOut.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation input from batchnorm must be virtual, output must be non virtual");
    }
}

void checkInferenceVarianceExtActivationTensors(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt& bnInfAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    // in_0 must be the batchnorm inference output (forward path)
    if(actAttr.in_0_tensor_uid() != bnInfAttr.y_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation in_0 must be the batchnorm inference output tensor (y)");
    }

    // Check for virtual tensors
    const auto& bnInfTensorX
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.x_tensor_uid());
    const auto& bnInfTensorMean
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.mean_tensor_uid());
    const auto& bnInfTensorVariance
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.variance_tensor_uid());
    const auto& bnInfTensorScale
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.scale_tensor_uid());
    const auto& bnInfTensorBias
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.bias_tensor_uid());
    const auto& bnInfTensorY
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.y_tensor_uid());

    if(bnInfTensorX.virtual_() || bnInfTensorMean.virtual_() || bnInfTensorVariance.virtual_()
       || bnInfTensorScale.virtual_() || bnInfTensorBias.virtual_() || !bnInfTensorY.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm inference input tensors must be non-virtual, output tensor must be virtual");
    }

    const auto& actTensorIn0
        = core::utils::findTensorAttributes(tensorMap, actAttr.in_0_tensor_uid());
    const auto& actTensorOut
        = core::utils::findTensorAttributes(tensorMap, actAttr.out_0_tensor_uid());

    if(!actTensorIn0.virtual_() || actTensorOut.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation input from batchnorm must be virtual, output must be non virtual");
    }
}

namespace
{

bool isNodeActivFwd(const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& attr)
{
    using PointwiseMode = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

    // Check if operation is supported for batchnorm fusion
    switch(attr.operation())
    {
    case PointwiseMode::IDENTITY:
        return true;
    case PointwiseMode::RELU_FWD:
        break; // Continue to parameter validation
    default:
        return false;
    }

    // Batchnorm training fusion supports:
    // - Standard ReLU (no parameters)
    // - Clipped ReLU (relu_upper_clip only)
    // - CLAMP (relu_lower_clip + relu_upper_clip)
    // But does NOT support Leaky ReLU (relu_lower_clip_slope)
    return !attr.relu_lower_clip_slope();
}

void checkRunningStatisticsTensorVirtuality(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& bnAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    // Optional running statistics tensors must be non-virtual if present
    if(bnAttr.prev_running_mean_tensor_uid().has_value())
    {
        const auto& bnTensorAttrPrevRunningMean = core::utils::findTensorAttributes(
            tensorMap, bnAttr.prev_running_mean_tensor_uid().value());
        if(bnTensorAttrPrevRunningMean.virtual_())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm prev_running_mean tensor must be non-virtual");
        }
    }

    if(bnAttr.prev_running_variance_tensor_uid().has_value())
    {
        const auto& bnTensorAttrPrevRunningVar = core::utils::findTensorAttributes(
            tensorMap, bnAttr.prev_running_variance_tensor_uid().value());
        if(bnTensorAttrPrevRunningVar.virtual_())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm prev_running_variance tensor must be non-virtual");
        }
    }

    if(bnAttr.next_running_mean_tensor_uid().has_value())
    {
        const auto& bnTensorAttrNextRunningMean = core::utils::findTensorAttributes(
            tensorMap, bnAttr.next_running_mean_tensor_uid().value());
        if(bnTensorAttrNextRunningMean.virtual_())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm next_running_mean tensor must be non-virtual");
        }
    }

    if(bnAttr.next_running_variance_tensor_uid().has_value())
    {
        const auto& bnTensorAttrNextRunningVar = core::utils::findTensorAttributes(
            tensorMap, bnAttr.next_running_variance_tensor_uid().value());
        if(bnTensorAttrNextRunningVar.virtual_())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm next_running_variance tensor must be non-virtual");
        }
    }
}

} // namespace

const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes&
    checkTrainingNode(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& node)
{
    if(node.attributesType()
       != hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "First node must be batchnorm");
    }

    const auto& bnAttr
        = node.attributesAs<hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes>();

    return bnAttr;
}

const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& checkTrainingActivationNode(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& node,
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& bnAttr)
{
    if(node.attributesType()
       != hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Second node must be pointwise");
    }

    const auto& activAttr
        = node.attributesAs<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>();

    if(!isNodeActivFwd(activAttr))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Unsupported activation mode for batchnorm fusion");
    }

    if(activAttr.in_0_tensor_uid() != bnAttr.y_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Activation input must match batchnorm output");
    }

    return activAttr;
}

void checkTrainingTensorVirtuality1Node(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& bnAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    // Check for virtual tensors - 1-node case (solo batchnorm training)
    const auto& bnTensorX = core::utils::findTensorAttributes(tensorMap, bnAttr.x_tensor_uid());
    const auto& bnTensorScale
        = core::utils::findTensorAttributes(tensorMap, bnAttr.scale_tensor_uid());
    const auto& bnTensorBias
        = core::utils::findTensorAttributes(tensorMap, bnAttr.bias_tensor_uid());
    const auto& bnTensorY = core::utils::findTensorAttributes(tensorMap, bnAttr.y_tensor_uid());

    if(bnTensorX.virtual_() || bnTensorScale.virtual_() || bnTensorBias.virtual_()
       || bnTensorY.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm training tensors must be non-virtual for 1-node graph");
    }

    // Optional mean/variance tensors must be non-virtual if present, except a saved pair that is
    // virtual in both tensors, which is not requested.
    const bool unrequestedSavedPair = hasUnrequestedSavedPair(bnAttr, tensorMap);
    if(!unrequestedSavedPair && bnAttr.mean_tensor_uid().has_value())
    {
        const auto& bnTensorMean
            = core::utils::findTensorAttributes(tensorMap, bnAttr.mean_tensor_uid().value());
        if(bnTensorMean.virtual_())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Batchnorm mean tensor must be non-virtual");
        }
    }

    if(!unrequestedSavedPair && bnAttr.inv_variance_tensor_uid().has_value())
    {
        const auto& bnTensorInvVar = core::utils::findTensorAttributes(
            tensorMap, bnAttr.inv_variance_tensor_uid().value());
        if(bnTensorInvVar.virtual_())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm inv_variance tensor must be non-virtual");
        }
    }

    checkRunningStatisticsTensorVirtuality(bnAttr, tensorMap);
}

void checkTrainingTensorVirtuality2Node(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& bnAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    // Check for virtual tensors - 2-node case (batchnorm training + activation)
    const auto& bnTensorX = core::utils::findTensorAttributes(tensorMap, bnAttr.x_tensor_uid());
    const auto& bnTensorScale
        = core::utils::findTensorAttributes(tensorMap, bnAttr.scale_tensor_uid());
    const auto& bnTensorBias
        = core::utils::findTensorAttributes(tensorMap, bnAttr.bias_tensor_uid());
    const auto& bnTensorY = core::utils::findTensorAttributes(tensorMap, bnAttr.y_tensor_uid());

    if(bnTensorX.virtual_() || bnTensorScale.virtual_() || bnTensorBias.virtual_()
       || !bnTensorY.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm training input tensors must be non-virtual, output tensor must be virtual");
    }

    // Optional mean/variance tensors must be non-virtual if present
    if(bnAttr.mean_tensor_uid().has_value())
    {
        const auto& bnTensorMean
            = core::utils::findTensorAttributes(tensorMap, bnAttr.mean_tensor_uid().value());
        if(bnTensorMean.virtual_())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Batchnorm mean tensor must be non-virtual");
        }
    }

    if(bnAttr.inv_variance_tensor_uid().has_value())
    {
        const auto& bnTensorInvVar = core::utils::findTensorAttributes(
            tensorMap, bnAttr.inv_variance_tensor_uid().value());
        if(bnTensorInvVar.virtual_())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm inv_variance tensor must be non-virtual");
        }
    }

    checkRunningStatisticsTensorVirtuality(bnAttr, tensorMap);

    const auto& actTensorIn0
        = core::utils::findTensorAttributes(tensorMap, actAttr.in_0_tensor_uid());
    const auto& actTensorOut
        = core::utils::findTensorAttributes(tensorMap, actAttr.out_0_tensor_uid());

    if(!actTensorIn0.virtual_() || actTensorOut.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation input from batchnorm must be virtual, output must be non virtual");
    }
}

void checkBackwardActivationTensors(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr,
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormBackwardAttributes& bnBwdAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    const auto actIn1Uid = actAttr.in_1_tensor_uid();
    if(!actIn1Uid.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation backward requires in_1 tensor (forward activation input)");
    }

    if(actIn1Uid.value() != bnInfAttr.y_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation in_1 must be the batchnorm inference output tensor (y)");
    }

    if(actAttr.out_0_tensor_uid() != bnBwdAttr.dy_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward dy input must be the activation output tensor");
    }

    if(bnBwdAttr.x_tensor_uid() != bnInfAttr.x_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward must use the same X tensor as batchnorm inference");
    }

    if(bnBwdAttr.mean_tensor_uid().has_value()
       && bnBwdAttr.mean_tensor_uid().value() != bnInfAttr.mean_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward must use the same mean tensor as batchnorm inference");
    }

    if(bnBwdAttr.inv_variance_tensor_uid().has_value()
       && bnBwdAttr.inv_variance_tensor_uid().value() != bnInfAttr.inv_variance_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward must use the same inv_variance tensor as batchnorm inference");
    }

    if(bnBwdAttr.scale_tensor_uid() != bnInfAttr.scale_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward must use the same scale tensor as batchnorm inference");
    }

    const auto& bnInfTensorX
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.x_tensor_uid());
    const auto& bnInfTensorMean
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.mean_tensor_uid());
    const auto& bnInfTensorInvVar
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.inv_variance_tensor_uid());
    const auto& bnInfTensorScale
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.scale_tensor_uid());
    const auto& bnInfTensorBias
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.bias_tensor_uid());
    const auto& bnInfTensorY
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.y_tensor_uid());

    if(bnInfTensorX.virtual_() || bnInfTensorMean.virtual_() || bnInfTensorInvVar.virtual_()
       || bnInfTensorScale.virtual_() || bnInfTensorBias.virtual_() || !bnInfTensorY.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm inference input tensors must be non-virtual, output tensor must be virtual");
    }

    const auto& actTensorIn0
        = core::utils::findTensorAttributes(tensorMap, actAttr.in_0_tensor_uid());
    if(actTensorIn0.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Activation in_0 (dy gradient) must be non-virtual");
    }

    const auto& actTensorIn1 = core::utils::findTensorAttributes(tensorMap, actIn1Uid.value());
    const auto& actTensorOut
        = core::utils::findTensorAttributes(tensorMap, actAttr.out_0_tensor_uid());
    if(!actTensorIn1.virtual_() || !actTensorOut.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation input from batchnorm must be virtual, output must be virtual");
    }

    const auto& bnBwdTensorDy
        = core::utils::findTensorAttributes(tensorMap, bnBwdAttr.dy_tensor_uid());
    const auto& bnBwdTensorDx
        = core::utils::findTensorAttributes(tensorMap, bnBwdAttr.dx_tensor_uid());
    const auto& bnBwdTensorDscale
        = core::utils::findTensorAttributes(tensorMap, bnBwdAttr.dscale_tensor_uid());
    const auto& bnBwdTensorDbias
        = core::utils::findTensorAttributes(tensorMap, bnBwdAttr.dbias_tensor_uid());

    if(!bnBwdTensorDy.virtual_() || bnBwdTensorDx.virtual_() || bnBwdTensorDscale.virtual_()
       || bnBwdTensorDbias.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward dy input must be virtual, output tensors must be non-virtual");
    }
}

} // namespace batchnorm

} // namespace hip_kernel_provider
