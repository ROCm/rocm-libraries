// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenBatchnormFwdTrainingPlan.hpp"
#include "MiopenUtils.hpp"
#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>

namespace miopen_plugin
{

// We have made the intentional decision to hardcode the batchnorm mode to miopenBNSpatial
// rather than making it configurable and adding extra complexity.
const miopenBatchNormMode_t MIOPEN_BATCHNORM_MODE_TRAINING = miopenBNSpatial;

BatchnormFwdTrainingParams::BatchnormFwdTrainingParams(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(miopen_utils::createBatchnormTensor(tensorMap, attributes.x_tensor_uid()))
    , _y(miopen_utils::createBatchnormTensor(tensorMap, attributes.y_tensor_uid()))
    , _scale(miopen_utils::createBatchnormTensor(tensorMap, attributes.scale_tensor_uid()))
    , _bias(miopen_utils::createBatchnormTensor(tensorMap, attributes.bias_tensor_uid()))
{
    // Extract epsilon value from pass-by-value tensor (cast to double for MIOpen compatibility)
    auto epsilonTensorAttr = tensorMap.at(attributes.epsilon_tensor_uid());
    _epsilonValue = miopen_utils::extractDoubleFromTensorValue(epsilonTensorAttr, "Epsilon");

    // Save mean and inv_variance are optional (controlled by MIO_SAVE_MEAN_VARIANCE)
    auto optMeanUid = attributes.mean_tensor_uid();
    if(optMeanUid.has_value())
    {
        _mean = miopen_utils::createBatchnormTensor(tensorMap, *optMeanUid);
    }

    auto optInvVarUid = attributes.inv_variance_tensor_uid();
    if(optInvVarUid.has_value())
    {
        _invVariance = miopen_utils::createBatchnormTensor(tensorMap, *optInvVarUid);
    }

    auto optPrevRunMeanUid = attributes.prev_running_mean_tensor_uid();
    auto optPrevRunVarUid = attributes.prev_running_variance_tensor_uid();
    auto optMomentumUid = attributes.momentum_tensor_uid();
    auto optNextRunMeanUid = attributes.next_running_mean_tensor_uid();
    auto optNextRunVarUid = attributes.next_running_variance_tensor_uid();

    if(optPrevRunMeanUid.has_value() && optPrevRunVarUid.has_value() && optMomentumUid.has_value()
       && optNextRunMeanUid.has_value() && optNextRunVarUid.has_value())
    {
        // Extract momentum value from pass-by-value tensor (cast to double for MIOpen compatibility)
        auto momentumTensorAttr = tensorMap.at(*optMomentumUid);
        _momentumValue = miopen_utils::extractDoubleFromTensorValue(momentumTensorAttr, "Momentum");

        _prevRunningMean = miopen_utils::createBatchnormTensor(tensorMap, *optPrevRunMeanUid);
        _prevRunningVariance = miopen_utils::createBatchnormTensor(tensorMap, *optPrevRunVarUid);
        _nextRunningMean = miopen_utils::createBatchnormTensor(tensorMap, *optNextRunMeanUid);
        _nextRunningVariance = miopen_utils::createBatchnormTensor(tensorMap, *optNextRunVarUid);
        _hasRunningStats = true;
    }
}

BatchnormFwdTrainingParams::BatchnormFwdTrainingParams(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& attributes,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(miopen_utils::createBatchnormTensor(tensorMap, attributes.x_tensor_uid()))
    , _y(miopen_utils::createBatchnormTensor(tensorMap, attributes.y_tensor_uid()))
    , _scale(miopen_utils::createBatchnormTensor(tensorMap, attributes.scale_tensor_uid()))
    , _bias(miopen_utils::createBatchnormTensor(tensorMap, attributes.bias_tensor_uid()))
    , _activationOut(
          miopen_utils::createBatchnormTensor(tensorMap, pointwiseAttributes.out_0_tensor_uid()))
{
    using namespace miopen_utils;

    // Extract epsilon value from pass-by-value tensor (cast to double for MIOpen compatibility)
    auto epsilonTensorAttr = tensorMap.at(attributes.epsilon_tensor_uid());
    _epsilonValue = extractDoubleFromTensorValue(epsilonTensorAttr, "Epsilon");

    // Validate that activation input matches batchnorm output
    if(pointwiseAttributes.in_0_tensor_uid() != attributes.y_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "BatchnormFwdTrainingParams: Activation input must match batchnorm output");
    }

    // Get activation parameters
    HIPDNN_PREPEND_MESSAGE_ON_THROW(_optActivation
                                    = mapPointwiseModeToMiopenActivation(pointwiseAttributes),
                                    "BatchnormFwdTrainingParams: ");

    // Save mean and inv_variance are optional (controlled by MIO_SAVE_MEAN_VARIANCE)
    auto optMeanUid = attributes.mean_tensor_uid();
    if(optMeanUid.has_value())
    {
        _mean = createBatchnormTensor(tensorMap, *optMeanUid);
    }

    auto optInvVarUid = attributes.inv_variance_tensor_uid();
    if(optInvVarUid.has_value())
    {
        _invVariance = createBatchnormTensor(tensorMap, *optInvVarUid);
    }

    auto optPrevRunMeanUid = attributes.prev_running_mean_tensor_uid();
    auto optPrevRunVarUid = attributes.prev_running_variance_tensor_uid();
    auto optMomentumUid = attributes.momentum_tensor_uid();
    auto optNextRunMeanUid = attributes.next_running_mean_tensor_uid();
    auto optNextRunVarUid = attributes.next_running_variance_tensor_uid();

    if(optPrevRunMeanUid.has_value() && optPrevRunVarUid.has_value() && optMomentumUid.has_value()
       && optNextRunMeanUid.has_value() && optNextRunVarUid.has_value())
    {
        // Extract momentum value from pass-by-value tensor (cast to double for MIOpen compatibility)
        auto momentumTensorAttr = tensorMap.at(*optMomentumUid);
        _momentumValue = miopen_utils::extractDoubleFromTensorValue(momentumTensorAttr, "Momentum");

        _prevRunningMean = miopen_utils::createBatchnormTensor(tensorMap, *optPrevRunMeanUid);
        _prevRunningVariance = miopen_utils::createBatchnormTensor(tensorMap, *optPrevRunVarUid);
        _nextRunningMean = miopen_utils::createBatchnormTensor(tensorMap, *optNextRunMeanUid);
        _nextRunningVariance = miopen_utils::createBatchnormTensor(tensorMap, *optNextRunVarUid);
        _hasRunningStats = true;
    }
}

const MiopenTensor& BatchnormFwdTrainingParams::x() const
{
    return _x;
}

const MiopenTensor& BatchnormFwdTrainingParams::y() const
{
    return _y;
}

const MiopenTensor& BatchnormFwdTrainingParams::scale() const
{
    return _scale;
}

const MiopenTensor& BatchnormFwdTrainingParams::bias() const
{
    return _bias;
}

double BatchnormFwdTrainingParams::epsilonValue() const
{
    return _epsilonValue;
}

bool BatchnormFwdTrainingParams::hasSaveMeanVariance() const
{
    return _mean.has_value() && _invVariance.has_value();
}

const MiopenTensor& BatchnormFwdTrainingParams::mean() const
{
    if(!_mean.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "mean() called but mean tensor was not set");
    }
    return *_mean;
}

const MiopenTensor& BatchnormFwdTrainingParams::invVariance() const
{
    if(!_invVariance.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "invVariance() called but tensor was not set");
    }
    return *_invVariance;
}

bool BatchnormFwdTrainingParams::hasRunningStats() const
{
    return _hasRunningStats;
}

const MiopenTensor& BatchnormFwdTrainingParams::prevRunningMean() const
{
    if(!_prevRunningMean.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "prevRunningMean() called but tensor was not set");
    }
    return *_prevRunningMean;
}

const MiopenTensor& BatchnormFwdTrainingParams::prevRunningVariance() const
{
    if(!_prevRunningVariance.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "prevRunningVariance() called but tensor was not set");
    }
    return *_prevRunningVariance;
}

double BatchnormFwdTrainingParams::momentumValue() const
{
    if(!_momentumValue.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "momentumValue() called but momentum was not set");
    }
    return *_momentumValue;
}

const MiopenTensor& BatchnormFwdTrainingParams::nextRunningMean() const
{
    if(!_nextRunningMean.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "nextRunningMean() called but tensor was not set");
    }
    return *_nextRunningMean;
}

const MiopenTensor& BatchnormFwdTrainingParams::nextRunningVariance() const
{
    if(!_nextRunningVariance.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "nextRunningVariance() called but tensor was not set");
    }
    return *_nextRunningVariance;
}

const std::optional<miopen_utils::ActivationParams>&
    BatchnormFwdTrainingParams::optActivation() const
{
    return _optActivation;
}

const std::optional<MiopenTensor>& BatchnormFwdTrainingParams::activationOut() const
{
    return _activationOut;
}

BatchnormFwdTrainingPlan::BatchnormFwdTrainingPlan(BatchnormFwdTrainingParams&& trainingParams,
                                                   const HipdnnMiopenSettings& executionSettings)
    : _trainingParams(std::move(trainingParams))
    , _executionSettings(executionSettings)
{
}

size_t BatchnormFwdTrainingPlan::getWorkspaceSize(
    [[maybe_unused]] const HipdnnMiopenHandle& handle) const
{
    // No workspace needed for batchnorm training
    return 0;
}

void BatchnormFwdTrainingPlan::execute(const HipdnnMiopenHandle& handle,
                                       const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                       uint32_t numDeviceBuffers,
                                       [[maybe_unused]] void* workspace) const
{
    // Set tuning policy based on benchmarking flag - RAII ensures restoration
    const ScopedTuningPolicy tuningGuard(handle.miopenHandle,
                                         _executionSettings.benchmarkingEnabled());

    float alpha = 1.0f;
    float beta = 0.0f;

    // Extract epsilon from pass-by-value tensor attribute (type-safe, no buffer lookup needed)
    // Note: Type validation already done in constructor
    const double epsilon = _trainingParams.epsilonValue();

    // Extract momentum from pass-by-value tensor attribute if running stats exist
    double expAvgFactor = 0.0;
    if(_trainingParams.hasRunningStats())
    {
        expAvgFactor = _trainingParams.momentumValue();
        HIPDNN_PLUGIN_LOG_INFO(
            "BatchnormFwdTrainingPlan: expAvgFactor (momentum) = " << expAvgFactor);
    }

    // Get all required device buffers
    auto xBuffer = miopen_utils::findDeviceBuffer(
        _trainingParams.x().uid(), deviceBuffers, numDeviceBuffers);
    auto scaleBuffer = miopen_utils::findDeviceBuffer(
        _trainingParams.scale().uid(), deviceBuffers, numDeviceBuffers);
    auto biasBuffer = miopen_utils::findDeviceBuffer(
        _trainingParams.bias().uid(), deviceBuffers, numDeviceBuffers);

    // Handle save mean/variance if provided (optional)
    void* resultSaveMeanPtr = nullptr;
    void* resultSaveInvVariancePtr = nullptr;
    miopenTensorDescriptor_t savedMeanDesc = nullptr;
    miopenTensorDescriptor_t savedVarDesc = nullptr;

    if(_trainingParams.hasSaveMeanVariance())
    {
        auto meanBuffer = miopen_utils::findDeviceBuffer(
            _trainingParams.mean().uid(), deviceBuffers, numDeviceBuffers);
        auto invVarianceBuffer = miopen_utils::findDeviceBuffer(
            _trainingParams.invVariance().uid(), deviceBuffers, numDeviceBuffers);

        resultSaveMeanPtr = meanBuffer.ptr;
        resultSaveInvVariancePtr = invVarianceBuffer.ptr;
        savedMeanDesc = _trainingParams.mean().tensorDescriptor();
        savedVarDesc = _trainingParams.invVariance().tensorDescriptor();
    }

    void* prevRunningMeanPtr = nullptr;
    void* prevRunningVariancePtr = nullptr;
    void* nextRunningMeanPtr = nullptr;
    void* nextRunningVariancePtr = nullptr;

    if(_trainingParams.hasRunningStats())
    {
        prevRunningMeanPtr = miopen_utils::findDeviceBuffer(_trainingParams.prevRunningMean().uid(),
                                                            deviceBuffers,
                                                            numDeviceBuffers)
                                 .ptr;
        prevRunningVariancePtr
            = miopen_utils::findDeviceBuffer(
                  _trainingParams.prevRunningVariance().uid(), deviceBuffers, numDeviceBuffers)
                  .ptr;
        nextRunningMeanPtr = miopen_utils::findDeviceBuffer(_trainingParams.nextRunningMean().uid(),
                                                            deviceBuffers,
                                                            numDeviceBuffers)
                                 .ptr;
        nextRunningVariancePtr
            = miopen_utils::findDeviceBuffer(
                  _trainingParams.nextRunningVariance().uid(), deviceBuffers, numDeviceBuffers)
                  .ptr;
    }

    // Check if activation fusion is enabled
    const auto& optActivation = _trainingParams.optActivation();
    const auto& optActivationOut = _trainingParams.activationOut();
    if(optActivation.has_value() && optActivationOut.has_value())
    {
        const auto& activOutTensor = *optActivationOut;

        // Use activation fusion API
        auto yBuffer
            = miopen_utils::findDeviceBuffer(activOutTensor.uid(), deviceBuffers, numDeviceBuffers);

        // Create activation descriptor
        miopenActivationDescriptor_t activationDesc;
        THROW_ON_MIOPEN_FAILURE(miopenCreateActivationDescriptor(&activationDesc));
        auto activationDescRes
            = hipdnn_data_sdk::utilities::ScopedResource<miopenActivationDescriptor_t>(
                activationDesc, [](miopenActivationDescriptor_t desc) {
                    auto status = miopenDestroyActivationDescriptor(desc);
                    if(status != miopenStatusSuccess)
                    {
                        HIPDNN_PLUGIN_LOG_ERROR("miopenDestroyActivationDescriptor failed in "
                                                "BatchnormFwdTrainingPlan::execute");
                    }
                });

        const auto& activParams = *optActivation;
        THROW_ON_MIOPEN_FAILURE(miopenSetActivationDescriptor(activationDesc,
                                                              activParams.mode,
                                                              activParams.alpha,
                                                              activParams.beta,
                                                              activParams.gamma));

        if(_trainingParams.hasRunningStats())
        {
            THROW_ON_MIOPEN_FAILURE(miopenBatchNormForwardTrainingActivation_V2(
                handle.miopenHandle,
                MIOPEN_BATCHNORM_MODE_TRAINING,
                &alpha,
                &beta,
                _trainingParams.x().tensorDescriptor(),
                xBuffer.ptr,
                activOutTensor.tensorDescriptor(),
                yBuffer.ptr,
                _trainingParams.scale().tensorDescriptor(),
                _trainingParams.bias().tensorDescriptor(),
                savedMeanDesc,
                savedVarDesc,
                scaleBuffer.ptr,
                biasBuffer.ptr,
                expAvgFactor,
                prevRunningMeanPtr,
                prevRunningVariancePtr,
                nextRunningMeanPtr,
                nextRunningVariancePtr,
                epsilon,
                resultSaveMeanPtr,
                resultSaveInvVariancePtr,
                activationDesc));
        }
        else
        {
            THROW_ON_MIOPEN_FAILURE(miopenBatchNormForwardTrainingActivation(
                handle.miopenHandle,
                MIOPEN_BATCHNORM_MODE_TRAINING,
                &alpha,
                &beta,
                _trainingParams.x().tensorDescriptor(),
                xBuffer.ptr,
                activOutTensor.tensorDescriptor(),
                yBuffer.ptr,
                _trainingParams.scale().tensorDescriptor(),
                _trainingParams.bias().tensorDescriptor(),
                savedMeanDesc,
                savedVarDesc,
                scaleBuffer.ptr,
                biasBuffer.ptr,
                expAvgFactor,
                nullptr, // resultRunningMean: nullptr means running mean is not saved
                nullptr, // resultRunningVariance: nullptr means running variance is not saved
                epsilon,
                resultSaveMeanPtr,
                resultSaveInvVariancePtr,
                activationDesc));
        }
    }
    else
    {
        // Use standard batchnorm training API (no activation)
        auto yBuffer = miopen_utils::findDeviceBuffer(
            _trainingParams.y().uid(), deviceBuffers, numDeviceBuffers);

        if(_trainingParams.hasRunningStats())
        {
            THROW_ON_MIOPEN_FAILURE(miopenBatchNormalizationForwardTraining_V3(
                handle.miopenHandle,
                MIOPEN_BATCHNORM_MODE_TRAINING,
                &alpha,
                &beta,
                _trainingParams.x().tensorDescriptor(),
                xBuffer.ptr,
                _trainingParams.y().tensorDescriptor(),
                yBuffer.ptr,
                _trainingParams.scale().tensorDescriptor(),
                _trainingParams.bias().tensorDescriptor(),
                savedMeanDesc,
                savedVarDesc,
                scaleBuffer.ptr,
                biasBuffer.ptr,
                expAvgFactor,
                prevRunningMeanPtr,
                prevRunningVariancePtr,
                nextRunningMeanPtr,
                nextRunningVariancePtr,
                epsilon,
                resultSaveMeanPtr,
                resultSaveInvVariancePtr));
        }
        else
        {
            THROW_ON_MIOPEN_FAILURE(miopenBatchNormalizationForwardTraining(
                handle.miopenHandle,
                MIOPEN_BATCHNORM_MODE_TRAINING,
                &alpha,
                &beta,
                _trainingParams.x().tensorDescriptor(),
                xBuffer.ptr,
                _trainingParams.y().tensorDescriptor(),
                yBuffer.ptr,
                _trainingParams.scale().tensorDescriptor(),
                scaleBuffer.ptr,
                biasBuffer.ptr,
                expAvgFactor,
                nullptr, // resultRunningMean: nullptr means running mean is not saved
                nullptr, // resultRunningVariance: nullptr means running variance is not saved
                epsilon,
                resultSaveMeanPtr,
                resultSaveInvVariancePtr));
        }
    }
}

}
