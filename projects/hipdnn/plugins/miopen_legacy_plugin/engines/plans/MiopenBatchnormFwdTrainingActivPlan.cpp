// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_sdk/utilities/ScopedResource.hpp>

#include "HipdnnEnginePluginHandle.hpp"
#include "MiopenBatchnormFwdTrainingActivPlan.hpp"

namespace miopen_legacy_plugin
{

// We have made the intentional decision to hardcode the batchnorm mode to miopenBNSpatial
// rather than making it configurable and adding extra complexity.
const miopenBatchNormMode_t MIOPEN_BATCHNORM_MODE = miopenBNSpatial;

BatchnormFwdTrainingActivParams::BatchnormFwdTrainingActivParams(
    const hipdnn_sdk::data_objects::BatchnormAttributes& bnAttr,
    const hipdnn_sdk::data_objects::PointwiseAttributes& activAttr,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap)
    : _x(miopen_utils::createTensor(tensorMap, bnAttr.x_tensor_uid()))
    , _y(miopen_utils::createTensor(tensorMap, activAttr.out_0_tensor_uid()))
    , _scale(miopen_utils::createTensor(tensorMap, bnAttr.scale_tensor_uid()))
    , _bias(miopen_utils::createTensor(tensorMap, bnAttr.bias_tensor_uid()))
{
    // Extract epsilon value from pass-by-value tensor (cast to double for MIOpen compatibility)
    auto epsilonTensorAttr = tensorMap.at(bnAttr.epsilon_tensor_uid());
    _epsilonValue = miopen_utils::extractDoubleFromTensorValue(epsilonTensorAttr, "Epsilon");

    // Validate that activation input matches batchnorm output
    if(activAttr.in_0_tensor_uid() != bnAttr.y_tensor_uid())
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "BatchnormFwdTrainingActivParams: Activation input must match batchnorm output");
    }

    // Get activation parameters
    const auto activParams = miopen_utils::mapPointwiseModeToMiopenActivation(activAttr);
    if(!activParams.has_value())
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "BatchnormFwdTrainingActivParams: Unsupported activation mode");
    }
    _activParams = activParams.value();

    // Handle optional saved mean/variance
    // For WITH_BATCH_STATS: these won't exist in graph (no UIDs)
    // For FULL_TRAINING: these will exist as outputs
    if(bnAttr.mean_tensor_uid().has_value())
    {
        _mean = miopen_utils::createTensor(tensorMap, bnAttr.mean_tensor_uid().value());
    }

    if(bnAttr.inv_variance_tensor_uid().has_value())
    {
        _invVariance
            = miopen_utils::createTensor(tensorMap, bnAttr.inv_variance_tensor_uid().value());
    }

    // TODO: Enable running statistics when MIOpen API supports separate input/output buffers
    // Currently commented out due to API mismatch - plan builder will reject graphs with running stats
    /*
    // Handle optional running statistics
    if(bnAttr.prev_running_mean_tensor_uid().has_value()
       && bnAttr.prev_running_variance_tensor_uid().has_value()
       && bnAttr.momentum_tensor_uid().has_value()
       && bnAttr.next_running_mean_tensor_uid().has_value()
       && bnAttr.next_running_variance_tensor_uid().has_value())
    {
        _prevRunningMean = miopen_utils::createTensor(
            tensorMap, bnAttr.prev_running_mean_tensor_uid().value());
        _prevRunningVariance = miopen_utils::createTensor(
            tensorMap, bnAttr.prev_running_variance_tensor_uid().value());
        
        auto momentumTensorAttr = tensorMap.at(bnAttr.momentum_tensor_uid().value());
        _momentumValue = miopen_utils::extractDoubleFromTensorValue(momentumTensorAttr, "Momentum");
        
        _nextRunningMean = miopen_utils::createTensor(
            tensorMap, bnAttr.next_running_mean_tensor_uid().value());
        _nextRunningVariance = miopen_utils::createTensor(
            tensorMap, bnAttr.next_running_variance_tensor_uid().value());
        
        _hasRunningStats = true;
    }
    */

    // Running statistics not supported - API mismatch between hipDNN and MIOpen
    // Defensive check: should have been rejected by plan builder
    if(bnAttr.prev_running_mean_tensor_uid().has_value()
       || bnAttr.prev_running_variance_tensor_uid().has_value()
       || bnAttr.momentum_tensor_uid().has_value()
       || bnAttr.next_running_mean_tensor_uid().has_value()
       || bnAttr.next_running_variance_tensor_uid().has_value())
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "Running statistics should have been rejected by plan builder");
    }
}

const MiopenTensor& BatchnormFwdTrainingActivParams::x() const
{
    return _x;
}

const MiopenTensor& BatchnormFwdTrainingActivParams::y() const
{
    return _y;
}

const MiopenTensor& BatchnormFwdTrainingActivParams::scale() const
{
    return _scale;
}

const MiopenTensor& BatchnormFwdTrainingActivParams::bias() const
{
    return _bias;
}

double BatchnormFwdTrainingActivParams::epsilonValue() const
{
    return _epsilonValue;
}

bool BatchnormFwdTrainingActivParams::hasSaveMeanVariance() const
{
    return _mean.has_value() && _invVariance.has_value();
}

const MiopenTensor& BatchnormFwdTrainingActivParams::mean() const
{
    return _mean.value();
}

const MiopenTensor& BatchnormFwdTrainingActivParams::invVariance() const
{
    return _invVariance.value();
}

bool BatchnormFwdTrainingActivParams::hasRunningStats() const
{
    return _hasRunningStats;
}

const MiopenTensor& BatchnormFwdTrainingActivParams::prevRunningMean() const
{
    return _prevRunningMean.value();
}

const MiopenTensor& BatchnormFwdTrainingActivParams::prevRunningVariance() const
{
    return _prevRunningVariance.value();
}

double BatchnormFwdTrainingActivParams::momentumValue() const
{
    return _momentumValue.value();
}

const MiopenTensor& BatchnormFwdTrainingActivParams::nextRunningMean() const
{
    return _nextRunningMean.value();
}

const MiopenTensor& BatchnormFwdTrainingActivParams::nextRunningVariance() const
{
    return _nextRunningVariance.value();
}

const miopen_utils::ActivationParams& BatchnormFwdTrainingActivParams::activParams() const
{
    return _activParams;
}

BatchnormFwdTrainingActivPlan::BatchnormFwdTrainingActivPlan(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle,
    BatchnormFwdTrainingActivParams&& params)
    : _params(std::move(params))
{
    // No initialization needed - miopenBatchNormForwardTrainingActivation doesn't require
    // pre-compilation like the fusion API does
}

size_t BatchnormFwdTrainingActivPlan::getWorkspaceSize(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle) const
{
    // No workspace needed for miopenBatchNormForwardTrainingActivation
    return 0;
}

void BatchnormFwdTrainingActivPlan::execute(const HipdnnEnginePluginHandle& handle,
                                            const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                            uint32_t numDeviceBuffers,
                                            [[maybe_unused]] void* workspace) const
{
    float alpha = 1.0f;
    float beta = 0.0f;

    auto xBuffer
        = miopen_utils::findDeviceBuffer(_params.x().uid(), deviceBuffers, numDeviceBuffers);
    auto yBuffer
        = miopen_utils::findDeviceBuffer(_params.y().uid(), deviceBuffers, numDeviceBuffers);
    auto scaleBuffer
        = miopen_utils::findDeviceBuffer(_params.scale().uid(), deviceBuffers, numDeviceBuffers);
    auto biasBuffer
        = miopen_utils::findDeviceBuffer(_params.bias().uid(), deviceBuffers, numDeviceBuffers);

    // Optional saved batch statistics (mean/invVariance)
    void* savedMeanPtr = nullptr;
    void* savedInvVariancePtr = nullptr;
    miopenTensorDescriptor_t savedMeanDesc = nullptr;
    miopenTensorDescriptor_t savedVarDesc = nullptr;

    if(_params.hasSaveMeanVariance())
    {
        auto meanBuffer
            = miopen_utils::findDeviceBuffer(_params.mean().uid(), deviceBuffers, numDeviceBuffers);
        auto invVarianceBuffer = miopen_utils::findDeviceBuffer(
            _params.invVariance().uid(), deviceBuffers, numDeviceBuffers);
        savedMeanPtr = meanBuffer.ptr;
        savedInvVariancePtr = invVarianceBuffer.ptr;
        savedMeanDesc = _params.mean().tensorDescriptor();
        savedVarDesc = _params.invVariance().tensorDescriptor();
    }

    // TODO: Enable running statistics when MIOpen API supports separate input/output buffers
    // Currently commented out due to API mismatch - plan builder will reject graphs with running stats
    /*
    void* runningMeanPtr = nullptr;
    void* runningVariancePtr = nullptr;
    double expAvgFactor = 0.0;

    if(_params.hasRunningStats())
    {
        auto prevMeanBuffer = miopen_utils::findDeviceBuffer(
            _params.prevRunningMean().uid(), deviceBuffers, numDeviceBuffers);
        auto prevVarBuffer = miopen_utils::findDeviceBuffer(
            _params.prevRunningVariance().uid(), deviceBuffers, numDeviceBuffers);
        auto nextMeanBuffer = miopen_utils::findDeviceBuffer(
            _params.nextRunningMean().uid(), deviceBuffers, numDeviceBuffers);
        auto nextVarBuffer = miopen_utils::findDeviceBuffer(
            _params.nextRunningVariance().uid(), deviceBuffers, numDeviceBuffers);

        // TODO: Copy prev to next buffers before calling MIOpen
        // This workaround is needed until MIOpen supports separate input/output buffers
        // hipMemcpy(nextMeanBuffer.ptr, prevMeanBuffer.ptr, size, hipMemcpyDeviceToDevice);
        // hipMemcpy(nextVarBuffer.ptr, prevVarBuffer.ptr, size, hipMemcpyDeviceToDevice);

        runningMeanPtr = nextMeanBuffer.ptr;
        runningVariancePtr = nextVarBuffer.ptr;
        expAvgFactor = _params.momentumValue();
    }
    */

    // Running statistics should have been rejected by plan builder
    if(_params.hasRunningStats())
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "Running statistics should have been rejected by plan builder");
    }

    void* runningMeanPtr = nullptr;
    void* runningVariancePtr = nullptr;
    double expAvgFactor = 0.0;

    // Create activation descriptor
    miopenActivationDescriptor_t activationDesc;
    THROW_ON_MIOPEN_FAILURE(miopenCreateActivationDescriptor(&activationDesc));
    auto activationDescRes = hipdnn_sdk::utilities::ScopedResource<miopenActivationDescriptor_t>(
        activationDesc, [](miopenActivationDescriptor_t desc) {
            auto status = miopenDestroyActivationDescriptor(desc);
            if(status != miopenStatusSuccess)
            {
                HIPDNN_LOG_ERROR("miopenDestroyActivationDescriptor failed in "
                                 "BatchnormFwdTrainingActivPlan destructor");
            }
        });

    THROW_ON_MIOPEN_FAILURE(miopenSetActivationDescriptor(activationDesc,
                                                          _params.activParams().mode,
                                                          _params.activParams().alpha,
                                                          _params.activParams().beta,
                                                          _params.activParams().gamma));

    THROW_ON_MIOPEN_FAILURE(
        miopenBatchNormForwardTrainingActivation(handle.miopenHandle,
                                                 MIOPEN_BATCHNORM_MODE,
                                                 &alpha,
                                                 &beta,
                                                 _params.x().tensorDescriptor(),
                                                 xBuffer.ptr,
                                                 _params.y().tensorDescriptor(),
                                                 yBuffer.ptr,
                                                 _params.scale().tensorDescriptor(),
                                                 _params.bias().tensorDescriptor(),
                                                 savedMeanDesc,
                                                 savedVarDesc,
                                                 scaleBuffer.ptr,
                                                 biasBuffer.ptr,
                                                 expAvgFactor,
                                                 runningMeanPtr,
                                                 runningVariancePtr,
                                                 _params.epsilonValue(),
                                                 savedMeanPtr,
                                                 savedInvVariancePtr,
                                                 activationDesc));
}

} // namespace miopen_legacy_plugin
