// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenBatchnormFwdInferenceWithVariancePlan.hpp"
#include "MiopenUtils.hpp"

#include <hipdnn_data_sdk/utilities/Constants.hpp>

namespace miopen_plugin
{

// We have made the intentional decision to hardcode the batchnorm mode to miopenBNSpatial
// rather than making it configurable and adding extra complexity.
const miopenBatchNormMode_t MIOPEN_BATCHNORM_MODE = miopenBNSpatial;

BatchnormFwdInferenceWithVarianceParams::BatchnormFwdInferenceWithVarianceParams(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(miopen_utils::createBatchnormTensor(tensorMap, attributes.x_tensor_uid()))
    , _y(miopen_utils::createBatchnormTensor(tensorMap, attributes.y_tensor_uid()))
    , _scale(miopen_utils::createBatchnormTensor(tensorMap, attributes.scale_tensor_uid()))
    , _bias(miopen_utils::createBatchnormTensor(tensorMap, attributes.bias_tensor_uid()))
    , _estMean(miopen_utils::createBatchnormTensor(tensorMap, attributes.mean_tensor_uid()))
    , _variance(miopen_utils::createBatchnormTensor(tensorMap, attributes.variance_tensor_uid()))
{
    // Extract epsilon value from pass-by-value tensor (cast to double for MIOpen compatibility)
    auto epsilonTensorAttr = tensorMap.at(attributes.epsilon_tensor_uid());
    _epsilonValue = miopen_utils::extractDoubleFromTensorValue(epsilonTensorAttr, "Epsilon");
}

BatchnormFwdInferenceWithVarianceParams::BatchnormFwdInferenceWithVarianceParams(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt&
        inferenceAttributes,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(miopen_utils::createBatchnormTensor(tensorMap, inferenceAttributes.x_tensor_uid()))
    , _y(miopen_utils::createBatchnormTensor(tensorMap, inferenceAttributes.y_tensor_uid()))
    , _scale(miopen_utils::createBatchnormTensor(tensorMap, inferenceAttributes.scale_tensor_uid()))
    , _bias(miopen_utils::createBatchnormTensor(tensorMap, inferenceAttributes.bias_tensor_uid()))
    , _estMean(
          miopen_utils::createBatchnormTensor(tensorMap, inferenceAttributes.mean_tensor_uid()))
    , _variance(
          miopen_utils::createBatchnormTensor(tensorMap, inferenceAttributes.variance_tensor_uid()))
    , _optActivation(pointwiseAttributes)
    , _activationOut(
          miopen_utils::createBatchnormTensor(tensorMap, pointwiseAttributes.out_0_tensor_uid()))
{
    // Extract epsilon value from pass-by-value tensor (cast to double for MIOpen compatibility)
    auto epsilonTensorAttr = tensorMap.at(inferenceAttributes.epsilon_tensor_uid());
    _epsilonValue = miopen_utils::extractDoubleFromTensorValue(epsilonTensorAttr, "Epsilon");
}

const MiopenTensor& BatchnormFwdInferenceWithVarianceParams::x() const
{
    return _x;
}

const MiopenTensor& BatchnormFwdInferenceWithVarianceParams::y() const
{
    return _y;
}

const MiopenTensor& BatchnormFwdInferenceWithVarianceParams::scale() const
{
    return _scale;
}

const MiopenTensor& BatchnormFwdInferenceWithVarianceParams::bias() const
{
    return _bias;
}

const MiopenTensor& BatchnormFwdInferenceWithVarianceParams::estMean() const
{
    return _estMean;
}

const MiopenTensor& BatchnormFwdInferenceWithVarianceParams::variance() const
{
    return _variance;
}

double BatchnormFwdInferenceWithVarianceParams::epsilonValue() const
{
    return _epsilonValue;
}

const std::optional<MiopenActivationDescriptor>&
    BatchnormFwdInferenceWithVarianceParams::optActivation() const
{
    return _optActivation;
}

const std::optional<MiopenTensor>& BatchnormFwdInferenceWithVarianceParams::activationOut() const
{
    return _activationOut;
}

BatchnormFwdInferenceWithVariancePlan::BatchnormFwdInferenceWithVariancePlan(
    BatchnormFwdInferenceWithVarianceParams&& inferenceParams,
    const HipdnnMiopenSettings& executionSettings)
    : _inferenceParams(std::move(inferenceParams))
    , _executionSettings(executionSettings)
{
}

size_t BatchnormFwdInferenceWithVariancePlan::getWorkspaceSize(
    [[maybe_unused]] const HipdnnMiopenHandle& handle) const
{
    // No workspace needed for batchnorm inference with variance
    return 0;
}

void BatchnormFwdInferenceWithVariancePlan::execute(const HipdnnMiopenHandle& handle,
                                                    const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                                    uint32_t numDeviceBuffers,
                                                    [[maybe_unused]] void* workspace) const
{
    // Set tuning policy based on benchmarking flag - RAII ensures restoration
    ScopedTuningPolicy tuningGuard(handle.miopenHandle, _executionSettings.benchmarkingEnabled());

    // Hardcoded values from bn_driver in miopen
    auto alpha = static_cast<float>(1);
    auto beta = static_cast<float>(0);
    double epsilon = _inferenceParams.epsilonValue();

    auto xBuffer = miopen_utils::findDeviceBuffer(
        _inferenceParams.x().uid(), deviceBuffers, numDeviceBuffers);
    auto scaleBuffer = miopen_utils::findDeviceBuffer(
        _inferenceParams.scale().uid(), deviceBuffers, numDeviceBuffers);
    auto biasBuffer = miopen_utils::findDeviceBuffer(
        _inferenceParams.bias().uid(), deviceBuffers, numDeviceBuffers);
    auto estMeanBuffer = miopen_utils::findDeviceBuffer(
        _inferenceParams.estMean().uid(), deviceBuffers, numDeviceBuffers);
    auto varianceBuffer = miopen_utils::findDeviceBuffer(
        _inferenceParams.variance().uid(), deviceBuffers, numDeviceBuffers);

    if(_inferenceParams.optActivation().has_value() && _inferenceParams.activationOut().has_value())
    {
        auto activationOutBuffer = miopen_utils::findDeviceBuffer(
            _inferenceParams.activationOut()->uid(), deviceBuffers, numDeviceBuffers);

        THROW_ON_MIOPEN_FAILURE(miopenBatchNormForwardInferenceActivation(
            handle.miopenHandle,
            MIOPEN_BATCHNORM_MODE,
            &alpha,
            &beta,
            _inferenceParams.x().tensorDescriptor(),
            xBuffer.ptr,
            _inferenceParams.activationOut().value().tensorDescriptor(),
            activationOutBuffer.ptr,
            _inferenceParams.scale().tensorDescriptor(),
            _inferenceParams.bias().tensorDescriptor(),
            _inferenceParams.estMean().tensorDescriptor(),
            _inferenceParams.variance().tensorDescriptor(),
            scaleBuffer.ptr,
            biasBuffer.ptr,
            estMeanBuffer.ptr,
            varianceBuffer.ptr,
            epsilon,
            _inferenceParams.optActivation().value().activationDescriptor()));
    }
    else
    {
        auto yBuffer = miopen_utils::findDeviceBuffer(
            _inferenceParams.y().uid(), deviceBuffers, numDeviceBuffers);
        THROW_ON_MIOPEN_FAILURE(miopenBatchNormalizationForwardInference_V2(
            handle.miopenHandle,
            MIOPEN_BATCHNORM_MODE,
            &alpha,
            &beta,
            _inferenceParams.x().tensorDescriptor(),
            xBuffer.ptr,
            _inferenceParams.y().tensorDescriptor(),
            yBuffer.ptr,
            _inferenceParams.scale().tensorDescriptor(),
            _inferenceParams.bias().tensorDescriptor(),
            _inferenceParams.estMean().tensorDescriptor(),
            _inferenceParams.variance().tensorDescriptor(),
            scaleBuffer.ptr,
            biasBuffer.ptr,
            estMeanBuffer.ptr,
            varianceBuffer.ptr,
            epsilon));
    }
}

} // namespace miopen_plugin
