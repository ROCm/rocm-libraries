// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenBatchnormBwdPlan.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "MiopenUtils.hpp"

#include <hipdnn_sdk/utilities/Constants.hpp>

namespace miopen_legacy_plugin
{

// We have made the intentional decision to hardcode the batchnorm mode to miopenBNSpatial
// rather than making it configurable and adding extra complexity.
const miopenBatchNormMode_t MIOPEN_BATCHNORM_MODE = miopenBNSpatial;

BatchnormBwdParams::BatchnormBwdParams(
    const hipdnn_sdk::data_objects::BatchnormBackwardAttributes& attributes,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap)
    : _x(miopen_utils::createTensor(tensorMap, attributes.x_tensor_uid()))
    , _dy(miopen_utils::createTensor(tensorMap, attributes.dy_tensor_uid()))
    , _dx(miopen_utils::createTensor(tensorMap, attributes.dx_tensor_uid()))
    , _scale(miopen_utils::createTensor(tensorMap, attributes.scale_tensor_uid()))
    , _dscale(miopen_utils::createTensor(tensorMap, attributes.dscale_tensor_uid()))
    , _dbias(miopen_utils::createTensor(tensorMap, attributes.dbias_tensor_uid()))
{
    if(attributes.mean_tensor_uid().has_value())
    {
        _optMean = miopen_utils::createTensor(tensorMap, attributes.mean_tensor_uid().value());
    }

    if(attributes.inv_variance_tensor_uid().has_value())
    {
        _optInvVariance
            = miopen_utils::createTensor(tensorMap, attributes.inv_variance_tensor_uid().value());
    }
}

const MiopenTensor& BatchnormBwdParams::x() const
{
    return _x;
}

const MiopenTensor& BatchnormBwdParams::dy() const
{
    return _dy;
}

const MiopenTensor& BatchnormBwdParams::dx() const
{
    return _dx;
}

const MiopenTensor& BatchnormBwdParams::scale() const
{
    return _scale;
}

const MiopenTensor& BatchnormBwdParams::dscale() const
{
    return _dscale;
}

const MiopenTensor& BatchnormBwdParams::dbias() const
{
    return _dbias;
}

const std::optional<MiopenTensor>& BatchnormBwdParams::optMean() const
{
    return _optMean;
}

const std::optional<MiopenTensor>& BatchnormBwdParams::optInvVariance() const
{
    return _optInvVariance;
}

BatchnormBwdPlan::BatchnormBwdPlan(BatchnormBwdParams&& params)
    : _params(std::move(params))
{
}

size_t BatchnormBwdPlan::getWorkspaceSize(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle) const
{
    // No workspace needed for batchnorm backward
    return 0;
}

void BatchnormBwdPlan::execute(const HipdnnEnginePluginHandle& handle,
                               const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                               uint32_t numDeviceBuffers,
                               [[maybe_unused]] void* workspace) const
{
    float alphaDataDiff = 1.0f;
    float betaDataDiff = 0.0f;
    float alphaParamDiff = 1.0f;
    float betaParamDiff = 0.0f;
    double epsilon = hipdnn_sdk::utilities::BATCHNORM_DEFAULT_EPSILON;

    auto xBuffer
        = miopen_utils::findDeviceBuffer(_params.x().uid(), deviceBuffers, numDeviceBuffers);
    auto dyBuffer
        = miopen_utils::findDeviceBuffer(_params.dy().uid(), deviceBuffers, numDeviceBuffers);
    auto dxBuffer
        = miopen_utils::findDeviceBuffer(_params.dx().uid(), deviceBuffers, numDeviceBuffers);
    auto scaleBuffer
        = miopen_utils::findDeviceBuffer(_params.scale().uid(), deviceBuffers, numDeviceBuffers);
    auto dscaleBuffer
        = miopen_utils::findDeviceBuffer(_params.dscale().uid(), deviceBuffers, numDeviceBuffers);
    auto dbiasBuffer
        = miopen_utils::findDeviceBuffer(_params.dbias().uid(), deviceBuffers, numDeviceBuffers);

    hipdnnPluginDeviceBuffer_t meanBuffer = {0, nullptr};
    if(_params.optMean().has_value())
    {
        meanBuffer = miopen_utils::findDeviceBuffer(
            _params.optMean().value().uid(), deviceBuffers, numDeviceBuffers);
    }

    hipdnnPluginDeviceBuffer_t invVarianceBuffer = {0, nullptr};
    if(_params.optInvVariance().has_value())
    {
        invVarianceBuffer = miopen_utils::findDeviceBuffer(
            _params.optInvVariance().value().uid(), deviceBuffers, numDeviceBuffers);
    }

    THROW_ON_MIOPEN_FAILURE(miopenBatchNormalizationBackward_V2(
        handle.miopenHandle,
        MIOPEN_BATCHNORM_MODE,
        &alphaDataDiff,
        &betaDataDiff,
        &alphaParamDiff,
        &betaParamDiff,
        _params.x().tensorDescriptor(),
        xBuffer.ptr,
        _params.dy().tensorDescriptor(),
        dyBuffer.ptr,
        _params.dx().tensorDescriptor(),
        dxBuffer.ptr,
        _params.scale().tensorDescriptor(),
        _params.scale().tensorDescriptor(),
        _params.optMean().has_value() ? _params.optMean().value().tensorDescriptor() : nullptr,
        _params.optInvVariance().has_value() ? _params.optInvVariance().value().tensorDescriptor()
                                             : nullptr,
        scaleBuffer.ptr,
        dscaleBuffer.ptr,
        dbiasBuffer.ptr,
        epsilon,
        meanBuffer.ptr,
        invVarianceBuffer.ptr));
}

} // namespace miopen_legacy_plugin
