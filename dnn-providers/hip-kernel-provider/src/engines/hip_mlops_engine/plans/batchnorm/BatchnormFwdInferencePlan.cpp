// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "BatchnormFwdInferencePlan.hpp"
#include "BatchnormKernelCompileOptions.hpp"
#include "engines/hip_mlops_engine/plans/PlanUtils.hpp"

#include "compilation/IKernelCompiler.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hip_kernel_provider::batchnorm
{

BatchnormFwdInferenceParams::BatchnormFwdInferenceParams(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(tensorMap.at(attributes.x_tensor_uid()))
    , _y(tensorMap.at(attributes.y_tensor_uid()))
    , _scale(tensorMap.at(attributes.scale_tensor_uid()))
    , _bias(tensorMap.at(attributes.bias_tensor_uid()))
    , _estMean(tensorMap.at(attributes.mean_tensor_uid()))
    , _invVariance(tensorMap.at(attributes.inv_variance_tensor_uid()))
    , _activationOut(nullptr)
{
}

BatchnormFwdInferenceParams::BatchnormFwdInferenceParams(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes& inferenceAttributes,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(tensorMap.at(inferenceAttributes.x_tensor_uid()))
    , _y(tensorMap.at(inferenceAttributes.y_tensor_uid()))
    , _scale(tensorMap.at(inferenceAttributes.scale_tensor_uid()))
    , _bias(tensorMap.at(inferenceAttributes.bias_tensor_uid()))
    , _estMean(tensorMap.at(inferenceAttributes.mean_tensor_uid()))
    , _invVariance(tensorMap.at(inferenceAttributes.inv_variance_tensor_uid()))
    , _optActivation(parseActivation(pointwiseAttributes))
    , _activationOut(tensorMap.at(pointwiseAttributes.out_0_tensor_uid()))
{
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* BatchnormFwdInferenceParams::x() const
{
    return _x;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* BatchnormFwdInferenceParams::y() const
{
    return _y;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceParams::scale() const
{
    return _scale;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceParams::bias() const
{
    return _bias;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceParams::estMean() const
{
    return _estMean;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceParams::invVariance() const
{
    return _invVariance;
}

const std::optional<ActivationParams>& BatchnormFwdInferenceParams::optActivation() const
{
    return _optActivation;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceParams::activationOut() const
{
    return _activationOut;
}

BatchnormFwdInferencePlan::BatchnormFwdInferencePlan(BatchnormFwdInferenceParams&& inferenceParams)
    : _inferenceParams(std::move(inferenceParams))
{
}

size_t BatchnormFwdInferencePlan::getWorkspaceSize([[maybe_unused]] const Handle& handle) const
{
    // No workspace needed for batchnorm inference
    return 0;
}

void BatchnormFwdInferencePlan::compile(const IKernelCompiler& kernelCompiler,
                                        const hipDeviceProp_t& deviceProperties)
{
    // Determine data type configuration
    auto xDataType = _inferenceParams.x()->data_type();
    auto scaleDataType = _inferenceParams.scale()->data_type();

    const bool useFp16Mix
        = (xDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::HALF
           && scaleDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    const bool useBfp16Mix
        = (xDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16
           && scaleDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    // Extract dimensions from x tensor
    const auto* xDims = _inferenceParams.x()->dims();
    const auto* xStrides = _inferenceParams.x()->strides();

    int n = 0;
    int c = 0;
    int h = 0;
    int w = 0;
    int nStride = 0;
    int cStride = 0;
    int wStride = 0;

    // Check if 4D (NCHW/NHWC) or 5D (NCDHW/NDHWC)
    if(xDims->size() == 4)
    {
        n = static_cast<int>(xDims->Get(0));
        c = static_cast<int>(xDims->Get(1));
        h = static_cast<int>(xDims->Get(2));
        w = static_cast<int>(xDims->Get(3));

        nStride = static_cast<int>(xStrides->Get(0));
        cStride = static_cast<int>(xStrides->Get(1));
        wStride = static_cast<int>(xStrides->Get(3));
    }
    else if(xDims->size() == 5)
    {
        n = static_cast<int>(xDims->Get(0));
        c = static_cast<int>(xDims->Get(1));
        auto d = static_cast<int>(xDims->Get(2));
        h = static_cast<int>(xDims->Get(3));
        w = static_cast<int>(xDims->Get(4));
        // For 5D, combine D*H*W into spatial dimension
        h = d * h;

        nStride = static_cast<int>(xStrides->Get(0));
        cStride = static_cast<int>(xStrides->Get(1));
        wStride = static_cast<int>(xStrides->Get(4));
    }
    else
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Unsupported tensor dimension: "
                                                           + std::to_string(xDims->size()));
    }

    auto inCstride = static_cast<unsigned int>(h * w);

    const bool isLayoutNhwc = isChannelLastLayout(_inferenceParams.x());

    // Calculate vector size based on layout
    auto vectorsize = computeVectorSize(isLayoutNhwc, c, inCstride);

    // Calculate block and grid dimensions
    size_t xlocalsize = 0;
    size_t xgridsize = 0;
    size_t ylocalsize = 0;
    size_t ygridsize = 0;
    size_t zlocalsize = 0;
    size_t zgridsize = 0;
    const size_t maxLocalsize = 256;

    if(isLayoutNhwc)
    {
        xlocalsize = std::min(static_cast<size_t>(c) / vectorsize, maxLocalsize);
        xgridsize
            = ((static_cast<size_t>(c) / vectorsize) + xlocalsize - 1) / xlocalsize * xlocalsize;

        ylocalsize = maxLocalsize / xlocalsize;
        ygridsize = (inCstride + ylocalsize - 1) / ylocalsize * ylocalsize;
    }
    else
    {
        xlocalsize = 1;
        xgridsize = ((static_cast<size_t>(c) + xlocalsize - 1) / xlocalsize) * xlocalsize;

        ylocalsize = maxLocalsize;
        ygridsize = ((inCstride / vectorsize + ylocalsize - 1) / ylocalsize) * ylocalsize;
    }

    zlocalsize = 1;
    const size_t activeThreadsXy = xgridsize * ygridsize;
    auto maxActiveThreads = static_cast<size_t>(deviceProperties.multiProcessorCount) * 32
                            * static_cast<size_t>(deviceProperties.warpSize);

    if(activeThreadsXy < maxActiveThreads)
    {
        zgridsize = std::min(maxActiveThreads / activeThreadsXy, static_cast<size_t>(n));
    }
    else
    {
        zgridsize = 1;
    }

    // Get activation mode
    auto activationMode = ActivationMode::PASTHRU;

    if(_inferenceParams.optActivation().has_value() && _inferenceParams.activationOut() != nullptr)
    {
        activationMode = (*_inferenceParams.optActivation()).mode;
    }

    // Prepare compilation options
    BatchnormKernelCompileOptions options(_inferenceParams.x(), deviceProperties, activationMode);
    options.update("HIP_PLUGIN_USE_FPMIX", useFp16Mix);
    options.update("HIP_PLUGIN_USE_BFPMIX", useBfp16Mix);
    options.update("HIP_PLUGIN_BN_GRP0", xlocalsize);
    options.update("HIP_PLUGIN_BN_GRP1", ylocalsize);
    options.update("HIP_PLUGIN_BN_GRP2", zlocalsize);
    options.update("HIP_PLUGIN_BN_VEC_SIZE", vectorsize);

    // Compile kernel and configure launch dimensions
    _compiledProgram = kernelCompiler.compile("BatchNormFwdInferSpatial.cpp", options);
    _runnableKernel = _compiledProgram->getKernel("BatchNormFwdInferSpatialEstInvVar");

    _runnableKernel->setBlockSize(static_cast<unsigned int>(xlocalsize),
                                  static_cast<unsigned int>(ylocalsize),
                                  static_cast<unsigned int>(zlocalsize));
    _runnableKernel->setGridSize(static_cast<unsigned int>(xgridsize / xlocalsize),
                                 static_cast<unsigned int>(ygridsize / ylocalsize),
                                 static_cast<unsigned int>(zgridsize / zlocalsize));

    // Store kernel launch parameters
    _channels = static_cast<unsigned int>(c);
    _inCstride = inCstride;
    _batchCount = static_cast<unsigned int>(n);
    _cStride = static_cast<unsigned int>(cStride);
    _hwStride = static_cast<unsigned int>(wStride);
    _batchStride = static_cast<unsigned int>(nStride);
}

void BatchnormFwdInferencePlan::execute(const Handle& handle,
                                        const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                        uint32_t numDeviceBuffers,
                                        [[maybe_unused]] void* workspace) const
{
    if(!_runnableKernel)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "BatchnormFwdInferencePlan::execute() called before compile()");
    }

    // Get device buffer pointers
    auto xBuffer = findDeviceBuffer(_inferenceParams.x()->uid(), deviceBuffers, numDeviceBuffers);
    auto scaleBuffer
        = findDeviceBuffer(_inferenceParams.scale()->uid(), deviceBuffers, numDeviceBuffers);
    auto biasBuffer
        = findDeviceBuffer(_inferenceParams.bias()->uid(), deviceBuffers, numDeviceBuffers);
    auto estMeanBuffer
        = findDeviceBuffer(_inferenceParams.estMean()->uid(), deviceBuffers, numDeviceBuffers);
    auto invVarianceBuffer
        = findDeviceBuffer(_inferenceParams.invVariance()->uid(), deviceBuffers, numDeviceBuffers);

    float activationAlpha = 0.0f;
    float activationBeta = 0.0f;

    // Launch kernel with appropriate output buffer
    if(_inferenceParams.optActivation().has_value() && _inferenceParams.activationOut() != nullptr)
    {
        auto activationOutBuffer = findDeviceBuffer(
            _inferenceParams.activationOut()->uid(), deviceBuffers, numDeviceBuffers);

        // Get activation parameters
        const auto& activation = *_inferenceParams.optActivation();
        activationAlpha = static_cast<float>(activation.alpha);
        activationBeta = static_cast<float>(activation.beta);

        _runnableKernel->launch(handle.getStream(),
                                xBuffer.ptr,
                                activationOutBuffer.ptr,
                                estMeanBuffer.ptr,
                                invVarianceBuffer.ptr,
                                scaleBuffer.ptr,
                                biasBuffer.ptr,
                                _channels,
                                _inCstride,
                                _batchCount,
                                _cStride,
                                _hwStride,
                                _batchStride,
                                activationAlpha,
                                activationBeta);
    }
    else
    {
        auto yBuffer
            = findDeviceBuffer(_inferenceParams.y()->uid(), deviceBuffers, numDeviceBuffers);

        _runnableKernel->launch(handle.getStream(),
                                xBuffer.ptr,
                                yBuffer.ptr,
                                estMeanBuffer.ptr,
                                invVarianceBuffer.ptr,
                                scaleBuffer.ptr,
                                biasBuffer.ptr,
                                _channels,
                                _inCstride,
                                _batchCount,
                                _cStride,
                                _hwStride,
                                _batchStride,
                                activationAlpha,
                                activationBeta);
    }
}

} // namespace hip_kernel_provider::batchnorm
