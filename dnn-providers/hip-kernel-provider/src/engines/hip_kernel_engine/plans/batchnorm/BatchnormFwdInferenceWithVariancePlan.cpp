// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "BatchnormFwdInferenceWithVariancePlan.hpp"
#include "engines/hip_kernel_engine/plans/PlanUtils.hpp"
#include "hip/HipKernelCompileOptions.hpp"

#include "hip/IKernelCompiler.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hip_kernel_provider::batchnorm
{

BatchnormFwdInferenceWithVarianceParams::BatchnormFwdInferenceWithVarianceParams(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(tensorMap.at(attributes.x_tensor_uid()))
    , _y(tensorMap.at(attributes.y_tensor_uid()))
    , _scale(tensorMap.at(attributes.scale_tensor_uid()))
    , _bias(tensorMap.at(attributes.bias_tensor_uid()))
    , _estMean(tensorMap.at(attributes.mean_tensor_uid()))
    , _estVariance(tensorMap.at(attributes.variance_tensor_uid()))
    , _activationOut(nullptr)
{
    // Extract epsilon value from pass-by-value tensor (cast to double for kernel compatibility)
    auto epsilonTensorAttr = tensorMap.at(attributes.epsilon_tensor_uid());
    _epsilonValue = hipdnn_flatbuffers_sdk::utilities::extractDoubleFromTensorValue(
        epsilonTensorAttr, "Epsilon");
}

BatchnormFwdInferenceWithVarianceParams::BatchnormFwdInferenceWithVarianceParams(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt&
        inferenceAttributes,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(tensorMap.at(inferenceAttributes.x_tensor_uid()))
    , _y(tensorMap.at(inferenceAttributes.y_tensor_uid()))
    , _scale(tensorMap.at(inferenceAttributes.scale_tensor_uid()))
    , _bias(tensorMap.at(inferenceAttributes.bias_tensor_uid()))
    , _estMean(tensorMap.at(inferenceAttributes.mean_tensor_uid()))
    , _estVariance(tensorMap.at(inferenceAttributes.variance_tensor_uid()))
    , _optActivation(hip_kernel_utils::parseActivation(pointwiseAttributes))
    , _activationOut(tensorMap.at(pointwiseAttributes.out_0_tensor_uid()))
{
    // Extract epsilon value from pass-by-value tensor (cast to double for kernel compatibility)
    auto epsilonTensorAttr = tensorMap.at(inferenceAttributes.epsilon_tensor_uid());
    _epsilonValue = hipdnn_flatbuffers_sdk::utilities::extractDoubleFromTensorValue(
        epsilonTensorAttr, "Epsilon");
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceWithVarianceParams::x() const
{
    return _x;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceWithVarianceParams::y() const
{
    return _y;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceWithVarianceParams::scale() const
{
    return _scale;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceWithVarianceParams::bias() const
{
    return _bias;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceWithVarianceParams::estMean() const
{
    return _estMean;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceWithVarianceParams::estVariance() const
{
    return _estVariance;
}

double BatchnormFwdInferenceWithVarianceParams::epsilonValue() const
{
    return _epsilonValue;
}

const std::optional<hip_kernel_utils::ActivationParams>&
    BatchnormFwdInferenceWithVarianceParams::optActivation() const
{
    return _optActivation;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceWithVarianceParams::activationOut() const
{
    return _activationOut;
}

BatchnormFwdInferenceWithVariancePlan::BatchnormFwdInferenceWithVariancePlan(
    BatchnormFwdInferenceWithVarianceParams&& inferenceParams)
    : _inferenceParams(std::move(inferenceParams))
{
}

size_t BatchnormFwdInferenceWithVariancePlan::getWorkspaceSize(
    [[maybe_unused]] const HipKernelHandle& handle) const
{
    // No workspace needed for batchnorm inference with variance
    return 0;
}

void BatchnormFwdInferenceWithVariancePlan::compile(const IKernelCompiler& kernelCompiler,
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

    // Detect layout: NHWC has C dimension (index 1) with stride 1, NCHW has stride H*W
    const bool isLayoutNHWC = hip_kernel_utils::isChannelLastLayout(_inferenceParams.x());

    // Calculate vector size based on layout
    auto vectorsize = computeVectorSize(isLayoutNHWC, c, inCstride);

    // Calculate block and grid dimensions
    size_t xlocalsize = 0;
    size_t xgridsize = 0;
    size_t ylocalsize = 0;
    size_t ygridsize = 0;
    size_t zlocalsize = 0;
    size_t zgridsize = 0;
    const size_t maxLocalsize = 256;

    if(isLayoutNHWC)
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

    // Detect GPU architecture
    const std::string archName(deviceProperties.gcnArchName);
    const bool isGfx103X = (archName.find("gfx103") == 0);
    const bool isGfx110X = (archName.find("gfx110") == 0);
    const bool isGfx120X = (archName.find("gfx120") == 0);
    const bool isGfx115X = (archName.find("gfx115") == 0);

    // Get activation mode
    auto activationMode = hip_kernel_utils::ActivationMode::PASTHRU;

    if(_inferenceParams.optActivation().has_value() && _inferenceParams.activationOut() != nullptr)
    {
        activationMode = (*_inferenceParams.optActivation()).mode;
    }

    // Prepare compilation options
    HipKernelCompileOptions options(_inferenceParams.x(), deviceProperties, activationMode);
    options.add("HIP_PLUGIN_USE_FPMIX", useFp16Mix);
    options.add("HIP_PLUGIN_USE_BFPMIX", useBfp16Mix);
    options.add("HIP_PLUGIN_BN_GRP0", xlocalsize);
    options.add("HIP_PLUGIN_BN_GRP1", ylocalsize);
    options.add("HIP_PLUGIN_BN_GRP2", zlocalsize);
    options.add("HIP_PLUGIN_BN_VEC_SIZE", vectorsize);
    options.add("HIP_PLUGIN_BN_GFX103X", isGfx103X);
    options.add("HIP_PLUGIN_BN_GFX110X", isGfx110X);
    options.add("HIP_PLUGIN_BN_GFX120X", isGfx120X);
    options.add("HIP_PLUGIN_BN_GFX115X", isGfx115X);

    // Compile kernel and configure launch dimensions
    _compiledProgram = kernelCompiler.compile("BatchNormFwdInferSpatial.cpp", options);
    _runnableKernel = _compiledProgram->getKernel("BatchNormFwdInferSpatialEst");

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

void BatchnormFwdInferenceWithVariancePlan::execute(const HipKernelHandle& handle,
                                                    const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                                    uint32_t numDeviceBuffers,
                                                    [[maybe_unused]] void* workspace) const
{
    if(!_runnableKernel)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "BatchnormFwdInferenceWithVariancePlan::execute() called before compile()");
    }

    // Get device buffer pointers
    auto xBuffer = hip_kernel_utils::findDeviceBuffer(
        _inferenceParams.x()->uid(), deviceBuffers, numDeviceBuffers);
    auto scaleBuffer = hip_kernel_utils::findDeviceBuffer(
        _inferenceParams.scale()->uid(), deviceBuffers, numDeviceBuffers);
    auto biasBuffer = hip_kernel_utils::findDeviceBuffer(
        _inferenceParams.bias()->uid(), deviceBuffers, numDeviceBuffers);
    auto estMeanBuffer = hip_kernel_utils::findDeviceBuffer(
        _inferenceParams.estMean()->uid(), deviceBuffers, numDeviceBuffers);
    auto estVarianceBuffer = hip_kernel_utils::findDeviceBuffer(
        _inferenceParams.estVariance()->uid(), deviceBuffers, numDeviceBuffers);

    // Get epsilon
    double epsilon = _inferenceParams.epsilonValue();

    float activationAlpha = 0.0f;
    float activationBeta = 0.0f;

    // Launch kernel with appropriate output buffer
    if(_inferenceParams.optActivation().has_value() && _inferenceParams.activationOut() != nullptr)
    {
        auto activationOutBuffer = hip_kernel_utils::findDeviceBuffer(
            _inferenceParams.activationOut()->uid(), deviceBuffers, numDeviceBuffers);

        // Get activation parameters
        const auto& activation = *_inferenceParams.optActivation();
        activationAlpha = static_cast<float>(activation.alpha);
        activationBeta = static_cast<float>(activation.beta);

        _runnableKernel->launch(handle.getStream(),
                                xBuffer.ptr,
                                activationOutBuffer.ptr,
                                estMeanBuffer.ptr,
                                estVarianceBuffer.ptr,
                                scaleBuffer.ptr,
                                biasBuffer.ptr,
                                epsilon,
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
        auto yBuffer = hip_kernel_utils::findDeviceBuffer(
            _inferenceParams.y()->uid(), deviceBuffers, numDeviceBuffers);

        _runnableKernel->launch(handle.getStream(),
                                xBuffer.ptr,
                                yBuffer.ptr,
                                estMeanBuffer.ptr,
                                estVarianceBuffer.ptr,
                                scaleBuffer.ptr,
                                biasBuffer.ptr,
                                epsilon,
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
