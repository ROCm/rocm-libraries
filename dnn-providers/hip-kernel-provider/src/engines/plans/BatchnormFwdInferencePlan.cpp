// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "BatchnormFwdInferencePlan.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <stdexcept>

namespace hip_kernel_provider
{

BatchnormFwdInferenceParams::BatchnormFwdInferenceParams(
    const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes& attributes,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
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
    const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes& inferenceAttributes,
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(tensorMap.at(inferenceAttributes.x_tensor_uid()))
    , _y(tensorMap.at(inferenceAttributes.y_tensor_uid()))
    , _scale(tensorMap.at(inferenceAttributes.scale_tensor_uid()))
    , _bias(tensorMap.at(inferenceAttributes.bias_tensor_uid()))
    , _estMean(tensorMap.at(inferenceAttributes.mean_tensor_uid()))
    , _invVariance(tensorMap.at(inferenceAttributes.inv_variance_tensor_uid()))
    , _optActivation(hip_kernel_utils::parseActivation(pointwiseAttributes))
    , _activationOut(tensorMap.at(pointwiseAttributes.out_0_tensor_uid()))
{
}

const hipdnn_data_sdk::data_objects::TensorAttributes* BatchnormFwdInferenceParams::x() const
{
    return _x;
}

const hipdnn_data_sdk::data_objects::TensorAttributes* BatchnormFwdInferenceParams::y() const
{
    return _y;
}

const hipdnn_data_sdk::data_objects::TensorAttributes* BatchnormFwdInferenceParams::scale() const
{
    return _scale;
}

const hipdnn_data_sdk::data_objects::TensorAttributes* BatchnormFwdInferenceParams::bias() const
{
    return _bias;
}

const hipdnn_data_sdk::data_objects::TensorAttributes* BatchnormFwdInferenceParams::estMean() const
{
    return _estMean;
}

const hipdnn_data_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceParams::invVariance() const
{
    return _invVariance;
}

const std::optional<hip_kernel_utils::ActivationParams>&
    BatchnormFwdInferenceParams::optActivation() const
{
    return _optActivation;
}

const hipdnn_data_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceParams::activationOut() const
{
    return _activationOut;
}

BatchnormFwdInferencePlan::BatchnormFwdInferencePlan(BatchnormFwdInferenceParams&& inferenceParams,
                                                     const IKernelCompiler& kernelCompiler)
    : _inferenceParams(std::move(inferenceParams))
    , _kernelCompiler(kernelCompiler)
{
}

size_t BatchnormFwdInferencePlan::getWorkspaceSize(
    [[maybe_unused]] const HipKernelHandle& handle) const
{
    // No workspace needed for batchnorm inference
    return 0;
}

namespace
{

size_t computeVectorSize(bool isLayoutNhwc, int channels, unsigned int inCstride)
{
    if(isLayoutNhwc)
    {
        if(channels % 4 == 0)
        {
            return 4;
        }
        return channels % 2 == 0 ? 2 : 1;
    }

    if(inCstride % 4 == 0)
    {
        return 4;
    }
    return inCstride % 2 == 0 ? 2 : 1;
}

} // namespace

void BatchnormFwdInferencePlan::compile(const hipDeviceProp_t& deviceProperties)
{
    // Determine data type configuration
    auto xDataType = _inferenceParams.x()->data_type();
    auto scaleDataType = _inferenceParams.scale()->data_type();

    bool useFp16Mix = (xDataType == hipdnn_data_sdk::data_objects::DataType::HALF
                       && scaleDataType == hipdnn_data_sdk::data_objects::DataType::FLOAT);
    bool useBfp16Mix = (xDataType == hipdnn_data_sdk::data_objects::DataType::BFLOAT16
                        && scaleDataType == hipdnn_data_sdk::data_objects::DataType::FLOAT);
    bool useFp32 = !useFp16Mix && !useBfp16Mix;

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
        throw std::runtime_error("Unsupported tensor dimension: " + std::to_string(xDims->size()));
    }

    auto inCstride = static_cast<unsigned int>(h * w);

    // Detect layout: NHWC has C dimension (index 1) with stride 1, NCHW has stride H*W
    bool isLayoutNhwc = (xStrides->Get(1) == 1);

    // Calculate vector size based on layout
    auto vectorsize = computeVectorSize(isLayoutNhwc, c, inCstride);

    // Calculate block and grid dimensions
    size_t xlocalsize = 0;
    size_t xgridsize = 0;
    size_t ylocalsize = 0;
    size_t ygridsize = 0;
    size_t zlocalsize = 0;
    size_t zgridsize = 0;
    size_t maxLocalsize = 256;

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
    size_t activeThreadsXy = xgridsize * ygridsize;
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
    std::string archName(deviceProperties.gcnArchName);
    bool isGfx103x = (archName.find("gfx103") == 0);
    bool isGfx110x = (archName.find("gfx110") == 0);
    bool isGfx120x = (archName.find("gfx120") == 0);
    bool isGfx115x = (archName.find("gfx115") == 0);

    // Get activation parameters
    int nrnOpId = 0;

    if(_inferenceParams.optActivation().has_value() && _inferenceParams.activationOut() != nullptr)
    {
        const auto& activation = *_inferenceParams.optActivation();
        nrnOpId = static_cast<int>(activation.mode);
        _activationAlpha = static_cast<float>(activation.alpha);
        _activationBeta = static_cast<float>(activation.beta);
    }

    // Prepare compilation options
    std::vector<std::string> options;
    options.emplace_back("-I/opt/rocm/include");
    options.emplace_back(std::string("-DHIP_PLUGIN_USE_FP32=") + (useFp32 ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_USE_FP16=") + (useFp16Mix ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_USE_BFP16=") + (useBfp16Mix ? "1" : "0"));
    options.emplace_back("-DHIP_PLUGIN_USE_RNE_BFLOAT16=1");
    options.emplace_back(std::string("-DHIP_PLUGIN_USE_FPMIX=") + (useFp16Mix ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_USE_BFPMIX=") + (useBfp16Mix ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GRP0=") + std::to_string(xlocalsize));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GRP1=") + std::to_string(ylocalsize));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GRP2=") + std::to_string(zlocalsize));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_VEC_SIZE=") + std::to_string(vectorsize));
    options.emplace_back(std::string("-DHIP_PLUGIN_LAYOUT_NHWC=") + (isLayoutNhwc ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GFX103X=") + (isGfx103x ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GFX110X=") + (isGfx110x ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GFX120X=") + (isGfx120x ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GFX115X=") + (isGfx115x ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_NRN_OP_ID=") + std::to_string(nrnOpId));
    options.emplace_back(std::string("--offload-arch=") + deviceProperties.gcnArchName);

    // Compile kernel and configure launch dimensions
    _compiledProgram = _kernelCompiler.compile("BatchNormFwdInferSpatial.cpp", options);
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

void BatchnormFwdInferencePlan::execute(const HipKernelHandle& handle,
                                        const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                        uint32_t numDeviceBuffers,
                                        [[maybe_unused]] void* workspace) const
{
    if(!_runnableKernel)
    {
        throw std::runtime_error("BatchnormFwdInferencePlan::execute() called before compile()");
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
    auto invVarianceBuffer = hip_kernel_utils::findDeviceBuffer(
        _inferenceParams.invVariance()->uid(), deviceBuffers, numDeviceBuffers);

    // Launch kernel with appropriate output buffer
    if(_inferenceParams.optActivation().has_value() && _inferenceParams.activationOut() != nullptr)
    {
        auto activationOutBuffer = hip_kernel_utils::findDeviceBuffer(
            _inferenceParams.activationOut()->uid(), deviceBuffers, numDeviceBuffers);

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
                                _activationAlpha,
                                _activationBeta);
    }
    else
    {
        auto yBuffer = hip_kernel_utils::findDeviceBuffer(
            _inferenceParams.y()->uid(), deviceBuffers, numDeviceBuffers);

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
                                _activationAlpha,
                                _activationBeta);
    }
}

}
