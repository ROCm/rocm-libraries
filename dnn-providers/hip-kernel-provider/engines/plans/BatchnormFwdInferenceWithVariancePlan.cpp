// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "BatchnormFwdInferenceWithVariancePlan.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "hip/HipKernel.hpp"
#include "hip/HipProgram.hpp"
#include "hip/HipUtils.hpp"

#include <hip/hip_runtime_api.h>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <sstream>
#include <stdexcept>

namespace hip_kernel_plugin
{

BatchnormFwdInferenceWithVarianceParams::BatchnormFwdInferenceWithVarianceParams(
    const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributesVarianceExt& attributes,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
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
    _epsilonValue
        = hipdnn_data_sdk::utilities::extractDoubleFromTensorValue(epsilonTensorAttr, "Epsilon");
}

BatchnormFwdInferenceWithVarianceParams::BatchnormFwdInferenceWithVarianceParams(
    const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributesVarianceExt&
        inferenceAttributes,
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
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
    _epsilonValue
        = hipdnn_data_sdk::utilities::extractDoubleFromTensorValue(epsilonTensorAttr, "Epsilon");
}

const hipdnn_data_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceWithVarianceParams::x() const
{
    return _x;
}

const hipdnn_data_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceWithVarianceParams::y() const
{
    return _y;
}

const hipdnn_data_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceWithVarianceParams::scale() const
{
    return _scale;
}

const hipdnn_data_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceWithVarianceParams::bias() const
{
    return _bias;
}

const hipdnn_data_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceWithVarianceParams::estMean() const
{
    return _estMean;
}

const hipdnn_data_sdk::data_objects::TensorAttributes*
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

const hipdnn_data_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceWithVarianceParams::activationOut() const
{
    return _activationOut;
}

BatchnormFwdInferenceWithVariancePlan::BatchnormFwdInferenceWithVariancePlan(
    BatchnormFwdInferenceWithVarianceParams&& inferenceParams, bool benchmarkingEnabled)
    : _inferenceParams(std::move(inferenceParams))
    , _benchmarkingEnabled(benchmarkingEnabled)
{
}

size_t BatchnormFwdInferenceWithVariancePlan::getWorkspaceSize(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle) const
{
    // No workspace needed for batchnorm inference with variance
    return 0;
}

void BatchnormFwdInferenceWithVariancePlan::execute(const HipdnnEnginePluginHandle& handle,
                                                    const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                                    uint32_t numDeviceBuffers,
                                                    [[maybe_unused]] void* workspace) const
{
    // Get device and properties
    int device;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));

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

    int n, c, h, w;
    int nStride, cStride, wStride;

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
        int d = static_cast<int>(xDims->Get(2));
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

    unsigned int in_cstride = static_cast<unsigned int>(h * w);

    // Detect layout: NHWC has C dimension (index 1) with stride 1, NCHW has stride H*W
    bool isLayoutNHWC = (xStrides->Get(1) == 1);

    // Calculate vector size based on layout
    size_t vectorsize = isLayoutNHWC ? (c % 4 == 0 ? 4 : (c % 2 == 0 ? 2 : 1))
                                     : (in_cstride % 4 == 0 ? 4 : (in_cstride % 2 == 0 ? 2 : 1));

    // Calculate block and grid dimensions
    size_t xlocalsize, xgridsize, ylocalsize, ygridsize, zlocalsize, zgridsize;
    size_t max_localsize = 256;

    if(isLayoutNHWC)
    {
        xlocalsize = std::min(static_cast<size_t>(c) / vectorsize, max_localsize);
        xgridsize
            = ((static_cast<size_t>(c) / vectorsize) + xlocalsize - 1) / xlocalsize * xlocalsize;

        ylocalsize = max_localsize / xlocalsize;
        ygridsize = (in_cstride + ylocalsize - 1) / ylocalsize * ylocalsize;
    }
    else
    {
        xlocalsize = 1;
        xgridsize = ((static_cast<size_t>(c) + xlocalsize - 1) / xlocalsize) * xlocalsize;

        ylocalsize = max_localsize;
        ygridsize = ((in_cstride / vectorsize + ylocalsize - 1) / ylocalsize) * ylocalsize;
    }

    zlocalsize = 1;
    size_t active_threads_xy = xgridsize * ygridsize;
    size_t max_active_threads
        = static_cast<size_t>(props.multiProcessorCount) * 32 * static_cast<size_t>(props.warpSize);

    if(active_threads_xy < max_active_threads)
    {
        zgridsize
            = std::min(size_t{max_active_threads / active_threads_xy}, static_cast<size_t>(n));
    }
    else
    {
        zgridsize = 1;
    }

    // Detect GPU architecture
    std::string archName(props.gcnArchName);
    bool isGfx103X = (archName.find("gfx103") == 0);
    bool isGfx110X = (archName.find("gfx110") == 0);
    bool isGfx120X = (archName.find("gfx120") == 0);
    bool isGfx115X = (archName.find("gfx115") == 0);

    // Get activation parameters
    int nrnOpId = 0;
    float alpha = 0.0f;
    float beta = 0.0f;

    if(_inferenceParams.optActivation().has_value() && _inferenceParams.activationOut() != nullptr)
    {
        const auto& activation = *_inferenceParams.optActivation();
        nrnOpId = static_cast<int>(activation.mode);
        alpha = static_cast<float>(activation.alpha);
        beta = static_cast<float>(activation.beta);
    }

    // Get epsilon
    double epsilon = _inferenceParams.epsilonValue();

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
    options.emplace_back(std::string("-DHIP_PLUGIN_LAYOUT_NHWC=") + (isLayoutNHWC ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GFX103X=") + (isGfx103X ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GFX110X=") + (isGfx110X ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GFX120X=") + (isGfx120X ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GFX115X=") + (isGfx115X ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_NRN_OP_ID=") + std::to_string(nrnOpId));
    options.emplace_back(std::string("--offload-arch=") + props.gcnArchName);

    // Create and configure kernel
    auto hipProgram = HipProgram("BatchNormFwdInferSpatial.cpp", options);
    auto hipKernel = HipKernel(hipProgram, "BatchNormFwdInferSpatialEst");

    hipKernel.SetBlockSize(static_cast<unsigned int>(xlocalsize),
                           static_cast<unsigned int>(ylocalsize),
                           static_cast<unsigned int>(zlocalsize));
    hipKernel.SetGridSize(static_cast<unsigned int>(xgridsize / xlocalsize),
                          static_cast<unsigned int>(ygridsize / ylocalsize),
                          static_cast<unsigned int>(zgridsize / zlocalsize));

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

    unsigned int hwStride = static_cast<unsigned int>(wStride);
    unsigned int batchStride = static_cast<unsigned int>(nStride);

    // Launch kernel with appropriate output buffer
    if(_inferenceParams.optActivation().has_value() && _inferenceParams.activationOut() != nullptr)
    {
        auto activationOutBuffer = hip_kernel_utils::findDeviceBuffer(
            _inferenceParams.activationOut()->uid(), deviceBuffers, numDeviceBuffers);

        hipKernel.Launch(handle.getStream(),
                         xBuffer.ptr,
                         activationOutBuffer.ptr,
                         estMeanBuffer.ptr,
                         estVarianceBuffer.ptr,
                         scaleBuffer.ptr,
                         biasBuffer.ptr,
                         epsilon,
                         static_cast<unsigned int>(c),
                         static_cast<unsigned int>(in_cstride),
                         static_cast<unsigned int>(n),
                         static_cast<unsigned int>(cStride),
                         hwStride,
                         batchStride,
                         alpha,
                         beta);
    }
    else
    {
        auto yBuffer = hip_kernel_utils::findDeviceBuffer(
            _inferenceParams.y()->uid(), deviceBuffers, numDeviceBuffers);

        hipKernel.Launch(handle.getStream(),
                         xBuffer.ptr,
                         yBuffer.ptr,
                         estMeanBuffer.ptr,
                         estVarianceBuffer.ptr,
                         scaleBuffer.ptr,
                         biasBuffer.ptr,
                         epsilon,
                         static_cast<unsigned int>(c),
                         static_cast<unsigned int>(in_cstride),
                         static_cast<unsigned int>(n),
                         static_cast<unsigned int>(cStride),
                         hwStride,
                         batchStride,
                         alpha,
                         beta);
    }
}

}
