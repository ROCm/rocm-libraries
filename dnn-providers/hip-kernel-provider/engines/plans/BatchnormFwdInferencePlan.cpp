// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "BatchnormFwdInferencePlan.hpp"
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
                                                     bool benchmarkingEnabled)
    : _inferenceParams(std::move(inferenceParams))
    , _benchmarkingEnabled(benchmarkingEnabled)
{
}

size_t BatchnormFwdInferencePlan::getWorkspaceSize(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle) const
{
    // No workspace needed for batchnorm inference
    return 0;
}

void BatchnormFwdInferencePlan::execute(const HipdnnEnginePluginHandle& handle,
                                        const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                        uint32_t numDeviceBuffers,
                                        [[maybe_unused]] void* workspace) const
{
    // TODO: This is an initial implementation to get batchnorm working. It needs to be enhanced
    // to match the full MIOpen solver capabilities.

    // Get device and properties
    int device;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));

    // Determine data type configuration (matching MIOpen solver logic)
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

    int n;
    int c;
    int hw;
    int nStride;
    int cStride;
    int wStride;

    // Check if 4D (NCHW/NHWC) or 5D (NCDHW/NDHWC)
    if(xDims->size() == 4)
    {
        // 4D tensor: N, C, H, W
        n = static_cast<int>(xDims->Get(0));
        c = static_cast<int>(xDims->Get(1));
        int h = static_cast<int>(xDims->Get(2));
        int w = static_cast<int>(xDims->Get(3));
        hw = h * w;

        nStride = static_cast<int>(xStrides->Get(0));
        cStride = static_cast<int>(xStrides->Get(1));
        wStride = static_cast<int>(xStrides->Get(3));
    }
    else if(xDims->size() == 5)
    {
        // 5D tensor: N, C, D, H, W
        n = static_cast<int>(xDims->Get(0));
        c = static_cast<int>(xDims->Get(1));
        int d = static_cast<int>(xDims->Get(2));
        int h = static_cast<int>(xDims->Get(3));
        int w = static_cast<int>(xDims->Get(4));
        hw = d * h * w; // For 5D, spatial volume is D*H*W

        nStride = static_cast<int>(xStrides->Get(0));
        cStride = static_cast<int>(xStrides->Get(1));
        wStride = static_cast<int>(xStrides->Get(4));
    }
    else
    {
        throw std::runtime_error("Unsupported tensor dimension: " + std::to_string(xDims->size()));
    }

    // Prepare options for compilation
    // For NCHW spatial mode: GRP0=1 (xlocalsize), GRP1=256 (ylocalsize), GRP2=1 (zlocalsize)
    std::vector<std::string> options;
    options.emplace_back("-I/opt/rocm/include");
    // Only ONE of these can be 1 (FP32, FP16Mix, or BFP16Mix)
    options.emplace_back(std::string("-DHIP_PLUGIN_USE_FP32=") + (useFp32 ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_USE_FP16=")
                         + "0"); // Not used for mixed precision
    options.emplace_back(std::string("-DHIP_PLUGIN_USE_BFP16=")
                         + "0"); // Not used for mixed precision
    options.emplace_back(std::string("-DHIP_PLUGIN_USE_FPMIX=") + (useFp16Mix ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_USE_BFPMIX=") + (useBfp16Mix ? "1" : "0"));
    options.emplace_back("-DMIO_BN_GRP0=1");
    options.emplace_back("-DMIO_BN_GRP1=256");
    options.emplace_back("-DMIO_BN_GRP2=1");
    options.emplace_back("-DMIO_BN_VEC_SIZE=1");
    options.emplace_back("-DMIO_LAYOUT_NHWC=0");

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
    options.emplace_back(std::string("-DHIP_PLUGIN_NRN_OP_ID=") + std::to_string(nrnOpId));
    options.emplace_back(std::string("--offload-arch=") + props.gcnArchName);

    auto hipProgram = HipProgram("BatchNormFwdInferSpatial.cpp", options);
    auto hipKernel = HipKernel(hipProgram, "BatchNormFwdInferSpatialEstInvVar");

    // Use tensor dimensions extracted earlier
    auto batchSize = static_cast<unsigned int>(n);
    auto channels = static_cast<unsigned int>(c);
    auto hwVolume = static_cast<unsigned int>(hw);

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

    // Calculate grid/block dimensions based on MIOpen solver logic
    // For NCHW spatial mode:
    // - x-dimension spans channels (c)
    // - y-dimension spans spatial elements (h*w)
    // - z-dimension spans batches (n)

    // Block dimensions (MIO_BN_GRP0, GRP1, GRP2)
    const unsigned int xlocalsize = 1; // For NCHW spatial
    const unsigned int ylocalsize = 256; // max_localsize
    const unsigned int zlocalsize = 1;
    hipKernel.SetBlockSize(xlocalsize, ylocalsize, zlocalsize);

    // Grid dimensions - must cover all channels, spatial elements, and batches
    const unsigned int vectorsize
        = 1; // TODO: Support vectorization (hwVolume % 4 == 0 ? 4 : hwVolume % 2 == 0 ? 2 : 1)
    unsigned int xgridsize = ((channels + xlocalsize - 1) / xlocalsize) * xlocalsize;
    unsigned int ygridsize = ((hwVolume / vectorsize + ylocalsize - 1) / ylocalsize) * ylocalsize;
    unsigned int zgridsize = batchSize; // TODO: Optimize based on GPU compute units

    // Convert to grid dimensions (in blocks, not threads)
    hipKernel.SetGridSize(xgridsize / xlocalsize, ygridsize / ylocalsize, zgridsize / zlocalsize);

    unsigned int hwStride = static_cast<unsigned int>(wStride);
    unsigned int batchStride = static_cast<unsigned int>(nStride);

    if(_inferenceParams.optActivation().has_value() && _inferenceParams.activationOut() != nullptr)
    {
        auto activationOutBuffer = hip_kernel_utils::findDeviceBuffer(
            _inferenceParams.activationOut()->uid(), deviceBuffers, numDeviceBuffers);

        hipKernel.Launch(handle.getStream(),
                         xBuffer.ptr,
                         activationOutBuffer.ptr,
                         estMeanBuffer.ptr,
                         invVarianceBuffer.ptr,
                         scaleBuffer.ptr,
                         biasBuffer.ptr,
                         channels,
                         hwVolume,
                         batchSize,
                         cStride,
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
                         invVarianceBuffer.ptr,
                         scaleBuffer.ptr,
                         biasBuffer.ptr,
                         channels,
                         hwVolume,
                         batchSize,
                         cStride,
                         hwStride,
                         batchStride,
                         alpha,
                         beta);
    }
}

}
