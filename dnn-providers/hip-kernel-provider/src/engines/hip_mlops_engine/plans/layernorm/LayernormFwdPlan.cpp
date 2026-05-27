// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "LayernormFwdPlan.hpp"

#include "HipKernelUtils.hpp"
#include "engines/hip_mlops_engine/plans/layernorm/LayernormUtilities.hpp"
#include "hip/IKernelCompiler.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hip_kernel_provider::layernorm
{

LayernormFwdParams::LayernormFwdParams(
    const hipdnn_flatbuffers_sdk::data_objects::LayernormAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(tensorMap.at(attributes.x_tensor_uid()))
    , _y(tensorMap.at(attributes.y_tensor_uid()))
    , _scale(tensorMap.at(attributes.scale_tensor_uid()))
    , _bias(tensorMap.at(attributes.bias_tensor_uid()))
    , _mean(attributes.mean_tensor_uid().has_value()
                ? tensorMap.at(attributes.mean_tensor_uid().value())
                : nullptr)
    , _invVariance(attributes.inv_variance_tensor_uid().has_value()
                       ? tensorMap.at(attributes.inv_variance_tensor_uid().value())
                       : nullptr)
    , _epsilon((tensorMap.at(attributes.epsilon_tensor_uid())))
{
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* LayernormFwdParams::x() const
{
    return _x;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* LayernormFwdParams::y() const
{
    return _y;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* LayernormFwdParams::scale() const
{
    return _scale;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* LayernormFwdParams::bias() const
{
    return _bias;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* LayernormFwdParams::mean() const
{
    return _mean;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    LayernormFwdParams::invVariance() const
{
    return _invVariance;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* LayernormFwdParams::epsilon() const
{
    return _epsilon;
}

LayernormFwdPlan::LayernormFwdPlan(LayernormFwdParams&& params)
    : _params(std::move(params))
{
}

size_t LayernormFwdPlan::getWorkspaceSize([[maybe_unused]] const HipKernelHandle& handle) const
{
    // No workspace needed for layernorm
    return 0;
}

void LayernormFwdPlan::compile(const IKernelCompiler& kernelCompiler,
                               const hipDeviceProp_t& deviceProperties)
{
    // Determine data type configuration
    auto xDataType = _params.x()->data_type();

    // Extract dimensions from x tensor
    const auto* xDims = _params.x()->dims();
    const auto* xStrides = _params.x()->strides();
    const auto strideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(
        std::vector<int64_t>(xStrides->begin(), xStrides->end()));

    const size_t normalizedDim
        = layernorm::guessNormalizedDim(_params.x(), _params.scale(), _params.mean());
    long outerSize = 1;
    long innerSize = 1;
    long stride = 1;
    const auto layoutNHWC = hipdnn_data_sdk::utilities::TensorLayout::NHWC;
    const auto layoutNDHWC = hipdnn_data_sdk::utilities::TensorLayout::NDHWC;

    if(normalizedDim > 1
       && (strideOrder == layoutNHWC.strideOrder || strideOrder == layoutNDHWC.strideOrder))
    {
        stride = static_cast<long>(xDims->Get(1));
    }

    for(unsigned int i = 0; i < xDims->size(); ++i)
    {
        if(i < normalizedDim)
        {
            if(stride == 1 || i != 1) // Don't add C to outerSize if there is a stride
            {
                outerSize *= static_cast<long>(xDims->Get(i));
            }
        }
        else
        {
            innerSize *= static_cast<long>(xDims->Get(i));
        }
    }

    const long xlocalsize = 1024;
    const long xgridsize = outerSize * stride;
    const long ylocalsize = 1;
    const long ygridsize = 1;
    const long zlocalsize = 1;
    const long zgridsize = 1;

    // Prepare compilation options
    std::vector<std::string> options;
    auto rocmPath
        = hipdnn_data_sdk::utilities::trim(hipdnn_data_sdk::utilities::getEnv("ROCM_PATH"));
    if(!rocmPath.empty())
    {
        auto rocmIncludeArg = "-I" + rocmPath + "/include";
        options.emplace_back(rocmIncludeArg);
        HIPDNN_PLUGIN_LOG_INFO(
            "LayernormFwdPlan: HIPRTC compile ROCm include path: " << rocmIncludeArg);
    }
    options.emplace_back(
        std::string("-DHIP_PLUGIN_USE_FP32=")
        + (xDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT ? "1" : "0"));
    options.emplace_back(
        std::string("-DHIP_PLUGIN_USE_FP16=")
        + (xDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::HALF ? "1" : "0"));
    options.emplace_back(
        std::string("-DHIP_PLUGIN_USE_BFP16=")
        + (xDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16 ? "1" : "0"));
    options.emplace_back(std::string("-DOUTER_SIZE=") + std::to_string(outerSize));
    options.emplace_back(std::string("-DINNER_SIZE=") + std::to_string(innerSize));
    options.emplace_back(std::string("-DSTRIDE=") + std::to_string(stride));
    options.emplace_back(std::string("-DLOCAL_SIZE=") + std::to_string(xlocalsize));
    options.emplace_back(std::string("--offload-arch=") + deviceProperties.gcnArchName);

    // Compile kernel and configure launch dimensions
    _compiledProgram = kernelCompiler.compile("LayernormFwd.cpp", options);
    _runnableKernel = _compiledProgram->getKernel("LayernormFwd");

    _runnableKernel->setBlockSize(static_cast<unsigned int>(xlocalsize),
                                  static_cast<unsigned int>(ylocalsize),
                                  static_cast<unsigned int>(zlocalsize));
    _runnableKernel->setGridSize(static_cast<unsigned int>(xgridsize),
                                 static_cast<unsigned int>(ygridsize),
                                 static_cast<unsigned int>(zgridsize));
}

void LayernormFwdPlan::execute(const HipKernelHandle& handle,
                               const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                               uint32_t numDeviceBuffers,
                               [[maybe_unused]] void* workspace) const
{
    if(!_runnableKernel)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "LayernormFwdPlan::execute() called before compile()");
    }

    // Get device buffer pointers
    auto xBuffer
        = hip_kernel_utils::findDeviceBuffer(_params.x()->uid(), deviceBuffers, numDeviceBuffers);
    auto yBuffer
        = hip_kernel_utils::findDeviceBuffer(_params.y()->uid(), deviceBuffers, numDeviceBuffers);
    auto scaleBuffer = hip_kernel_utils::findDeviceBuffer(
        _params.scale()->uid(), deviceBuffers, numDeviceBuffers);
    auto biasBuffer = hip_kernel_utils::findDeviceBuffer(
        _params.bias()->uid(), deviceBuffers, numDeviceBuffers);
    auto meanBuffer = _params.mean() != nullptr
                          ? hip_kernel_utils::findDeviceBuffer(
                                _params.mean()->uid(), deviceBuffers, numDeviceBuffers)
                          : hipdnnPluginDeviceBuffer_t{-1, nullptr};
    auto invVarianceBuffer = _params.invVariance() != nullptr
                                 ? hip_kernel_utils::findDeviceBuffer(_params.invVariance()->uid(),
                                                                      deviceBuffers,
                                                                      numDeviceBuffers)
                                 : hipdnnPluginDeviceBuffer_t{-1, nullptr};

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT epsilonTensor;
    _params.epsilon()->UnPackTo(&epsilonTensor);
    double epsilon
        = hipdnn_flatbuffers_sdk::utilities::extractDoubleFromTensorValue(epsilonTensor, "Epsilon");

    // Launch kernel
    _runnableKernel->launch(handle.getStream(),
                            xBuffer.ptr,
                            yBuffer.ptr,
                            scaleBuffer.ptr,
                            biasBuffer.ptr,
                            meanBuffer.ptr,
                            invVarianceBuffer.ptr,
                            static_cast<float>(epsilon));
}

} // namespace hip_kernel_provider::layernorm
