// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "RMSnormFwdPlan.hpp"
#include "RMSnormApplicabilityChecks.hpp"

#include "HipKernelUtils.hpp"
#include "hip/IKernelCompiler.hpp"

#include <cstdint>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hip_kernel_provider
{

RMSnormFwdParams::RMSnormFwdParams(
    const hipdnn_data_sdk::data_objects::RMSNormAttributes& attributes,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(tensorMap.at(attributes.x_tensor_uid()))
    , _scale(tensorMap.at(attributes.scale_tensor_uid()))
    , _bias(attributes.bias_tensor_uid().has_value()
                ? tensorMap.at(attributes.bias_tensor_uid().value())
                : nullptr)
    , _y(tensorMap.at(attributes.y_tensor_uid()))
    , _invRMS(attributes.inv_rms_tensor_uid().has_value()
                  ? tensorMap.at(attributes.inv_rms_tensor_uid().value())
                  : nullptr)
{
    auto epsilonTensorAttr = tensorMap.at(attributes.epsilon_tensor_uid());
    _epsilonValue = static_cast<float>(
        hipdnn_data_sdk::utilities::extractDoubleFromTensorValue(epsilonTensorAttr, "Epsilon"));
}

const hipdnn_data_sdk::data_objects::TensorAttributes* RMSnormFwdParams::x() const
{
    return _x;
}

const hipdnn_data_sdk::data_objects::TensorAttributes* RMSnormFwdParams::scale() const
{
    return _scale;
}

float RMSnormFwdParams::epsilon() const
{
    return _epsilonValue;
}

const hipdnn_data_sdk::data_objects::TensorAttributes* RMSnormFwdParams::bias() const
{
    return _bias;
}

const hipdnn_data_sdk::data_objects::TensorAttributes* RMSnormFwdParams::y() const
{
    return _y;
}

const hipdnn_data_sdk::data_objects::TensorAttributes* RMSnormFwdParams::invRMS() const
{
    return _invRMS;
}

RMSnormFwdPlan::RMSnormFwdPlan(RMSnormFwdParams&& params)
    : _params(std::move(params))
{
}

size_t RMSnormFwdPlan::getWorkspaceSize([[maybe_unused]] const HipKernelHandle& handle) const
{
    // No workspace needed for RMS norrm
    return 0;
}

void RMSnormFwdPlan::compile(const IKernelCompiler& kernelCompiler,
                             const hipDeviceProp_t& deviceProperties)
{
    // Determine input/output data type configuration
    auto ioDataType = _params.x()->data_type();
    const bool useFp16 = ioDataType == hipdnn_data_sdk::data_objects::DataType::HALF;
    const bool useBfp16 = ioDataType == hipdnn_data_sdk::data_objects::DataType::BFLOAT16;
    const bool useFp32 = !useFp16 && !useBfp16;
    std::string ioTypeString;
    switch(ioDataType)
    {
    case hipdnn_data_sdk::data_objects::DataType::HALF:
        ioTypeString = "half";
        break;
    case hipdnn_data_sdk::data_objects::DataType::BFLOAT16:
        ioTypeString = "ushort";
        break;
    default:
        ioTypeString = "float";
        break;
    }

    // Extract dimensions from x tensor
    const auto* xDims = _params.x()->dims();
    const auto* xStrides = _params.x()->strides();
    int64_t cStride = xStrides->Get(1);
    int64_t cSize = xDims->Get(1);
    int64_t nSize = xDims->Get(0);

    if(xDims->size() < 4 || xDims->size() > 5)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Unsupported tensor dimension: "
                                                           + std::to_string(xDims->size()));
    }

    // Calculate block and grid dimensions
    size_t xlocalsize = LOCAL_SIZE;
    size_t xgridsize = static_cast<size_t>(nSize * cStride) * xlocalsize;
    size_t ylocalsize = 1;
    size_t ygridsize = 1;
    size_t zlocalsize = 1;
    size_t zgridsize = 1;

    // Detect GPU architecture
    std::string archName(deviceProperties.gcnArchName);
    bool isGfx103x = (archName.find("gfx103") == 0);
    bool isGfx110x = (archName.find("gfx110") == 0);
    bool isGfx120x = (archName.find("gfx120") == 0);
    bool isGfx115x = (archName.find("gfx115") == 0);

    // Prepare compilation options
    std::vector<std::string> options;
    auto rocmPath
        = hipdnn_data_sdk::utilities::trim(hipdnn_data_sdk::utilities::getEnv("ROCM_PATH"));
    if(!rocmPath.empty())
    {
        auto rocmIncludeArg = "-I" + rocmPath + "/include";
        options.emplace_back(rocmIncludeArg);
        HIPDNN_PLUGIN_LOG_INFO(
            "RMSnormFwdPlan: HIPRTC compile ROCm include path: " << rocmIncludeArg);
    }

    options.emplace_back(std::string("-DHIP_PLUGIN_USE_FP32=") + (useFp32 ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_USE_FP16=") + (useFp16 ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_USE_BFP16=") + (useBfp16 ? "1" : "0"));
    options.emplace_back("-DHIP_PLUGIN_USE_RNE_BFLOAT16=1");
    options.emplace_back(std::string("-DHIP_PLUGIN_RMSNORM_C_STRIDE=") + std::to_string(cStride));
    options.emplace_back(std::string("-DHIP_PLUGIN_RMSNORM_C_SIZE=") + std::to_string(cSize));
    options.emplace_back(std::string("-DHIP_PLUGIN_RMSNORM_IO_TYPE=") + ioTypeString);
    options.emplace_back(std::string("-DHIP_PLUGIN_RMSNORM_LOCAL_SIZE=")
                         + std::to_string(LOCAL_SIZE));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GFX103X=") + (isGfx103x ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GFX110X=") + (isGfx110x ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GFX120X=") + (isGfx120x ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GFX115X=") + (isGfx115x ? "1" : "0"));
    options.emplace_back(std::string("--offload-arch=") + deviceProperties.gcnArchName);

    // Compile kernel and configure launch dimensions
    _compiledProgram = kernelCompiler.compile("RMSNormFwd.cpp", options);
    _runnableKernel = _compiledProgram->getKernel("RMSnormFwd");

    _runnableKernel->setBlockSize(static_cast<unsigned int>(xlocalsize),
                                  static_cast<unsigned int>(ylocalsize),
                                  static_cast<unsigned int>(zlocalsize));
    _runnableKernel->setGridSize(static_cast<unsigned int>(xgridsize / xlocalsize),
                                 static_cast<unsigned int>(ygridsize / ylocalsize),
                                 static_cast<unsigned int>(zgridsize / zlocalsize));
}

void RMSnormFwdPlan::execute(const HipKernelHandle& handle,
                             const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                             uint32_t numDeviceBuffers,
                             [[maybe_unused]] void* workspace) const
{
    if(!_runnableKernel)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "RMSnormFwdPlan::execute() called before compile()");
    }

    // Get device buffer pointers
    auto xBuffer
        = hip_kernel_utils::findDeviceBuffer(_params.x()->uid(), deviceBuffers, numDeviceBuffers);
    auto scaleBuffer = hip_kernel_utils::findDeviceBuffer(
        _params.scale()->uid(), deviceBuffers, numDeviceBuffers);
    auto yBuffer
        = hip_kernel_utils::findDeviceBuffer(_params.y()->uid(), deviceBuffers, numDeviceBuffers);

    void* biasBufferPtr = (_params.bias() == nullptr)
                              ? nullptr
                              : hip_kernel_utils::findDeviceBuffer(
                                    _params.bias()->uid(), deviceBuffers, numDeviceBuffers)
                                    .ptr;
    void* invRMSBufferPtr = (_params.invRMS() == nullptr)
                                ? nullptr
                                : hip_kernel_utils::findDeviceBuffer(
                                      _params.invRMS()->uid(), deviceBuffers, numDeviceBuffers)
                                      .ptr;

    auto epsilonValue = _params.epsilon();

    _runnableKernel->launch(handle.getStream(),
                            xBuffer.ptr,
                            scaleBuffer.ptr,
                            biasBufferPtr,
                            yBuffer.ptr,
                            invRMSBufferPtr,
                            epsilonValue);
}

}
