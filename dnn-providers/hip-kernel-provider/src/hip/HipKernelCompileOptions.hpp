// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "HipKernelUtils.hpp"

namespace hip_kernel_provider
{

class HipKernelCompileOptions
{
public:
    HipKernelCompileOptions(const hipdnn_data_sdk::data_objects::TensorAttributes* inputTensorAttrs,
                            const hipDeviceProp_t& deviceProps,
                            const std::optional<hip_kernel_utils::ActivationMode>& optActivationMode
                            = std::nullopt)
    {
        // Add rocm include path if ROCM_PATH env variable is set
        auto rocmPath
            = hipdnn_data_sdk::utilities::trim(hipdnn_data_sdk::utilities::getEnv("ROCM_PATH"));
        if(!rocmPath.empty())
        {
            auto rocmIncludeArg = "-I" + rocmPath + "/include";
            _compileOptions.emplace_back(rocmIncludeArg);
            HIPDNN_PLUGIN_LOG_INFO(
                "HipKernelProvider: HIPRTC compile ROCm include path: " << rocmIncludeArg);
        }
        else
        {
            HIPDNN_PLUGIN_LOG_WARN("HipKernelProvider: ROCM_PATH not set or empty, HIPRTC compile "
                                   "may fail if ROCm headers are not in standard include paths");
        }

        // Add device arch to compile options
        _compileOptions.emplace_back(std::string("--offload-arch=") + deviceProps.gcnArchName);

        // Add data type and layout options
        addDataTypeAndLayoutOptions(inputTensorAttrs);

        // Add activation options if activation is fused
        if(optActivationMode.has_value())
        {
            int nrnOpId = static_cast<int>(optActivationMode.value());
            _compileOptions.emplace_back(std::string("-DHIP_PLUGIN_NRN_OP_ID=")
                                         + std::to_string(nrnOpId));
        }
    }

    ~HipKernelCompileOptions() = default;

    HipKernelCompileOptions(const HipKernelCompileOptions&) = delete;
    HipKernelCompileOptions& operator=(const HipKernelCompileOptions&) = delete;
    HipKernelCompileOptions(HipKernelCompileOptions&&) = default;
    HipKernelCompileOptions& operator=(HipKernelCompileOptions&&) = default;

    template <typename T,
              typename = std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>>>
    void add(const std::string& name, T value)
    {
        _compileOptions.emplace_back("-D" + name + "=" + std::to_string(value));
    }

    void add(const std::string& name, const std::string& value)
    {
        _compileOptions.emplace_back("-D" + name + "=" + value);
    }

    void add(const std::string& name, bool value)
    {
        _compileOptions.emplace_back("-D" + name + "=" + (value ? "1" : "0"));
    }

    operator const auto &() const
    {
        return _compileOptions;
    }

private:
    void addDataTypeAndLayoutOptions(
        const hipdnn_data_sdk::data_objects::TensorAttributes* tensorAttrs)
    {
        auto inputDataType = tensorAttrs->data_type();
        auto isLayoutNhwc = hip_kernel_utils::isChannelLastLayout(tensorAttrs);

        // Add data type options
        bool useFp32 = (inputDataType == hipdnn_data_sdk::data_objects::DataType::FLOAT);
        bool useFp16 = (inputDataType == hipdnn_data_sdk::data_objects::DataType::HALF);
        bool useBfp16 = (inputDataType == hipdnn_data_sdk::data_objects::DataType::BFLOAT16);

        _compileOptions.emplace_back(std::string("-DHIP_PLUGIN_USE_FP32=") + (useFp32 ? "1" : "0"));
        _compileOptions.emplace_back(std::string("-DHIP_PLUGIN_USE_FP16=") + (useFp16 ? "1" : "0"));
        _compileOptions.emplace_back(std::string("-DHIP_PLUGIN_USE_BFP16=")
                                     + (useBfp16 ? "1" : "0"));
        _compileOptions.emplace_back("-DHIP_PLUGIN_USE_RNE_BFLOAT16=1");

        // Add layout option
        _compileOptions.emplace_back(std::string("-DHIP_PLUGIN_LAYOUT_NHWC=")
                                     + (isLayoutNhwc ? "1" : "0"));
    }

    std::vector<std::string> _compileOptions;
};

} // namespace hip_kernel_provider
