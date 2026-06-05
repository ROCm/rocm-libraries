// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "core/Utils.hpp"

namespace hip_kernel_provider::compilation
{

class KernelCompileOptions
{
public:
    KernelCompileOptions(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* inputTensorAttrs,
        const hipDeviceProp_t& deviceProps)
    {
        // Kernels use C++17 features (if constexpr, scoped-enum brace-init).
        _baseCompileOptions.emplace_back("-std=c++17");

        // Add device arch to compile options
        _baseCompileOptions.emplace_back(std::string("--offload-arch=") + deviceProps.gcnArchName);

        // Add data type and layout options
        addDataTypeAndLayoutOptions(inputTensorAttrs);
        addArchName(deviceProps.gcnArchName);
    }

    ~KernelCompileOptions() = default;

    KernelCompileOptions(const KernelCompileOptions&) = delete;
    KernelCompileOptions& operator=(const KernelCompileOptions&) = delete;
    KernelCompileOptions(KernelCompileOptions&&) = default;
    KernelCompileOptions& operator=(KernelCompileOptions&&) = default;

    operator std::vector<std::string>() const
    {
        std::vector<std::string> compileOptions = _baseCompileOptions;

        for(const auto& [name, value] : _mutableCompileOptionsMap)
        {
            std::string option = "-D";
            option += name;
            option += '=';
            option += value;
            compileOptions.emplace_back(std::move(option));
        }
        return compileOptions;
    }

    template <typename T,
              typename = std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>>>
    void add(const std::string& name, T value)
    {
        _mutableCompileOptionsMap[name] = std::to_string(value);
    }

    void add(const std::string& name, const std::string& value)
    {
        _mutableCompileOptionsMap[name] = value;
    }

    void add(const std::string& name, bool value)
    {
        _mutableCompileOptionsMap[name] = value ? "1" : "0";
    }

    template <typename T,
              typename = std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>>>
    void update(const std::string& name, T value)
    {
        updateIfExists(name, std::to_string(value));
    }

    void update(const std::string& name, const std::string& value)
    {
        updateIfExists(name, value);
    }

    void update(const std::string& name, bool value)
    {
        updateIfExists(name, value ? "1" : "0");
    }

    void remove(const std::string& name)
    {
        _mutableCompileOptionsMap.erase(name);
    }

private:
    void addDataTypeAndLayoutOptions(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* tensorAttrs)
    {
        auto inputDataType = tensorAttrs->data_type();
        auto isLayoutNhwc = core::utils::isChannelLastLayout(tensorAttrs);

        add("HIP_PLUGIN_USE_FP32",
            inputDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
        add("HIP_PLUGIN_USE_FP16",
            inputDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
        add("HIP_PLUGIN_USE_BFP16",
            inputDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16);
        add("HIP_PLUGIN_USE_RNE_BFLOAT16", true);
        add("HIP_PLUGIN_USE_FPMIX", false);
        add("HIP_PLUGIN_USE_BFPMIX", false);
        add("HIP_PLUGIN_LAYOUT_NHWC", isLayoutNhwc);
        add("HIP_PLUGIN_USE_AMDGCN", 0);
    }

    void addArchName(const std::string& archName)
    {
        const bool isGfx103X = (archName.find("gfx103") == 0);
        add("HIP_PLUGIN_GFX103X", isGfx103X);

        const bool isGfx110X = (archName.find("gfx110") == 0);
        add("HIP_PLUGIN_GFX110X", isGfx110X);

        const bool isGfx120X = (archName.find("gfx120") == 0);
        add("HIP_PLUGIN_GFX120X", isGfx120X);

        const bool isGfx115X = (archName.find("gfx115") == 0);
        add("HIP_PLUGIN_GFX115X", isGfx115X);
    }

    void updateIfExists(const std::string& name, std::string value)
    {
        auto it = _mutableCompileOptionsMap.find(name);
        if(it != _mutableCompileOptionsMap.end())
        {
            it->second = std::move(value);
        }
    }

protected:
    std::vector<std::string> _baseCompileOptions;
    std::unordered_map<std::string, std::string> _mutableCompileOptionsMap;
};

} // namespace hip_kernel_provider::compilation
