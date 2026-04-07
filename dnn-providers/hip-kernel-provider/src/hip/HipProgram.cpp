// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipProgram.hpp"
#include "HipUtils.hpp"

#include "kernel_includes.hpp"
#include "kernel_sources.hpp"
#include <hip/hiprtc.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace hip_kernel_provider
{

HipProgram::HipProgram(std::string kernelFileName, const std::vector<std::string>& options)
    : _programName(std::move(kernelFileName))
{
    // Load source/includes
    auto kernelSrc = hip_plugin::getKernelSrc(_programName.c_str());
    std::vector<std::string_view> includeTexts;
    std::vector<const char*> includeNames;
    hip_plugin::getKernelIncList(includeTexts, includeNames);

    // Convert includes
    std::vector<const char*> headersData;
    headersData.reserve(includeTexts.size());
    for(const auto& h : includeTexts)
    {
        headersData.emplace_back(h.data());
    }

    // Create program
    hiprtcProgram prog;
    HIPRTC_CHECK(hiprtcCreateProgram(&prog,
                                     kernelSrc.data(),
                                     _programName.c_str(),
                                     static_cast<int>(headersData.size()),
                                     headersData.data(),
                                     includeNames.data()));

    // Compile
    std::vector<const char*> optPtrs;
    optPtrs.reserve(options.size());
    for(const auto& opt : options)
    {
        optPtrs.push_back(opt.c_str());
    }

    auto result = hiprtcCompileProgram(prog, static_cast<int>(optPtrs.size()), optPtrs.data());
    if(result != HIPRTC_SUCCESS)
    {
        // Get compilation log
        size_t logSize = 0;
        hiprtcGetProgramLogSize(prog, &logSize);
        std::string log;
        if(logSize > 1)
        {
            log.resize(logSize);
            hiprtcGetProgramLog(prog, log.data());
        }
        hiprtcDestroyProgram(&prog);
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "hiprtcCompileProgram failed for " + _programName + ": " + hiprtcGetErrorString(result)
                + "\nCompilation log:\n" + log);
    }

    // Extract binary
    size_t codeSize;
    HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));
    _binary.resize(codeSize);
    HIPRTC_CHECK(hiprtcGetCode(prog, _binary.data()));

    // Cleanup rtc program (no longer needed)
    hiprtcDestroyProgram(&prog);

    // Load module
    HIP_CHECK(hipModuleLoadData(&_module, _binary.data()));
}

hipFunction_t HipProgram::getKernel(const std::string& kernelName) const
{
    hipFunction_t kernel = nullptr;
    HIP_CHECK(hipModuleGetFunction(&kernel, _module, kernelName.c_str()));
    return kernel;
}

HipProgram::~HipProgram()
{
    if(_module != nullptr)
    {
        auto result = hipModuleUnload(_module);
        if(result != hipSuccess)
        {
            // Log the error for debugging purposes
            HIPDNN_PLUGIN_LOG_WARN("hipModuleUnload failed: " << hipGetErrorString(result));
        }
    }
}

} // namespace hip_kernel_provider
