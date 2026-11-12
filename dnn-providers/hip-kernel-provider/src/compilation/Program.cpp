// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Program.hpp"
#include "Utils.hpp"
#include "device/ScopedDevice.hpp"

#include "kernel_includes.hpp"
#include "kernel_sources.hpp"
#include <hip/hiprtc.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include <utility>

namespace hip_kernel_provider::compilation
{

Program::Program(const std::string& kernelFileName, const std::vector<std::string>& options)
    : Program(
          kernelFileName, compileBinary(kernelFileName, options), device::currentDeviceOrdinal())
{
}

Program::Program(std::string programName, Binary binary, int deviceOrdinal)
    : _programName(std::move(programName))
    , _binary(std::move(binary))
    , _deviceOrdinal(deviceOrdinal)
{
    if(_binary == nullptr || _binary->empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "no compiled code object to load for "
                                                           + _programName);
    }

    // Bound before the load, not after: the device current at hipModuleLoadData is the
    // one the module belongs to for the rest of its life. A refused bind fails the load,
    // because a module cached under one ordinal and resident on another is a wrong answer
    // every later dispatch reuses.
    const device::ScopedDevice binding(_deviceOrdinal);
    if(!binding.bound())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "cannot make device "
                                                           + std::to_string(_deviceOrdinal)
                                                           + " current to load " + _programName);
    }

    const hipError_t status = hipModuleLoadData(&_module, _binary->data());
    if(status != hipSuccess)
    {
        // A device of the compiled architecture can still refuse the object when its
        // feature settings (xnack, sramecc) differ from the device it was compiled for.
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "hipModuleLoadData rejected " + _programName + " on device "
                + std::to_string(_deviceOrdinal) + ": " + hipGetErrorString(status));
    }
}

Program::Binary Program::compileBinary(const std::string& kernelFileName,
                                       const std::vector<std::string>& options)
{
    // Load source/includes
    const auto kernelSrc = hip_plugin::getKernelSrc(kernelFileName.c_str());
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
    // Kernel sources are generated from R-string literals (null-terminated) — see
    // kernels/templates/kernel_sources.cpp.in.
    // NOLINTNEXTLINE(bugprone-suspicious-stringview-data-usage)
    HIPRTC_CHECK(hiprtcCreateProgram(&prog,
                                     kernelSrc.data(),
                                     kernelFileName.c_str(),
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
        // Get compilation log (check return values gracefully — we are already
        // on the error path, so avoid throwing from log retrieval itself).
        std::string log;
        size_t logSize = 0;
        if(hiprtcGetProgramLogSize(prog, &logSize) == HIPRTC_SUCCESS && logSize > 1)
        {
            log.resize(logSize);
            if(hiprtcGetProgramLog(prog, log.data()) != HIPRTC_SUCCESS)
            {
                log = "(failed to retrieve compilation log)";
            }
        }
        hiprtcDestroyProgram(&prog);
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "hiprtcCompileProgram failed for " + kernelFileName + ": "
                + hiprtcGetErrorString(result) + "\nCompilation log:\n" + log);
    }

    // Extract binary
    size_t codeSize;
    HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));
    auto binary = std::make_shared<std::vector<char>>(codeSize);
    HIPRTC_CHECK(hiprtcGetCode(prog, binary->data()));

    // Cleanup rtc program (no longer needed)
    hiprtcDestroyProgram(&prog);

    return binary;
}

hipFunction_t Program::getKernel(const std::string& kernelName) const
{
    // Resolved on the module's own device, the one its launches will run on.
    const device::ScopedDevice binding(_deviceOrdinal);
    if(!binding.bound())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "cannot make device " + std::to_string(_deviceOrdinal) + " current to resolve '"
                + kernelName + "' from " + _programName);
    }

    hipFunction_t kernel = nullptr;
    HIP_CHECK(hipModuleGetFunction(&kernel, _module, kernelName.c_str()));
    return kernel;
}

Program::~Program()
{
    if(_module != nullptr)
    {
        // HIP binds the unload to the device current at the call, as it does the load. A
        // refused bind is logged by ScopedDevice; the unload is still attempted, as a
        // destructor has nowhere to report a refusal.
        const device::ScopedDevice binding(_deviceOrdinal);
        auto result = hipModuleUnload(_module);
        if(result != hipSuccess)
        {
            // Log the error for debugging purposes
            HIPDNN_PLUGIN_LOG_WARN("hipModuleUnload failed: " << hipGetErrorString(result));
        }
    }
}

} // namespace hip_kernel_provider::compilation
