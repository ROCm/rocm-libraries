// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Program.hpp"
#include "Utils.hpp"

#include "kernel_includes.hpp"
#include "kernel_sources.hpp"
#include <hip/hiprtc.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace hip_kernel_provider::compilation
{

Program::Program(std::string kernelFileName, const std::vector<std::string>& options)
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

    // Kernel sources and includes are generated from R-string literals
    // (null-terminated) — see kernels/templates/kernel_sources.cpp.in.
    // NOLINTNEXTLINE(bugprone-suspicious-stringview-data-usage)
    compileAndLoad(kernelSrc.data(), headersData, includeNames, options);
}

Program::Program(std::string sourceText,
                 std::string programName,
                 std::vector<std::pair<std::string, std::string>> headers,
                 const std::vector<std::string>& options)
    : _programName(std::move(programName))
{
    // The pairs stay alive for the whole constructor, so the pointers hipRTC is handed
    // stay valid for the one call that reads them.
    std::vector<const char*> headerTexts;
    std::vector<const char*> headerNames;
    headerTexts.reserve(headers.size());
    headerNames.reserve(headers.size());
    for(const auto& [name, text] : headers)
    {
        headerNames.emplace_back(name.c_str());
        headerTexts.emplace_back(text.c_str());
    }

    compileAndLoad(sourceText.c_str(), headerTexts, headerNames, options);
}

void Program::compileAndLoad(const char* sourceText,
                             const std::vector<const char*>& headerTexts,
                             const std::vector<const char*>& headerNames,
                             const std::vector<std::string>& options)
{
    // Create program
    hiprtcProgram prog;
    HIPRTC_CHECK(hiprtcCreateProgram(&prog,
                                     sourceText,
                                     _programName.c_str(),
                                     static_cast<int>(headerTexts.size()),
                                     headerTexts.data(),
                                     headerNames.data()));

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

hipFunction_t Program::getKernel(const std::string& kernelName) const
{
    hipFunction_t kernel = nullptr;
    HIP_CHECK(hipModuleGetFunction(&kernel, _module, kernelName.c_str()));
    return kernel;
}

Program::~Program()
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

} // namespace hip_kernel_provider::compilation
