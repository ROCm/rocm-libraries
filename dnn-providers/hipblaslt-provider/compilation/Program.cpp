// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <string_view>
#include <utility>

#include <hip/hiprtc.h>
#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "HipblasltUtils.hpp"
#include "Program.hpp"
#include "kernel_sources.hpp"

namespace hipblaslt_plugin::compilation
{
namespace
{

void throwOnHiprtcFailure(hiprtcResult result, const char* call, const std::string& programName)
{
    if(result != HIPRTC_SUCCESS)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            std::string(call) + " failed for " + programName + ": " + hiprtcGetErrorString(result));
    }
}

// Already on the error path, so this reports what it can read and never throws.
std::string compileLog(hiprtcProgram program)
{
    size_t logSize = 0;
    if(hiprtcGetProgramLogSize(program, &logSize) != HIPRTC_SUCCESS || logSize <= 1)
    {
        return "";
    }
    std::string log(logSize, '\0');
    if(hiprtcGetProgramLog(program, log.data()) != HIPRTC_SUCCESS)
    {
        return "(failed to retrieve the compilation log)";
    }
    return log;
}

} // namespace

std::vector<char> compileKernelSource(const std::string& sourceName,
                                      const std::vector<std::string>& options)
{
    // Embedded texts are null-terminated raw string literals
    // (templates/kernel_sources.cpp.in), so their data() can be passed to HIPRTC.
    // NOLINTNEXTLINE(bugprone-suspicious-stringview-data-usage)
    const char* const source = getKernelSrc(sourceName.c_str()).data();
    std::vector<std::string_view> includeTexts;
    std::vector<const char*> includeNames;
    getKernelIncList(includeTexts, includeNames);

    std::vector<const char*> headers;
    headers.reserve(includeTexts.size());
    for(const auto& text : includeTexts)
    {
        // NOLINTNEXTLINE(bugprone-suspicious-stringview-data-usage)
        headers.push_back(text.data());
    }

    hiprtcProgram program = nullptr;
    throwOnHiprtcFailure(hiprtcCreateProgram(&program,
                                             source,
                                             sourceName.c_str(),
                                             static_cast<int>(headers.size()),
                                             headers.data(),
                                             includeNames.data()),
                         "hiprtcCreateProgram",
                         sourceName);
    // Destroyed on every path out of this function, a throw included.
    const hipdnn_data_sdk::utilities::ScopedResource<hiprtcProgram> programGuard(
        program, [](hiprtcProgram p) { static_cast<void>(hiprtcDestroyProgram(&p)); });

    std::vector<const char*> optionPtrs;
    optionPtrs.reserve(options.size());
    for(const auto& option : options)
    {
        optionPtrs.push_back(option.c_str());
    }
    const auto result
        = hiprtcCompileProgram(program, static_cast<int>(optionPtrs.size()), optionPtrs.data());
    if(result != HIPRTC_SUCCESS)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "hiprtcCompileProgram failed for " + sourceName + ": " + hiprtcGetErrorString(result)
                + "\nCompilation log:\n" + compileLog(program));
    }

    size_t codeSize = 0;
    throwOnHiprtcFailure(hiprtcGetCodeSize(program, &codeSize), "hiprtcGetCodeSize", sourceName);
    std::vector<char> code(codeSize);
    throwOnHiprtcFailure(hiprtcGetCode(program, code.data()), "hiprtcGetCode", sourceName);
    return code;
}

Program::Program(std::string sourceName, const std::vector<std::string>& options)
    : _programName(std::move(sourceName))
    , _binary(compileKernelSource(_programName, options))
{
    THROW_ON_HIP_FAILURE(hipModuleLoadData(&_module, _binary.data()));
}

Program::~Program()
{
    if(_module != nullptr)
    {
        const auto result = hipModuleUnload(_module);
        if(result != hipSuccess)
        {
            HIPDNN_PLUGIN_LOG_WARN("hipModuleUnload failed for " << _programName << ": "
                                                                 << hipGetErrorString(result));
        }
    }
}

hipFunction_t Program::getKernel(const std::string& kernelName) const
{
    hipFunction_t kernel = nullptr;
    const auto result = hipModuleGetFunction(&kernel, _module, kernelName.c_str());
    if(result != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "kernel '" + kernelName + "' not found in "
                                                           + _programName + ": "
                                                           + hipGetErrorName(result));
    }
    return kernel;
}

} // namespace hipblaslt_plugin::compilation
