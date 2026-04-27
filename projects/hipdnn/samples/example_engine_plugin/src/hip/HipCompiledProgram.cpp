// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipCompiledProgram.hpp"

#include "HipRunnableKernel.hpp"
#include "HipUtils.hpp"
#include "kernel_includes.hpp"
#include "kernel_sources.hpp"

#include <hip/hiprtc.h>
#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include <vector>

namespace example_provider
{

HipCompiledProgram::HipCompiledProgram(const std::string& kernelFileName,
                                       const std::vector<std::string>& compilerOptions)
{
    HIPDNN_PLUGIN_LOG_INFO("Compiling kernel: " << kernelFileName);

    // Load embedded kernel source and include headers (convert to std::string
    // to guarantee null-termination required by hiprtcCreateProgram)
    std::string kernelSrc(getKernelSrc(kernelFileName.c_str()));

    std::vector<std::string_view> includeTexts;
    std::vector<const char*> includeNames;
    getKernelIncList(includeTexts, includeNames);

    // Convert include texts to C-strings for HIPRTC
    std::vector<const char*> includeTextPtrs;
    includeTextPtrs.reserve(includeTexts.size());
    for(const auto& text : includeTexts)
    {
        includeTextPtrs.push_back(text.data());
    }

    // Create HIPRTC program with source and headers
    hiprtcProgram program = nullptr;
    HIPRTC_CHECK(hiprtcCreateProgram(&program,
                                     kernelSrc.data(),
                                     kernelFileName.c_str(),
                                     static_cast<int>(includeTextPtrs.size()),
                                     includeTextPtrs.data(),
                                     includeNames.data()));

    const hipdnn_data_sdk::utilities::ScopedResource programGuard(
        program, [](hiprtcProgram p) { hiprtcDestroyProgram(&p); });

    // Convert compiler options to C-strings
    std::vector<const char*> optionPtrs;
    optionPtrs.reserve(compilerOptions.size());
    for(const auto& opt : compilerOptions)
    {
        optionPtrs.push_back(opt.c_str());
    }

    // Compile the program
    const hiprtcResult compileResult
        = hiprtcCompileProgram(program, static_cast<int>(optionPtrs.size()), optionPtrs.data());

    if(compileResult != HIPRTC_SUCCESS)
    {
        // Retrieve compilation log for diagnostics
        size_t logSize = 0;
        hiprtcGetProgramLogSize(program, &logSize);
        std::string compileLog(logSize, '\0');
        hiprtcGetProgramLog(program, compileLog.data());

        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "HIPRTC compilation failed for "
                                                           + kernelFileName + ": " + compileLog);
    }

    // Extract compiled binary code
    size_t codeSize = 0;
    HIPRTC_CHECK(hiprtcGetCodeSize(program, &codeSize));
    std::vector<char> code(codeSize);
    HIPRTC_CHECK(hiprtcGetCode(program, code.data()));

    // Load the compiled binary as a HIP module
    HIP_CHECK(hipModuleLoadData(&_module, code.data()));

    HIPDNN_PLUGIN_LOG_INFO("Kernel compiled and loaded: " << kernelFileName);
}

HipCompiledProgram::~HipCompiledProgram()
{
    if(_module != nullptr)
    {
        auto result = hipModuleUnload(_module);
        if(result != hipSuccess)
        {
            HIPDNN_PLUGIN_LOG_WARN("hipModuleUnload failed: " << hipGetErrorString(result));
        }
    }
}

std::unique_ptr<IRunnableKernel>
    HipCompiledProgram::getRunnableKernel(const std::string& kernelFunctionName) const
{
    hipFunction_t function = nullptr;
    HIP_CHECK(hipModuleGetFunction(&function, _module, kernelFunctionName.c_str()));
    return std::make_unique<HipRunnableKernel>(function, kernelFunctionName);
}

} // namespace example_provider
