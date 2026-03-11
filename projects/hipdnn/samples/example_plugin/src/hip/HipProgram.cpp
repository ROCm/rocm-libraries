// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipProgram.hpp"

#include "HipUtils.hpp"
#include "kernel_includes.hpp"
#include "kernel_sources.hpp"

#include <hip/hiprtc.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include <vector>

namespace example_plugin
{

HipProgram::HipProgram(const std::string& kernelFileName,
                       const std::vector<std::string>& compilerOptions)
{
    HIPDNN_PLUGIN_LOG_INFO("Compiling kernel: " << kernelFileName);

    // Load embedded kernel source and include headers
    auto kernelSrc = getKernelSrc(kernelFileName.c_str());

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

    // Convert compiler options to C-strings
    std::vector<const char*> optionPtrs;
    optionPtrs.reserve(compilerOptions.size());
    for(const auto& opt : compilerOptions)
    {
        optionPtrs.push_back(opt.c_str());
    }

    // Compile the program
    hiprtcResult compileResult
        = hiprtcCompileProgram(program, static_cast<int>(optionPtrs.size()), optionPtrs.data());

    if(compileResult != HIPRTC_SUCCESS)
    {
        // Retrieve compilation log for diagnostics
        size_t logSize = 0;
        hiprtcGetProgramLogSize(program, &logSize);
        std::string compileLog(logSize, '\0');
        hiprtcGetProgramLog(program, compileLog.data());

        hiprtcDestroyProgram(&program);

        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "HIPRTC compilation failed for "
                                                           + kernelFileName + ": " + compileLog);
    }

    // Extract compiled binary code
    size_t codeSize = 0;
    HIPRTC_CHECK(hiprtcGetCodeSize(program, &codeSize));
    std::vector<char> code(codeSize);
    HIPRTC_CHECK(hiprtcGetCode(program, code.data()));
    HIPRTC_CHECK(hiprtcDestroyProgram(&program));

    // Load the compiled binary as a HIP module
    HIP_CHECK(hipModuleLoadData(&_module, code.data()));

    HIPDNN_PLUGIN_LOG_INFO("Kernel compiled and loaded: " << kernelFileName);
}

HipProgram::~HipProgram()
{
    if(_module != nullptr)
    {
        static_cast<void>(hipModuleUnload(_module));
    }
}

hipFunction_t HipProgram::getKernel(const std::string& kernelName) const
{
    hipFunction_t function = nullptr;
    HIP_CHECK(hipModuleGetFunction(&function, _module, kernelName.c_str()));
    return function;
}

} // namespace example_plugin
