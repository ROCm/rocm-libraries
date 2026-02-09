// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipProgram.hpp"
#include "HipUtils.hpp"

#include "kernel_includes.hpp"
#include "kernel_sources.hpp"
#include <hip/hiprtc.h>

HipProgram::HipProgram(std::string kernelFileName, std::vector<std::string> options)
    : programName_(kernelFileName)
{
    // Load source/includes
    auto kernelSrc = hip_plugin::getKernelSrc(kernelFileName.c_str());
    std::vector<std::string_view> includeTexts;
    std::vector<const char*> includeNames;
    hip_plugin::getKernelIncList(includeTexts, includeNames);

    // Convert includes
    std::vector<const char*> headersData;
    headersData.reserve(includeTexts.size());
    for(const auto& h : includeTexts)
        headersData.emplace_back(h.data());

    // Create program
    hiprtcProgram prog;
    HIPRTC_CHECK(hiprtcCreateProgram(&prog,
                                     kernelSrc.data(),
                                     kernelFileName.c_str(),
                                     static_cast<int>(headersData.size()),
                                     headersData.data(),
                                     includeNames.data()));

    // Compile
    std::vector<const char*> optPtrs;
    for(const auto& opt : options)
        optPtrs.push_back(opt.c_str());

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
        throw std::runtime_error("hiprtcCompileProgram failed for " + programName_ + ": "
                                 + hiprtcGetErrorString(result) + "\nCompilation log:\n" + log);
    }

    // Extract binary
    size_t codeSize;
    HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));
    binary_.resize(codeSize);
    HIPRTC_CHECK(hiprtcGetCode(prog, binary_.data()));

    // Cleanup rtc program (no longer needed)
    hiprtcDestroyProgram(&prog);

    // Load module
    HIP_CHECK(hipModuleLoadData(&module_, binary_.data()));
}

hipFunction_t HipProgram::GetKernel(const std::string& kernelName) const
{
    hipFunction_t kernel = nullptr;
    HIP_CHECK(hipModuleGetFunction(&kernel, module_, kernelName.c_str()));
    return kernel;
}

HipProgram::~HipProgram()
{
    if(module_)
    {
        [[maybe_unused]] auto result = hipModuleUnload(module_);
    }
}
