// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_gpu_ref/detail/GpuRefKernelCompiler.hpp>

#include "GpuRefKernelSources.hpp"
#include <hip/hiprtc.h>
#include <hipdnn_gpu_ref/detail/GpuRefHipError.hpp>
#include <stdexcept>
#include <string>

namespace hipdnn_gpu_ref::detail
{

namespace
{

void throwOnRtcError(hiprtcResult err, const char* call)
{
    if(err != HIPRTC_SUCCESS)
    {
        throw std::runtime_error(std::string(call) + " failed: " + hiprtcGetErrorString(err));
    }
}

} // namespace

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define GPU_REF_HIP_CHECK(call) ::hipdnn_gpu_ref::detail::throwOnHipError((call), #call)
#define GPU_REF_RTC_CHECK(call) throwOnRtcError((call), #call)
// NOLINTEND(cppcoreguidelines-macro-usage)

CompiledKernel::CompiledKernel(const std::string& sourceName,
                               const std::vector<std::string>& compileDefines,
                               const std::string& functionName)
{
    // Get the kernel source by name — convert to std::string for null-terminated c_str(),
    // since hiprtcCreateProgram requires null-terminated C strings.
    auto kernelSrc = std::string(hipdnn_gpu_ref::getGpuRefKernelSrc(sourceName.c_str()));

    // Get include headers — likewise convert string_views to strings for null termination.
    std::vector<std::string_view> includeTexts;
    std::vector<const char*> includeNames;
    hipdnn_gpu_ref::getGpuRefKernelIncList(includeTexts, includeNames);

    std::vector<std::string> includeStrings;
    std::vector<const char*> headersData;
    includeStrings.reserve(includeTexts.size());
    headersData.reserve(includeTexts.size());
    for(const auto& h : includeTexts)
    {
        includeStrings.emplace_back(h);
        headersData.emplace_back(includeStrings.back().c_str());
    }

    // Create program
    hiprtcProgram prog;
    GPU_REF_RTC_CHECK(hiprtcCreateProgram(&prog,
                                          kernelSrc.c_str(),
                                          sourceName.c_str(),
                                          static_cast<int>(headersData.size()),
                                          headersData.data(),
                                          includeNames.data()));

    // Build compile options from defines vector
    std::vector<const char*> optPtrs;
    optPtrs.reserve(compileDefines.size());
    for(const auto& def : compileDefines)
    {
        optPtrs.push_back(def.c_str());
    }

    auto result = hiprtcCompileProgram(prog, static_cast<int>(optPtrs.size()), optPtrs.data());
    if(result != HIPRTC_SUCCESS)
    {
        size_t logSize = 0;
        hiprtcGetProgramLogSize(prog, &logSize);
        std::string log;
        if(logSize > 1)
        {
            log.resize(logSize);
            hiprtcGetProgramLog(prog, log.data());
        }
        hiprtcDestroyProgram(&prog);

        std::string defStr;
        for(const auto& def : compileDefines)
        {
            defStr += " " + def;
        }
        throw std::runtime_error("HipRTC compilation failed for " + sourceName + " with defines"
                                 + defStr + ": " + hiprtcGetErrorString(result)
                                 + "\nCompilation log:\n" + log);
    }

    // Extract binary
    size_t codeSize;
    GPU_REF_RTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));
    _binary.resize(codeSize);
    GPU_REF_RTC_CHECK(hiprtcGetCode(prog, _binary.data()));

    hiprtcDestroyProgram(&prog);

    // Load module and get function
    GPU_REF_HIP_CHECK(hipModuleLoadData(&_module, _binary.data()));
    GPU_REF_HIP_CHECK(hipModuleGetFunction(&_function, _module, functionName.c_str()));
}

CompiledKernel::~CompiledKernel()
{
    if(_module != nullptr)
    {
        static_cast<void>(hipModuleUnload(_module));
    }
}

GpuRefKernelCompiler& GpuRefKernelCompiler::instance()
{
    static GpuRefKernelCompiler s_instance;
    return s_instance;
}

const CompiledKernel&
    GpuRefKernelCompiler::getOrCompile(const std::string& sourceName,
                                       const std::vector<std::string>& compileDefines,
                                       const std::string& functionName)
{
    // Build cache key: sourceName::define1,define2,...::functionName
    std::string key = sourceName + "::";
    for(size_t i = 0; i < compileDefines.size(); ++i)
    {
        if(i > 0)
        {
            key += ",";
        }
        key += compileDefines[i];
    }
    key += "::" + functionName;

    const std::lock_guard<std::mutex> lock(_mutex);
    auto it = _cache.find(key);
    if(it != _cache.end())
    {
        return *it->second;
    }

    auto kernel = std::make_unique<CompiledKernel>(sourceName, compileDefines, functionName);
    auto& ref = *kernel;
    _cache.emplace(std::move(key), std::move(kernel));
    return ref;
}

} // namespace hipdnn_gpu_ref::detail
