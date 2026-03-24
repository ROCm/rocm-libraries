// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_gpu_ref/detail/GpuRefKernelCompiler.hpp>

#include "GpuRefKernelSources.hpp"
#include <hip/hiprtc.h>
#include <stdexcept>
#include <string>

namespace hipdnn_gpu_ref::detail
{

namespace
{

void throwOnHipError(hipError_t err, const char* call)
{
    if(err != hipSuccess)
    {
        throw std::runtime_error(std::string(call) + " failed: " + hipGetErrorString(err));
    }
}

void throwOnRtcError(hiprtcResult err, const char* call)
{
    if(err != HIPRTC_SUCCESS)
    {
        throw std::runtime_error(std::string(call) + " failed: " + hiprtcGetErrorString(err));
    }
}

} // namespace

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define GPU_REF_HIP_CHECK(call) throwOnHipError((call), #call)
#define GPU_REF_RTC_CHECK(call) throwOnRtcError((call), #call)
// NOLINTEND(cppcoreguidelines-macro-usage)

CompiledKernel::CompiledKernel(const std::string& typeDefine, const std::string& functionName)
{
    // Get the kernel source
    auto kernelSrc = hipdnn_gpu_ref::getGpuRefKernelSrc("GpuRefConvFwd.cpp");

    // Get include headers
    std::vector<std::string_view> includeTexts;
    std::vector<const char*> includeNames;
    hipdnn_gpu_ref::getGpuRefKernelIncList(includeTexts, includeNames);

    std::vector<const char*> headersData;
    headersData.reserve(includeTexts.size());
    for(const auto& h : includeTexts)
    {
        headersData.emplace_back(h.data());
    }

    // Create program
    hiprtcProgram prog;
    GPU_REF_RTC_CHECK(hiprtcCreateProgram(&prog,
                                          kernelSrc.data(),
                                          "GpuRefConvFwd.cpp",
                                          static_cast<int>(headersData.size()),
                                          headersData.data(),
                                          includeNames.data()));

    // Build compile options
    std::string typeOpt = "-DDATA_TYPE=" + typeDefine;
    std::vector<const char*> optPtrs = {typeOpt.c_str()};

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
        throw std::runtime_error("HipRTC compilation failed for DATA_TYPE=" + typeDefine + ": "
                                 + hiprtcGetErrorString(result) + "\nCompilation log:\n" + log);
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

const CompiledKernel& GpuRefKernelCompiler::getOrCompile(const std::string& typeDefine,
                                                         const std::string& functionName)
{
    std::string key = typeDefine + "::" + functionName;

    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _cache.find(key);
    if(it != _cache.end())
    {
        return *it->second;
    }

    auto kernel = std::make_unique<CompiledKernel>(typeDefine, functionName);
    auto& ref = *kernel;
    _cache.emplace(std::move(key), std::move(kernel));
    return ref;
}

} // namespace hipdnn_gpu_ref::detail
