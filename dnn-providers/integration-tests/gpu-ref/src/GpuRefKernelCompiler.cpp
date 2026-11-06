// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp>

#include "GpuRefKernelSources.hpp"
#include <array>
#include <cstdlib>
#include <filesystem>
#include <hip/hiprtc.h>
#include <hipdnn-gpu-ref/detail/GpuRefHipError.hpp>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

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

bool hasLowPrecisionHeaders(const std::filesystem::path& includeDir)
{
    return std::filesystem::exists(includeDir / "hip" / "hip_fp4.h")
           && std::filesystem::exists(includeDir / "hip" / "hip_fp6.h")
           && std::filesystem::exists(includeDir / "hip" / "hip_fp8.h");
}

std::optional<std::filesystem::path> environmentPath(const char* variable)
{
#if defined(_WIN32)
    char* value = nullptr;
    size_t length = 0;
    if(_dupenv_s(&value, &length, variable) != 0)
    {
        throw std::runtime_error(std::string("Failed to read environment variable ") + variable);
    }
    const std::unique_ptr<char, decltype(&std::free)> ownedValue(value, &std::free);
    if(value == nullptr)
    {
        return std::nullopt;
    }
    return std::filesystem::path(value);
#else
    if(const char* value = std::getenv(variable))
    {
        return std::filesystem::path(value);
    }
    return std::nullopt;
#endif
}

std::filesystem::path hipIncludeDir()
{
    for(const char* variable : {"HIP_PATH", "ROCM_PATH"})
    {
        if(const auto prefix = environmentPath(variable))
        {
            auto includeDir = *prefix / "include";
            if(hasLowPrecisionHeaders(includeDir))
            {
                return includeDir;
            }
        }
    }

#if defined(_WIN32)
    HMODULE module = nullptr;
    if(GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                              | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCSTR>(&hiprtcCompileProgram),
                          &module)
       != 0)
    {
        std::array<char, 32768> libraryPath{};
        const auto length = GetModuleFileNameA(
            module, libraryPath.data(), static_cast<DWORD>(libraryPath.size()));
        if(length > 0 && length < libraryPath.size())
        {
            auto includeDir = std::filesystem::path(
                                  std::string(libraryPath.data(), static_cast<size_t>(length)))
                                  .parent_path()
                                  .parent_path()
                              / "include";
            if(hasLowPrecisionHeaders(includeDir))
            {
                return includeDir;
            }
        }
    }
#else
    Dl_info libraryInfo{};
    if(dladdr(reinterpret_cast<const void*>(&hiprtcCompileProgram), &libraryInfo) != 0
       && libraryInfo.dli_fname != nullptr)
    {
        auto includeDir
            = std::filesystem::path(libraryInfo.dli_fname).parent_path().parent_path() / "include";
        if(hasLowPrecisionHeaders(includeDir))
        {
            return includeDir;
        }
    }
#endif

    throw std::runtime_error(
        "Unable to locate HIP FP4/FP6/FP8 headers next to hiprtc or via HIP_PATH/ROCM_PATH.");
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

    const auto includeOption = std::string("-I") + hipIncludeDir().string();
    std::vector<const char*> optPtrs;
    optPtrs.reserve(compileDefines.size() + 1);
    for(const auto& define : compileDefines)
    {
        optPtrs.push_back(define.c_str());
    }
    optPtrs.push_back(includeOption.c_str());

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

    // Manual error check (instead of GPU_REF_HIP_CHECK) because module must be
    // unloaded before throwing to avoid memory leak.
    auto funcResult = hipModuleGetFunction(&_function, _module, functionName.c_str());
    if(funcResult != hipSuccess)
    {
        static_cast<void>(hipModuleUnload(_module));
        _module = nullptr;
        throwOnHipError(funcResult, "hipModuleGetFunction(...)");
    }
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
