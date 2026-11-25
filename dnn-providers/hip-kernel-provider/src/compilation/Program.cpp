// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Program.hpp"
#include "CompileCounters.hpp"
#include "HiprtcDiskCache.hpp"
#include "ProgramFetchRecording.hpp"
#include "Utils.hpp"

#include <chrono>
#include <string_view>

#include "kernel_includes.hpp"
#include "kernel_sources.hpp"
#include <hip/hiprtc.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace hip_kernel_provider::compilation
{

namespace
{

/// Everything the code object depends on: the compiler, the source, every embedded include,
/// and every option -- the target architecture among them.
std::string cacheKey(const std::string& programName,
                     std::string_view source,
                     const std::vector<std::string_view>& includeTexts,
                     const std::vector<const char*>& includeNames,
                     const std::vector<std::string>& options)
{
    return hiprtc_cache::codeObjectKey(
        hiprtc_cache::compilerIdentity(), programName, source, includeTexts, includeNames, options);
}

/// A test hook (TEST_FAIL_COMPILE_ENV): whether @p program is to fail to compile.
bool compileFailureInjected(const std::string& program)
{
    const auto pattern = hipdnn_data_sdk::utilities::getEnv(TEST_FAIL_COMPILE_ENV);
    return !pattern.empty() && program.find(pattern) != std::string::npos;
}

/// DISABLE_HIPRTC_ENV is set to a true value.
bool hiprtcDisabled()
{
    const auto raw = hipdnn_data_sdk::utilities::toLower(
        hipdnn_data_sdk::utilities::trim(hipdnn_data_sdk::utilities::getEnv(DISABLE_HIPRTC_ENV)));
    return raw == "1" || raw == "true" || raw == "on" || raw == "yes";
}

double millisecondsSince(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}

} // namespace

std::string Program::codeObjectKey(const std::string& kernelFileName,
                                   const std::vector<std::string>& options)
{
    const auto kernelSrc = hip_plugin::getKernelSrc(kernelFileName.c_str());
    std::vector<std::string_view> includeTexts;
    std::vector<const char*> includeNames;
    hip_plugin::getKernelIncList(includeTexts, includeNames);
    return cacheKey(kernelFileName, kernelSrc, includeTexts, includeNames, options);
}

Program::Program(std::string kernelFileName, const std::vector<std::string>& options)
    : _programName(std::move(kernelFileName))
{
    const auto started = std::chrono::steady_clock::now();

    // Load source/includes
    auto kernelSrc = hip_plugin::getKernelSrc(_programName.c_str());
    std::vector<std::string_view> includeTexts;
    std::vector<const char*> includeNames;
    hip_plugin::getKernelIncList(includeTexts, includeNames);

    // A failure injected for tests skips the cache, so a cached object cannot hide it, and
    // compiles a source HIPRTC rejects, so it fails the way a real compile error does.
    const bool failureInjected = compileFailureInjected(_programName);
    const std::string injectedSource
        = std::string("#error \"compile failure injected by ") + TEST_FAIL_COMPILE_ENV + "\"\n";

    // A code object compiled earlier, by this process or another, from exactly these inputs.
    const auto cachePath = hiprtc_cache::codeObjectPath(
        cacheKey(_programName, kernelSrc, includeTexts, includeNames, options));
    if(!cachePath.empty() && !failureInjected)
    {
        auto cached = hiprtc_cache::readCodeObject(cachePath);
        if(cached.outcome == hiprtc_cache::ReadOutcome::VALID
           && hipModuleLoadData(&_module, cached.codeObject.data()) == hipSuccess)
        {
            _binary = std::move(cached.codeObject);
            ProgramFetchRecording::record(
                {_programName, CodeSource::DISK_CACHE, millisecondsSince(started)});
            return;
        }
        // Absent, damaged or not loadable on this device: compile and replace it below. A
        // failed load leaves HIP's last error set; clear it, since the fallback recovers and a
        // caller checking hipGetLastError later must not see a failure that did not happen.
        if(cached.outcome == hiprtc_cache::ReadOutcome::CORRUPT)
        {
            countCorruptEntry();
            HIPDNN_PLUGIN_LOG_WARN("hipRTC cache: " << cachePath.string() << " " << cached.problem
                                                    << "; recompiling " << _programName
                                                    << " and replacing it");
        }
        else if(cached.outcome == hiprtc_cache::ReadOutcome::VALID)
        {
            static_cast<void>(hipGetLastError());
        }
        _module = nullptr;
    }

    // Past here the program needs a compile.
    if(hiprtcDisabled())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
            "HIPRTC compilation is disabled (" + std::string(DISABLE_HIPRTC_ENV) + ") and "
                + _programName + " has no disk-cache entry for these options");
    }

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
                                     failureInjected ? injectedSource.c_str() : kernelSrc.data(),
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
        countCompileFailure();
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

    if(!cachePath.empty())
    {
        hiprtc_cache::writeCodeObject(cachePath, _binary);
    }
    ProgramFetchRecording::record({_programName, CodeSource::COMPILED, millisecondsSince(started)});
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
