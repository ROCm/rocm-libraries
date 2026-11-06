// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Program.hpp"
#include "Utils.hpp"
#include "utilities/Digest.hpp"

#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <string_view>
#include <system_error>

#include "kernel_includes.hpp"
#include "kernel_sources.hpp"
#include <hip/hiprtc.h>
#include <hipdnn_data_sdk/utilities/CacheRoot.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace hip_kernel_provider::compilation
{

namespace
{

/// Bumped when the cache's key or file format changes, so stale entries are never read.
constexpr std::string_view CACHE_FORMAT = "hip-kernel-provider-hiprtc-v1";

/// Appends @p field with its length first, so no two field sequences hash alike.
void appendField(std::string& key, std::string_view field)
{
    key.append(std::to_string(field.size()));
    key.push_back(':');
    key.append(field);
}

/// Everything the code object depends on: the compiler, the source, every embedded include,
/// and every option -- the target architecture among them.
std::string cacheKey(const std::string& programName,
                     std::string_view source,
                     const std::vector<std::string_view>& includeTexts,
                     const std::vector<const char*>& includeNames,
                     const std::vector<std::string>& options)
{
    int major = 0;
    int minor = 0;
    if(hiprtcVersion(&major, &minor) != HIPRTC_SUCCESS)
    {
        major = -1;
    }

    std::string key;
    appendField(key, CACHE_FORMAT);
    appendField(key, std::to_string(major) + "." + std::to_string(minor));
    appendField(key, programName);
    appendField(key, source);
    for(size_t i = 0; i < includeTexts.size(); ++i)
    {
        appendField(key, includeNames[i]);
        appendField(key, includeTexts[i]);
    }
    for(const auto& option : options)
    {
        appendField(key, option);
    }
    return utilities::sha256Hex(key.data(), key.size());
}

/// Where the code object for @p key lives, or an empty path when the on-disk cache is off
/// (HIPDNN_DISABLE_CACHE) or its root cannot be made.
std::filesystem::path cachedCodeObjectPath(const std::string& key)
{
    const auto root = hipdnn_data_sdk::utilities::cacheRoot();
    if(root.empty())
    {
        return {};
    }
    return root / "hip-kernel-provider" / "hiprtc" / (key + ".co");
}

std::optional<std::vector<char>> readCodeObject(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if(!file)
    {
        return std::nullopt;
    }
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    if(bytes.empty())
    {
        return std::nullopt;
    }
    return bytes;
}

/// Best effort: a file another process is writing is never seen half-written, since each
/// writer renames its own complete temporary file into place. Failures are logged and dropped;
/// the next process simply compiles again.
void writeCodeObject(const std::filesystem::path& path, const std::vector<char>& bytes)
{
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if(error)
    {
        HIPDNN_PLUGIN_LOG_INFO("hipRTC cache: cannot create " << path.parent_path().string() << ": "
                                                              << error.message());
        return;
    }

    std::random_device random;
    auto temporary = path;
    temporary += "." + std::to_string(random()) + ".tmp";
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if(!file)
        {
            HIPDNN_PLUGIN_LOG_INFO("hipRTC cache: cannot write " << temporary.string());
            file.close();
            std::filesystem::remove(temporary, error);
            return;
        }
    }
    std::filesystem::rename(temporary, path, error);
    if(error)
    {
        HIPDNN_PLUGIN_LOG_INFO("hipRTC cache: cannot place " << path.string() << ": "
                                                             << error.message());
        std::filesystem::remove(temporary, error);
    }
}

} // namespace

Program::Program(std::string kernelFileName, const std::vector<std::string>& options)
    : _programName(std::move(kernelFileName))
{
    // Load source/includes
    auto kernelSrc = hip_plugin::getKernelSrc(_programName.c_str());
    std::vector<std::string_view> includeTexts;
    std::vector<const char*> includeNames;
    hip_plugin::getKernelIncList(includeTexts, includeNames);

    // A code object compiled earlier, by this process or another, from exactly these inputs.
    const auto cachePath = cachedCodeObjectPath(
        cacheKey(_programName, kernelSrc, includeTexts, includeNames, options));
    if(!cachePath.empty())
    {
        auto cached = readCodeObject(cachePath);
        if(cached.has_value() && hipModuleLoadData(&_module, cached->data()) == hipSuccess)
        {
            _binary = std::move(*cached);
            return;
        }
        // Absent, unreadable or not loadable on this device: compile and replace it below. A
        // failed load leaves HIP's last error set; clear it, since the fallback recovers and a
        // caller checking hipGetLastError later must not see a failure that did not happen.
        if(cached.has_value())
        {
            static_cast<void>(hipGetLastError());
        }
        _module = nullptr;
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
                                     kernelSrc.data(),
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
        writeCodeObject(cachePath, _binary);
    }
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
