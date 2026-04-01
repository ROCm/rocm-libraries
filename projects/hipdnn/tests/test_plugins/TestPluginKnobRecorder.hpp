// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

namespace hipdnn_tests
{

/// RAII wrapper for loading a test plugin and accessing its knob recording functions.
/// Platform-agnostic (Linux dlopen / Windows LoadLibrary) via data_sdk PlatformUtils.
///
/// The test plugin must export three C functions:
///   - hipdnnTestKnobsPluginGetReceivedKnobsCount() -> uint32_t
///   - hipdnnTestKnobsPluginGetReceivedKnobsAt(uint32_t) -> const char*
///   - hipdnnTestKnobsPluginResetReceivedKnobs() -> void
///
/// Use findLoadedPluginPath() to obtain the exact path the backend used when loading
/// the plugin, ensuring dlopen returns a handle to the same loaded library.
class TestPluginKnobRecorder
{
public:
    /// Opens the plugin library at the given absolute path and resolves recording symbols.
    /// The path should be the exact resolved path the backend used to load the plugin;
    /// use findLoadedPluginPath() to obtain it from hipdnnGetLoadedEnginePluginPaths_ext.
    /// Throws std::runtime_error on failure.
    explicit TestPluginKnobRecorder(const std::filesystem::path& pluginPath)
    {
        _handle = hipdnn_data_sdk::utilities::openLibrary(pluginPath);

        _fnGetCount = resolveSymbol<GetCountFn>("hipdnnTestKnobsPluginGetReceivedKnobsCount");
        _fnGetAt = resolveSymbol<GetAtFn>("hipdnnTestKnobsPluginGetReceivedKnobsAt");
        _fnReset = resolveSymbol<ResetFn>("hipdnnTestKnobsPluginResetReceivedKnobs");
    }

    /// Queries the backend for all loaded plugin paths and returns the one whose
    /// filename contains the given substring. This ensures we dlopen the exact same
    /// path the backend used (important because plugins are loaded with RTLD_LOCAL).
    /// Throws std::runtime_error if no matching plugin is found.
    static std::filesystem::path findLoadedPluginPath(hipdnnHandle_t handle,
                                                      const std::string& pluginNameSubstring)
    {
        size_t numPlugins = 0;
        size_t maxPathLength = 0;
        auto status
            = hipdnnGetLoadedEnginePluginPaths_ext(handle, &numPlugins, nullptr, &maxPathLength);
        if(status != HIPDNN_STATUS_SUCCESS)
        {
            throw std::runtime_error(
                "TestPluginKnobRecorder: hipdnnGetLoadedEnginePluginPaths_ext failed");
        }

        if(numPlugins == 0)
        {
            throw std::runtime_error("TestPluginKnobRecorder: no plugins loaded");
        }

        std::vector<std::vector<char>> pathBuffers(numPlugins, std::vector<char>(maxPathLength));
        std::vector<char*> pathPtrs(numPlugins);
        for(size_t i = 0; i < numPlugins; ++i)
        {
            pathPtrs[i] = pathBuffers[i].data();
        }

        status = hipdnnGetLoadedEnginePluginPaths_ext(
            handle, &numPlugins, pathPtrs.data(), &maxPathLength);
        if(status != HIPDNN_STATUS_SUCCESS)
        {
            throw std::runtime_error(
                "TestPluginKnobRecorder: hipdnnGetLoadedEnginePluginPaths_ext failed");
        }

        for(size_t i = 0; i < numPlugins; ++i)
        {
            std::string path(pathPtrs[i]);
            if(path.find(pluginNameSubstring) != std::string::npos)
            {
                return std::filesystem::path(path);
            }
        }

        throw std::runtime_error("TestPluginKnobRecorder: no loaded plugin matching '"
                                 + pluginNameSubstring + "'");
    }

    TestPluginKnobRecorder(TestPluginKnobRecorder&& other) noexcept
        : _handle(other._handle)
        , _fnGetCount(other._fnGetCount)
        , _fnGetAt(other._fnGetAt)
        , _fnReset(other._fnReset)
    {
        other._handle = nullptr;
        other._fnGetCount = nullptr;
        other._fnGetAt = nullptr;
        other._fnReset = nullptr;
    }

    TestPluginKnobRecorder& operator=(TestPluginKnobRecorder&& other) noexcept
    {
        if(this != &other)
        {
            cleanup();
            _handle = other._handle;
            _fnGetCount = other._fnGetCount;
            _fnGetAt = other._fnGetAt;
            _fnReset = other._fnReset;
            other._handle = nullptr;
            other._fnGetCount = nullptr;
            other._fnGetAt = nullptr;
            other._fnReset = nullptr;
        }
        return *this;
    }

    TestPluginKnobRecorder(const TestPluginKnobRecorder&) = delete;
    TestPluginKnobRecorder& operator=(const TestPluginKnobRecorder&) = delete;

    ~TestPluginKnobRecorder()
    {
        cleanup();
    }

    /// Returns the number of knob setting entries recorded since last reset.
    uint32_t count() const
    {
        return _fnGetCount();
    }

    /// Returns the nth recorded knob setting string.
    /// Throws std::out_of_range if index >= count().
    std::string at(uint32_t index) const
    {
        const char* raw = _fnGetAt(index);
        if(raw == nullptr)
        {
            throw std::out_of_range("TestPluginKnobRecorder::at: index " + std::to_string(index)
                                    + " out of range (count=" + std::to_string(count()) + ")");
        }
        return std::string{raw};
    }

    /// Returns all recorded knob settings as a vector of strings.
    std::vector<std::string> getAll() const
    {
        std::vector<std::string> result;
        const uint32_t n = count();
        result.reserve(n);
        for(uint32_t i = 0; i < n; ++i)
        {
            result.push_back(at(i));
        }
        return result;
    }

    /// Returns the last recorded knob setting (convenience).
    /// Returns empty string if no knobs have been recorded.
    std::string last() const
    {
        const uint32_t n = count();
        return n > 0 ? at(n - 1) : std::string();
    }

    /// Clears all recorded knob settings in the plugin.
    void reset()
    {
        _fnReset();
    }

private:
    using GetCountFn = uint32_t (*)();
    using GetAtFn = const char* (*)(uint32_t);
    using ResetFn = void (*)();

    template <typename T>
    T resolveSymbol(const char* name)
    {
        void* sym = hipdnn_data_sdk::utilities::getSymbol(_handle, name);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        return reinterpret_cast<T>(sym);
    }

    void cleanup() noexcept
    {
        if(_handle != nullptr)
        {
            hipdnn_data_sdk::utilities::closeLibrary(_handle);
            _handle = nullptr;
        }
    }

    hipdnn_data_sdk::utilities::LibHandle _handle = nullptr;
    GetCountFn _fnGetCount = nullptr;
    GetAtFn _fnGetAt = nullptr;
    ResetFn _fnReset = nullptr;
};

} // namespace hipdnn_tests
