// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace hipdnn_tests
{

using hipdnn_flatbuffers_sdk::data_objects::EngineConfigT;
using hipdnn_flatbuffers_sdk::data_objects::UnPackEngineConfig;

/// RAII wrapper for loading a test plugin and accessing its knob recording functions.
///
/// The test plugin must export four C functions:
///   - hipdnnTestKnobsPluginGetReceivedKnobsCount() -> uint32_t
///   - hipdnnTestKnobsPluginGetReceivedKnobsDataAt(uint32_t) -> const uint8_t*
///   - hipdnnTestKnobsPluginGetReceivedKnobsSizeAt(uint32_t) -> uint32_t
///   - hipdnnTestKnobsPluginResetReceivedKnobs() -> void
///
/// The pluginPath passed to the constructor should be the exact resolved path the
/// backend used when loading the plugin. Use hipdnn_frontend::getLoadedEnginePluginPaths()
/// to obtain it, ensuring dlopen returns a handle to the same loaded library.
class TestPluginKnobRecorder
{
public:
    /// Opens the plugin library at the given absolute path and resolves recording symbols.
    /// Throws std::runtime_error on failure.
    explicit TestPluginKnobRecorder(const std::filesystem::path& pluginPath)
    {
        // Re-open the already-loaded plugin to get a handle for symbol lookup.
        // The backend loads plugins with RTLD_LOCAL (Linux), so symbols are not
        // visible via dlsym(RTLD_DEFAULT). This just bumps the refcount.
#ifdef _WIN32
        _handle = LoadLibraryW(pluginPath.wstring().c_str());
        if(_handle == nullptr)
        {
            throw std::runtime_error("Failed to load plugin: " + pluginPath.string()
                                     + " (Error Code: " + std::to_string(GetLastError()) + ")");
        }
#else
        _handle = dlopen(pluginPath.string().c_str(), RTLD_NOW | RTLD_LOCAL);
        if(_handle == nullptr)
        {
            const char* error = dlerror();
            throw std::runtime_error("Failed to load plugin: " + pluginPath.string() + " ("
                                     + (error != nullptr ? std::string(error) : "Unknown error")
                                     + ")");
        }
#endif
        try
        {
            _fnGetCount = resolveSymbol<GetCountFn>("hipdnnTestKnobsPluginGetReceivedKnobsCount");
            _fnGetDataAt
                = resolveSymbol<GetDataAtFn>("hipdnnTestKnobsPluginGetReceivedKnobsDataAt");
            _fnGetSizeAt
                = resolveSymbol<GetSizeAtFn>("hipdnnTestKnobsPluginGetReceivedKnobsSizeAt");
            _fnReset = resolveSymbol<ResetFn>("hipdnnTestKnobsPluginResetReceivedKnobs");
        }
        catch(...)
        {
            cleanup();
            throw;
        }
    }

    TestPluginKnobRecorder(const TestPluginKnobRecorder&) = delete;
    TestPluginKnobRecorder& operator=(const TestPluginKnobRecorder&) = delete;
    TestPluginKnobRecorder(TestPluginKnobRecorder&&) = delete;
    TestPluginKnobRecorder& operator=(TestPluginKnobRecorder&&) = delete;

    ~TestPluginKnobRecorder()
    {
        cleanup();
    }

    /// Returns the number of knob setting entries recorded since last reset.
    uint32_t count() const
    {
        return _fnGetCount();
    }

    /// Returns the nth recorded EngineConfig, unpacked from flatbuffer bytes.
    /// Knobs are sorted by knob_id for deterministic comparison.
    /// Throws std::out_of_range if index >= count().
    EngineConfigT at(uint32_t index) const
    {
        const uint8_t* data = _fnGetDataAt(index);
        const uint32_t size = _fnGetSizeAt(index);
        if(data == nullptr || size == 0)
        {
            throw std::out_of_range("TestPluginKnobRecorder::at: index " + std::to_string(index)
                                    + " out of range (count=" + std::to_string(count()) + ")");
        }
        auto config = UnPackEngineConfig(data);
        sortKnobs(*config);
        return std::move(*config);
    }

    /// Returns all recorded EngineConfigs as a vector.
    std::vector<EngineConfigT> getAll() const
    {
        std::vector<EngineConfigT> result;
        const uint32_t n = count();
        result.reserve(n);
        for(uint32_t i = 0; i < n; ++i)
        {
            result.push_back(at(i));
        }
        return result;
    }

    /// Returns the last recorded EngineConfig, or std::nullopt if none recorded.
    std::optional<EngineConfigT> last() const
    {
        const uint32_t n = count();
        if(n == 0)
        {
            return std::nullopt;
        }
        return at(n - 1);
    }

    /// Clears all recorded knob settings in the plugin.
    void reset()
    {
        _fnReset();
    }

private:
    using GetCountFn = uint32_t (*)();
    using GetDataAtFn = const uint8_t* (*)(uint32_t);
    using GetSizeAtFn = uint32_t (*)(uint32_t);
    using ResetFn = void (*)();

    static void sortKnobs(EngineConfigT& config)
    {
        std::sort(config.knobs.begin(), config.knobs.end(), [](const auto& a, const auto& b) {
            return a->knob_id < b->knob_id;
        });
    }

    template <typename T>
    T resolveSymbol(const char* name)
    {
#ifdef _WIN32
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        auto* sym = reinterpret_cast<void*>(GetProcAddress(_handle, name));
#else
        void* sym = dlsym(_handle, name);
#endif
        if(sym == nullptr)
        {
#ifdef _WIN32
            throw std::runtime_error("Failed to get symbol: " + std::string(name)
                                     + " (Error Code: " + std::to_string(GetLastError()) + ")");
#else
            const char* error = dlerror();
            throw std::runtime_error("Failed to get symbol: " + std::string(name) + " ("
                                     + (error != nullptr ? std::string(error) : "Unknown error")
                                     + ")");
#endif
        }
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        return reinterpret_cast<T>(sym);
    }

    void cleanup() noexcept
    {
        if(_handle != nullptr)
        {
#ifdef _WIN32
            FreeLibrary(_handle);
#else
            dlclose(_handle);
#endif
            _handle = nullptr;
        }
    }

#ifdef _WIN32
    HMODULE _handle = nullptr;
#else
    void* _handle = nullptr;
#endif
    GetCountFn _fnGetCount = nullptr;
    GetDataAtFn _fnGetDataAt = nullptr;
    GetSizeAtFn _fnGetSizeAt = nullptr;
    ResetFn _fnReset = nullptr;
};

} // namespace hipdnn_tests
