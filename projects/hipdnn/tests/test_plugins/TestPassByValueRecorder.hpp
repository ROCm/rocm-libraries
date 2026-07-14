// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>

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

/// RAII reader for the runtime pass-by-value recorder plugin
/// (TestPassByValueRecorderPlugin). It re-opens the already-loaded plugin to
/// read back the (uid, value) pairs the plugin resolved from device_buffers at
/// execute via the shared plugin SDK helper resolveScalarOperand().
///
/// The test plugin must export four C functions:
///   - hipdnnTestPbvPluginGetReceivedCount() -> uint32_t
///   - hipdnnTestPbvPluginGetReceivedUidAt(uint32_t) -> int64_t
///   - hipdnnTestPbvPluginGetReceivedValueAt(uint32_t) -> double
///   - hipdnnTestPbvPluginReset() -> void
///
/// The pluginPath passed to the constructor should be the exact resolved path the
/// backend used when loading the plugin. Use
/// hipdnn_frontend::getLoadedEnginePluginPaths() to obtain it, ensuring the
/// dynamic loader returns a handle to the same loaded library.
class TestPassByValueRecorder
{
public:
    /// Opens the plugin library at the given absolute path and resolves recording
    /// symbols. Throws std::runtime_error on failure.
    explicit TestPassByValueRecorder(const std::filesystem::path& pluginPath)
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
            _fnGetCount = resolveSymbol<GetCountFn>("hipdnnTestPbvPluginGetReceivedCount");
            _fnGetUidAt = resolveSymbol<GetUidAtFn>("hipdnnTestPbvPluginGetReceivedUidAt");
            _fnGetValueAt = resolveSymbol<GetValueAtFn>("hipdnnTestPbvPluginGetReceivedValueAt");
            _fnReset = resolveSymbol<ResetFn>("hipdnnTestPbvPluginReset");
        }
        catch(...)
        {
            cleanup();
            throw;
        }
    }

    TestPassByValueRecorder(const TestPassByValueRecorder&) = delete;
    TestPassByValueRecorder& operator=(const TestPassByValueRecorder&) = delete;
    TestPassByValueRecorder(TestPassByValueRecorder&&) = delete;
    TestPassByValueRecorder& operator=(TestPassByValueRecorder&&) = delete;

    ~TestPassByValueRecorder()
    {
        cleanup();
    }

    /// Number of (uid, value) pairs recorded since the last reset.
    uint32_t count() const
    {
        return _fnGetCount();
    }

    /// The tensor uid of the nth recorded scalar. Throws if out of range.
    int64_t uidAt(uint32_t index) const
    {
        throwIfOutOfRange(index);
        return _fnGetUidAt(index);
    }

    /// The resolved (host-delivered) value of the nth recorded scalar, as a
    /// double (resolveScalarOperand's return type). Throws if out of range.
    double valueAt(uint32_t index) const
    {
        throwIfOutOfRange(index);
        return _fnGetValueAt(index);
    }

    /// Returns the value recorded for a specific uid, or std::nullopt if that uid
    /// was not delivered.
    std::optional<double> valueForUid(int64_t uid) const
    {
        const uint32_t n = count();
        for(uint32_t i = 0; i < n; ++i)
        {
            if(_fnGetUidAt(i) == uid)
            {
                return _fnGetValueAt(i);
            }
        }
        return std::nullopt;
    }

    /// Clears all recorded scalars (and any pending operands) in the plugin.
    void reset()
    {
        _fnReset();
    }

private:
    using GetCountFn = uint32_t (*)();
    using GetUidAtFn = int64_t (*)(uint32_t);
    using GetValueAtFn = double (*)(uint32_t);
    using ResetFn = void (*)();

    void throwIfOutOfRange(uint32_t index) const
    {
        if(index >= count())
        {
            throw std::out_of_range("TestPassByValueRecorder: index " + std::to_string(index)
                                    + " out of range (count=" + std::to_string(count()) + ")");
        }
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
    GetUidAtFn _fnGetUidAt = nullptr;
    GetValueAtFn _fnGetValueAt = nullptr;
    ResetFn _fnReset = nullptr;
};

} // namespace hipdnn_tests
