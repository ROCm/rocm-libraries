// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>

#include <test_plugins/TestPluginCommon.hpp>

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
///
/// Handle lifecycle (dlopen/dlsym and cross-platform loading) is owned by the
/// composed ScopedTestPluginLibrary; this class only adds the typed accessors.
class TestPassByValueRecorder
{
public:
    /// Opens the plugin library at the given absolute path and resolves recording
    /// symbols. Throws std::runtime_error on failure.
    explicit TestPassByValueRecorder(const std::filesystem::path& pluginPath)
        : _lib(pluginPath.string())
    {
        _fnGetCount = _lib.requireSymbol<GetCountFn>("hipdnnTestPbvPluginGetReceivedCount");
        _fnGetUidAt = _lib.requireSymbol<GetUidAtFn>("hipdnnTestPbvPluginGetReceivedUidAt");
        _fnGetValueAt = _lib.requireSymbol<GetValueAtFn>("hipdnnTestPbvPluginGetReceivedValueAt");
        _fnReset = _lib.requireSymbol<ResetFn>("hipdnnTestPbvPluginReset");
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

    ScopedTestPluginLibrary _lib;
    GetCountFn _fnGetCount = nullptr;
    GetUidAtFn _fnGetUidAt = nullptr;
    GetValueAtFn _fnGetValueAt = nullptr;
    ResetFn _fnReset = nullptr;
};

} // namespace hipdnn_tests
