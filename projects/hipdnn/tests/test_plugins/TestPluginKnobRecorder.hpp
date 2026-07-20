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

#include <test_plugins/TestPluginCommon.hpp>

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
///
/// Handle lifecycle (dlopen/dlsym and cross-platform loading) is owned by the
/// composed ScopedTestPluginLibrary; this class only adds the typed accessors.
class TestPluginKnobRecorder
{
public:
    /// Opens the plugin library at the given absolute path and resolves recording symbols.
    /// Throws std::runtime_error on failure.
    explicit TestPluginKnobRecorder(const std::filesystem::path& pluginPath)
        : _lib(pluginPath.string())
    {
        _fnGetCount = _lib.requireSymbol<GetCountFn>("hipdnnTestKnobsPluginGetReceivedKnobsCount");
        _fnGetDataAt
            = _lib.requireSymbol<GetDataAtFn>("hipdnnTestKnobsPluginGetReceivedKnobsDataAt");
        _fnGetSizeAt
            = _lib.requireSymbol<GetSizeAtFn>("hipdnnTestKnobsPluginGetReceivedKnobsSizeAt");
        _fnReset = _lib.requireSymbol<ResetFn>("hipdnnTestKnobsPluginResetReceivedKnobs");
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

    ScopedTestPluginLibrary _lib;
    GetCountFn _fnGetCount = nullptr;
    GetDataAtFn _fnGetDataAt = nullptr;
    GetSizeAtFn _fnGetSizeAt = nullptr;
    ResetFn _fnReset = nullptr;
};

} // namespace hipdnn_tests
