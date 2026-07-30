// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <set>
#include <string>

#include "EnginePlugin.hpp"
#include "HipdnnException.hpp"
#include "PluginCore.hpp"
#include <hipdnn_plugin_sdk/engine_api_version.h>

namespace hipdnn_backend::plugin
{

class EnginePluginManager : public PluginManagerBase<EnginePlugin>
{
public:
    EnginePluginManager()
        : PluginManagerBase<EnginePlugin>(getPluginSearchPaths(
              "HIPDNN_PLUGIN_DIR", {std::filesystem::path("hipdnn_plugins/engines/")}))
    {
    }

protected:
    void validateBeforeAdding(const EnginePlugin& plugin) override
    {
        // Reject plugins whose `apiVersion()` string fails to parse before
        // dispatch can observe them.
        const auto parsedVersion = plugin.parsedApiVersion();
        if(!parsedVersion.has_value())
        {
            throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                                  "Plugin " + plugin.cachedName()
                                      + " reports an unparseable API version ('"
                                      + std::string(plugin.apiVersion())
                                      + "'); rejecting at load time so dispatch is not exposed "
                                        "to the malformed string on every graph execute.");
        }

        // Validate engine C ABI major version against the engine API version
        // (RFC 0008: engine plugin API has independent versioning from backend,
        // mirroring the heuristic plugin pattern from RFC 0007).
        //
        // ONE-OFF transitional shim for the 0.x -> 1.0.0 bump: also accept
        // major == 0 so plugins built against the pre-1.0.0 SDK still load.
        // REMOVE this legacy clause (and the `&& pluginMajor != 0` below)
        // at the next major bump (1.x -> 2.0.0). The static_assert is a
        // tautology by design — the literal-equality check exists only to
        // break the build when the macro changes, forcing this file to be
        // revisited so the legacy clause can be dropped.
        // NOLINTNEXTLINE(misc-redundant-expression)
        static_assert(HIPDNN_ENGINE_API_VERSION_MAJOR == 1,
                      "Engine API major changed; drop the legacy major=0 "
                      "acceptance in EnginePluginManager.hpp.");
        const auto pluginMajor = parsedVersion->major;
        if(pluginMajor != HIPDNN_ENGINE_API_VERSION_MAJOR && pluginMajor != 0)
        {
            throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                                  "ERROR: ENGINE PLUGIN ABI VALIDATION FAILED\n"
                                  "Plugin "
                                      + plugin.cachedName() + "'s major API version ("
                                      + std::string(plugin.apiVersion())
                                      + ") does not match expected engine API major version ("
                                      + std::to_string(HIPDNN_ENGINE_API_VERSION_MAJOR) + ")\n"
                                      + "Expected API version: " HIPDNN_ENGINE_API_VERSION);
        }
        if(pluginMajor == 0 && pluginMajor != HIPDNN_ENGINE_API_VERSION_MAJOR)
        {
            // Per-load (not per-dispatch) notice: this branch is the
            // transitional shim above and will be removed at the next major
            // bump. Logging once per loaded plugin is appropriate.
            HIPDNN_BACKEND_LOG_INFO(
                "Accepting legacy major-0 plugin '{}' under transitional shim; this will be "
                "removed in the next major version.",
                plugin.cachedName());
        }

        auto engineIds = plugin.getAllEngineIds();
        for(const auto id : engineIds)
        {
            if(_engineIds.find(id) != _engineIds.end())
            {
                throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                                      "Engine ID " + std::to_string(id)
                                          + " already exists in the list");
            }
        }
    }

    void actionAfterAdding(const EnginePlugin& plugin) override
    {
        auto engineIds = plugin.getAllEngineIds();
        _engineIds.insert(engineIds.begin(), engineIds.end());
    }

    void actionAfterClearing() override
    {
        _engineIds.clear();
    }

    std::set<int64_t> _engineIds;
};

} // namespace hipdnn_backend::plugin
