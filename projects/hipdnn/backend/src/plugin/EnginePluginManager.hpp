// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "EnginePlugin.hpp"
#include "HipdnnException.hpp"
#include "PluginCore.hpp"
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
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

    /// @brief Every engine that survived load-time admission, by engine ID.
    ///
    /// An engine is dropped when its plugin-reported name does not hash to its
    /// engine ID (RFC 0003), or when an already-loaded plugin owns the same ID.
    /// Dropped engines are absent here, so enumeration, routing and dispatch all
    /// treat them as nonexistent. Keying by ID is what makes an engine's owner
    /// unique by construction rather than by a check.
    [[nodiscard]] virtual const std::unordered_map<int64_t, const EnginePlugin*>&
        liveEngines() const
    {
        return _engineOwner;
    }

    /// @brief The plugin that provides @p engineId, or nullptr when no loaded
    /// plugin does: the ID was never declared, or the engine was dropped.
    [[nodiscard]] const EnginePlugin* engineOwner(int64_t engineId) const
    {
        const auto& live = liveEngines();
        const auto it = live.find(engineId);
        return it == live.end() ? nullptr : it->second;
    }

    /// @brief The live engine IDs a single plugin contributes, ascending.
    ///
    /// A view over liveEngines() for the callers that walk one plugin at a time.
    /// Ordering is imposed here because the underlying map has none.
    [[nodiscard]] virtual std::vector<int64_t> acceptedEngineIds(const EnginePlugin& plugin) const
    {
        std::vector<int64_t> engineIds;
        for(const auto& [id, owner] : liveEngines())
        {
            if(owner == &plugin)
            {
                engineIds.push_back(id);
            }
        }

        std::sort(engineIds.begin(), engineIds.end());
        return engineIds;
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

        // A plugin with no engines, or with duplicate IDs within itself, is
        // malformed rather than conflicting and is rejected whole. Conflicts
        // between plugins are left to actionAfterAdding, which drops only the
        // offending engine; throwing here would reject the plugin. The call also
        // caches the IDs, so actionAfterAdding cannot throw after the plugin is in
        // the list.
        std::ignore = plugin.getAllEngineIds();
    }

    /// Applies the per-engine admission rules. Runs after the plugin is in the
    /// list, so a rejected engine costs that engine only.
    void actionAfterAdding(const EnginePlugin& plugin) override
    {
        namespace utilities = hipdnn_data_sdk::utilities;

        const auto engineIds = plugin.getAllEngineIds();

        for(const auto id : engineIds)
        {
            // The name entry point is optional, so an engine that reports no name
            // is exempt; plugins predating it keep loading unchanged. Standing in
            // the engine's own ID for the absent name keeps that engine on the
            // uniqueness check below while it skips the hash check.
            const auto engineName = plugin.getEngineName(id);
            const auto nameId
                = engineName.has_value() ? utilities::engineNameToId(*engineName) : id;
            if(nameId != id)
            {
                // A name determines an ID, so whichever engine holds hash(name) is
                // the one the name belongs to. Naming that engine turns an opaque
                // hash mismatch into the collision the reader is looking at. The
                // plugin's own list is consulted too because IDs arrive sorted, so
                // the rightful owner may not have been admitted yet.
                std::string holder = "no loaded engine claims that ID";
                if(const auto claimant = _engineOwner.find(nameId); claimant != _engineOwner.end())
                {
                    holder = "plugin '" + claimant->second->cachedName() + "' provides that engine";
                }
                else if(std::find(engineIds.begin(), engineIds.end(), nameId) != engineIds.end())
                {
                    holder = "this plugin also declares that engine";
                }

                HIPDNN_BACKEND_LOG_ERROR(
                    "Plugin '{}' reports engine name '{}' for engine {}, but that name is the name "
                    "of engine {} ({}). An engine ID must equal the hash of its name; dropping the "
                    "engine.",
                    plugin.cachedName(),
                    *engineName,
                    utilities::formatEngineIdHex(id),
                    utilities::formatEngineIdHex(nameId),
                    holder);
                continue;
            }

            // One ID admits one engine, so the insertion is the uniqueness check.
            // Names hash to IDs, so it covers duplicate names too.
            const auto [owner, admitted] = _engineOwner.emplace(id, &plugin);
            if(!admitted)
            {
                HIPDNN_BACKEND_LOG_ERROR(
                    "Plugin '{}' declares engine {}, which plugin '{}' already provides. The "
                    "first plugin to declare an engine keeps it; dropping the duplicate.",
                    plugin.cachedName(),
                    utilities::formatEngineIdHex(id),
                    owner->second->cachedName());
            }
        }
    }

    void actionAfterClearing() override
    {
        _engineOwner.clear();
    }

    // The plugin pointers stay valid because PluginManagerBase holds the plugins
    // by shared_ptr and only ever drops them all at once.
    std::unordered_map<int64_t, const EnginePlugin*> _engineOwner;
};

} // namespace hipdnn_backend::plugin
