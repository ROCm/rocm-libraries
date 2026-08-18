// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <optional>
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
    /// treat them as nonexistent.
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

    /// @brief The name @p engineId reported through its plugin's
    /// `hipdnnEnginePluginGetEngineName`, or nullopt when it reported none or the
    /// engine is not live.
    ///
    /// Resolved once at admission, so naming an engine never re-enters the plugin.
    [[nodiscard]] virtual std::optional<std::string> engineEntryPointName(int64_t engineId) const
    {
        const auto it = _engineNames.find(engineId);
        return it == _engineNames.end() ? std::optional<std::string>{} : it->second;
    }

    /// @brief The live engine IDs a single plugin contributes, ascending.
    ///
    /// Ordering is imposed here because liveEngines() is an unordered map.
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
        // malformed and is rejected whole by the throw here. Conflicts between
        // plugins are instead left to actionAfterAdding, which drops only the
        // offending engine. This call also caches the IDs, so actionAfterAdding
        // cannot throw once the plugin is in the list.
        std::ignore = plugin.getAllEngineIds();
    }

    /// Applies the per-engine admission rules. Runs after the plugin is in the
    /// list, so a rejected engine costs that engine only.
    void actionAfterAdding(const EnginePlugin& plugin) override
    {
        namespace utilities = hipdnn_data_sdk::utilities;

        const auto engineIds = plugin.getAllEngineIds();

        // A plugin with no entry point cannot answer for any engine.
        const bool pluginNamesEngines = plugin.hasEngineName();
        size_t namedEngines = 0;

        for(const auto id : engineIds)
        {
            // The name entry point is optional, so an engine reporting no name is
            // exempt and plugins predating it keep loading. Substituting its own ID
            // skips the hash check while keeping it on the uniqueness check below.
            // Failing outright is a defect, but this hook must not throw once the
            // plugin is in the list, so it costs that one engine.
            std::optional<std::string> engineName;
            if(pluginNamesEngines)
            {
                try
                {
                    engineName = plugin.getEngineName(id);
                }
                catch(const HipdnnException& e)
                {
                    HIPDNN_BACKEND_LOG_ERROR("Plugin '{}' failed to report a name for engine {}: "
                                             "{}. Dropping the engine.",
                                             plugin.cachedName(),
                                             utilities::formatEngineIdHex(id),
                                             e.what());
                    continue;
                }
            }

            // Counted where the plugin answers, so the warning below reports what
            // it supplied rather than what survived admission.
            namedEngines += engineName.has_value() ? 1 : 0;

            const auto nameId
                = engineName.has_value() ? utilities::engineNameToId(*engineName) : id;
            if(nameId != id)
            {
                // A name determines an ID, so whichever engine holds hash(name) owns
                // the name; naming it turns an opaque hash mismatch into a legible
                // collision. The plugin's own list is consulted too because IDs
                // arrive sorted, so the rightful owner may not be admitted yet.
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
                continue;
            }

            // Cached alongside the ownership it was validated against, so later
            // lookups never call back into the plugin.
            if(engineName.has_value())
            {
                _engineNames.emplace(id, std::move(*engineName));
            }
        }

        // Still supported, but such a plugin's engines can only be addressed by ID.
        // The two ways to arrive here take different fixes, so name the one that applies.
        if(namedEngines == 0)
        {
            const char* const remedy
                = pluginNamesEngines
                      ? "Its engine container supplies no name for any of them; give it a static "
                        "getEngineName."
                      : "It does not export hipdnnEnginePluginGetEngineName, which engine plugin "
                        "API 1.4.0 added.";

            HIPDNN_BACKEND_LOG_WARN(
                "Plugin '{}' names none of its engines, so they are identified by their IDs. {}",
                plugin.cachedName(),
                remedy);
        }
    }

    void actionAfterClearing() override
    {
        _engineOwner.clear();
        _engineNames.clear();
    }

    // The plugin pointers stay valid because PluginManagerBase holds the plugins
    // by shared_ptr and only ever drops them all at once.
    std::unordered_map<int64_t, const EnginePlugin*> _engineOwner;

    // Only engines in _engineOwner, and only those their plugin named.
    std::unordered_map<int64_t, std::string> _engineNames;
};

} // namespace hipdnn_backend::plugin
