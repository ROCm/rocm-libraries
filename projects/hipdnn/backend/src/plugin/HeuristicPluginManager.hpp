// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <set>
#include <string>

#include "HeuristicPlugin.hpp"
#include "HipdnnException.hpp"
#include "PluginCore.hpp"
#include <hipdnn_backend/version.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_plugin_sdk/heuristic_api_version.h>

namespace hipdnn_backend::plugin
{

/**
 * @brief Manager for loading and validating heuristic plugin shared libraries.
 *
 * This class extends PluginManagerBase to provide heuristic-specific plugin
 * discovery, loading, and validation. It uses a separate search path from
 * engine plugins and validates heuristic-specific constraints.
 *
 * Validation includes:
 * - Heuristic C ABI major version compatibility
 * - Unique policy IDs across all loaded heuristic plugins
 * - Policy name is provided via hipdnnPluginGetName()
 */
class HeuristicPluginManager : public PluginManagerBase<HeuristicPlugin>
{
public:
    HeuristicPluginManager()
        : PluginManagerBase<HeuristicPlugin>(getPluginSearchPaths(
              "HIPDNN_HEURISTIC_PLUGIN_DIR", {std::filesystem::path("hipdnn_plugins/heuristics/")}))
    {
    }

protected:
    void validateBeforeAdding(const HeuristicPlugin& plugin) override
    {
        using hipdnn_data_sdk::utilities::Version;

        // Validate heuristic C ABI major version against the heuristic API version
        // (RFC 0007: heuristic plugin API has independent versioning from backend)
        if(Version{plugin.apiVersion()}.major != HIPDNN_HEURISTIC_API_VERSION_MAJOR)
        {
            throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                                  "ERROR: HEURISTIC PLUGIN ABI VALIDATION FAILED\n"
                                  "Plugin API major version ("
                                      + std::string(plugin.apiVersion())
                                      + ") does not match expected heuristic API major version ("
                                      + std::to_string(HIPDNN_HEURISTIC_API_VERSION_MAJOR) + ")\n"
                                      + "The plugin was compiled against an incompatible version "
                                        "of HeuristicsPluginApi.h\n"
                                      + "Expected API version: " HIPDNN_HEURISTIC_API_VERSION);
        }

        // Validate unique policy ID
        const int64_t policyId = plugin.policyId();
        if(_policyIds.find(policyId) != _policyIds.end())
        {
            throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                                  "ERROR: HEURISTIC PLUGIN VALIDATION FAILED\n"
                                  "Policy ID "
                                      + std::to_string(policyId)
                                      + " already exists in the list of loaded heuristic plugins.\n"
                                      + "Each heuristic plugin must have a unique policy ID.");
        }

        // Validate policy name is provided (required for all heuristic plugins)
        auto policyNameView = plugin.name();
        if(policyNameView.empty())
        {
            throw HipdnnException(
                HIPDNN_STATUS_PLUGIN_ERROR,
                "ERROR: HEURISTIC PLUGIN VALIDATION FAILED\n"
                "Policy name is required but was not provided.\n"
                "Plugin must implement hipdnnPluginGetName() and return a non-empty policy name.");
        }
    }

    void actionAfterAdding(const HeuristicPlugin& plugin) override
    {
        _policyIds.insert(plugin.policyId());
    }

private:
    std::set<int64_t> _policyIds;
};

} // namespace hipdnn_backend::plugin
