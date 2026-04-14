// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <set>
#include <string>

#include "EnginePlugin.hpp"
#include "HipdnnException.hpp"
#include "PluginCore.hpp"
#include <hipdnn_backend/version.h>

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
        using hipdnn_data_sdk::utilities::Version;
        if(Version{plugin.apiVersion()}.major != HIPDNN_BACKEND_VERSION_MAJOR)
        {
            throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                                  "Plugin " + std::string(plugin.name()) + "'s major API version ("
                                      + plugin.apiVersion().data()
                                      + ") does not match backend major version ("
                                      + std::to_string(HIPDNN_BACKEND_VERSION_MAJOR) + ")");
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

    std::set<int64_t> _engineIds;
};

} // namespace hipdnn_backend::plugin
