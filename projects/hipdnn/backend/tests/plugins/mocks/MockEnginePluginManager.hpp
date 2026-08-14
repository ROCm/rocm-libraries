// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "plugin/EnginePlugin.hpp"
#include "plugin/EnginePluginManager.hpp"
#include "plugin/PluginCore.hpp"

#include <filesystem>
#include <gmock/gmock.h>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace hipdnn_backend::plugin
{

class MockEnginePluginManager : public EnginePluginManager
{
public:
    MOCK_METHOD(void,
                loadPlugins,
                (const std::set<std::filesystem::path>& customPaths,
                 hipdnnPluginLoadingMode_ext_t mode),
                (override));

    MOCK_METHOD(const std::vector<std::shared_ptr<EnginePlugin>>&,
                getPlugins,
                (),
                (const, override));

    MOCK_METHOD(const std::set<std::filesystem::path>&,
                getLoadedPluginFiles,
                (),
                (const, override));

    /// Declares what admission left of a plugin. Mock plugins never go through the
    /// real admission hooks, so by default one contributes everything it declares
    /// and only tests exercising a drop need this.
    void setAcceptedEngineIds(const EnginePlugin& plugin, std::vector<int64_t> engineIds)
    {
        _accepted[&plugin] = std::move(engineIds);
    }

    std::vector<int64_t> acceptedEngineIds(const EnginePlugin& plugin) const override
    {
        const auto it = _accepted.find(&plugin);
        return it != _accepted.end() ? it->second : plugin.getAllEngineIds();
    }

private:
    std::map<const EnginePlugin*, std::vector<int64_t>> _accepted;
};

} // namespace hipdnn_backend::plugin
