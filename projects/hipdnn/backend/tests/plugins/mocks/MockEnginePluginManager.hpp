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
#include <unordered_map>
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
    /// real admission hooks, so by default one contributes everything it declares.
    void setAcceptedEngineIds(const EnginePlugin& plugin, std::vector<int64_t> engineIds)
    {
        _accepted[&plugin] = std::move(engineIds);
    }

    std::vector<int64_t> acceptedEngineIds(const EnginePlugin& plugin) const override
    {
        const auto it = _accepted.find(&plugin);
        return it != _accepted.end() ? it->second : plugin.getAllEngineIds();
    }

    /// Ownership is what name resolution reads, and the inherited live set is
    /// filled by admission hooks a mock plugin never runs. Stand in for them on
    /// the same terms acceptedEngineIds() does: a plugin owns what it contributes,
    /// and the first plugin to claim an ID keeps it.
    const std::unordered_map<int64_t, const EnginePlugin*>& liveEngines() const override
    {
        _live.clear();
        for(const auto& plugin : getPlugins())
        {
            for(const auto id : acceptedEngineIds(*plugin))
            {
                _live.emplace(id, plugin.get());
            }
        }
        return _live;
    }

private:
    std::map<const EnginePlugin*, std::vector<int64_t>> _accepted;
    mutable std::unordered_map<int64_t, const EnginePlugin*> _live;
};

} // namespace hipdnn_backend::plugin
