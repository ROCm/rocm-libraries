// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "plugin/EnginePlugin.hpp"
#include "plugin/EnginePluginManager.hpp"
#include "plugin/PluginCore.hpp"

#include <filesystem>
#include <gmock/gmock.h>
#include <set>
#include <vector>

namespace hipdnn_backend
{
namespace plugin
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
};

} // namespace plugin
} // namespace hipdnn_backend
