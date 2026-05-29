// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <string>

namespace hipdnn_backend::plugin_constants
{
// Test plugin directory constants relative to backend library location
inline const std::string& getTestPluginDefaultDir()
{

    static const std::string s_defaultDir
#if defined(_WIN32)
        = "bin/test_plugins";
#else
        = "lib/test_plugins";
#endif
    return s_defaultDir;
}

// Helper to get full path to a heuristic test plugin in the custom directory
inline std::filesystem::path getHeuristicPluginPath(const char* pluginName)
{
    const auto moduleDir
        = hipdnn_backend::platform_utilities::getCurrentModuleDirectory().parent_path();
    const auto pluginDir = moduleDir / getTestPluginDefaultDir() / "custom";
    return pluginDir / hipdnn_data_sdk::utilities::getLibraryName(pluginName);
}

} // namespace hipdnn_backend::plugin_constants
