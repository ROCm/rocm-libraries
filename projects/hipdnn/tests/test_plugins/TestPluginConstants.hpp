// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "TestPluginEngineIdMap.hpp"
#include <filesystem>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <stdexcept>
#include <string>

namespace hipdnn_tests
{
namespace plugin_constants
{
// Test plugin directory constants relative to backend library location
inline const std::string& getTestPluginDefaultDir()
{
    static const std::string s_defaultDir = "./test_plugins/default";
    return s_defaultDir;
}

inline const std::string& getTestPluginCustomDir()
{
    static const std::string s_customDir = "./test_plugins/custom";
    return s_customDir;
}

// Compose full plugin path with existence checking
inline std::string getPluginPath(const char* pluginName)
{
    namespace fs = std::filesystem;

    fs::path pluginFile
        = fs::path(getTestPluginCustomDir()) / hipdnn_sdk::utilities::getLibraryName(pluginName);

    return pluginFile.string();
}

inline std::string getDefaultPluginPath()
{
    namespace fs = std::filesystem;
    return (fs::path(getTestPluginDefaultDir())
            / hipdnn_sdk::utilities::getLibraryName(TEST_GOOD_DEFAULT_PLUGIN_NAME))
        .string();
}

inline const std::string& testGoodPluginPath()
{
    static const std::string s_testGoodPluginPath = getPluginPath(TEST_GOOD_PLUGIN_NAME);
    return s_testGoodPluginPath;
}

inline const std::string& testExecuteFailsPluginPath()
{
    static const std::string s_testExecuteFailsPluginPath
        = getPluginPath(TEST_EXECUTE_FAILS_PLUGIN_NAME);
    return s_testExecuteFailsPluginPath;
}

inline const std::string& testNoApplicableEnginesAPluginPath()
{
    static const std::string s_testNoApplicableEnginesPluginPath
        = getPluginPath(TEST_NO_APPLICABLE_ENGINES_A_PLUGIN_NAME);
    return s_testNoApplicableEnginesPluginPath;
}

inline const std::string& testNoApplicableEnginesBPluginPath()
{
    static const std::string s_testNoApplicableEnginesPluginPath
        = getPluginPath(TEST_NO_APPLICABLE_ENGINES_B_PLUGIN_NAME);
    return s_testNoApplicableEnginesPluginPath;
}

inline const std::string& testDuplicateIdAPluginPath()
{
    static const std::string s_testDuplicateIdAPluginPath
        = getPluginPath(TEST_DUPLICATE_ID_A_PLUGIN_NAME);
    return s_testDuplicateIdAPluginPath;
}

inline const std::string& testDuplicateIdBPluginPath()
{
    static const std::string s_testDuplicateIdBPluginPath
        = getPluginPath(TEST_DUPLICATE_ID_B_PLUGIN_NAME);
    return s_testDuplicateIdBPluginPath;
}

inline const std::string& testIncompleteApiPluginPath()
{
    static const std::string s_testIncompleteApiPluginPath
        = getPluginPath(TEST_INCOMPLETE_API_PLUGIN_NAME);
    return s_testIncompleteApiPluginPath;
}

} // namespace plugin_constants
} // namespace hipdnn_tests
