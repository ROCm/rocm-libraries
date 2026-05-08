// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "TestPluginEngineIdMap.hpp"
#include <filesystem>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <stdexcept>
#include <string>

namespace hipdnn_tests::plugin_constants
{
// Test plugin directory constants relative to backend library location
inline const std::string& getTestPluginDefaultDir()
{
    static const std::string s_defaultDir
        = (std::filesystem::path(".") / "test_plugins" / "default").string();
    return s_defaultDir;
}

inline const std::string& getTestPluginCustomDir()
{
    static const std::string s_customDir
        = (std::filesystem::path(".") / "test_plugins" / "custom").string();
    return s_customDir;
}

// Compose full plugin path with existence checking
inline std::string getTestCustomFilepathForPlugin(const char* pluginName)
{
    namespace fs = std::filesystem;

    const fs::path pluginFile = fs::path(getTestPluginCustomDir())
                                / hipdnn_data_sdk::utilities::getLibraryName(pluginName);

    return pluginFile.string();
}

inline std::string testDefaultGoodPluginPath()
{
    namespace fs = std::filesystem;
    return (fs::path(getTestPluginDefaultDir())
            / hipdnn_data_sdk::utilities::getLibraryName(TEST_GOOD_DEFAULT_PLUGIN_NAME))
        .string();
}

inline const std::string& testGoodPluginPath()
{
    static const std::string s_testGoodPluginPath
        = getTestCustomFilepathForPlugin(TEST_GOOD_PLUGIN_NAME);
    return s_testGoodPluginPath;
}

inline const std::string& testExecuteFailsPluginPath()
{
    static const std::string s_testExecuteFailsPluginPath
        = getTestCustomFilepathForPlugin(TEST_EXECUTE_FAILS_PLUGIN_NAME);
    return s_testExecuteFailsPluginPath;
}

inline const std::string& testNoApplicableEnginesAPluginPath()
{
    static const std::string s_testNoApplicableEnginesPluginPath
        = getTestCustomFilepathForPlugin(TEST_NO_APPLICABLE_ENGINES_A_PLUGIN_NAME);
    return s_testNoApplicableEnginesPluginPath;
}

inline const std::string& testNoApplicableEnginesBPluginPath()
{
    static const std::string s_testNoApplicableEnginesPluginPath
        = getTestCustomFilepathForPlugin(TEST_NO_APPLICABLE_ENGINES_B_PLUGIN_NAME);
    return s_testNoApplicableEnginesPluginPath;
}

inline const std::string& testDuplicateIdAPluginPath()
{
    static const std::string s_testDuplicateIdAPluginPath
        = getTestCustomFilepathForPlugin(TEST_DUPLICATE_ID_A_PLUGIN_NAME);
    return s_testDuplicateIdAPluginPath;
}

inline const std::string& testDuplicateIdBPluginPath()
{
    static const std::string s_testDuplicateIdBPluginPath
        = getTestCustomFilepathForPlugin(TEST_DUPLICATE_ID_B_PLUGIN_NAME);
    return s_testDuplicateIdBPluginPath;
}

inline const std::string& testIncompleteApiPluginPath()
{
    static const std::string s_testIncompleteApiPluginPath
        = getTestCustomFilepathForPlugin(TEST_INCOMPLETE_API_PLUGIN_NAME);
    return s_testIncompleteApiPluginPath;
}

inline const std::string& testKnobsPluginPath()
{
    static const std::string s_testKnobsPluginPath
        = getTestCustomFilepathForPlugin(TEST_KNOBS_PLUGIN_NAME);
    return s_testKnobsPluginPath;
}

inline const std::string& testKnobConstraintValidationPluginPath()
{
    static const std::string s_testKnobConstraintValidationPluginPath
        = getTestCustomFilepathForPlugin(TEST_KNOB_CONSTRAINT_VALIDATION_PLUGIN_NAME);
    return s_testKnobConstraintValidationPluginPath;
}

inline const std::string& testIncompatibleVersionPluginPath()
{
    static const std::string s_testIncompatibleVersionPluginPath
        = getTestCustomFilepathForPlugin(TEST_INCOMPATIBLE_VERSION_PLUGIN_NAME);
    return s_testIncompatibleVersionPluginPath;
}

// Override-execute fake-plugin paths.
inline const std::string& testOverrideImplementingPluginPath()
{
    static const std::string s_path
        = getTestCustomFilepathForPlugin(TEST_OVERRIDE_IMPLEMENTING_PLUGIN_NAME);
    return s_path;
}

inline const std::string& testOverrideOmittingPluginPath()
{
    static const std::string s_path
        = getTestCustomFilepathForPlugin(TEST_OVERRIDE_OMITTING_PLUGIN_NAME);
    return s_path;
}

inline const std::string& testVersionLiarPluginPath()
{
    static const std::string s_path = getTestCustomFilepathForPlugin(TEST_VERSION_LIAR_PLUGIN_NAME);
    return s_path;
}

inline const std::string& testSecondOverridePluginPath()
{
    static const std::string s_path
        = getTestCustomFilepathForPlugin(TEST_SECOND_OVERRIDE_PLUGIN_NAME);
    return s_path;
}

inline const std::string& testMalformedVersionPluginPath()
{
    static const std::string s_path
        = getTestCustomFilepathForPlugin(TEST_MALFORMED_VERSION_PLUGIN_NAME);
    return s_path;
}

inline const std::string& testVersionZeroPluginPath()
{
    static const std::string s_path = getTestCustomFilepathForPlugin(TEST_VERSION_ZERO_PLUGIN_NAME);
    return s_path;
}
} // namespace hipdnn_tests::plugin_constants
