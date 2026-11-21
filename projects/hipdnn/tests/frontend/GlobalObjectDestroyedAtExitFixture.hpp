// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// A handle or built graph owned by a namespace-scope object is destroyed during process exit,
// after everything the program created in main, including the frontend's backend instance and
// the loaded plugins. Each test re-runs itself in a child process that leaves such an object
// behind, and expects the child to exit with status 0.
//
// The child loads only the test_good engine plugin and runs with HIPDNN_PLUGIN_UNLOAD_EAGER, so
// a late destroy that reaches the plugins releases the last plugin manager and unloads the
// plugin. A PluginUnloadProbe destroyed after the object checks that the plugin is still loaded
// and exits with PLUGIN_RELEASED_AT_EXIT otherwise, so a late plugin call fails the test even
// when it does not crash.

#pragma once

#include <array>
#include <cstdlib>
#include <string>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/ProcessUtils.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "test_plugins/TestPluginConstants.hpp"

namespace hipdnn_tests
{

// Child exit statuses other than a crash.
inline constexpr int PLUGIN_RELEASED_AT_EXIT = 3;
inline constexpr int PROBE_FAILED_AT_EXIT = 4;
inline constexpr int DESTROY_FAILED_AT_EXIT = 5;

/// Define at namespace scope before the objects under test, so it is destroyed after them.
class PluginUnloadProbe
{
public:
    PluginUnloadProbe() = default;
    PluginUnloadProbe(const PluginUnloadProbe&) = delete;
    PluginUnloadProbe& operator=(const PluginUnloadProbe&) = delete;
    PluginUnloadProbe(PluginUnloadProbe&&) = delete;
    PluginUnloadProbe& operator=(PluginUnloadProbe&&) = delete;

    ~PluginUnloadProbe()
    {
        try
        {
            if(!_pluginPath.empty() && !hipdnn_test_sdk::utilities::isLibraryLoaded(_pluginPath))
            {
                std::_Exit(PLUGIN_RELEASED_AT_EXIT);
            }
        }
        catch(...)
        {
            std::_Exit(PROBE_FAILED_AT_EXIT);
        }
    }

    void arm(const std::string& pluginPath)
    {
        _pluginPath = pluginPath;
    }

private:
    std::string _pluginPath;
};

class IntegrationGlobalObjectDestroyedAtExit : public ::testing::Test
{
protected:
    static constexpr const char* CHILD_ENV = "HIPDNN_TEST_EXIT_ORDER_CHILD";

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
    }

    static bool isChild()
    {
        return !hipdnn_data_sdk::utilities::getEnv(CHILD_ENV).empty();
    }

    static void useTestGoodEnginePlugin()
    {
        const std::array<const char*, 1> paths = {plugin_constants::testGoodPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
    }

    // Configures the child before its first handle: the engine plugin is unloaded as soon as
    // nothing holds it, and the probe checks at exit that something still does.
    static void prepareChild(PluginUnloadProbe& probe)
    {
        ASSERT_NO_FATAL_FAILURE(useTestGoodEnginePlugin());
        ASSERT_EQ(hipdnnSetPluginUnloadMode_ext(HIPDNN_PLUGIN_UNLOAD_EAGER), HIPDNN_STATUS_SUCCESS);
        probe.arm(plugin_constants::testGoodPluginPath());
    }

    static bool isTestGoodPluginLoaded()
    {
        return hipdnn_test_sdk::utilities::isLibraryLoaded(plugin_constants::testGoodPluginPath());
    }

    // Re-runs the current test in a child process and expects it to exit with status 0.
    static void expectChildExitsCleanly()
    {
        const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter child(CHILD_ENV, "1");
        // Objects released during exit are left to the operating system, which LeakSanitizer
        // would report as leaks.
        const std::string asanOptions = hipdnn_data_sdk::utilities::getEnv("ASAN_OPTIONS");
        const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter noLeakCheck(
            "ASAN_OPTIONS",
            asanOptions.empty() ? "detect_leaks=0" : asanOptions + ":detect_leaks=0");

        const int status = hipdnn_test_sdk::utilities::runCurrentTestInChild();
        EXPECT_EQ(status, 0) << "the child did not exit cleanly (exit " << PLUGIN_RELEASED_AT_EXIT
                             << ": the engine plugin was released during exit; "
                             << DESTROY_FAILED_AT_EXIT
                             << ": a destroy failed during exit; otherwise a crash)";
    }
};

} // namespace hipdnn_tests
