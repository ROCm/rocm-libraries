// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <thread>

#include "PlatformUtils.hpp"
#include <gtest/gtest.h>
#include <utility>

#include "plugin/PluginCore.hpp"
#include <hipdnn_sdk/test_utilities/FileUtilities.hpp>
#include <hipdnn_sdk/test_utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>

using namespace hipdnn_backend;
using namespace hipdnn_sdk::test_utilities;

namespace
{

class Plugin : public plugin::PluginBase
{
public:
    // Forward constructor to base class
    explicit Plugin(plugin::SharedLibrary&& lib)
        : PluginBase(std::move(lib))
    {
    }

    static hipdnnPluginType_t getPluginType()
    {
        return HIPDNN_PLUGIN_TYPE_UNSPECIFIED;
    }

    using PluginBase::_lib;
    using PluginBase::getLastErrorString;

private:
    friend class plugin::PluginManagerBase<Plugin>;
};

class TestPluginManager : public plugin::PluginManagerBase<Plugin>
{
public:
    TestPluginManager()
        : plugin::PluginManagerBase<Plugin>({"test_plugins_dir"})
    {
    }
    TestPluginManager(std::set<std::filesystem::path> paths)
        : plugin::PluginManagerBase<Plugin>(std::move(paths))
    {
    }

    using plugin::PluginManagerBase<Plugin>::getPluginSearchPaths;
};

class TestPluginCallback : public ::testing::Test
{
protected:
    void SetUp() override
    {
        s_callbackCalled = false;
    }

    static void dummyCallback(hipdnnSeverity_t sev, const char* msg)
    {
        (void)sev;
        (void)msg;
        s_callbackCalled = true;
    }

    static bool s_callbackCalled; //NOLINT
};

bool TestPluginCallback::s_callbackCalled = false;

const std::filesystem::path TEST_PLUGIN_DIR = "lib/test_plugins";

const auto PLUGIN_PATH1 = ".." / TEST_PLUGIN_DIR / TEST_PLUGIN1_NAME;
const auto PLUGIN_PATH2 = ".." / TEST_PLUGIN_DIR / TEST_PLUGIN2_NAME;

const auto FULL_PLUGIN_PATH1
    = hipdnn_backend::platform_utilities::getCurrentModuleDirectory().parent_path()
      / TEST_PLUGIN_DIR / hipdnn_sdk::utilities::getLibraryName(TEST_PLUGIN1_NAME);
const auto FULL_PLUGIN_PATH2
    = hipdnn_backend::platform_utilities::getCurrentModuleDirectory().parent_path()
      / TEST_PLUGIN_DIR / hipdnn_sdk::utilities::getLibraryName(TEST_PLUGIN2_NAME);

} // namespace

TEST(TestPluginManager, LoadPlugins)
{
    // Create a PluginManager instance
    TestPluginManager pluginManager;

    // Create a list of paths to plugins
    std::set<std::filesystem::path> pluginPaths = {PLUGIN_PATH1, PLUGIN_PATH2};

    // Load the plugins
    pluginManager.loadPlugins(pluginPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = pluginManager.getPlugins();
    ASSERT_EQ(plugins.size(), 2); // Ensure two plugins are loaded

    // Check that the plugins have the correct names
    ASSERT_EQ(plugins[0]->name(), "Plugin1");
    ASSERT_EQ(plugins[1]->name(), "Plugin2");

    // Check that the plugins have the correct versions
    ASSERT_EQ(plugins[0]->version(), "1.0");
    ASSERT_EQ(plugins[1]->version(), "2.0");

    // Check that the plugins have the correct types
    ASSERT_EQ(plugins[0]->type(), HIPDNN_PLUGIN_TYPE_UNSPECIFIED);
    ASSERT_EQ(plugins[1]->type(), HIPDNN_PLUGIN_TYPE_UNSPECIFIED);
}

TEST(TestPluginManager, LoadPluginsFromDirectory)
{
    std::filesystem::path tempPluginDir
        = hipdnn_backend::platform_utilities::getCurrentModuleDirectory() /= "temp_plugin_dir";
    ScopedDirectory tempDir(tempPluginDir);

    std::filesystem::copy_file(
        FULL_PLUGIN_PATH1, tempDir.path() / std::filesystem::path(FULL_PLUGIN_PATH1).filename());
    std::filesystem::copy_file(
        FULL_PLUGIN_PATH2, tempDir.path() / std::filesystem::path(FULL_PLUGIN_PATH2).filename());

    TestPluginManager pluginManager;
    pluginManager.loadPlugins({tempDir.path()}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = pluginManager.getPlugins();
    ASSERT_EQ(plugins.size(), 2);

    std::set<std::string_view> pluginNames;
    for(const auto& p : plugins)
    {
        pluginNames.insert(p->name());
    }
    EXPECT_TRUE(pluginNames.find("Plugin1") != pluginNames.end());
    EXPECT_TRUE(pluginNames.find("Plugin2") != pluginNames.end());
}

TEST(TestPluginManager, LoadPluginsAbsolute)
{
    TestPluginManager pluginManager;
    pluginManager.loadPlugins({PLUGIN_PATH1}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    ASSERT_EQ(pluginManager.getPlugins().size(), 1);

    pluginManager.loadPlugins({PLUGIN_PATH2}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    const auto& plugins = pluginManager.getPlugins();
    ASSERT_EQ(plugins.size(), 1);
    EXPECT_EQ(plugins[0]->name(), "Plugin2");
}

TEST(TestPluginManager, LoadPluginsAdditive)
{
    TestPluginManager pluginManager;
    pluginManager.loadPlugins({PLUGIN_PATH1}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    ASSERT_EQ(pluginManager.getPlugins().size(), 1);

    pluginManager.loadPlugins({PLUGIN_PATH1, PLUGIN_PATH2}, HIPDNN_PLUGIN_LOADING_ADDITIVE);
    const auto& plugins = pluginManager.getPlugins();
    EXPECT_EQ(plugins.size(), 2);
}

TEST(TestPluginManager, LoadPluginsAdditiveAccumulates)
{
    TestPluginManager pluginManager;
    pluginManager.loadPlugins({PLUGIN_PATH1}, HIPDNN_PLUGIN_LOADING_ADDITIVE);
    ASSERT_EQ(pluginManager.getPlugins().size(), 1);

    pluginManager.loadPlugins({PLUGIN_PATH2}, HIPDNN_PLUGIN_LOADING_ADDITIVE);
    const auto& plugins = pluginManager.getPlugins();
    ASSERT_EQ(plugins.size(), 2);

    std::set<std::string_view> pluginNames;
    for(const auto& p : plugins)
    {
        pluginNames.insert(p->name());
    }
    EXPECT_TRUE(pluginNames.find("Plugin1") != pluginNames.end());
    EXPECT_TRUE(pluginNames.find("Plugin2") != pluginNames.end());
}

TEST(TestPluginManager, LoadPluginsAbsoluteReplaces)
{
    TestPluginManager pluginManager;
    pluginManager.loadPlugins({PLUGIN_PATH1}, HIPDNN_PLUGIN_LOADING_ADDITIVE);
    ASSERT_EQ(pluginManager.getPlugins().size(), 1);

    pluginManager.loadPlugins({PLUGIN_PATH2}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    const auto& plugins = pluginManager.getPlugins();
    ASSERT_EQ(plugins.size(), 1);
    EXPECT_EQ(plugins[0]->name(), "Plugin2");
}

TEST(TestPluginManager, LoadPluginsAdditiveWithDefault)
{
    std::filesystem::path tempPluginDir
        = hipdnn_backend::platform_utilities::getCurrentModuleDirectory() /= "test_plugins_dir";
    ScopedDirectory defaultDir(tempPluginDir);

    // Place a plugin in the default directory
    std::filesystem::copy_file(
        FULL_PLUGIN_PATH1, defaultDir.path() / std::filesystem::path(FULL_PLUGIN_PATH1).filename());

    TestPluginManager pluginManager;
    pluginManager.loadPlugins({PLUGIN_PATH2}, HIPDNN_PLUGIN_LOADING_ADDITIVE);

    const auto& plugins = pluginManager.getPlugins();
    ASSERT_EQ(plugins.size(), 2);

    // Verify both plugins (default and custom) were loaded
    std::set<std::string_view> pluginNames;
    for(const auto& p : plugins)
    {
        pluginNames.insert(p->name());
    }
    EXPECT_TRUE(pluginNames.find("Plugin1") != pluginNames.end());
    EXPECT_TRUE(pluginNames.find("Plugin2") != pluginNames.end());
}

TEST(TestPluginManager, LoadPluginsCombinedFileAndDirectory)
{
    std::filesystem::path tempPluginDir
        = hipdnn_backend::platform_utilities::getCurrentModuleDirectory()
        /= "temp_plugin_dir_combined";

    ScopedDirectory tempDir(tempPluginDir);

    std::filesystem::copy_file(
        FULL_PLUGIN_PATH1, tempDir.path() / std::filesystem::path(FULL_PLUGIN_PATH1).filename());

    TestPluginManager pluginManager;
    pluginManager.loadPlugins({tempDir.path(), PLUGIN_PATH2}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = pluginManager.getPlugins();
    ASSERT_EQ(plugins.size(), 2);

    std::set<std::string_view> pluginNames;
    for(const auto& p : plugins)
    {
        pluginNames.insert(p->name());
    }
    EXPECT_TRUE(pluginNames.find("Plugin1") != pluginNames.end());
    EXPECT_TRUE(pluginNames.find("Plugin2") != pluginNames.end());
}

TEST(TestPluginManager, LastError)
{
    TestPluginManager pluginManager;

    std::set<std::filesystem::path> pluginPaths = {PLUGIN_PATH1};
    pluginManager.loadPlugins(pluginPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = pluginManager.getPlugins();
    ASSERT_EQ(plugins.size(), 1);

    using FuncType = hipdnnPluginStatus_t (*)(const char**);
    auto funcGetName = plugins[0]->_lib.getSymbol<FuncType>("hipdnnPluginGetName");

    ASSERT_TRUE(plugins[0]->getLastErrorString().empty());

    ASSERT_NE(funcGetName(nullptr), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(plugins[0]->getLastErrorString(), "name is null");
}

TEST(TestPluginManager, LastErrorMultithreaded)
{
    TestPluginManager pluginManager;

    std::set<std::filesystem::path> pluginPaths = {PLUGIN_PATH1};
    pluginManager.loadPlugins(pluginPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = pluginManager.getPlugins();
    ASSERT_EQ(plugins.size(), 1);

    using FuncType = hipdnnPluginStatus_t (*)(const char**);
    auto funcGetName = plugins[0]->_lib.getSymbol<FuncType>("hipdnnPluginGetName");

    auto checkGetName = [&]() {
        if(!plugins[0]->getLastErrorString().empty())
        {
            return false;
        }

        return funcGetName(nullptr) != HIPDNN_PLUGIN_STATUS_SUCCESS
               && plugins[0]->getLastErrorString() == "name is null";
    };

    ASSERT_EQ(checkGetName(), true);

    bool result1 = false;
    bool result2 = false;

    std::thread thread1([&] { result1 = checkGetName(); });

    std::thread thread2([&] { result2 = checkGetName(); });

    thread1.join();
    thread2.join();

    ASSERT_EQ(result1, true);
    ASSERT_EQ(result2, true);
}

TEST(TestPluginManager, LastErrorOnSecondLoad)
{
    using FuncType = hipdnnPluginStatus_t (*)(const char**);
    const auto funcName = "hipdnnPluginGetName";

    std::set<std::filesystem::path> pluginPaths = {PLUGIN_PATH1};

    {
        TestPluginManager pluginManager;
        pluginManager.loadPlugins(pluginPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

        const auto& plugins = pluginManager.getPlugins();
        ASSERT_EQ(plugins.size(), 1);

        auto funcGetName = plugins[0]->_lib.getSymbol<FuncType>(funcName);
        funcGetName(nullptr);
    }

    {
        TestPluginManager pluginManager;
        pluginManager.loadPlugins(pluginPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

        const auto& plugins = pluginManager.getPlugins();
        ASSERT_EQ(plugins.size(), 1);

        auto funcGetName = plugins[0]->_lib.getSymbol<FuncType>(funcName);

        ASSERT_TRUE(plugins[0]->getLastErrorString().empty());
        ASSERT_NE(funcGetName(nullptr), HIPDNN_PLUGIN_STATUS_SUCCESS);
        ASSERT_EQ(plugins[0]->getLastErrorString(), "name is null");
    }
}

TEST_F(TestPluginCallback, SetLoggingCallback)
{

    plugin::SharedLibrary lib(PLUGIN_PATH1);

    Plugin plugin(std::move(lib));

    EXPECT_EQ(plugin.setLoggingCallback(dummyCallback), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_TRUE(s_callbackCalled);
    EXPECT_EQ(plugin.setLoggingCallback(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestPluginCore, GetPluginSearchPathsWithEnvVar)
{
    const char* envVarName = "TEST_PLUGIN_PATH";
    const std::string testPath = "/custom/plugin/path";

    ScopedEnvironmentVariableSetter envSetter(envVarName, testPath);

    std::set<std::filesystem::path> defaultPaths = {"/default/path1", "/default/path2"};

    auto result = TestPluginManager::getPluginSearchPaths(envVarName, defaultPaths);

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.begin()->string(), testPath);
}

TEST(TestPluginCore, GetPluginSearchPathsWithoutEnvVar)
{
    const char* envVarName = "TEST_PLUGIN_PATH_UNSET";

    std::set<std::filesystem::path> defaultPaths = {"/default/path1", "/default/path2"};

    auto result = TestPluginManager::getPluginSearchPaths(envVarName, defaultPaths);

    ASSERT_EQ(result.size(), 2);
    EXPECT_EQ(result, defaultPaths);
}

TEST(TestPluginCore, GetPluginSearchPathsWithEmptyEnvVar)
{
    const char* envVarName = "TEST_PLUGIN_PATH_EMPTY";

    ScopedEnvironmentVariableSetter envSetter(envVarName);

    std::set<std::filesystem::path> defaultPaths
        = {"/default/path1", "/default/path2", "/default/path3"};

    auto result = TestPluginManager::getPluginSearchPaths(envVarName, defaultPaths);

    ASSERT_EQ(result.size(), 3);
    EXPECT_EQ(result, defaultPaths);
}
