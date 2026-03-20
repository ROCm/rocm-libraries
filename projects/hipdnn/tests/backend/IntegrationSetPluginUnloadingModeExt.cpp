// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include "hipdnn_backend.h"
#include <gtest/gtest.h>
#include <test_plugins/TestPluginConstants.hpp>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <fstream>
#endif

namespace
{
// Check if a shared library is currently loaded in the process
// Uses /proc/self/maps on Linux to check for loaded libraries by filename,
// GetModuleHandle on Windows
bool isSharedLibraryLoaded(const char* libraryPath)
{
#ifdef _WIN32
    // On Windows, GetModuleHandle returns NULL if the module is not loaded
    HMODULE handle = GetModuleHandleA(libraryPath);
    return handle != nullptr;
#else
    // Extract the filename from the path for matching
    const std::filesystem::path path(libraryPath);
    const std::string filename = path.filename().string();

    // Read /proc/self/maps to find all loaded shared objects
    std::ifstream maps("/proc/self/maps");
    if(!maps.is_open())
    {
        return false;
    }

    std::string line;
    while(std::getline(maps, line))
    {
        // Each line in /proc/self/maps contains the mapped file path at the end
        // Check if the line contains our library filename
        if(line.find(filename) != std::string::npos)
        {
            return true;
        }
    }
    return false;
#endif
}
} // namespace

class IntegrationSetPluginUnloadingModeExt : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Set up test plugin paths
        const std::array<const char*, 1> paths
            = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        // Reset to default mode after each test
        hipdnnSetPluginUnloadMode_ext(HIPDNN_DEFAULT_PLUGIN_UNLOADING_MODE);
    }

    static const std::string& getPluginPath()
    {
        return hipdnn_tests::plugin_constants::testGoodPluginPath();
    }
};

TEST_F(IntegrationSetPluginUnloadingModeExt, InvalidModeReturnsBadParam)
{
    const hipdnnStatus_t status
        = hipdnnSetPluginUnloadMode_ext(static_cast<hipdnnPluginUnloadingMode_ext_t>(-1));
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(IntegrationSetPluginUnloadingModeExt, LazyModeKeepsPluginLoaded)
{
    // Set lazy mode
    hipdnnStatus_t status = hipdnnSetPluginUnloadMode_ext(HIPDNN_PLUGIN_UNLOAD_LAZY);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    // Create first handle - this loads plugins
    hipdnnHandle_t handle1 = nullptr;
    status = hipdnnCreate(&handle1);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle1, nullptr);

    // Plugin should now be loaded
    EXPECT_TRUE(isSharedLibraryLoaded(getPluginPath().c_str()));

    // Destroy first handle
    EXPECT_EQ(hipdnnDestroy(handle1), HIPDNN_STATUS_SUCCESS);

    // In lazy mode, plugin should still be loaded after handle destruction
    EXPECT_TRUE(isSharedLibraryLoaded(getPluginPath().c_str()));

    // Create second handle - should still be using the same loaded plugin
    hipdnnHandle_t handle2 = nullptr;
    status = hipdnnCreate(&handle2);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle2, nullptr);

    EXPECT_TRUE(isSharedLibraryLoaded(getPluginPath().c_str()));

    EXPECT_EQ(hipdnnDestroy(handle2), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationSetPluginUnloadingModeExt, EagerModeUnloadsPlugin)
{
    // Set eager mode
    hipdnnStatus_t status = hipdnnSetPluginUnloadMode_ext(HIPDNN_PLUGIN_UNLOAD_EAGER);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    // Plugin should not be loaded yet (no handle created)
    EXPECT_FALSE(isSharedLibraryLoaded(getPluginPath().c_str()));

    // Create first handle - this loads plugins
    hipdnnHandle_t handle1 = nullptr;
    status = hipdnnCreate(&handle1);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle1, nullptr);

    // Plugin should now be loaded
    EXPECT_TRUE(isSharedLibraryLoaded(getPluginPath().c_str()));

    // Destroy first handle - plugins unload in eager mode
    EXPECT_EQ(hipdnnDestroy(handle1), HIPDNN_STATUS_SUCCESS);

    // In eager mode, plugin should be unloaded after handle destruction
    EXPECT_FALSE(isSharedLibraryLoaded(getPluginPath().c_str()));

    // Create second handle - should reload plugins
    hipdnnHandle_t handle2 = nullptr;
    status = hipdnnCreate(&handle2);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle2, nullptr);

    EXPECT_TRUE(isSharedLibraryLoaded(getPluginPath().c_str()));

    EXPECT_EQ(hipdnnDestroy(handle2), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationSetPluginUnloadingModeExt, SwitchingToEagerModeUnloadsPlugins)
{
    // Start with lazy mode
    hipdnnStatus_t status = hipdnnSetPluginUnloadMode_ext(HIPDNN_PLUGIN_UNLOAD_LAZY);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    // Create and destroy handle in lazy mode - plugin stays loaded
    hipdnnHandle_t handle1 = nullptr;
    status = hipdnnCreate(&handle1);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnDestroy(handle1), HIPDNN_STATUS_SUCCESS);

    // Plugin should still be loaded (lazy mode)
    EXPECT_TRUE(isSharedLibraryLoaded(getPluginPath().c_str()));

    // Switch to eager mode - should unload plugins
    status = hipdnnSetPluginUnloadMode_ext(HIPDNN_PLUGIN_UNLOAD_EAGER);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    // Plugin should now be unloaded (no handles exist, eager mode clears persistentPmPtr)
    EXPECT_FALSE(isSharedLibraryLoaded(getPluginPath().c_str()));
}

TEST_F(IntegrationSetPluginUnloadingModeExt, SetPluginPathsUnloadsPluginsInLazyMode)
{
    // Start with lazy mode
    hipdnnStatus_t status = hipdnnSetPluginUnloadMode_ext(HIPDNN_PLUGIN_UNLOAD_LAZY);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    // Create and destroy handle - plugin stays loaded in lazy mode
    hipdnnHandle_t handle1 = nullptr;
    status = hipdnnCreate(&handle1);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnDestroy(handle1), HIPDNN_STATUS_SUCCESS);

    // Plugin should still be loaded (lazy mode)
    EXPECT_TRUE(isSharedLibraryLoaded(getPluginPath().c_str()));

    // Set plugin paths again - should clear persistentPmPtr and unload plugins
    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::testExecuteFailsPluginPath().c_str()};
    status = hipdnnSetEnginePluginPaths_ext(
        paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    // Plugin should now be unloaded (setPluginPaths clears persistentPmPtr)
    EXPECT_FALSE(isSharedLibraryLoaded(getPluginPath().c_str()));
}
