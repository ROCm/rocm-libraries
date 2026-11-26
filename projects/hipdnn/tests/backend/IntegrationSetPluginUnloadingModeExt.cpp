// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include "hipdnn_backend.h"
#include <array>
#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/ProcessUtils.hpp>
#include <test_plugins/TestPluginConstants.hpp>

using hipdnn_test_sdk::utilities::isLibraryLoaded;

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
    EXPECT_TRUE(isLibraryLoaded(getPluginPath()));

    // Destroy first handle
    EXPECT_EQ(hipdnnDestroy(handle1), HIPDNN_STATUS_SUCCESS);

    // In lazy mode, plugin should still be loaded after handle destruction
    EXPECT_TRUE(isLibraryLoaded(getPluginPath()));

    // Create second handle - should still be using the same loaded plugin
    hipdnnHandle_t handle2 = nullptr;
    status = hipdnnCreate(&handle2);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle2, nullptr);

    EXPECT_TRUE(isLibraryLoaded(getPluginPath()));

    EXPECT_EQ(hipdnnDestroy(handle2), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationSetPluginUnloadingModeExt, EagerModeUnloadsPlugin)
{
    // Set eager mode
    hipdnnStatus_t status = hipdnnSetPluginUnloadMode_ext(HIPDNN_PLUGIN_UNLOAD_EAGER);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    // Plugin should not be loaded yet (no handle created)
    EXPECT_FALSE(isLibraryLoaded(getPluginPath()));

    // Create first handle - this loads plugins
    hipdnnHandle_t handle1 = nullptr;
    status = hipdnnCreate(&handle1);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle1, nullptr);

    // Plugin should now be loaded
    EXPECT_TRUE(isLibraryLoaded(getPluginPath()));

    // Destroy first handle - plugins unload in eager mode
    EXPECT_EQ(hipdnnDestroy(handle1), HIPDNN_STATUS_SUCCESS);

    // In eager mode, plugin should be unloaded after handle destruction
    EXPECT_FALSE(isLibraryLoaded(getPluginPath()));

    // Create second handle - should reload plugins
    hipdnnHandle_t handle2 = nullptr;
    status = hipdnnCreate(&handle2);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle2, nullptr);

    EXPECT_TRUE(isLibraryLoaded(getPluginPath()));

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
    EXPECT_TRUE(isLibraryLoaded(getPluginPath()));

    // Switch to eager mode - should unload plugins
    status = hipdnnSetPluginUnloadMode_ext(HIPDNN_PLUGIN_UNLOAD_EAGER);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    // Plugin should now be unloaded (no handles exist, eager mode clears persistentPmPtr)
    EXPECT_FALSE(isLibraryLoaded(getPluginPath()));
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
    EXPECT_TRUE(isLibraryLoaded(getPluginPath()));

    // Set plugin paths again - should clear persistentPmPtr and unload plugins
    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::testExecuteFailsPluginPath().c_str()};
    status = hipdnnSetEnginePluginPaths_ext(
        paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    // Plugin should now be unloaded (setPluginPaths clears persistentPmPtr)
    EXPECT_FALSE(isLibraryLoaded(getPluginPath()));
}
