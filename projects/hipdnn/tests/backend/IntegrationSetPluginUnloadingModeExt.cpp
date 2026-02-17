// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include "hipdnn_backend.h"
#include <gtest/gtest.h>

class IntegrationSetPluginUnloadingModeExt : public ::testing::Test
{
protected:
    void TearDown() override
    {
        // Reset to default mode after each test
        hipdnnSetPluginUnloadingMode_ext(HIPDNN_DEFAULT_PLUGIN_UNLOADING_MODE);
    }
};

TEST_F(IntegrationSetPluginUnloadingModeExt, SetLazyModeSucceeds)
{
    hipdnnStatus_t status = hipdnnSetPluginUnloadingMode_ext(HIPDNN_PLUGIN_UNLOAD_LAZY);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationSetPluginUnloadingModeExt, SetEagerModeSucceeds)
{
    hipdnnStatus_t status = hipdnnSetPluginUnloadingMode_ext(HIPDNN_PLUGIN_UNLOAD_EAGER);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationSetPluginUnloadingModeExt, InvalidModeReturnsBadParam)
{
    hipdnnStatus_t status
        = hipdnnSetPluginUnloadingMode_ext(static_cast<hipdnnPluginUnloadingMode_ext_t>(-1));
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(IntegrationSetPluginUnloadingModeExt, CanSetModeBeforeHandleCreation)
{
    // Set lazy mode before creating any handles
    hipdnnStatus_t status = hipdnnSetPluginUnloadingMode_ext(HIPDNN_PLUGIN_UNLOAD_LAZY);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    // Create and destroy a handle - should work normally
    hipdnnHandle_t handle = nullptr;
    status = hipdnnCreate(&handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationSetPluginUnloadingModeExt, CanSetModeAfterHandleDestroyed)
{
    // Create and destroy a handle first
    hipdnnHandle_t handle = nullptr;
    hipdnnStatus_t status = hipdnnCreate(&handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);

    // Now set mode - should succeed
    status = hipdnnSetPluginUnloadingMode_ext(HIPDNN_PLUGIN_UNLOAD_EAGER);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationSetPluginUnloadingModeExt, CanSetModeWhileHandleExists)
{
    // Create a handle
    hipdnnHandle_t handle = nullptr;
    hipdnnStatus_t status = hipdnnCreate(&handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    // Set mode while handle exists - should succeed
    status = hipdnnSetPluginUnloadingMode_ext(HIPDNN_PLUGIN_UNLOAD_LAZY);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    status = hipdnnSetPluginUnloadingMode_ext(HIPDNN_PLUGIN_UNLOAD_EAGER);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationSetPluginUnloadingModeExt, LazyModeAllowsMultipleHandleCycles)
{
    // Set lazy mode
    hipdnnStatus_t status = hipdnnSetPluginUnloadingMode_ext(HIPDNN_PLUGIN_UNLOAD_LAZY);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    // Create first handle - this loads plugins
    hipdnnHandle_t handle1 = nullptr;
    status = hipdnnCreate(&handle1);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle1, nullptr);

    // Destroy first handle
    EXPECT_EQ(hipdnnDestroy(handle1), HIPDNN_STATUS_SUCCESS);

    // Create second handle - should succeed (lazy mode keeps plugins loaded)
    hipdnnHandle_t handle2 = nullptr;
    status = hipdnnCreate(&handle2);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle2, nullptr);

    EXPECT_EQ(hipdnnDestroy(handle2), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationSetPluginUnloadingModeExt, EagerModeAllowsMultipleHandleCycles)
{
    // Set eager mode
    hipdnnStatus_t status = hipdnnSetPluginUnloadingMode_ext(HIPDNN_PLUGIN_UNLOAD_EAGER);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    // Create first handle
    hipdnnHandle_t handle1 = nullptr;
    status = hipdnnCreate(&handle1);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle1, nullptr);

    // Destroy first handle - plugins unload in eager mode
    EXPECT_EQ(hipdnnDestroy(handle1), HIPDNN_STATUS_SUCCESS);

    // Create second handle - should succeed (plugins reload)
    hipdnnHandle_t handle2 = nullptr;
    status = hipdnnCreate(&handle2);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle2, nullptr);

    EXPECT_EQ(hipdnnDestroy(handle2), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationSetPluginUnloadingModeExt, CanSwitchModesBetweenHandleCycles)
{
    // Start with lazy mode
    hipdnnStatus_t status = hipdnnSetPluginUnloadingMode_ext(HIPDNN_PLUGIN_UNLOAD_LAZY);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    // Create and destroy handle in lazy mode
    hipdnnHandle_t handle1 = nullptr;
    status = hipdnnCreate(&handle1);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnDestroy(handle1), HIPDNN_STATUS_SUCCESS);

    // Switch to eager mode
    status = hipdnnSetPluginUnloadingMode_ext(HIPDNN_PLUGIN_UNLOAD_EAGER);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    // Create and destroy handle in eager mode
    hipdnnHandle_t handle2 = nullptr;
    status = hipdnnCreate(&handle2);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnDestroy(handle2), HIPDNN_STATUS_SUCCESS);

    // Switch back to lazy mode
    status = hipdnnSetPluginUnloadingMode_ext(HIPDNN_PLUGIN_UNLOAD_LAZY);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    // Create handle - should succeed
    hipdnnHandle_t handle3 = nullptr;
    status = hipdnnCreate(&handle3);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnDestroy(handle3), HIPDNN_STATUS_SUCCESS);
}
