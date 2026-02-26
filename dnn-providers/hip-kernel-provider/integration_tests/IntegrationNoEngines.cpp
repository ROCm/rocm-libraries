// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/PluginApi.h>

#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

namespace {

// ============================================================================
// Test fixture for verifying the HIP kernel plugin has no engines
// ============================================================================

class IntegrationHipKernelNoEngines : public ::testing::Test {};

}  // namespace

// ============================================================================
// Verify that the HIP kernel plugin reports zero engines
// ============================================================================

TEST_F(IntegrationHipKernelNoEngines, GetAllEngineIdsReturnsZero) {
    // Query the total number of engines (maxEngines=0 means query-only)
    uint32_t numEngines = 0;
    auto status = hipdnnEnginePluginGetAllEngineIds(nullptr, 0, &numEngines);

    ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(numEngines, 0u) << "HIP kernel plugin should have no engines registered yet";
}

TEST_F(IntegrationHipKernelNoEngines, CreateAndDestroyHandle) {
    hipdnnEnginePluginHandle_t handle = nullptr;
    auto status = hipdnnEnginePluginCreate(&handle);
    ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    status = hipdnnEnginePluginDestroy(handle);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST_F(IntegrationHipKernelNoEngines, GetPluginNameAndVersion) {
    const char* name = nullptr;
    auto status = hipdnnPluginGetName(&name);
    ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(name, "hip_kernel_provider_plugin");

    const char* version = nullptr;
    status = hipdnnPluginGetVersion(&version);
    ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(version, "1.0.0");
}

TEST_F(IntegrationHipKernelNoEngines, GetPluginType) {
    hipdnnPluginType_t type = static_cast<hipdnnPluginType_t>(0);
    auto status = hipdnnPluginGetType(&type);
    ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(type, HIPDNN_PLUGIN_TYPE_ENGINE);
}
