// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_test_sdk/utilities/FrontendGraphFactory.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

namespace
{

// ============================================================================
// Test fixture for verifying the HIP kernel plugin has no engines.
// Uses the frontend API to load the plugin dynamically and attempt graph builds.
// ============================================================================

class IntegrationHipKernelNoEngines : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(hipInit(0), hipSuccess);

        // Plugin paths must be set before creating the hipdnn handle
        auto pluginPath = std::filesystem::weakly_canonical(
            hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / PLUGIN_PATH);
        const std::string pluginPathStr = pluginPath.string();
        const std::array<const char*, 1> paths = {pluginPathStr.c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        ASSERT_EQ(hipdnnSetStream(_handle, _stream), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
        if(_stream != nullptr)
        {
            ASSERT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }

    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
};

} // namespace

// ============================================================================
// Verify that building a batchnorm inference graph fails (no engines registered)
// ============================================================================

TEST_F(IntegrationHipKernelNoEngines, BatchnormInferenceGraphBuildFails)
{
    auto graph = hipdnn_test_sdk::utilities::FrontendGraphFactory::createBatchnormInferenceGraph();

    auto result = graph.build(_handle);
    EXPECT_TRUE(result.is_bad()) << "Expected build to fail since no engines are registered";
}
