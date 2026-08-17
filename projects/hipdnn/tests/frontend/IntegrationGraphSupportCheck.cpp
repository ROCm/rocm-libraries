// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/FrontendGraphFactory.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <test_plugins/TestPluginConstants.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using hipdnn_test_sdk::utilities::FrontendGraphFactory;
using hipdnn_test_sdk::utilities::OperationType;

namespace
{

class IntegrationGraphSupportCheck : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // is_supported_ext drives finalize(), which reads the device from the
        // handle's stream — bind a real one and skip on no-GPU runners.
        SKIP_IF_NO_DEVICES();
        loadPlugins({hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()});
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
            _stream = nullptr;
        }
    }

    static void loadPlugins(std::initializer_list<const char*> pluginPaths)
    {
        const std::vector<const char*> paths(pluginPaths);
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
    }

    void recreateHandleWithPlugins(std::initializer_list<const char*> pluginPaths)
    {
        ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        _handle = nullptr;

        loadPlugins(pluginPaths);
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnSetStream(_handle, _stream), HIPDNN_STATUS_SUCCESS);
    }

    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
};

TEST_F(IntegrationGraphSupportCheck, SupportedWithGoodPlugin)
{
    Graph graph = FrontendGraphFactory::create(OperationType::CONV_FORWARD);

    auto result = graph.is_supported_ext(_handle);
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(IntegrationGraphSupportCheck, NotSupportedWhenNoApplicableEngines)
{
    recreateHandleWithPlugins(
        {hipdnn_tests::plugin_constants::testNoApplicableEnginesAPluginPath().c_str(),
         hipdnn_tests::plugin_constants::testNoApplicableEnginesBPluginPath().c_str()});

    Graph graph = FrontendGraphFactory::create(OperationType::CONV_FORWARD);

    auto result = graph.is_supported_ext(_handle);
    EXPECT_FALSE(result.is_good()) << "Expected failure when no engines are applicable";
}

TEST_F(IntegrationGraphSupportCheck, SupportedWithMixedPlugins)
{
    recreateHandleWithPlugins(
        {hipdnn_tests::plugin_constants::testNoApplicableEnginesAPluginPath().c_str(),
         hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()});

    Graph graph = FrontendGraphFactory::create(OperationType::CONV_FORWARD);

    auto result = graph.is_supported_ext(_handle);
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(IntegrationGraphSupportCheck, AutoBuildsGraphIfNotBuilt)
{
    // Create graph but do NOT call validate() or build_operation_graph()
    Graph graph = FrontendGraphFactory::create(OperationType::CONV_FORWARD);

    // is_supported_ext should auto-validate and auto-build
    auto result = graph.is_supported_ext(_handle);
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(IntegrationGraphSupportCheck, SkipsBuildIfAlreadyBuilt)
{
    Graph graph = FrontendGraphFactory::create(OperationType::CONV_FORWARD);

    // Explicitly validate and build first
    auto result = graph.validate();
    ASSERT_TRUE(result.is_good()) << result.get_message();

    result = graph.build_operation_graph(_handle);
    ASSERT_TRUE(result.is_good()) << result.get_message();

    // is_supported_ext should work on a pre-built graph
    result = graph.is_supported_ext(_handle);
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(IntegrationGraphSupportCheck, SupportedAfterIsSupportedDoesNotCorruptState)
{
    Graph graph = FrontendGraphFactory::create(OperationType::CONV_FORWARD);

    // First call is_supported_ext
    auto result = graph.is_supported_ext(_handle);
    ASSERT_TRUE(result.is_good()) << "is_supported_ext failed: " << result.get_message();

    // Then proceed with full execution plan flow
    result = graph.create_execution_plans();
    ASSERT_TRUE(result.is_good()) << "create_execution_plans failed: " << result.get_message();

    result = graph.check_support();
    ASSERT_TRUE(result.is_good()) << "check_support failed: " << result.get_message();

    result = graph.build_plans();
    ASSERT_TRUE(result.is_good()) << "build_plans failed: " << result.get_message();
}

} // namespace
