// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend.hpp>
#include <test_plugins/TestPluginConstants.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

class IntegrationGraphSupportCheck : public ::testing::Test
{
protected:
    void SetUp() override
    {
        loadPlugins({hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()});
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
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
    }

    static Graph createSimplePointwiseGraph()
    {
        const std::vector<int64_t> dims = {2, 3, 4, 4};

        Graph graph;
        graph.set_compute_data_type(DataType::FLOAT).set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_name("X")
            .set_uid(1)
            .set_dim(dims)
            .set_stride({dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1})
            .set_data_type(DataType::FLOAT);

        PointwiseAttributes attrs;
        attrs.set_mode(PointwiseMode::RELU_FWD);

        auto y = graph.pointwise(x, attrs);
        y->set_name("Y").set_uid(2).set_data_type(DataType::FLOAT).set_output(true);

        return graph;
    }

    hipdnnHandle_t _handle = nullptr;
};

TEST_F(IntegrationGraphSupportCheck, SupportedWithGoodPlugin)
{
    Graph graph = createSimplePointwiseGraph();

    auto result = graph.is_supported_ext(_handle);
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(IntegrationGraphSupportCheck, NotSupportedWhenNoApplicableEngines)
{
    recreateHandleWithPlugins(
        {hipdnn_tests::plugin_constants::testNoApplicableEnginesAPluginPath().c_str(),
         hipdnn_tests::plugin_constants::testNoApplicableEnginesBPluginPath().c_str()});

    Graph graph = createSimplePointwiseGraph();

    auto result = graph.is_supported_ext(_handle);
    EXPECT_FALSE(result.is_good()) << "Expected failure when no engines are applicable";
}

TEST_F(IntegrationGraphSupportCheck, SupportedWithMixedPlugins)
{
    recreateHandleWithPlugins(
        {hipdnn_tests::plugin_constants::testNoApplicableEnginesAPluginPath().c_str(),
         hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()});

    Graph graph = createSimplePointwiseGraph();

    auto result = graph.is_supported_ext(_handle);
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(IntegrationGraphSupportCheck, AutoBuildsGraphIfNotBuilt)
{
    // Create graph but do NOT call validate() or build_operation_graph()
    Graph graph = createSimplePointwiseGraph();

    // is_supported_ext should auto-validate and auto-build
    auto result = graph.is_supported_ext(_handle);
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(IntegrationGraphSupportCheck, SkipsBuildIfAlreadyBuilt)
{
    Graph graph = createSimplePointwiseGraph();

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
    Graph graph = createSimplePointwiseGraph();

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
