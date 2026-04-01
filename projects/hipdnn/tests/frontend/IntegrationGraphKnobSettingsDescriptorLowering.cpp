// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <filesystem>

#include <gtest/gtest.h>
#include <hipdnn_frontend.hpp>
#include <test_plugins/TestKnobExpectation.hpp>
#include <test_plugins/TestPluginConstants.hpp>
#include <test_plugins/TestPluginKnobRecorder.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

std::string serializeExpectedKnobSettings(int64_t engineId,
                                          const std::vector<KnobSetting>& settings)
{
    std::vector<hipdnn_tests::knob_expectation::CanonicalKnobSetting> canonicalSettings;
    canonicalSettings.reserve(settings.size());

    for(const auto& setting : settings)
    {
        canonicalSettings.push_back(hipdnn_tests::knob_expectation::makeExpectedKnobSetting(
            setting.knobId(), setting.value()));
    }

    return hipdnn_tests::knob_expectation::serializeExpectedKnobSettings(
        engineId, std::move(canonicalSettings));
}

class TestableGraph : public Graph
{
public:
    using Graph::build_operation_graph_via_descriptors;
    using Graph::create_execution_plan_ext;
};

class IntegrationGraphKnobsDescriptorLowering : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const std::array<const char*, 1> paths
            = {hipdnn_tests::plugin_constants::testKnobsPluginPath().c_str()};

        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);

        // Query the exact plugin paths the backend resolved when loading,
        // then find the knobs plugin by name. This ensures we dlopen the same
        // library instance (plugins are loaded with RTLD_LOCAL).
        std::vector<std::filesystem::path> loadedPaths;
        auto pathResult = getLoadedEnginePluginPaths(_handle, loadedPaths);
        ASSERT_TRUE(pathResult.is_good()) << pathResult.get_message();

        auto it = std::find_if(loadedPaths.begin(), loadedPaths.end(), [](const auto& path) {
            return path.string().find(TEST_KNOBS_PLUGIN_NAME) != std::string::npos;
        });
        ASSERT_NE(it, loadedPaths.end()) << "TestKnobsPlugin not found in loaded plugins";

        _knobRecorder = std::make_unique<hipdnn_tests::TestPluginKnobRecorder>(*it);
        _knobRecorder->reset();
    }

    void TearDown() override
    {
        _knobRecorder.reset();
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
    }

    TestableGraph createAndBuildSimpleGraph()
    {
        TestableGraph graph;
        graph.set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_name("X").set_dim({2, 3, 4, 4}).set_stride({48, 16, 4, 1});

        PointwiseAttributes attrs;
        attrs.set_mode(PointwiseMode::RELU_FWD);
        auto y = graph.pointwise(x, attrs);
        y->set_name("Y");

        auto result = graph.validate();
        EXPECT_TRUE(result.is_good()) << result.get_message();

        result = graph.build_operation_graph_via_descriptors(_handle);
        EXPECT_TRUE(result.is_good()) << result.get_message();

        return graph;
    }

    hipdnnHandle_t _handle = nullptr;
    std::unique_ptr<hipdnn_tests::TestPluginKnobRecorder> _knobRecorder;
};

TEST_F(IntegrationGraphKnobsDescriptorLowering, CreateExecutionPlanWithIntKnob)
{
    TestableGraph graph = createAndBuildSimpleGraph();

    const int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    const std::vector<KnobSetting> settings = {KnobSetting("test.int_knob", int64_t{80})};

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(_knobRecorder->last(), serializeExpectedKnobSettings(engineId, settings));
}

TEST_F(IntegrationGraphKnobsDescriptorLowering, CreateExecutionPlanWithFloatKnob)
{
    TestableGraph graph = createAndBuildSimpleGraph();

    const int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    const std::vector<KnobSetting> settings = {KnobSetting("test.float_knob", 0.75)};

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(_knobRecorder->last(), serializeExpectedKnobSettings(engineId, settings));
}

TEST_F(IntegrationGraphKnobsDescriptorLowering, CreateExecutionPlanWithStringKnob)
{
    TestableGraph graph = createAndBuildSimpleGraph();

    const int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    const std::vector<KnobSetting> settings
        = {KnobSetting("test.string_knob", std::string("accurate"))};

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(_knobRecorder->last(), serializeExpectedKnobSettings(engineId, settings));
}

TEST_F(IntegrationGraphKnobsDescriptorLowering, CreateExecutionPlanWithMultipleKnobs)
{
    TestableGraph graph = createAndBuildSimpleGraph();

    const int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    const std::vector<KnobSetting> settings
        = {KnobSetting("test.int_knob", int64_t{80}),
           KnobSetting("test.float_knob", 0.75),
           KnobSetting("test.string_knob", std::string("accurate"))};

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(_knobRecorder->last(), serializeExpectedKnobSettings(engineId, settings));
}

TEST_F(IntegrationGraphKnobsDescriptorLowering, CreateExecutionPlanWithSharedKnob)
{
    TestableGraph graph = createAndBuildSimpleGraph();

    const int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPluginEngineB>();
    const std::vector<KnobSetting> settings
        = {KnobSetting("test.shared.deterministic", int64_t{1})};

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(_knobRecorder->last(), serializeExpectedKnobSettings(engineId, settings));
}

TEST_F(IntegrationGraphKnobsDescriptorLowering, CreateExecutionPlanWithDeprecatedKnob)
{
    TestableGraph graph = createAndBuildSimpleGraph();

    const int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    const std::vector<KnobSetting> settings = {KnobSetting("test.deprecated_knob", int64_t{5})};

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(_knobRecorder->last(), serializeExpectedKnobSettings(engineId, settings));
}

TEST_F(IntegrationGraphKnobsDescriptorLowering, CreateExecutionPlanWithEmptyKnobs)
{
    TestableGraph graph = createAndBuildSimpleGraph();

    const int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    const std::vector<KnobSetting> settings;

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(_knobRecorder->last(), serializeExpectedKnobSettings(engineId, settings));
}

TEST_F(IntegrationGraphKnobsDescriptorLowering, CreateExecutionPlanFiltersUnsupportedKnob)
{
    TestableGraph graph = createAndBuildSimpleGraph();

    const int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    const std::vector<KnobSetting> settings = {KnobSetting("nonexistent.knob", int64_t{42})};
    const std::vector<KnobSetting> expectedSettings;

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(_knobRecorder->last(), serializeExpectedKnobSettings(engineId, expectedSettings));
}

} // namespace
