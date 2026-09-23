// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <filesystem>

#include <gtest/gtest.h>
#include <hipdnn_frontend.hpp>
#include <test_plugins/TestPluginConstants.hpp>
#include <test_plugins/TestPluginKnobRecorder.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using hipdnn_flatbuffers_sdk::data_objects::EngineConfigT;
using hipdnn_flatbuffers_sdk::data_objects::FloatValueT;
using hipdnn_flatbuffers_sdk::data_objects::IntValueT;
using hipdnn_flatbuffers_sdk::data_objects::KnobSettingT;
using hipdnn_flatbuffers_sdk::data_objects::StringValueT;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::TestableGraphKnobLowering;

namespace
{

EngineConfigT buildExpectedEngineConfig(int64_t engineId, const std::vector<KnobSetting>& settings)
{
    EngineConfigT config;
    config.engine_id = engineId;

    for(const auto& setting : settings)
    {
        auto knob = std::make_unique<KnobSettingT>();
        knob->knob_id = setting.knobId();
        std::visit(
            [&knob](const auto& val) {
                using T = std::decay_t<decltype(val)>;
                if constexpr(std::is_same_v<T, int64_t>)
                {
                    IntValueT intVal;
                    intVal.value = val;
                    knob->value.Set(intVal);
                }
                else if constexpr(std::is_same_v<T, double>)
                {
                    FloatValueT floatVal;
                    floatVal.value = val;
                    knob->value.Set(floatVal);
                }
                else if constexpr(std::is_same_v<T, std::string>)
                {
                    StringValueT stringVal;
                    stringVal.value = val;
                    knob->value.Set(std::move(stringVal));
                }
            },
            setting.value());
        config.knobs.push_back(std::move(knob));
    }

    std::sort(config.knobs.begin(), config.knobs.end(), [](const auto& a, const auto& b) {
        return a->knob_id < b->knob_id;
    });
    return config;
}

class IntegrationGraphKnobsDescriptorLowering : public IntegrationTestFixture
{
protected:
    void SetUp() override
    {
        IntegrationTestFixture::SetUp();
        // GTEST_SKIP() in the base only unwinds the base SetUp() frame; without this
        // guard, the post-base setup below would dereference a null _handle.
        if(IsSkipped())
        {
            return;
        }

        // Query the exact plugin paths the backend resolved when loading,
        // then find the knobs plugin by name. This ensures we dlopen the same
        // library instance (plugins are loaded with RTLD_LOCAL).
        std::vector<std::filesystem::path> loadedPaths;
        auto pathResult = getLoadedEnginePluginPaths(_handle, loadedPaths);
        ASSERT_TRUE(pathResult.is_good()) << pathResult.get_message();

        auto it = std::find_if(loadedPaths.begin(), loadedPaths.end(), [](const auto& path) {
            return path.string().find(TEST_KNOBS_PLUGIN_NAME) != std::string::npos;
        });
        ASSERT_NE(it, loadedPaths.end())
            << "TestKnobsPlugin not found in loaded plugins"; // NOLINT(readability-implicit-bool-conversion)

        _knobRecorder = std::make_unique<hipdnn_tests::TestPluginKnobRecorder>(*it);
        _knobRecorder->reset();
    }

    std::vector<std::string> getPluginPaths() const override
    {
        return {hipdnn_tests::plugin_constants::testKnobsPluginPath()};
    }

    void TearDown() override
    {
        _knobRecorder.reset();
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
    }

    TestableGraphKnobLowering createAndBuildSimpleGraph()
    {
        TestableGraphKnobLowering graph;
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

    std::unique_ptr<hipdnn_tests::TestPluginKnobRecorder> _knobRecorder;
};

TEST_F(IntegrationGraphKnobsDescriptorLowering, CreateExecutionPlanWithIntKnob)
{
    TestableGraphKnobLowering graph = createAndBuildSimpleGraph();

    const int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    const std::vector<KnobSetting> settings = {KnobSetting("test.int_knob", int64_t{80})};

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(_knobRecorder->last(), buildExpectedEngineConfig(engineId, settings));
}

TEST_F(IntegrationGraphKnobsDescriptorLowering, CreateExecutionPlanWithFloatKnob)
{
    TestableGraphKnobLowering graph = createAndBuildSimpleGraph();

    const int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    const std::vector<KnobSetting> settings = {KnobSetting("test.float_knob", 0.75)};

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(_knobRecorder->last(), buildExpectedEngineConfig(engineId, settings));
}

TEST_F(IntegrationGraphKnobsDescriptorLowering, CreateExecutionPlanWithStringKnob)
{
    TestableGraphKnobLowering graph = createAndBuildSimpleGraph();

    const int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    const std::vector<KnobSetting> settings
        = {KnobSetting("test.string_knob", std::string("accurate"))};

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(_knobRecorder->last(), buildExpectedEngineConfig(engineId, settings));
}

TEST_F(IntegrationGraphKnobsDescriptorLowering, CreateExecutionPlanWithMultipleKnobs)
{
    TestableGraphKnobLowering graph = createAndBuildSimpleGraph();

    const int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    const std::vector<KnobSetting> settings
        = {KnobSetting("test.int_knob", int64_t{80}),
           KnobSetting("test.float_knob", 0.75),
           KnobSetting("test.string_knob", std::string("accurate"))};

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(_knobRecorder->last(), buildExpectedEngineConfig(engineId, settings));
}

TEST_F(IntegrationGraphKnobsDescriptorLowering, CreateExecutionPlanWithSharedKnob)
{
    TestableGraphKnobLowering graph = createAndBuildSimpleGraph();

    const int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPluginEngineB>();
    const std::vector<KnobSetting> settings
        = {KnobSetting("test.shared.deterministic", int64_t{1})};

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(_knobRecorder->last(), buildExpectedEngineConfig(engineId, settings));
}

TEST_F(IntegrationGraphKnobsDescriptorLowering, CreateExecutionPlanWithDeprecatedKnob)
{
    TestableGraphKnobLowering graph = createAndBuildSimpleGraph();

    const int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    const std::vector<KnobSetting> settings = {KnobSetting("test.deprecated_knob", int64_t{5})};

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(_knobRecorder->last(), buildExpectedEngineConfig(engineId, settings));
}

TEST_F(IntegrationGraphKnobsDescriptorLowering, CreateExecutionPlanWithEmptyKnobs)
{
    TestableGraphKnobLowering graph = createAndBuildSimpleGraph();

    const int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    const std::vector<KnobSetting> settings;

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(_knobRecorder->last(), buildExpectedEngineConfig(engineId, settings));
}

TEST_F(IntegrationGraphKnobsDescriptorLowering, CreateExecutionPlanFiltersUnsupportedKnob)
{
    TestableGraphKnobLowering graph = createAndBuildSimpleGraph();

    const int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    const std::vector<KnobSetting> settings = {KnobSetting("nonexistent.knob", int64_t{42})};
    const std::vector<KnobSetting> expectedSettings;

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(_knobRecorder->last(), buildExpectedEngineConfig(engineId, expectedSettings));
}

} // namespace
