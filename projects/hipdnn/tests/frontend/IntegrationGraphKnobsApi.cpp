// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/LoggingUtils.hpp>
#include <test_plugins/TestPluginConstants.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

struct KnobQueryTestCase
{
    std::string description;
    int64_t engineId;
    size_t minKnobCount;
    std::vector<std::string> requiredKnobIds;

    friend std::ostream& operator<<(std::ostream& os, const KnobQueryTestCase& tc)
    {
        os << "KnobQueryTestCase{description: " << tc.description << ", engineId: " << tc.engineId
           << ", minKnobCount: " << tc.minKnobCount << ", requiredKnobIds: [";
        for(size_t i = 0; i < tc.requiredKnobIds.size(); ++i)
        {
            if(i > 0)
            {
                os << ", ";
            }
            os << tc.requiredKnobIds[i];
        }
        os << "]}";
        return os;
    }
};

class IntegrationGraphKnobsApi : public ::testing::TestWithParam<KnobQueryTestCase>
{
protected:
    void SetUp() override
    {
        // Load test plugins: knobs plugin, constraint validation plugin, and good plugin
        const std::array<const char*, 3> paths
            = {hipdnn_tests::plugin_constants::testKnobsPluginPath().c_str(),
               hipdnn_tests::plugin_constants::testKnobConstraintValidationPluginPath().c_str(),
               hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};

        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
    }

    // Create and build a simple graph for testing
    Graph createAndBuildSimpleGraph()
    {
        Graph graph;
        graph.set_compute_data_type(DataType::FLOAT).set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(1).set_name("X").set_dim({2, 3, 4, 4});

        PointwiseAttributes attrs;
        attrs.set_mode(PointwiseMode::RELU_FWD);
        auto y = graph.pointwise(x, attrs);
        y->set_uid(2);

        auto result = graph.build_operation_graph(_handle);
        EXPECT_TRUE(result.is_good()) << result.get_message();

        return graph;
    }

    hipdnnHandle_t _handle = nullptr;
};

INSTANTIATE_TEST_SUITE_P(
    ,
    IntegrationGraphKnobsApi,
    ::testing::Values(
        KnobQueryTestCase{"KnobsPluginHasFiveKnobs",
                          hipdnn_tests::plugin_constants::engineId<KnobsPlugin>(),
                          5,
                          {"test.int_knob",
                           "test.float_knob",
                           "test.string_knob",
                           "test.deprecated_knob",
                           "test.shared.deterministic"}},
        KnobQueryTestCase{
            "GoodPluginHasNoKnobs", hipdnn_tests::plugin_constants::engineId<GoodPlugin>(), 0, {}},
        KnobQueryTestCase{
            "KnobsPluginEngineBHasThreeKnobs",
            hipdnn_tests::plugin_constants::engineId<KnobsPluginEngineB>(),
            3,
            {"test.engine_b.block_size", "test.engine_b.algorithm", "test.shared.deterministic"}}),
    [](const ::testing::TestParamInfo<KnobQueryTestCase>& info) { return info.param.description; });

TEST_P(IntegrationGraphKnobsApi, QueryKnobsFromEngine)
{
    const auto& testCase = GetParam();

    Graph graph = createAndBuildSimpleGraph();

    std::vector<Knob> knobs;
    auto result = graph.get_knobs_for_engine(testCase.engineId, knobs);

    ASSERT_TRUE(result.is_good()) << result.get_message();

    EXPECT_EQ(knobs.size(), testCase.minKnobCount) << "Engine returned unexpected number of knobs";

    // Verify all required knob IDs are present
    for(const auto& requiredId : testCase.requiredKnobIds)
    {
        auto it = std::find_if(knobs.begin(), knobs.end(), [&requiredId](const Knob& knob) {
            return knob.knobId() == requiredId;
        });
        EXPECT_NE(it, knobs.end())
            << "Required knob '" << requiredId << "' not found in engine " << testCase.engineId;
    }
}

TEST_P(IntegrationGraphKnobsApi, CreateExecutionPlanWithEmptyKnobs)
{
    const auto& testCase = GetParam();

    Graph graph = createAndBuildSimpleGraph();

    // Create execution plan with no knob settings (should use defaults)
    std::vector<KnobSetting> emptySettings;

    auto result = graph.create_execution_plan_ext(testCase.engineId, emptySettings);
    EXPECT_TRUE(result.is_good()) << "Engine " << testCase.engineId
                                  << " should accept empty knob settings: " << result.get_message();
}

TEST_F(IntegrationGraphKnobsApi, CreateExecutionPlanWithValidKnobs)
{
    Graph graph = createAndBuildSimpleGraph();

    int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    std::vector<KnobSetting> settings;
    settings.emplace_back("test.int_knob", static_cast<int64_t>(80));
    settings.emplace_back("test.float_knob", 0.75);
    settings.emplace_back("test.string_knob", std::string("accurate"));

    auto result = graph.create_execution_plan_ext(engineId, settings);

    GTEST_SKIP()
        << "KNOWN ISSUE: initializeEngineConfig() finalizes descriptor before knobs can be set";

    // Once the issue is fixed, these expectations should pass:
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(IntegrationGraphKnobsApi, CreateExecutionPlanWithInvalidIntKnob)
{
    Graph graph = createAndBuildSimpleGraph();

    // Try to set int knob with value outside range (min=0, max=100)
    int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    std::vector<KnobSetting> settings;
    settings.emplace_back("test.int_knob", static_cast<int64_t>(150)); // Above max

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_FALSE(result.is_good()) << "Should reject int value above maximum";
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
}

TEST_F(IntegrationGraphKnobsApi, CreateExecutionPlanWithInvalidStringKnob)
{
    Graph graph = createAndBuildSimpleGraph();

    // Try to set string knob with invalid choice (valid: "fast", "accurate", "balanced")
    int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    std::vector<KnobSetting> settings;
    settings.emplace_back("test.string_knob", std::string("invalid_choice"));

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_FALSE(result.is_good()) << "Should reject invalid string choice";
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
}

TEST_F(IntegrationGraphKnobsApi, CreateExecutionPlanWithUnsupportedKnob)
{
    Graph graph = createAndBuildSimpleGraph();

    // Try to set knob that doesn't exist on this engine
    int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    std::vector<KnobSetting> settings;
    settings.emplace_back("nonexistent.knob", static_cast<int64_t>(42));

    auto result = graph.create_execution_plan_ext(engineId, settings);
    // Should succeed - unsupported knobs are ignored with warning
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(IntegrationGraphKnobsApi, CreateExecutionPlanWithDeprecatedKnob)
{
    // Start recording logs
    hipdnn_test_sdk::utilities::LogRecorder recorder;

    Graph graph = createAndBuildSimpleGraph();

    int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    std::vector<KnobSetting> settings;
    settings.emplace_back("test.deprecated_knob", static_cast<int64_t>(5));

    auto result = graph.create_execution_plan_ext(engineId, settings);

    EXPECT_TRUE(result.is_good()) << result.get_message();

    std::string expectedLog = "Knob test.deprecated_knob has been marked as deprecated.";
    EXPECT_TRUE(
        hipdnn_test_sdk::utilities::LogRecorder::hasLogContaining(HIPDNN_SEV_WARN, expectedLog))
        << "Expected warning-level deprecation log '" << expectedLog << "' was not found in logs:\n"
        << hipdnn_test_sdk::utilities::LogRecorder::getRecordedLogsAsString();
}

TEST_F(IntegrationGraphKnobsApi, CreateExecutionPlanWithSharedKnob)
{
    Graph graph = createAndBuildSimpleGraph();

    // Set shared knob for Engine A
    int64_t engineIdA = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    std::vector<KnobSetting> settingsA;
    settingsA.emplace_back("test.shared.deterministic", static_cast<int64_t>(1));

    auto result = graph.create_execution_plan_ext(engineIdA, settingsA);

    GTEST_SKIP()
        << "KNOWN ISSUE: initializeEngineConfig() finalizes descriptor before knobs can be set";

    // Once the issue is fixed, these expectations should pass:
    EXPECT_TRUE(result.is_good()) << "Engine A should accept shared knob: " << result.get_message();

    // Set same shared knob for Engine B
    int64_t engineIdB = hipdnn_tests::plugin_constants::engineId<KnobsPluginEngineB>();
    std::vector<KnobSetting> settingsB;
    settingsB.emplace_back("test.shared.deterministic", static_cast<int64_t>(1));

    result = graph.create_execution_plan_ext(engineIdB, settingsB);
    EXPECT_TRUE(result.is_good()) << "Engine B should accept shared knob: " << result.get_message();
}

TEST_F(IntegrationGraphKnobsApi, QueryKnobsBeforeGraphBuilt)
{
    // Intentionally create graph but DON'T build it (testing error case)
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT).set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_name("X").set_dim({2, 3, 4, 4});

    PointwiseAttributes attrs;
    attrs.set_mode(PointwiseMode::RELU_FWD);
    auto y = graph.pointwise(x, attrs);
    y->set_uid(2);

    // Try to query knobs WITHOUT building the graph first
    int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    std::vector<Knob> knobs;
    auto result = graph.get_knobs_for_engine(engineId, knobs);

    EXPECT_FALSE(result.is_good()) << "Should fail when graph not built";
    EXPECT_EQ(result.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_NE(result.get_message().find("Graph has not been built"), std::string::npos);
}

} // namespace
