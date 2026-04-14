// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <test_plugins/TestPluginConstants.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

struct KnobQueryTestCase
{
    std::string description;
    int64_t engineId;
    std::vector<std::string> requiredKnobIds;

    friend std::ostream& operator<<(std::ostream& os, const KnobQueryTestCase& tc)
    {
        os << "KnobQueryTestCase{description: " << tc.description << ", engineId: " << tc.engineId
           << ", requiredKnobIds: [";
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

        result = graph.build_operation_graph(_handle);
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
                          {"test.int_knob",
                           "test.float_knob",
                           "test.string_knob",
                           "test.deprecated_knob",
                           "test.shared.deterministic"}},
        KnobQueryTestCase{
            "GoodPluginHasNoKnobs", hipdnn_tests::plugin_constants::engineId<GoodPlugin>(), {}},
        KnobQueryTestCase{
            "KnobsPluginEngineBHasThreeKnobs",
            hipdnn_tests::plugin_constants::engineId<KnobsPluginEngineB>(),
            {"test.engine_b.block_size", "test.engine_b.algorithm", "test.shared.deterministic"}}),
    [](const ::testing::TestParamInfo<KnobQueryTestCase>& info) { return info.param.description; });

TEST_P(IntegrationGraphKnobsApi, QueryKnobsFromEngine)
{
    const auto& testCase = GetParam();

    const Graph graph = createAndBuildSimpleGraph();

    std::vector<Knob> knobs;
    auto result = graph.get_knobs_for_engine(testCase.engineId, knobs);

    ASSERT_TRUE(result.is_good()) << result.get_message();

    EXPECT_EQ(knobs.size(), testCase.requiredKnobIds.size())
        << "Engine returned unexpected number of knobs";

    // Verify all required knob IDs are present
    for(const auto& requiredId : testCase.requiredKnobIds)
    {
        auto it = std::find_if(knobs.begin(), knobs.end(), [&requiredId](const Knob& knob) {
            return knob.knobId() == requiredId;
        });
        EXPECT_NE(it, knobs.end())
            << "Required knob '" << requiredId << "' not found in engine " << testCase.engineId;
    }

    // Additional constraint validation for integer and floating-point knobs
    for(const auto& knob : knobs)
    {
        if(knob.knobId() == "test.int_knob")
        {
            ASSERT_NE(knob.constraint(), nullptr) << "test.int_knob should have a constraint";
            auto* intConstraint = dynamic_cast<const IntConstraint*>(knob.constraint());
            ASSERT_NE(intConstraint, nullptr) << "test.int_knob constraint should be IntConstraint";
            EXPECT_EQ(intConstraint->getMinValue(), 0) << "test.int_knob minimum value mismatch";
            EXPECT_EQ(intConstraint->getMaxValue(), 100) << "test.int_knob maximum value mismatch";
            EXPECT_EQ(intConstraint->getStep(), 10) << "test.int_knob step value mismatch";
        }
        else if(knob.knobId() == "test.float_knob")
        {
            ASSERT_NE(knob.constraint(), nullptr) << "test.float_knob should have a constraint";
            auto* floatConstraint = dynamic_cast<const FloatConstraint*>(knob.constraint());
            ASSERT_NE(floatConstraint, nullptr)
                << "test.float_knob constraint should be FloatConstraint";
            EXPECT_DOUBLE_EQ(floatConstraint->getMinValue(), 0.0)
                << "test.float_knob minimum value mismatch";
            EXPECT_DOUBLE_EQ(floatConstraint->getMaxValue(), 1.0)
                << "test.float_knob maximum value mismatch";
        }
        else if(knob.knobId() == "test.shared.deterministic")
        {
            ASSERT_NE(knob.constraint(), nullptr)
                << "test.shared.deterministic should have a constraint";
            auto* intConstraint = dynamic_cast<const IntConstraint*>(knob.constraint());
            ASSERT_NE(intConstraint, nullptr)
                << "test.shared.deterministic constraint should be IntConstraint";
            EXPECT_EQ(intConstraint->getMinValue(), 0)
                << "test.shared.deterministic minimum value mismatch";
            EXPECT_EQ(intConstraint->getMaxValue(), 1)
                << "test.shared.deterministic maximum value mismatch";
            EXPECT_EQ(intConstraint->getStep(), 1)
                << "test.shared.deterministic step value mismatch";
        }
        else if(knob.knobId() == "test.engine_b.block_size")
        {
            ASSERT_NE(knob.constraint(), nullptr)
                << "test.engine_b.block_size should have a constraint";
            auto* intConstraint = dynamic_cast<const IntConstraint*>(knob.constraint());
            ASSERT_NE(intConstraint, nullptr)
                << "test.engine_b.block_size constraint should be IntConstraint";
            // This knob uses valid values {8, 16, 32, 64} rather than min/max
            const auto& validValues = intConstraint->getValidValues();
            EXPECT_FALSE(validValues.empty())
                << "test.engine_b.block_size should have valid values";
            const std::unordered_set<int64_t> expectedValues = {8, 16, 32, 64};
            EXPECT_EQ(validValues, expectedValues)
                << "test.engine_b.block_size valid values mismatch";
            EXPECT_EQ(intConstraint->getStep(), 1)
                << "test.engine_b.block_size step value mismatch";
        }
    }
}

TEST_P(IntegrationGraphKnobsApi, CreateExecutionPlanWithEmptyKnobs)
{
    const auto& testCase = GetParam();

    Graph graph = createAndBuildSimpleGraph();

    // Create execution plan with no knob settings (should use defaults)
    const std::vector<KnobSetting> emptySettings;

    auto result = graph.create_execution_plan_ext(testCase.engineId, emptySettings);
    EXPECT_TRUE(result.is_good()) << "Engine " << testCase.engineId
                                  << " should accept empty knob settings: " << result.get_message();
}

TEST_F(IntegrationGraphKnobsApi, CreateExecutionPlanWithValidKnobs)
{
    Graph graph = createAndBuildSimpleGraph();

    const int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    std::vector<KnobSetting> settings;
    settings.emplace_back("test.int_knob", static_cast<int64_t>(80));
    settings.emplace_back("test.float_knob", 0.75);
    settings.emplace_back("test.string_knob", std::string("accurate"));

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(IntegrationGraphKnobsApi, CreateExecutionPlanWithValidBoundaryKnobs)
{
    Graph graph = createAndBuildSimpleGraph();

    const int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();

    // Query knobs to get constraint information
    std::vector<Knob> knobs;
    auto queryResult = graph.get_knobs_for_engine(engineId, knobs);
    ASSERT_TRUE(queryResult.is_good()) << queryResult.get_message();

    // Find int_knob and float_knob to get their constraint information
    auto intKnobIt = std::find_if(knobs.begin(), knobs.end(), [](const Knob& knob) {
        return knob.knobId() == "test.int_knob";
    });
    ASSERT_NE(intKnobIt, knobs.end()) << "Could not find test.int_knob";
    auto* intConstraint = dynamic_cast<const IntConstraint*>(intKnobIt->constraint());
    ASSERT_NE(intConstraint, nullptr) << "test.int_knob should have IntConstraint";

    auto floatKnobIt = std::find_if(knobs.begin(), knobs.end(), [](const Knob& knob) {
        return knob.knobId() == "test.float_knob";
    });
    ASSERT_NE(floatKnobIt, knobs.end()) << "Could not find test.float_knob";
    auto* floatConstraint = dynamic_cast<const FloatConstraint*>(floatKnobIt->constraint());
    ASSERT_NE(floatConstraint, nullptr) << "test.float_knob should have FloatConstraint";

    // Test with minimum values
    {
        std::vector<KnobSetting> settings;
        settings.emplace_back("test.int_knob", intConstraint->getMinValue());
        settings.emplace_back("test.float_knob", floatConstraint->getMinValue());
        settings.emplace_back("test.string_knob", std::string("fast"));

        auto result = graph.create_execution_plan_ext(engineId, settings);
        EXPECT_TRUE(result.is_good())
            << "Minimum values should be accepted: " << result.get_message();
    }

    // Test with maximum values
    {
        std::vector<KnobSetting> settings;
        settings.emplace_back("test.int_knob", intConstraint->getMaxValue());
        settings.emplace_back("test.float_knob", floatConstraint->getMaxValue());
        settings.emplace_back("test.string_knob", std::string("balanced"));

        auto result = graph.create_execution_plan_ext(engineId, settings);
        EXPECT_TRUE(result.is_good())
            << "Maximum values should be accepted: " << result.get_message();
    }
}

TEST_F(IntegrationGraphKnobsApi, CreateExecutionPlanWithOutOfRangeIntKnob)
{
    Graph graph = createAndBuildSimpleGraph();

    // Try to set int knob with value outside range (min=0, max=100)
    const int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    std::vector<KnobSetting> settings;
    settings.emplace_back("test.int_knob", static_cast<int64_t>(150)); // Above max

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_FALSE(result.is_good()) << "Should reject int value above maximum";
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
}

TEST_F(IntegrationGraphKnobsApi, CreateExecutionPlanWithMisalignedIntKnobStep)
{
    Graph graph = createAndBuildSimpleGraph();

    // test.int_knob has step=10, so valid values are 0, 10, 20, ..., 100
    const int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    std::vector<KnobSetting> settings;
    settings.emplace_back("test.int_knob", static_cast<int64_t>(15)); // Not aligned with step

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_FALSE(result.is_good()) << "Should reject int value not aligned with step";
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
}

TEST_F(IntegrationGraphKnobsApi, CreateExecutionPlanWithInvalidStringKnob)
{
    Graph graph = createAndBuildSimpleGraph();

    // Try to set string knob with invalid choice (valid: "fast", "accurate", "balanced")
    const int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    std::vector<KnobSetting> settings;
    settings.emplace_back("test.string_knob", std::string("invalid_choice"));

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_FALSE(result.is_good()) << "Should reject invalid string choice";
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
}

TEST_F(IntegrationGraphKnobsApi, CreateExecutionPlanWithUnsupportedKnob)
{
    // Set up log recorder to capture warning about deprecated knob
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);

    Graph graph = createAndBuildSimpleGraph();

    // Try to set knob that doesn't exist on this engine
    const int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    std::vector<KnobSetting> settings;
    settings.emplace_back("nonexistent.knob", static_cast<int64_t>(42));

    auto result = graph.create_execution_plan_ext(engineId, settings);
    // Should succeed - unsupported knobs are ignored with warning
    EXPECT_TRUE(result.is_good()) << result.get_message();

    // Verify warning log is emitted for unsupported knob
    EXPECT_TRUE(recorder.hasLogContaining(
        HIPDNN_SEV_WARN, "Ignoring knob nonexistent.knob when creating execution plan for graph"))
        << "Expected warning log 'Ignoring knob nonexistent.knob when creating execution plan for "
           "graph'\nCaptured logs:\n"
        << recorder.getRecordedLogsAsString();
}

TEST_F(IntegrationGraphKnobsApi, CreateExecutionPlanWithDeprecatedKnob)
{
    // Set up log recorder to capture warning about deprecated knob
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);

    Graph graph = createAndBuildSimpleGraph();

    const int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    std::vector<KnobSetting> settings;
    settings.emplace_back("test.deprecated_knob", static_cast<int64_t>(5));

    auto result = graph.create_execution_plan_ext(engineId, settings);
    // Should succeed - deprecated knobs are used with warning
    EXPECT_TRUE(result.is_good()) << result.get_message();

    // Verify warning log is emitted for deprecated knob
    EXPECT_TRUE(recorder.hasLogContaining(
        HIPDNN_SEV_WARN, "Knob test.deprecated_knob has been marked as deprecated."))
        << "Expected warning log 'Knob test.deprecated_knob has been marked as "
           "deprecated.'\nCaptured logs:\n"
        << recorder.getRecordedLogsAsString();
}

TEST_F(IntegrationGraphKnobsApi, CreateExecutionPlanWithSharedKnob)
{
    // Set up log recorder to verify no deprecation warning is emitted
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);

    Graph graph = createAndBuildSimpleGraph();

    // Set shared knob for Engine A
    const int64_t engineIdA = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    std::vector<KnobSetting> settingsA;
    settingsA.emplace_back("test.shared.deterministic", static_cast<int64_t>(1));

    auto result = graph.create_execution_plan_ext(engineIdA, settingsA);

    EXPECT_TRUE(result.is_good()) << "Engine A should accept shared knob: " << result.get_message();

    // Set same shared knob for Engine B
    const int64_t engineIdB = hipdnn_tests::plugin_constants::engineId<KnobsPluginEngineB>();
    std::vector<KnobSetting> settingsB;
    settingsB.emplace_back("test.shared.deterministic", static_cast<int64_t>(1));

    result = graph.create_execution_plan_ext(engineIdB, settingsB);
    EXPECT_TRUE(result.is_good()) << "Engine B should accept shared knob: " << result.get_message();

    // Verify no deprecation warning is emitted for shared knob
    EXPECT_FALSE(recorder.hasLogContaining("has been marked as deprecated"))
        << "Shared knob should not trigger deprecation warning.\nCaptured logs:\n"
        << recorder.getRecordedLogsAsString();
}

TEST_F(IntegrationGraphKnobsApi, QueryKnobsBeforeGraphBuilt)
{
    // Intentionally create graph but DON'T build it (testing error case)
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT).set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("X").set_dim({2, 3, 4, 4});

    PointwiseAttributes attrs;
    attrs.set_mode(PointwiseMode::RELU_FWD);
    auto y = graph.pointwise(x, attrs);
    y->set_name("Y");

    // Try to query knobs WITHOUT building the graph first
    const int64_t engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    std::vector<Knob> knobs;
    auto result = graph.get_knobs_for_engine(engineId, knobs);

    EXPECT_FALSE(result.is_good()) << "Should fail when graph not built";
    EXPECT_EQ(result.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_NE(result.get_message().find("Graph has not been built"), std::string::npos);
}

} // namespace
