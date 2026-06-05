// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <test_plugins/TestPluginConstants.hpp>

#include <unordered_set>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::TestableGraphKnobs;

namespace
{
/// Integration tests for the descriptor-based knob lifting path.
/// Tests exercise detail::unpackKnobsFromDescriptors via
/// TestableGraphKnobs::get_knobs_for_engine_via_descriptors().
class IntegrationGraphKnobsDescriptorLifting : public IntegrationTestFixture
{
protected:
    std::vector<std::string> getPluginPaths() const override
    {
        return {hipdnn_tests::plugin_constants::testKnobsPluginPath(),
                hipdnn_tests::plugin_constants::testKnobConstraintValidationPluginPath(),
                hipdnn_tests::plugin_constants::testGoodPluginPath()};
    }

    TestableGraphKnobs createAndBuildSimpleGraph()
    {
        TestableGraphKnobs graph;
        graph.set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_name("X").set_dim({2, 3, 4, 4}).set_stride({48, 16, 4, 1});

        PointwiseAttributes attrs;
        attrs.set_mode(PointwiseMode::RELU_FWD);
        auto y = graph.pointwise(x, attrs);
        y->set_name("Y");

        auto validateResult = graph.validate();
        EXPECT_TRUE(validateResult.is_good()) << validateResult.get_message();

        auto result = graph.build_operation_graph(_handle);
        EXPECT_TRUE(result.is_good()) << result.get_message();

        return graph;
    }

    static const Knob* findKnob(const std::vector<Knob>& knobs, const std::string& knobId)
    {
        auto it = std::find_if(
            knobs.begin(), knobs.end(), [&](const Knob& knob) { return knob.knobId() == knobId; });
        return it == knobs.end() ? nullptr : &(*it);
    }
};

TEST_F(IntegrationGraphKnobsDescriptorLifting, KnobsPluginHasExpectedKnobs)
{
    const auto engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();

    auto graph = createAndBuildSimpleGraph();

    std::vector<Knob> knobs;
    auto result = graph.get_knobs_for_engine_via_descriptors(engineId, knobs);
    ASSERT_TRUE(result.is_good()) << result.get_message();

    ASSERT_EQ(knobs.size(), 5u);

    {
        const auto* knob = findKnob(knobs, "test.int_knob");
        ASSERT_NE(knob, nullptr);
        EXPECT_EQ(knob->valueType(), KnobValueType::INT64);
        EXPECT_EQ(std::get<int64_t>(knob->defaultValue()), 50);
        EXPECT_EQ(knob->description(), "Test integer knob with range 0-100");
        EXPECT_FALSE(knob->isDeprecated());

        auto* constraint = dynamic_cast<const IntConstraint*>(knob->constraint());
        ASSERT_NE(constraint, nullptr);
        EXPECT_EQ(constraint->getMinValue(), 0);
        EXPECT_EQ(constraint->getMaxValue(), 100);
        EXPECT_EQ(constraint->getStep(), 10);
        EXPECT_TRUE(constraint->getValidValues().empty());
    }

    {
        const auto* knob = findKnob(knobs, "test.float_knob");
        ASSERT_NE(knob, nullptr);
        EXPECT_EQ(knob->valueType(), KnobValueType::FLOAT64);
        EXPECT_DOUBLE_EQ(std::get<double>(knob->defaultValue()), 0.5);
        EXPECT_EQ(knob->description(), "Test float knob with range 0.0-1.0");
        EXPECT_FALSE(knob->isDeprecated());

        auto* constraint = dynamic_cast<const FloatConstraint*>(knob->constraint());
        ASSERT_NE(constraint, nullptr);
        EXPECT_DOUBLE_EQ(constraint->getMinValue(), 0.0);
        EXPECT_DOUBLE_EQ(constraint->getMaxValue(), 1.0);
    }

    {
        const auto* knob = findKnob(knobs, "test.string_knob");
        ASSERT_NE(knob, nullptr);
        EXPECT_EQ(knob->valueType(), KnobValueType::STRING);
        EXPECT_EQ(std::get<std::string>(knob->defaultValue()), "fast");
        EXPECT_EQ(knob->description(), "Test string knob with enum values");
        EXPECT_FALSE(knob->isDeprecated());

        auto* constraint = dynamic_cast<const StringConstraint*>(knob->constraint());
        ASSERT_NE(constraint, nullptr);
        EXPECT_EQ(constraint->getMaxLength(), 0);
        EXPECT_EQ(constraint->getValidValues(),
                  (std::unordered_set<std::string>{"fast", "accurate", "balanced"}));
    }

    {
        const auto* knob = findKnob(knobs, "test.deprecated_knob");
        ASSERT_NE(knob, nullptr);
        EXPECT_EQ(knob->valueType(), KnobValueType::INT64);
        EXPECT_EQ(std::get<int64_t>(knob->defaultValue()), 0);
        EXPECT_EQ(knob->description(), "Deprecated knob for testing");
        EXPECT_TRUE(knob->isDeprecated());

        auto* constraint = dynamic_cast<const IntConstraint*>(knob->constraint());
        ASSERT_NE(constraint, nullptr);
        EXPECT_EQ(constraint->getMinValue(), 0);
        EXPECT_EQ(constraint->getMaxValue(), 10);
        EXPECT_EQ(constraint->getStep(), 1);
        EXPECT_TRUE(constraint->getValidValues().empty());
    }

    {
        const auto* knob = findKnob(knobs, "test.shared.deterministic");
        ASSERT_NE(knob, nullptr);
        EXPECT_EQ(knob->valueType(), KnobValueType::INT64);
        EXPECT_EQ(std::get<int64_t>(knob->defaultValue()), 0);
        EXPECT_EQ(knob->description(), "Enable deterministic execution (shared across engines)");
        EXPECT_FALSE(knob->isDeprecated());

        auto* constraint = dynamic_cast<const IntConstraint*>(knob->constraint());
        ASSERT_NE(constraint, nullptr);
        EXPECT_EQ(constraint->getMinValue(), 0);
        EXPECT_EQ(constraint->getMaxValue(), 1);
        EXPECT_EQ(constraint->getStep(), 1);
        EXPECT_TRUE(constraint->getValidValues().empty());
    }
}

TEST_F(IntegrationGraphKnobsDescriptorLifting, EngineBKnobs)
{
    const auto engineId = hipdnn_tests::plugin_constants::engineId<KnobsPluginEngineB>();

    auto graph = createAndBuildSimpleGraph();

    std::vector<Knob> knobs;
    auto result = graph.get_knobs_for_engine_via_descriptors(engineId, knobs);
    ASSERT_TRUE(result.is_good()) << result.get_message();

    {
        ASSERT_EQ(knobs.size(), 3u);

        const auto* blockSizeKnob = findKnob(knobs, "test.engine_b.block_size");
        ASSERT_NE(blockSizeKnob, nullptr);
        EXPECT_EQ(blockSizeKnob->valueType(), KnobValueType::INT64);
        EXPECT_EQ(std::get<int64_t>(blockSizeKnob->defaultValue()), 16);
        EXPECT_EQ(blockSizeKnob->description(), "Block size for engine B (power of 2)");
        EXPECT_FALSE(blockSizeKnob->isDeprecated());

        auto* blockSizeConstraint = dynamic_cast<const IntConstraint*>(blockSizeKnob->constraint());
        ASSERT_NE(blockSizeConstraint, nullptr);
        EXPECT_EQ(blockSizeConstraint->getStep(), 1);
        EXPECT_EQ(blockSizeConstraint->getValidValues(),
                  (std::unordered_set<int64_t>{8, 16, 32, 64}));

        const auto* algorithmKnob = findKnob(knobs, "test.engine_b.algorithm");
        ASSERT_NE(algorithmKnob, nullptr);
        EXPECT_EQ(algorithmKnob->valueType(), KnobValueType::STRING);
        EXPECT_EQ(std::get<std::string>(algorithmKnob->defaultValue()), "winograd");
        EXPECT_EQ(algorithmKnob->description(), "Algorithm selection for engine B");
        EXPECT_FALSE(algorithmKnob->isDeprecated());

        auto* algorithmConstraint
            = dynamic_cast<const StringConstraint*>(algorithmKnob->constraint());
        ASSERT_NE(algorithmConstraint, nullptr);
        EXPECT_EQ(algorithmConstraint->getMaxLength(), 0);
        EXPECT_EQ(algorithmConstraint->getValidValues(),
                  (std::unordered_set<std::string>{"direct", "winograd", "fft"}));

        const auto* deterministicKnob = findKnob(knobs, "test.shared.deterministic");
        ASSERT_NE(deterministicKnob, nullptr);
        EXPECT_EQ(deterministicKnob->valueType(), KnobValueType::INT64);
        EXPECT_EQ(std::get<int64_t>(deterministicKnob->defaultValue()), 0);

        auto* deterministicConstraint
            = dynamic_cast<const IntConstraint*>(deterministicKnob->constraint());
        ASSERT_NE(deterministicConstraint, nullptr);
        EXPECT_EQ(deterministicConstraint->getMinValue(), 0);
        EXPECT_EQ(deterministicConstraint->getMaxValue(), 1);
        EXPECT_EQ(deterministicConstraint->getStep(), 1);
    }
}

TEST_F(IntegrationGraphKnobsDescriptorLifting, EngineWithNoKnobs)
{
    const auto engineId = hipdnn_tests::plugin_constants::engineId<GoodPlugin>();

    auto graph = createAndBuildSimpleGraph();

    std::vector<Knob> knobs;
    auto result = graph.get_knobs_for_engine_via_descriptors(engineId, knobs);
    ASSERT_TRUE(result.is_good()) << result.get_message();

    EXPECT_TRUE(knobs.empty())
        << "GoodPlugin should have no knobs"; // NOLINT(readability-implicit-bool-conversion)
}

TEST_F(IntegrationGraphKnobsDescriptorLifting, IntConstraintWithZeroMin)
{
    const auto engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();

    auto graph = createAndBuildSimpleGraph();

    std::vector<Knob> knobs;
    auto result = graph.get_knobs_for_engine_via_descriptors(engineId, knobs);
    ASSERT_TRUE(result.is_good()) << result.get_message();

    // test.int_knob has min=0, max=100, step=10
    auto it = std::find_if(
        knobs.begin(), knobs.end(), [](const Knob& k) { return k.knobId() == "test.int_knob"; });
    ASSERT_NE(it, knobs.end());

    auto* intConstraint = dynamic_cast<const IntConstraint*>(it->constraint());
    ASSERT_NE(intConstraint, nullptr);
    EXPECT_EQ(intConstraint->getMinValue(), 0)
        << "Zero min should be preserved"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_EQ(intConstraint->getMaxValue(), 100);
    EXPECT_EQ(intConstraint->getStep(), 10);
}

TEST_F(IntegrationGraphKnobsDescriptorLifting, FloatConstraintWithZeroMin)
{
    const auto engineId = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();

    auto graph = createAndBuildSimpleGraph();

    std::vector<Knob> knobs;
    auto result = graph.get_knobs_for_engine_via_descriptors(engineId, knobs);
    ASSERT_TRUE(result.is_good()) << result.get_message();

    // test.float_knob has min=0.0, max=1.0
    auto it = std::find_if(
        knobs.begin(), knobs.end(), [](const Knob& k) { return k.knobId() == "test.float_knob"; });
    ASSERT_NE(it, knobs.end());

    auto* floatConstraint = dynamic_cast<const FloatConstraint*>(it->constraint());
    ASSERT_NE(floatConstraint, nullptr);
    EXPECT_DOUBLE_EQ(floatConstraint->getMinValue(), 0.0)
        << "Zero min should be preserved"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_DOUBLE_EQ(floatConstraint->getMaxValue(), 1.0);
}

} // namespace
