// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <deque>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineDetailsWrapper.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "engines/MiopenEngine.hpp"
#include "mocks/MockHipdnnMiopenContext.hpp"
#include "mocks/MockPlanBuilder.hpp"
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>

using namespace miopen_plugin;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

TEST(TestMiopenEngine, ConstructorAndId)
{
    SKIP_IF_NO_DEVICES();

    const MiopenEngine engine(42);
    EXPECT_EQ(engine.id(), 42);
}

TEST(TestMiopenEngine, WorkspaceSizeReturnsZeroIfNoPlanBuilders)
{
    SKIP_IF_NO_DEVICES();

    const MiopenEngine engine(1);

    const HipdnnMiopenHandle dummyHandle;
    const MockGraph mockGraph;
    const MockEngineConfig mockConfig;
    EXPECT_CALL(mockConfig, isValid()).WillOnce(::testing::Return(false));

    EXPECT_EQ(engine.getMaxWorkspaceSize(dummyHandle, mockGraph, mockConfig), 0u);
}

TEST(TestMiopenEngine, WorkspaceSizeReturnsPlanBuilderWorkspace)
{
    SKIP_IF_NO_DEVICES();

    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder,
                initializeExecutionSettings(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1);
    EXPECT_CALL(*mockPlanBuilder, getMaxWorkspaceSize(::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(1337u));

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    const HipdnnMiopenHandle dummyHandle;
    const MockGraph mockGraph;
    const MockEngineConfig mockConfig;
    EXPECT_CALL(mockConfig, isValid()).WillOnce(::testing::Return(false));

    EXPECT_EQ(engine.getMaxWorkspaceSize(dummyHandle, mockGraph, mockConfig), 1337u);
}

TEST(TestMiopenEngine, WorkspaceSizeReturnsMaxPlanBuilderWorkspace)
{
    SKIP_IF_NO_DEVICES();

    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder,
                initializeExecutionSettings(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1);
    EXPECT_CALL(*mockPlanBuilder, getMaxWorkspaceSize(::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(1337u));

    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder2,
                initializeExecutionSettings(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1);
    EXPECT_CALL(*mockPlanBuilder2, getMaxWorkspaceSize(::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(45000u));

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    const HipdnnMiopenHandle dummyHandle;
    const MockGraph mockGraph;
    const MockEngineConfig mockConfig;
    EXPECT_CALL(mockConfig, isValid()).WillRepeatedly(::testing::Return(false));

    EXPECT_EQ(engine.getMaxWorkspaceSize(dummyHandle, mockGraph, mockConfig), 45000u);
}

TEST(TestMiopenEngine, WorkspaceSizeReturnsZeroIfNoPlanBuilderApplicable)
{
    SKIP_IF_NO_DEVICES();

    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    const HipdnnMiopenHandle dummyHandle;
    const MockGraph mockGraph;
    const MockEngineConfig mockConfig;
    EXPECT_CALL(mockConfig, isValid()).WillOnce(::testing::Return(false));

    EXPECT_EQ(engine.getMaxWorkspaceSize(dummyHandle, mockGraph, mockConfig), 0u);
}

TEST(TestMiopenEngine, IsApplicableReturnsTrueIfAnyPlanBuilderApplicable)
{
    SKIP_IF_NO_DEVICES();

    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));

    MiopenEngine engine(0);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    const MockGraph mockGraph;
    auto graphBuilder = hipdnn_test_sdk::utilities::createEmptyValidGraph();

    HipdnnMiopenHandle dummyHandle;
    EXPECT_TRUE(engine.isApplicable(dummyHandle, mockGraph));
}

TEST(TestMiopenEngine, IsApplicableReturnsAfterTheFirstApplicablePlanBuilder)
{
    SKIP_IF_NO_DEVICES();

    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_)).Times(0);

    MiopenEngine engine(0);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    const MockGraph mockGraph;
    auto graphBuilder = hipdnn_test_sdk::utilities::createEmptyValidGraph();

    HipdnnMiopenHandle dummyHandle;
    EXPECT_TRUE(engine.isApplicable(dummyHandle, mockGraph));
}

TEST(TestMiopenEngine, IsApplicableReturnsFalseIfNoPlanBuilders)
{
    SKIP_IF_NO_DEVICES();

    const MiopenEngine engine(0);

    const MockGraph mockGraph;
    auto graphBuilder = hipdnn_test_sdk::utilities::createEmptyValidGraph();

    HipdnnMiopenHandle dummyHandle;
    EXPECT_FALSE(engine.isApplicable(dummyHandle, mockGraph));
}

TEST(TestMiopenEngine, IsApplicableReturnsFalseIfNoPlanBuilderApplicable)
{
    SKIP_IF_NO_DEVICES();

    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));

    MiopenEngine engine(0);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    const MockGraph mockGraph;
    auto graphBuilder = hipdnn_test_sdk::utilities::createEmptyValidGraph();

    HipdnnMiopenHandle dummyHandle;
    EXPECT_FALSE(engine.isApplicable(dummyHandle, mockGraph));
}

TEST(TestMiopenEngine, GetDetailsReturnsSerializedEngineDetails)
{
    SKIP_IF_NO_DEVICES();

    const MiopenEngine engine(1);
    HipdnnMiopenHandle dummyHandle;
    const MockGraph mockGraph;

    hipdnnPluginConstData_t result;
    engine.getDetails(dummyHandle, mockGraph, result);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineDetailsWrapper engineDetails(
        result.ptr, result.size);
    EXPECT_EQ(engineDetails.engineId(), 1);
}

TEST(TestMiopenEngine, GetDetailsContainsBenchmarkingKnob)
{
    SKIP_IF_NO_DEVICES();

    const MiopenEngine engine(1);
    HipdnnMiopenHandle dummyHandle;
    const MockGraph mockGraph;

    hipdnnPluginConstData_t result;
    engine.getDetails(dummyHandle, mockGraph, result);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineDetailsWrapper engineDetails(
        result.ptr, result.size);
    ASSERT_EQ(engineDetails.knobCount(), 1u);

    const auto& knob = engineDetails.getKnobByName("global.benchmarking");
    EXPECT_EQ(knob.knobId(), "global.benchmarking");
    EXPECT_EQ(knob.description(), "Enable benchmarking");

    ASSERT_TRUE(knob.hasDefaultValue());
    EXPECT_EQ(knob.defaultValueType(), hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
    const auto& defaultValue
        = knob.defaultValueAs<hipdnn_flatbuffers_sdk::data_objects::IntValue>();
    EXPECT_EQ(defaultValue.value(), 0);

    ASSERT_TRUE(knob.hasConstraint());
    EXPECT_EQ(knob.constraintType(),
              hipdnn_flatbuffers_sdk::data_objects::KnobConstraint::IntConstraint);
    const auto& constraint
        = knob.constraintAs<hipdnn_flatbuffers_sdk::data_objects::IntConstraint>();
    EXPECT_EQ(constraint.min_value(), 0);
    EXPECT_EQ(constraint.max_value(), 1);
    EXPECT_EQ(constraint.step(), 1);
}

TEST(TestMiopenEngine, GetDetailsOnlyUsesFirstPlanBuilderCustomKnobs)
{
    SKIP_IF_NO_DEVICES();

    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    // Set up first plan builder to return a custom knob
    hipdnn_flatbuffers_sdk::data_objects::KnobT knob1;
    knob1.knob_id = "custom.knob1";
    knob1.description = "First custom knob";
    hipdnn_flatbuffers_sdk::data_objects::IntValueT defaultValue1;
    defaultValue1.value = 1;
    knob1.default_value.Set(defaultValue1);

    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> customKnobs1;
    customKnobs1.push_back(knob1);

    EXPECT_CALL(*mockPlanBuilder1, getCustomKnobs(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(customKnobs1));

    // Set up second plan builder to also return a custom knob (this should be ignored)
    hipdnn_flatbuffers_sdk::data_objects::KnobT knob2;
    knob2.knob_id = "custom.knob2";
    knob2.description = "Second custom knob";
    hipdnn_flatbuffers_sdk::data_objects::IntValueT defaultValue2;
    defaultValue2.value = 2;
    knob2.default_value.Set(defaultValue2);

    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> customKnobs2;
    customKnobs2.push_back(knob2);

    // This should NOT be called because we break after first non-empty custom knobs
    EXPECT_CALL(*mockPlanBuilder2, getCustomKnobs(::testing::_, ::testing::_)).Times(0);

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    HipdnnMiopenHandle dummyHandle;
    const MockGraph mockGraph;

    hipdnnPluginConstData_t result;
    engine.getDetails(dummyHandle, mockGraph, result);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineDetailsWrapper engineDetails(
        result.ptr, result.size);

    // Should have 2 knobs: benchmarking (always present) + custom.knob1 (from first builder)
    ASSERT_EQ(engineDetails.knobCount(), 2u);

    // Verify benchmarking knob is present
    const auto& benchmarkingKnob = engineDetails.getKnobByName("global.benchmarking");
    EXPECT_EQ(benchmarkingKnob.knobId(), "global.benchmarking");

    // Verify first custom knob is present
    const auto& customKnob1 = engineDetails.getKnobByName("custom.knob1");
    EXPECT_EQ(customKnob1.knobId(), "custom.knob1");
    EXPECT_EQ(customKnob1.description(), "First custom knob");

    // Verify second custom knob is NOT present (would throw if we tried to access it)
    EXPECT_THROW(engineDetails.getKnobByName("custom.knob2"), std::out_of_range);
}

TEST(TestMiopenEngine, InitializeExecutionContextInvokesFirstApplicablePlanBuilder)
{
    SKIP_IF_NO_DEVICES();

    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    // Only the first plan builder is applicable
    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_, ::testing::_))
        .Times(2)
        .WillRepeatedly(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder1,
                initializeExecutionSettings(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1);
    EXPECT_CALL(*mockPlanBuilder1,
                buildPlan(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1);
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(*mockPlanBuilder2,
                buildPlan(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    const MockGraph mockGraph;
    const HipdnnMiopenHandle dummyHandle;
    MockHipdnnMiopenContext ctx;
    const MockEngineConfig mockConfig;
    EXPECT_CALL(mockConfig, isValid()).WillRepeatedly(::testing::Return(false));

    engine.initializeExecutionContext(dummyHandle, mockGraph, mockConfig, ctx);
}

TEST(TestMiopenEngine, InitializeExecutionContextThrowsOnInvalidBenchmarkingKnobType)
{
    SKIP_IF_NO_DEVICES();

    const MiopenEngine engine(1);
    const MockGraph mockGraph;
    const HipdnnMiopenHandle dummyHandle;
    MockHipdnnMiopenContext ctx;

    flatbuffers::FlatBufferBuilder builder;
    auto knobIdOffset = builder.CreateString("global.benchmarking");
    auto stringValueOffset = builder.CreateString("invalid_value");
    auto knobValue
        = hipdnn_flatbuffers_sdk::data_objects::CreateStringValue(builder, stringValueOffset);
    hipdnn_flatbuffers_sdk::data_objects::KnobSettingBuilder knobSettingBuilder(builder);
    knobSettingBuilder.add_knob_id(knobIdOffset);
    knobSettingBuilder.add_value_type(hipdnn_flatbuffers_sdk::data_objects::KnobValue::StringValue);
    knobSettingBuilder.add_value(knobValue.Union());
    auto knobSetting = knobSettingBuilder.Finish();

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>> knobsVector;
    knobsVector.push_back(knobSetting);
    auto knobs = builder.CreateVector(knobsVector);

    auto engineConfig = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(builder, 1, knobs);
    builder.Finish(engineConfig);

    auto buffer = builder.Release();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper configWrapper(
        buffer.data(), buffer.size());

    EXPECT_THROW(engine.initializeExecutionContext(dummyHandle, mockGraph, configWrapper, ctx),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

/// Everything benchmarking-related, sharing one engine/graph/context and one
/// EngineConfig builder.
///
/// The environment variable is cleared for every case by default: each asserts what the
/// knob alone decides, so a runner carrying a stray HIPDNN_FORCE_BENCHMARKING must not
/// be able to flip the result. Cases that are about the override call
/// forceBenchmarking() to set it explicitly.
class TestMiopenEngineBenchmarking : public ::testing::Test
{
protected:
    /// Gates every case on a device. HipdnnMiopenHandle's constructor calls
    /// miopenCreate() and throws without one, so the handle cannot be a plain member:
    /// gtest builds members before SetUp() runs, and the throw would escape as a
    /// failure on a device-less runner instead of a skip.
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _handle = std::make_unique<HipdnnMiopenHandle>();
    }

    /// Valid for the whole test body; SetUp() skipped the case otherwise.
    HipdnnMiopenHandle& handle()
    {
        return *_handle;
    }

    MiopenEngine _engine{1};
    MockGraph _graph;
    MockHipdnnMiopenContext _context;

    /// Replaces the cleared default with an explicit HIPDNN_FORCE_BENCHMARKING value
    /// for the rest of the case.
    void forceBenchmarking(const std::string& value)
    {
        _guard.emplace(hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME, value);
    }

    /// An EngineConfig carrying no knobs at all.
    const IEngineConfig& configWithNoKnobs()
    {
        auto& builder = newBuilder();
        builder.Finish(hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(builder, 1, 0));
        return storeConfig(builder);
    }

    /// An EngineConfig carrying "global.benchmarking" set to @p value.
    const IEngineConfig& configWithBenchmarkingKnob(int64_t value)
    {
        auto& builder = newBuilder();
        auto knobIdOffset = builder.CreateString(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME);
        auto knobValue = hipdnn_flatbuffers_sdk::data_objects::CreateIntValue(builder, value);
        hipdnn_flatbuffers_sdk::data_objects::KnobSettingBuilder knobSettingBuilder(builder);
        knobSettingBuilder.add_knob_id(knobIdOffset);
        knobSettingBuilder.add_value_type(
            hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
        knobSettingBuilder.add_value(knobValue.Union());
        auto knobSetting = knobSettingBuilder.Finish();

        const std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>>
            knobsVector{knobSetting};
        auto knobs = builder.CreateVector(knobsVector);

        builder.Finish(hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(builder, 1, knobs));
        return storeConfig(builder);
    }

    bool initializeAndReadBenchmarkingEnabled(const IEngineConfig& engineConfig)
    {
        _engine.initializeExecutionContext(handle(), _graph, engineConfig, _context);
        return _context.executionSettings().benchmarkingEnabled();
    }

private:
    /// A fresh builder per config. One builder cannot be Finish()ed twice (flatbuffers
    /// asserts, and that assert compiles out under NDEBUG), and continuing to build
    /// moves GetBufferPointer(), which would strand an already-returned wrapper.
    flatbuffers::FlatBufferBuilder& newBuilder()
    {
        return _builders.emplace_back();
    }

    /// Wraps the builder's own buffer rather than a released one: a DetachedBuffer local
    /// to a helper would be freed before the wrapper is read. Both deques only ever grow,
    /// so every reference handed out stays valid for the fixture's life.
    const IEngineConfig& storeConfig(const flatbuffers::FlatBufferBuilder& builder)
    {
        return _configs.emplace_back(builder.GetBufferPointer(), builder.GetSize());
    }

    std::unique_ptr<HipdnnMiopenHandle> _handle;
    std::deque<flatbuffers::FlatBufferBuilder> _builders;
    std::deque<EngineConfigWrapper> _configs;
    std::optional<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter> _guard{
        std::in_place, hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME};
};

// The knob alone, with the override cleared.

TEST_F(TestMiopenEngineBenchmarking, InitializeExecutionContextSetsBenchmarkingEnabled)
{
    EXPECT_TRUE(initializeAndReadBenchmarkingEnabled(configWithBenchmarkingKnob(1)));
}

TEST_F(TestMiopenEngineBenchmarking, InitializeExecutionContextSetsBenchmarkingDisabled)
{
    EXPECT_FALSE(initializeAndReadBenchmarkingEnabled(configWithBenchmarkingKnob(0)));
}

TEST_F(TestMiopenEngineBenchmarking,
       InitializeExecutionContextDefaultsBenchmarkingDisabledWhenConfigInvalid)
{
    const MockEngineConfig invalidConfig;
    EXPECT_CALL(invalidConfig, isValid()).WillRepeatedly(::testing::Return(false));

    EXPECT_FALSE(initializeAndReadBenchmarkingEnabled(invalidConfig));
}

TEST_F(TestMiopenEngineBenchmarking,
       InitializeExecutionContextDefaultsBenchmarkingDisabledWhenNoKnobs)
{
    EXPECT_FALSE(initializeAndReadBenchmarkingEnabled(configWithNoKnobs()));
}

// HIPDNN_FORCE_BENCHMARKING is honoured outside the isValid() branch, so it also
// reaches the plain-execute path (no knob, or an invalid config).

TEST_F(TestMiopenEngineBenchmarking, ForceBenchmarkingOnSetsBenchmarkingEnabledWithNoKnob)
{
    forceBenchmarking("1");

    EXPECT_TRUE(initializeAndReadBenchmarkingEnabled(configWithNoKnobs()));
}

/// An invalid config is the plain-execute path, which the override must still reach.
TEST_F(TestMiopenEngineBenchmarking, ForceBenchmarkingOnSetsBenchmarkingEnabledWithAnInvalidConfig)
{
    forceBenchmarking("true");

    const MockEngineConfig invalidConfig;
    EXPECT_CALL(invalidConfig, isValid()).WillRepeatedly(::testing::Return(false));

    EXPECT_TRUE(initializeAndReadBenchmarkingEnabled(invalidConfig));
}

/// `0` forces off even when the knob asked for on, which an `||` composition could not
/// express.
TEST_F(TestMiopenEngineBenchmarking, ForceBenchmarkingOffOverridesAKnobEnabledRun)
{
    forceBenchmarking("0");

    EXPECT_FALSE(initializeAndReadBenchmarkingEnabled(configWithBenchmarkingKnob(1)));
}

TEST(TestMiopenEngine, InitializeExecutionContextSkipsNonApplicableBuilders)
{
    SKIP_IF_NO_DEVICES();

    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    // First plan builder not applicable, second is
    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_, ::testing::_))
        .Times(2)
        .WillRepeatedly(::testing::Return(false));
    EXPECT_CALL(*mockPlanBuilder1,
                buildPlan(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_))
        .Times(2)
        .WillRepeatedly(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder2,
                initializeExecutionSettings(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1);
    EXPECT_CALL(*mockPlanBuilder2,
                buildPlan(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1);

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    const MockGraph mockGraph;
    const HipdnnMiopenHandle dummyHandle;
    MockHipdnnMiopenContext ctx;
    const MockEngineConfig mockConfig;
    EXPECT_CALL(mockConfig, isValid()).WillRepeatedly(::testing::Return(false));

    engine.initializeExecutionContext(dummyHandle, mockGraph, mockConfig, ctx);
}

TEST(TestMiopenEngine, InitializeExecutionContextDoesNotCallBuildPlanIfNoApplicableBuilders)
{
    SKIP_IF_NO_DEVICES();

    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_, ::testing::_))
        .Times(2)
        .WillRepeatedly(::testing::Return(false));
    EXPECT_CALL(*mockPlanBuilder1,
                buildPlan(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_))
        .Times(2)
        .WillRepeatedly(::testing::Return(false));
    EXPECT_CALL(*mockPlanBuilder2,
                buildPlan(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    const MockGraph mockGraph;
    const HipdnnMiopenHandle dummyHandle;
    MockHipdnnMiopenContext ctx;
    const MockEngineConfig mockConfig;
    EXPECT_CALL(mockConfig, isValid()).WillRepeatedly(::testing::Return(false));

    engine.initializeExecutionContext(dummyHandle, mockGraph, mockConfig, ctx);
}
