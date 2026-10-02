// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/RankingMetrics.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineDetailsWrapper.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>

#include "MiopenContainer.hpp"
#include "engines/MiopenEngine.hpp"
#include "mocks/MockHipdnnMiopenContext.hpp"
#include "mocks/MockPlanBuilder.hpp"
#include "version.h"
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include <hipdnn_plugin_sdk/heuristics/uhd/FeatureExtractor.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/GbdtModelTestBuilder.hpp>
#include <nlohmann/json.hpp>
#endif

using namespace miopen_plugin;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

TEST(TestMiopenEngine, ConstructorAndId)
{
    SKIP_IF_NO_DEVICES();

    const MiopenEngine engine(42, "test:miopen", {});
    EXPECT_EQ(engine.id(), 42);
}

TEST(TestMiopenEngine, WorkspaceSizeReturnsZeroIfNoPlanBuilders)
{
    SKIP_IF_NO_DEVICES();

    const MiopenEngine engine(1, "test:miopen", {});

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

    MiopenEngine engine(1, "test:miopen", {});
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

    MiopenEngine engine(1, "test:miopen", {});
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

    MiopenEngine engine(1, "test:miopen", {});
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

    MiopenEngine engine(0, "test:miopen", {});
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

    MiopenEngine engine(0, "test:miopen", {});
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

    const MiopenEngine engine(0, "test:miopen", {});

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

    MiopenEngine engine(0, "test:miopen", {});
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    const MockGraph mockGraph;
    auto graphBuilder = hipdnn_test_sdk::utilities::createEmptyValidGraph();

    HipdnnMiopenHandle dummyHandle;
    EXPECT_FALSE(engine.isApplicable(dummyHandle, mockGraph));
}

TEST(TestMiopenEngine, GetDetailsReturnsSerializedEngineDetails)
{
    SKIP_IF_NO_DEVICES();

    const MiopenEngine engine(1, "test:miopen", {});
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

    const MiopenEngine engine(1, "test:miopen", {});
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

    MiopenEngine engine(1, "test:miopen", {});
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

    MiopenEngine engine(1, "test:miopen", {});
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

    const MiopenEngine engine(1, "test:miopen", {});
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

    MiopenEngine _engine{1, "test:miopen", {}};
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

    MiopenEngine engine(1, "test:miopen", {});
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

    MiopenEngine engine(1, "test:miopen", {});
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    const MockGraph mockGraph;
    const HipdnnMiopenHandle dummyHandle;
    MockHipdnnMiopenContext ctx;
    const MockEngineConfig mockConfig;
    EXPECT_CALL(mockConfig, isValid()).WillRepeatedly(::testing::Return(false));

    engine.initializeExecutionContext(dummyHandle, mockGraph, mockConfig, ctx);
}

TEST(TestMiopenEngine, ReportsNoEstimateWhenItsDeclaredL1ModelIsNotDeployed)
{
    SKIP_IF_NO_DEVICES();

    const MiopenEngine engine(
        1, "test:miopen", {{"tflops", "0f4d2c8b-6a19-4e73-9d05-8b1746ca3e2f"}});

    auto builder = createValidBatchnormInferenceGraph();
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    const EngineConfigWrapper config(nullptr, 0);
    HipdnnMiopenHandle handle;

    hipdnn_flatbuffers_sdk::data_objects::EnginePredictionT prediction;
    ASSERT_NO_THROW(prediction = engine.getPrediction(
                        handle, graph, config, HIPDNN_ENGINE_PREDICTION_ENGINE, true));
    EXPECT_EQ(prediction.status,
              hipdnn_flatbuffers_sdk::data_objects::PredictionStatus::UNAVAILABLE);
    EXPECT_EQ(prediction.engine_id, 1);
    EXPECT_EQ(prediction.kind, hipdnn_flatbuffers_sdk::data_objects::PredictionKind::ENGINE);
    // A configuration naming no metric asks in the default one, and the answer says so.
    EXPECT_EQ(prediction.metric, "tflops");
}

/// The decline must carry the requested metric: the backend rejects any answer, a decline
/// included, whose metric differs from the request.
TEST(TestMiopenEngine, DeclinesTheConfigurationPredictionQuery)
{
    SKIP_IF_NO_DEVICES();

    const MiopenEngine engine(1, "test:miopen", {});

    auto builder = createValidBatchnormInferenceGraph();
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    flatbuffers::FlatBufferBuilder configBuilder;
    configBuilder.Finish(hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfigDirect(
        configBuilder, 1, nullptr, "time"));
    const EngineConfigWrapper config(configBuilder.GetBufferPointer(), configBuilder.GetSize());
    HipdnnMiopenHandle handle;

    const auto prediction
        = engine.getPrediction(handle, graph, config, HIPDNN_ENGINE_PREDICTION_CONFIGURATION, true);
    EXPECT_EQ(prediction.status,
              hipdnn_flatbuffers_sdk::data_objects::PredictionStatus::UNAVAILABLE);
    EXPECT_EQ(prediction.kind, hipdnn_flatbuffers_sdk::data_objects::PredictionKind::CONFIGURATION);
    EXPECT_EQ(prediction.metric, "time");
}

/// One id shared by two declarations would silently make one answer with the other's model.
TEST(TestMiopenEngine, EachEngineDeclaresADistinctWellFormedModelIdPerMetric)
{
    std::set<std::string> seen;
    for(const auto* declared : {&MIOPEN_ENGINE_L1_MODELS, &MIOPEN_ENGINE_DETERMINISTIC_L1_MODELS})
    {
        std::set<std::string> metrics;
        for(const auto& [metric, id] : *declared)
        {
            // Neither a bad metric nor a malformed UUID literal would fail the build.
            EXPECT_NE(hipdnn_data_sdk::utilities::findRankingMetric(metric), nullptr) << metric;
            EXPECT_NO_THROW(static_cast<void>(hipdnn_flatbuffers_sdk::utilities::parseUuid(id)))
                << id;
            EXPECT_TRUE(seen.insert(id).second) << "id declared twice: " << id;
            metrics.insert(metric);
        }
        EXPECT_EQ(metrics, (std::set<std::string>{"tflops", "time"}));
    }
}

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
namespace
{

using hipdnn_flatbuffers_sdk::data_objects::EnginePredictionT;
using hipdnn_flatbuffers_sdk::data_objects::PredictionStatus;

/// The ENGINE prediction for a small conv forward graph, asked in @p metric.
EnginePredictionT predictConv(const MiopenEngine& engine,
                              HipdnnMiopenHandle& handle,
                              const std::string& metric,
                              bool evaluate)
{
    auto graphBuilder = createValidConvFwdGraph();
    const GraphWrapper graph(graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    flatbuffers::FlatBufferBuilder configBuilder;
    configBuilder.Finish(hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfigDirect(
        configBuilder, 1, nullptr, metric.c_str()));
    const EngineConfigWrapper config(configBuilder.GetBufferPointer(), configBuilder.GetSize());
    return engine.getPrediction(handle, graph, config, HIPDNN_ENGINE_PREDICTION_ENGINE, evaluate);
}

nlohmann::json describedBinding(const MiopenEngine& engine,
                                HipdnnMiopenHandle& handle,
                                const std::string& metric = "tflops")
{
    return nlohmann::json::parse(predictConv(engine, handle, metric, false).binding_json);
}

std::string describedRevision(const MiopenEngine& engine, HipdnnMiopenHandle& handle)
{
    return describedBinding(engine, handle).at("selector_revision").get<std::string>();
}

std::vector<std::string> revisionSegments(const std::string& revision)
{
    std::vector<std::string> parts;
    std::istringstream stream(revision);
    for(std::string part; std::getline(stream, part, '/');)
    {
        parts.push_back(part);
    }
    return parts;
}

/// @p revision with segment @p index replaced.
std::string withSegment(const std::string& revision, size_t index, const std::string& value)
{
    auto parts = revisionSegments(revision);
    parts.at(index) = value;
    std::string joined;
    for(const auto& part : parts)
    {
        joined += (joined.empty() ? "" : "/") + part;
    }
    return joined;
}

bool isSemver(const std::string& text)
{
    return std::regex_match(text, std::regex(R"(\d+\.\d+\.\d+)"));
}

} // namespace

/// The selector revision is a model's compatibility key, so it may carry only deliberately
/// versioned inputs; a commit in it would invalidate a model at the next commit.
TEST(TestMiopenEngine, SelectorRevisionIsTheVersionedPolicyAndNeverTheCommit)
{
    SKIP_IF_NO_DEVICES();

    HipdnnMiopenHandle handle;
    const auto binding = describedBinding(MiopenEngine(1, "test:miopen", {}), handle);
    const auto revision = binding.at("selector_revision").get<std::string>();
    // Collection records trained_against from the same string the loader compares.
    EXPECT_EQ(binding.at("trained_against").at("selector_revision").get<std::string>(), revision);

    // `miopen-provider/<semver>/<engine>-<policy>/miopen-<x.y.z>`.
    const auto parts = revisionSegments(revision);
    ASSERT_EQ(parts.size(), 4u) << revision;
    EXPECT_EQ(parts[0], "miopen-provider");
    EXPECT_TRUE(isSemver(parts[1])) << revision;
    EXPECT_EQ(parts[2].rfind("test:miopen-", 0), 0u) << revision;
    EXPECT_GT(parts[2].size(), std::string("test:miopen-").size()) << revision;
    ASSERT_EQ(parts[3].rfind("miopen-", 0), 0u) << revision;
    EXPECT_TRUE(isSemver(parts[3].substr(std::string("miopen-").size()))) << revision;
    if(const std::string commit = MIOPEN_PROVIDER_VERSION_TWEAK; commit != "unknown")
    {
        EXPECT_EQ(revision.find(commit), std::string::npos) << revision;
    }

    // Stable across instances; only the engine segment separates two engines.
    EXPECT_EQ(describedRevision(MiopenEngine(2, "test:miopen", {}), handle), revision);
    const auto other = describedRevision(MiopenEngine(1, "test:other", {}), handle);
    EXPECT_NE(other, revision);
    EXPECT_EQ(withSegment(other, 2, parts[2]), revision);
}

/// The description names the requested metric's declared id before anything is deployed.
TEST(TestMiopenEngine, DescriptionNamesTheModelIdDeclaredForTheRequestedMetric)
{
    SKIP_IF_NO_DEVICES();

    const std::string tflopsId = "aceee89e-ae84-4e79-abfd-4cfed80eaf87";
    const std::string timeId = "d318a8c5-dc1c-4a5e-8e4a-13f820ddc2fc";
    HipdnnMiopenHandle handle;
    const MiopenEngine engine(1, "test:miopen", {{"tflops", tflopsId}, {"time", timeId}});

    EXPECT_EQ(describedBinding(engine, handle, "tflops").at("uhd_id").get<std::string>(), tflopsId);
    EXPECT_EQ(describedBinding(engine, handle, "time").at("uhd_id").get<std::string>(), timeId);

    const MiopenEngine tflopsOnly(1, "test:miopen", {{"tflops", tflopsId}});
    EXPECT_FALSE(describedBinding(tflopsOnly, handle, "time").contains("uhd_id"));
}

namespace
{

/// A calibrated tree_data UHD over one feature every graph binds, answering 42 in @p metric.
nlohmann::json deployedL1Document(const std::string& id,
                                  const std::string& metric,
                                  const std::string& selectorRevision)
{
    const std::vector<nlohmann::json> signature = {"$graph.node_count"};
    return {{"version", "1.0"},
            {"id", id},
            {"name", "miopen " + metric},
            {"adapter", "tree_data"},
            {"tree_data", {{"artifact", "model.fb"}}},
            {"features_signature", signature},
            {"features_hash", hipdnn_plugin_sdk::uhd::FeatureExtractor::computeHash(signature)},
            {"objective",
             std::string(hipdnn_data_sdk::utilities::objectiveOf(
                 *hipdnn_data_sdk::utilities::findRankingMetric(metric)))},
            {"score", {{"metric", metric}, {"calibrated", true}, {"transform", "log1p"}}},
            {"trained_against", {{"selector_revision", selectorRevision}}}};
}

/// Deploys models for the declared ids under HIPDNN_DESCRIPTOR_DIR and queries the real
/// engines. Returns failed expectations, one per line. Must run in a process where no
/// engine has declared an id yet: descriptor roots are parsed once per process.
std::string deployedModelScenario()
{
    using namespace hipdnn_data_sdk::utilities;
    std::ostringstream failures;
    const auto expect = [&failures](bool ok, const std::string& what) {
        if(!ok)
        {
            failures << what << '\n';
        }
    };

    HipdnnMiopenHandle handle;
    // Engines declaring nothing do not parse the catalog yet.
    const auto engineBinding
        = describedBinding(MiopenEngine(MIOPEN_ENGINE_ID, MIOPEN_ENGINE_NAME, {}), handle);
    const auto engineRevision = engineBinding.at("selector_revision").get<std::string>();
    const auto arch = engineBinding.at("arch").get<std::string>();
    const auto deterministicRevision = describedRevision(
        MiopenEngine(MIOPEN_ENGINE_DETERMINISTIC_ID, MIOPEN_ENGINE_DETERMINISTIC_NAME, {}), handle);
    const auto testRevision = describedRevision(MiopenEngine(1, "test:miopen", {}), handle);

    const hipdnn_test_sdk::utilities::ScopedDirectory root(
        std::filesystem::temp_directory_path()
        / ("miopen_declared_l1_" + std::to_string(std::random_device{}())));
    if(!hipdnn_test_sdk::utilities::GbdtModelTestBuilder()
            .setNumFeatures(1)
            .setFeaturesHash(
                hipdnn_plugin_sdk::uhd::FeatureExtractor::computeHash({"$graph.node_count"}))
            .setTrainingArches({arch})
            .setBaseScore(std::log1p(42.0))
            .addTree(hipdnn_test_sdk::utilities::makeLeafTreeSpec(0.0))
            .buildToFile((root.path() / "model.fb").string()))
    {
        return "cannot write the model artifact";
    }
    const std::string misdeployedId = "faa2b9e6-21b6-4e5b-a80b-171b33c12917";
    const std::vector<nlohmann::json> documents = {
        // Recorded against exactly what the engine describes: must answer.
        deployedL1Document(MIOPEN_ENGINE_L1_MODELS.at("tflops"), "tflops", engineRevision),
        // Recorded against another MIOpen release: must be refused.
        deployedL1Document(MIOPEN_ENGINE_L1_MODELS.at("time"),
                           "time",
                           withSegment(engineRevision, 3, "miopen-999.0.0")),
        // Recorded under a different selector policy: must be refused.
        deployedL1Document(MIOPEN_ENGINE_DETERMINISTIC_L1_MODELS.at("tflops"),
                           "tflops",
                           withSegment(deterministicRevision,
                                       2,
                                       revisionSegments(deterministicRevision).at(2) + "-bumped")),
        // A tflops model published under an id declared for time.
        deployedL1Document(misdeployedId, "tflops", testRevision)};
    for(const auto& document : documents)
    {
        std::ofstream(root.path() / (document.at("id").get<std::string>() + ".uhd.json"))
            << document.dump(2);
    }

    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter descriptorDir(
        "HIPDNN_DESCRIPTOR_DIR", root.path().string());
    // Additive roots from the runner would add models this case did not write.
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter runtimeDir(
        "HIPDNN_DESCRIPTOR_RUNTIME_DIR");
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter searchPath(
        "HIPDNN_DESCRIPTOR_PATH");

    const MiopenEngine engine(MIOPEN_ENGINE_ID, MIOPEN_ENGINE_NAME, MIOPEN_ENGINE_L1_MODELS);
    const auto tflops = predictConv(engine, handle, "tflops", true);
    expect(tflops.status == PredictionStatus::AVAILABLE,
           "MIOPEN_ENGINE tflops is not AVAILABLE: " + tflops.reason);
    expect(std::abs(tflops.value - 42.0) < 1e-9,
           "MIOPEN_ENGINE tflops value " + std::to_string(tflops.value) + " != 42");
    expect(tflops.uhd_id == MIOPEN_ENGINE_L1_MODELS.at("tflops"),
           "MIOPEN_ENGINE tflops answered with model '" + tflops.uhd_id + "'");

    const auto otherRelease = predictConv(engine, handle, "time", true);
    expect(otherRelease.status == PredictionStatus::UNAVAILABLE
               && otherRelease.reason.find("trained against") != std::string::npos,
           "a model recorded against another MIOpen release was not refused: "
               + otherRelease.reason);

    const MiopenEngine deterministic(MIOPEN_ENGINE_DETERMINISTIC_ID,
                                     MIOPEN_ENGINE_DETERMINISTIC_NAME,
                                     MIOPEN_ENGINE_DETERMINISTIC_L1_MODELS);
    const auto bumped = predictConv(deterministic, handle, "tflops", true);
    expect(bumped.status == PredictionStatus::UNAVAILABLE
               && bumped.reason.find("trained against") != std::string::npos,
           "a model recorded under another selector policy was not refused: " + bumped.reason);

    const MiopenEngine misdeclared(1, "test:miopen", {{"time", misdeployedId}});
    const auto misdeployed = predictConv(misdeclared, handle, "tflops", true);
    expect(misdeployed.status == PredictionStatus::INVALID
               && misdeployed.reason.find("declared for metric 'time'") != std::string::npos,
           "a tflops model under the time id was not refused: " + misdeployed.reason);

    return failures.str();
}

} // namespace

/// A model matching the described revision answers; one recorded against another MIOpen
/// release or selector policy, or under another metric's id, is refused.
TEST(TestMiopenEngine, DeployedModelsForTheDeclaredIdsAnswerThroughTheEngine)
{
    SKIP_IF_NO_DEVICES();

    // Descriptor roots are parsed once per process, so this case needs a fresh one;
    // "threadsafe" re-executes the binary rather than forking an already-parsed process.
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    // EXPECT_EXIT's expansion has a switch with no default label; -isystem doesn't cover it.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wswitch-default"
    EXPECT_EXIT(
        {
            const auto failures = deployedModelScenario();
            std::cerr << failures;
            std::exit(failures.empty() ? 0 : 1);
        },
        testing::ExitedWithCode(0),
        "");
#pragma clang diagnostic pop
}
#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
