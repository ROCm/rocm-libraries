// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnBackendFlatbufferData.h"
#include "TestMacros.hpp"
#include "descriptors/EngineConfigDescriptor.hpp"
#include "descriptors/EngineDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/KnobSettingDescriptor.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockDescriptor.hpp"
#include "mocks/MockEnginePluginResourceManager.hpp"
#include "mocks/MockHandle.hpp"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_prediction_generated.h>

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

using namespace hipdnn_backend;
using namespace plugin;
using namespace hipdnn_backend::test_utilities;
using namespace ::testing;

using ::testing::Return;

class TestEngineConfigDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<EngineConfigDescriptor> getEngineConfigDescriptor() const
    {
        return _engineConfigWrapper->asDescriptor<EngineConfigDescriptor>();
    }

    std::shared_ptr<MockEngineDescriptor> getMockEngine() const
    {
        return MockDescriptorUtility::asDescriptorUnsafe<MockEngineDescriptor>(
            _mockEngineWrapper.get());
    }

    std::shared_ptr<MockEngineDescriptor> getMockEngineBadType() const
    {
        return MockDescriptorUtility::asDescriptorUnsafe<MockEngineDescriptor>(
            _mockEngineBadTypeWrapper.get());
    }

    std::shared_ptr<MockGraphDescriptor> getMockGraphDescriptor() const
    {
        return MockDescriptorUtility::asDescriptorUnsafe<MockGraphDescriptor>(
            _mockGraphWrapper.get());
    }

    void setEngine() const
    {
        EXPECT_CALL(*getMockEngine(), isFinalized()).WillOnce(Return(true));
        ASSERT_NO_THROW(getEngineConfigDescriptor()->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper));
    }

    void makeEngineConfigFinalized() const
    {
        // WARNING: Mock expectations set here apply to all tests that call this helper.
        // Setting expectations on the same mock methods *before* this call causes
        // undefined behavior (gmock uses the last matching expectation). Likewise,
        // expectations added *after* this call on the same methods may silently
        // shadow the ones below. Keep this in mind when writing new tests.
        EXPECT_CALL(*getMockEngine(), isFinalized()).WillRepeatedly(Return(true));
        EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));
        EXPECT_CALL(*getMockEngine(), getGraph()).WillRepeatedly(Return(getMockGraphDescriptor()));
        EXPECT_CALL(*getMockGraphDescriptor(), getHandle()).WillOnce(Return(_mockHandle.get()));
        EXPECT_CALL(*_mockHandle, getPluginResourceManager())
            .WillOnce(Return(_mockEnginePluginResourceManager));
        EXPECT_CALL(*_mockEnginePluginResourceManager, getWorkspaceSize(_, _, _))
            .WillOnce(Return(1024));

        setEngine();
        ASSERT_NO_THROW(getEngineConfigDescriptor()->finalize());
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _engineConfigWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockEngineWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockEngineBadTypeWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockWrongTypeWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockGraphWrapper = nullptr;
    std::unique_ptr<MockHandle> _mockHandle = nullptr;
    std::shared_ptr<MockEnginePluginResourceManager> _mockEnginePluginResourceManager = nullptr;

    void SetUp() override
    {
        _engineConfigWrapper = createDescriptor<EngineConfigDescriptor>();
        _mockEngineWrapper = createDescriptor<MockEngineDescriptor>();
        _mockEngineBadTypeWrapper = createDescriptor<MockEngineDescriptor>();
        _mockWrongTypeWrapper = createDescriptor<MockDescriptor<EngineConfigDescriptor>>();
        _mockGraphWrapper = createDescriptor<MockGraphDescriptor>();
        _mockHandle = std::make_unique<MockHandle>();
        _mockEnginePluginResourceManager = std::make_shared<MockEnginePluginResourceManager>();
    }
};

TEST_F(TestEngineConfigDescriptor, HeuristicResultDefersSelectionUntilWorkspaceIsRequested)
{
    EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));
    EXPECT_CALL(*getMockEngine(), getGraph()).WillRepeatedly(Return(getMockGraphDescriptor()));
    EXPECT_CALL(*getMockGraphDescriptor(), getHandle()).WillOnce(Return(_mockHandle.get()));
    EXPECT_CALL(*_mockHandle, getPluginResourceManager())
        .WillOnce(Return(_mockEnginePluginResourceManager));
    size_t selections = 0;
    EXPECT_CALL(*_mockEnginePluginResourceManager, getWorkspaceSize(_, _, _))
        .WillOnce([&](auto, auto, auto) {
            ++selections;
            return size_t{8192};
        });
    setEngine();
    hipdnn_flatbuffers_sdk::data_objects::EngineConfigT scored;
    scored.engine_id = 1;
    auto config = getEngineConfigDescriptor();
    config->setEngineConfig(scored, true);
    config->finalize();
    EXPECT_EQ(selections, 0u);
    EXPECT_EQ(config->getEngine()->getEngineId(), 1);
    EXPECT_EQ(selections, 0u);
    int64_t workspace = 0;
    config->getAttribute(
        HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, &workspace);
    EXPECT_EQ(workspace, 8192);
    EXPECT_EQ(selections, 1u);
    config->getAttribute(
        HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, &workspace);
    EXPECT_EQ(workspace, 8192);
    EXPECT_EQ(selections, 1u);
}

TEST_F(TestEngineConfigDescriptor, ScoredKnobsSurviveConfigSerializationRoundTrip)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));
    setEngine();
    EngineConfigT scored;
    scored.engine_id = 1;
    auto knob = std::make_unique<KnobSettingT>();
    knob->knob_id = "tile";
    IntValueT value;
    value.value = 128;
    knob->value.Set(value);
    scored.knobs.push_back(std::move(knob));
    getEngineConfigDescriptor()->setEngineConfig(scored);
    const auto bytes = getEngineConfigDescriptor()->getSerializedEngineConfig();
    auto restored = UnPackEngineConfig(bytes.ptr);

    // Deserialization must restore the exact scored knobs, not synthesize
    // an engine-only config that would run the selector again.
    auto second = std::make_shared<EngineConfigDescriptor>();
    EXPECT_CALL(*getMockEngine(), isFinalized()).WillOnce(Return(true));
    second->setAttribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper);
    second->setEngineConfig(*restored);
    scored.knobs.front()->value.AsIntValue()->value = 256;
    restored.reset();
    const auto roundTrip = second->getSerializedEngineConfig();
    const auto* result = GetEngineConfig(roundTrip.ptr);
    EXPECT_EQ(result->engine_id(), 1);
    ASSERT_NE(result->knobs(), nullptr);
    ASSERT_EQ(result->knobs()->size(), 1u);
    EXPECT_EQ(result->knobs()->Get(0)->knob_id()->str(), "tile");
    EXPECT_EQ(result->knobs()->Get(0)->value_as_IntValue()->value(), 128);
}

TEST_F(TestEngineConfigDescriptor, RejectsForeignOrAmbiguousScoredConfigurations)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));
    setEngine();
    EngineConfigT config;
    config.engine_id = 2;
    EXPECT_THROW(getEngineConfigDescriptor()->setEngineConfig(config), HipdnnException);
    config.engine_id = 1;
    config.knobs.push_back(nullptr);
    EXPECT_THROW(getEngineConfigDescriptor()->setEngineConfig(config), HipdnnException);
    config.knobs.clear();
    auto knob = std::make_unique<KnobSettingT>();
    knob->knob_id = "tile";
    IntValueT value;
    value.value = 128;
    knob->value.Set(value);
    config.knobs.push_back(std::make_unique<KnobSettingT>(*knob));
    config.knobs.push_back(std::move(knob));
    EXPECT_THROW(getEngineConfigDescriptor()->setEngineConfig(config), HipdnnException);
    // A heuristic plugin's config naming an unregistered metric cannot reach plan build.
    config.knobs.pop_back();
    config.ranking_metric = "flops";
    EXPECT_THROW(getEngineConfigDescriptor()->setEngineConfig(config), HipdnnException);
}

// RFC 0019 §11.4: refused when unregistered, readable before finalize, unset reads as default.
TEST_F(TestEngineConfigDescriptor, RankingMetricRoundTripsIntoTheSerializedConfig)
{
    EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));
    setEngine();
    auto config = getEngineConfigDescriptor();

    std::array<char, 16> readBack{};
    int64_t count = 0;
    config->getAttribute(HIPDNN_ATTR_ENGINECFG_RANKING_METRIC_EXT,
                         HIPDNN_TYPE_CHAR,
                         static_cast<int64_t>(readBack.size()),
                         &count,
                         readBack.data());
    EXPECT_STREQ(readBack.data(), "tflops");

    const std::string unregistered = "flops";
    ASSERT_THROW_HIPDNN_STATUS(config->setAttribute(HIPDNN_ATTR_ENGINECFG_RANKING_METRIC_EXT,
                                                    HIPDNN_TYPE_CHAR,
                                                    static_cast<int64_t>(unregistered.size()),
                                                    unregistered.data()),
                               HIPDNN_STATUS_BAD_PARAM);

    const std::string metric = "time";
    config->setAttribute(HIPDNN_ATTR_ENGINECFG_RANKING_METRIC_EXT,
                         HIPDNN_TYPE_CHAR,
                         static_cast<int64_t>(metric.size()),
                         metric.data());
    config->getAttribute(HIPDNN_ATTR_ENGINECFG_RANKING_METRIC_EXT,
                         HIPDNN_TYPE_CHAR,
                         static_cast<int64_t>(readBack.size()),
                         &count,
                         readBack.data());
    EXPECT_STREQ(readBack.data(), "time");
    const auto bytes = config->getSerializedEngineConfig();
    const auto* serialized = hipdnn_flatbuffers_sdk::data_objects::GetEngineConfig(bytes.ptr);
    ASSERT_NE(serialized->ranking_metric(), nullptr);
    EXPECT_EQ(serialized->ranking_metric()->string_view(), "time");
}

TEST_F(TestEngineConfigDescriptor, EngineConfigRejectsEngineCatalogInspectionAttributes)
{
    // Catalog enumeration belongs to the engine descriptor, never an engine config (RFC 0017 §3).
    auto config = getEngineConfigDescriptor();
    const int64_t value = 1;
    for(const auto attribute : {HIPDNN_ATTR_ENGINE_CANDIDATE_OFFSET_EXT,
                                HIPDNN_ATTR_ENGINE_CANDIDATE_LIMIT_EXT,
                                HIPDNN_ATTR_ENGINE_CANDIDATES_EXT})
    {
        ASSERT_THROW_HIPDNN_STATUS(config->setAttribute(attribute, HIPDNN_TYPE_INT64, 1, &value),
                                   HIPDNN_STATUS_NOT_SUPPORTED);
    }
}

TEST_F(TestEngineConfigDescriptor, PredictionEvaluateFlagRejectsValuesOutsideZeroOrOne)
{
    auto config = getEngineConfigDescriptor();
    const int64_t evaluate = 2;
    ASSERT_THROW_HIPDNN_STATUS(
        config->setAttribute(
            HIPDNN_ATTR_ENGINECFG_PREDICTION_EVALUATE_EXT, HIPDNN_TYPE_INT64, 1, &evaluate),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineConfigDescriptor, CreateEngineConfigDescriptor)
{
    auto engineConfig = getEngineConfigDescriptor();
    ASSERT_NE(engineConfig, nullptr);
    ASSERT_FALSE(engineConfig->isFinalized());
    ASSERT_EQ(engineConfig->getType(), HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
}

TEST_F(TestEngineConfigDescriptor, SetEngineConfigDescriptorEngine)
{
    auto engineConfig = getEngineConfigDescriptor();

    EXPECT_CALL(*getMockEngineBadType(), isFinalized()).Times(1);
    EXPECT_CALL(*getMockEngine(), getEngineId()).Times(AnyNumber());
    EXPECT_CALL(*getMockEngine(), isFinalized()).WillOnce(Return(false)).WillOnce(Return(true));

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper),
        HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    ASSERT_NO_THROW(engineConfig->setAttribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper));

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_INT64, 1, &_mockEngineWrapper),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &_mockEngineWrapper),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipdnnBackendDescriptor_t engine = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->setAttribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                          1,
                                                          &_mockEngineBadTypeWrapper),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->setAttribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                          1,
                                                          &_mockWrongTypeWrapper),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineConfigDescriptor, SetAttrOnFinalizedEngineConfigDescriptor)
{
    auto engineConfig = getEngineConfigDescriptor();
    makeEngineConfigFinalized();

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestEngineConfigDescriptor, FinalizeEngineConfigDescriptor)
{
    auto engineConfig = getEngineConfigDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(engineConfig->finalize(), HIPDNN_STATUS_BAD_PARAM);

    makeEngineConfigFinalized();
}

TEST_F(TestEngineConfigDescriptor, GetAttrOnUnfinalizedEngineConfigDescriptor)
{
    auto engineConfig = getEngineConfigDescriptor();
    hipdnnBackendDescriptor_t dummyEngine = nullptr;

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->getAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, &dummyEngine),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestEngineConfigDescriptor, GetEngineConfigDescriptorUnsupportedAttr)
{
    auto engineConfig = getEngineConfigDescriptor();
    hipdnnBackendDescriptor_t dummy = nullptr;

    makeEngineConfigFinalized();

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->getAttribute(HIPDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO,
                                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                          1,
                                                          nullptr,
                                                          &dummy),
                               HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestEngineConfigDescriptor, GetEngineConfigDescriptorEngine)
{
    auto engineConfig = getEngineConfigDescriptor();
    ScopedDescriptor engine;
    ScopedDescriptor engine2;
    makeEngineConfigFinalized();

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->getAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_INT64, 1, nullptr, engine.getPtr()),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->getAttribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                          2,
                                                          nullptr,
                                                          engine.getPtr()),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->getAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(engineConfig->getAttribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                               1,
                                               nullptr,
                                               static_cast<void*>(engine.getPtr())));
    ASSERT_EQ(*engine.get(), *(_mockEngineWrapper.get()));

    int64_t count;
    ASSERT_NO_THROW(engineConfig->getAttribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                               1,
                                               &count,
                                               static_cast<void*>(engine2.getPtr())));
    ASSERT_EQ(count, 1);
}

TEST_F(TestEngineConfigDescriptor, GetEngineThrowsIfNotFinalized)
{
    auto engineConfig = getEngineConfigDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(engineConfig->getEngine(), HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestEngineConfigDescriptor, GetEngineReturnsPointerIfFinalized)
{
    auto engineConfig = getEngineConfigDescriptor();
    makeEngineConfigFinalized();
    auto enginePtr = engineConfig->getEngine();
    ASSERT_NE(enginePtr, nullptr);
    ASSERT_EQ(static_cast<const IBackendDescriptor*>(enginePtr.get()),
              static_cast<const IBackendDescriptor*>(getMockEngine().get()));
}

TEST_F(TestEngineConfigDescriptor, GetEngineDescriptorMaxWorkspaceSize)
{
    auto engineConfig = getEngineConfigDescriptor();
    int64_t workspaceSize = 0;

    makeEngineConfigFinalized();

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->getAttribute(HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE,
                                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                          1,
                                                          nullptr,
                                                          &workspaceSize),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->getAttribute(
            HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 2, nullptr, &workspaceSize),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->getAttribute(
            HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(engineConfig->getAttribute(
        HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, &workspaceSize));
    ASSERT_EQ(workspaceSize, 1024);

    int64_t count;
    ASSERT_NO_THROW(engineConfig->getAttribute(
        HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, &count, &workspaceSize));
    ASSERT_EQ(count, 1);
}

// Helper function to create a serialized KnobSetting
static flatbuffers::DetachedBuffer createSerializedKnobSetting(const std::string& knobId,
                                                               int64_t value)
{
    flatbuffers::FlatBufferBuilder builder;
    auto knobIdOffset = builder.CreateString(knobId);
    auto intValue = hipdnn_flatbuffers_sdk::data_objects::CreateIntValue(builder, value);
    auto knobSetting = hipdnn_flatbuffers_sdk::data_objects::CreateKnobSetting(
        builder,
        knobIdOffset,
        hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue,
        intValue.Union());
    builder.Finish(knobSetting);
    return builder.Release();
}

TEST_F(TestEngineConfigDescriptor, PredictionConstraintsDoNotRequireEngineMaterialization)
{
    auto config = getEngineConfigDescriptor();
    auto knobBuffer = createSerializedKnobSetting("tile", 128);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};
    config->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                         HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                         1,
                         &knobData);
    const auto predictionConfig = config->getEngineConfigForPrediction(73, nullptr);
    EXPECT_EQ(predictionConfig.engine_id, 73);
    ASSERT_EQ(predictionConfig.knobs.size(), 1u);
    EXPECT_EQ(predictionConfig.knobs.front()->knob_id, "tile");
    ASSERT_NE(predictionConfig.knobs.front()->value.AsIntValue(), nullptr);
    EXPECT_EQ(predictionConfig.knobs.front()->value.AsIntValue()->value, 128);
    EXPECT_FALSE(config->isFinalized());
}

TEST_F(TestEngineConfigDescriptor, PredictionAttributeValidatesTheRequestBeforeAnyPluginQuery)
{
    auto config = getEngineConfigDescriptor();
    EXPECT_CALL(*_mockEnginePluginResourceManager, getEnginePrediction(_, _, _, _)).Times(0);
    hipdnnBackendFlatbufferData_t data{};
    int64_t count = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        config->getAttribute(
            HIPDNN_ATTR_ENGINECFG_PREDICTION_EXT, HIPDNN_TYPE_INT64, 1, &count, &data),
        HIPDNN_STATUS_BAD_PARAM);
    for(const int64_t requested : {int64_t{-1}, int64_t{2}})
    {
        ASSERT_THROW_HIPDNN_STATUS(config->getAttribute(HIPDNN_ATTR_ENGINECFG_PREDICTION_EXT,
                                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                                        requested,
                                                        &count,
                                                        &data),
                                   HIPDNN_STATUS_BAD_PARAM);
    }

    // A count query answers without building anything, so it needs no engine.
    ASSERT_NO_THROW(config->getAttribute(HIPDNN_ATTR_ENGINECFG_PREDICTION_EXT,
                                         HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                         0,
                                         &count,
                                         nullptr));
    EXPECT_EQ(count, 1);

    ASSERT_THROW_HIPDNN_STATUS(config->getAttribute(HIPDNN_ATTR_ENGINECFG_PREDICTION_EXT,
                                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                                    1,
                                                    &count,
                                                    nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    // A well-formed read still cannot name an engine to ask.
    ASSERT_THROW_HIPDNN_STATUS(config->getAttribute(HIPDNN_ATTR_ENGINECFG_PREDICTION_EXT,
                                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                                    1,
                                                    nullptr,
                                                    &data),
                               HIPDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(data.ptr, nullptr);
}

// The request carries every constraint set so far.
TEST_F(TestEngineConfigDescriptor, PredictionAsksForTheConfigurationLayerOnceAndCachesIt)
{
    namespace fb = hipdnn_flatbuffers_sdk::data_objects;
    EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));
    EXPECT_CALL(*getMockEngine(), getGraph()).WillRepeatedly(Return(getMockGraphDescriptor()));
    setEngine();
    auto config = getEngineConfigDescriptor();
    auto knobBuffer = createSerializedKnobSetting("tile", 128);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};
    config->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                         HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                         1,
                         &knobData);
    const std::string metric = "time";
    config->setAttribute(HIPDNN_ATTR_ENGINECFG_RANKING_METRIC_EXT,
                         HIPDNN_TYPE_CHAR,
                         static_cast<int64_t>(metric.size()),
                         metric.data());

    const std::array<uint8_t, 4> graphBytes{1, 2, 3, 4};
    EXPECT_CALL(*getMockGraphDescriptor(), getHandle()).WillOnce(Return(_mockHandle.get()));
    EXPECT_CALL(*_mockHandle, getPluginResourceManager())
        .WillOnce(Return(_mockEnginePluginResourceManager));
    EXPECT_CALL(*getMockGraphDescriptor(), getSerializedGraph())
        .WillOnce(Return(hipdnnPluginConstData_t{graphBytes.data(), graphBytes.size()}));
    EXPECT_CALL(*_mockEnginePluginResourceManager, getEnginePrediction(_, _, _, _))
        .WillOnce(Invoke([&graphBytes](const hipdnnPluginConstData_t& engineConfig,
                                       const hipdnnPluginConstData_t& opGraph,
                                       hipdnnEnginePredictionKind_t kind,
                                       bool evaluate) {
            EXPECT_EQ(kind, HIPDNN_ENGINE_PREDICTION_CONFIGURATION);
            EXPECT_TRUE(evaluate) << "Evaluation is the default";
            EXPECT_EQ(opGraph.ptr, graphBytes.data());
            const auto* request = fb::GetEngineConfig(engineConfig.ptr);
            EXPECT_EQ(request->engine_id(), 1);
            EXPECT_EQ(request->ranking_metric()->string_view(), "time");
            EXPECT_EQ(request->knobs()->size(), 1u);
            EXPECT_EQ(request->knobs()->Get(0)->knob_id()->str(), "tile");
            EXPECT_EQ(request->knobs()->Get(0)->value_as_IntValue()->value(), 128);
            fb::EnginePredictionT prediction;
            prediction.engine_id = 1;
            prediction.kind = fb::PredictionKind::CONFIGURATION;
            prediction.status = fb::PredictionStatus::AVAILABLE;
            prediction.value = 3.5;
            prediction.metric = "time";
            return prediction;
        }));

    hipdnnBackendFlatbufferData_t first{};
    int64_t count = 0;
    ASSERT_NO_THROW(config->getAttribute(HIPDNN_ATTR_ENGINECFG_PREDICTION_EXT,
                                         HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                         1,
                                         &count,
                                         &first));
    EXPECT_EQ(count, 1);
    ASSERT_NE(first.ptr, nullptr);
    const auto* published = fb::GetEnginePrediction(first.ptr);
    EXPECT_EQ(published->kind(), fb::PredictionKind::CONFIGURATION);
    EXPECT_EQ(published->status(), fb::PredictionStatus::AVAILABLE);
    EXPECT_DOUBLE_EQ(published->value(), 3.5);
    EXPECT_EQ(published->metric()->string_view(), "time");

    hipdnnBackendFlatbufferData_t second{};
    ASSERT_NO_THROW(config->getAttribute(HIPDNN_ATTR_ENGINECFG_PREDICTION_EXT,
                                         HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                         1,
                                         nullptr,
                                         &second));
    EXPECT_EQ(second.ptr, first.ptr);
    EXPECT_EQ(second.size, first.size);
    EXPECT_FALSE(config->isFinalized()) << "Reading a prediction must not finalize the config";
}

TEST_F(TestEngineConfigDescriptor, ChangingTheConfigRebuildsThePredictionAndKeepsEarlierBytesValid)
{
    namespace fb = hipdnn_flatbuffers_sdk::data_objects;
    EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));
    EXPECT_CALL(*getMockEngine(), getGraph()).WillRepeatedly(Return(getMockGraphDescriptor()));
    setEngine();
    auto config = getEngineConfigDescriptor();

    const std::array<uint8_t, 4> graphBytes{1, 2, 3, 4};
    EXPECT_CALL(*getMockGraphDescriptor(), getHandle()).WillRepeatedly(Return(_mockHandle.get()));
    EXPECT_CALL(*_mockHandle, getPluginResourceManager())
        .WillRepeatedly(Return(_mockEnginePluginResourceManager));
    EXPECT_CALL(*getMockGraphDescriptor(), getSerializedGraph())
        .WillRepeatedly(Return(hipdnnPluginConstData_t{graphBytes.data(), graphBytes.size()}));
    std::vector<bool> evaluateFlags;
    EXPECT_CALL(*_mockEnginePluginResourceManager, getEnginePrediction(_, _, _, _))
        .Times(2)
        .WillRepeatedly(Invoke([&evaluateFlags](const hipdnnPluginConstData_t&,
                                                const hipdnnPluginConstData_t&,
                                                hipdnnEnginePredictionKind_t,
                                                bool evaluate) {
            evaluateFlags.push_back(evaluate);
            fb::EnginePredictionT prediction;
            prediction.engine_id = 1;
            prediction.kind = fb::PredictionKind::CONFIGURATION;
            prediction.status = fb::PredictionStatus::AVAILABLE;
            prediction.value = static_cast<double>(evaluateFlags.size());
            prediction.metric = "tflops";
            return prediction;
        }));

    const auto read = [&config] {
        hipdnnBackendFlatbufferData_t data{};
        config->getAttribute(HIPDNN_ATTR_ENGINECFG_PREDICTION_EXT,
                             HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                             1,
                             nullptr,
                             &data);
        return data;
    };

    const auto before = read();
    ASSERT_NE(before.ptr, nullptr);
    const std::vector<uint8_t> beforeBytes(static_cast<const uint8_t*>(before.ptr),
                                           static_cast<const uint8_t*>(before.ptr) + before.size);

    const int64_t describeOnly = 0;
    config->setAttribute(
        HIPDNN_ATTR_ENGINECFG_PREDICTION_EVALUATE_EXT, HIPDNN_TYPE_INT64, 1, &describeOnly);
    const auto after = read();
    ASSERT_NE(after.ptr, nullptr);
    EXPECT_DOUBLE_EQ(fb::GetEnginePrediction(after.ptr)->value(), 2.0);
    EXPECT_EQ(evaluateFlags, (std::vector<bool>{true, false}))
        << "The evaluate flag must reach the plugin, and changing it must re-query";

    // The first answer is retired, not freed: its bytes are unchanged and still verify.
    ASSERT_EQ(before.size, beforeBytes.size());
    EXPECT_TRUE(std::equal(
        beforeBytes.begin(), beforeBytes.end(), static_cast<const uint8_t*>(before.ptr)));
    EXPECT_DOUBLE_EQ(fb::GetEnginePrediction(before.ptr)->value(), 1.0);

    // The rebuilt answer is cached in turn.
    EXPECT_EQ(read().ptr, after.ptr);
}

TEST_F(TestEngineConfigDescriptor, PredictionConfigMustBelongToItsEngineAndGraph)
{
    EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));
    EXPECT_CALL(*getMockEngine(), getGraph()).WillRepeatedly(Return(getMockGraphDescriptor()));
    setEngine();
    auto config = getEngineConfigDescriptor();
    const auto* graph = getMockGraphDescriptor().get();

    ASSERT_THROW_HIPDNN_STATUS(std::ignore = config->getEngineConfigForPrediction(2, graph),
                               HIPDNN_STATUS_BAD_PARAM);
    ASSERT_THROW_HIPDNN_STATUS(std::ignore = config->getEngineConfigForPrediction(1, nullptr),
                               HIPDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(config->getEngineConfigForPrediction(1, graph).engine_id, 1);
}

TEST_F(TestEngineConfigDescriptor, ScoredConfigRejectsNonFiniteFloatKnobs)
{
    namespace fb = hipdnn_flatbuffers_sdk::data_objects;
    EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));
    setEngine();
    auto config = getEngineConfigDescriptor();
    const auto withFloatKnob = [](double value) {
        fb::EngineConfigT scored;
        scored.engine_id = 1;
        auto knob = std::make_unique<fb::KnobSettingT>();
        knob->knob_id = "alpha";
        fb::FloatValueT floatValue;
        floatValue.value = value;
        knob->value.Set(floatValue);
        scored.knobs.push_back(std::move(knob));
        return scored;
    };

    for(const double bad : {std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::infinity(),
                            -std::numeric_limits<double>::infinity()})
    {
        ASSERT_THROW_HIPDNN_STATUS(config->setEngineConfig(withFloatKnob(bad)),
                                   HIPDNN_STATUS_BAD_PARAM);
    }

    ASSERT_NO_THROW(config->setEngineConfig(withFloatKnob(0.5)));
    const auto bytes = config->getSerializedEngineConfig();
    const auto* serialized = fb::GetEngineConfig(bytes.ptr);
    ASSERT_EQ(serialized->knobs()->size(), 1u);
    EXPECT_DOUBLE_EQ(serialized->knobs()->Get(0)->value_as_FloatValue()->value(), 0.5);
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceInvalidType)
{
    auto engineConfig = getEngineConfigDescriptor();

    auto knobBuffer = createSerializedKnobSetting("test_knob_100", 42);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    // Wrong attribute type
    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE, HIPDNN_TYPE_INT64, 1, &knobData),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceInvalidCount)
{
    auto engineConfig = getEngineConfigDescriptor();

    auto knobBuffer = createSerializedKnobSetting("test_knob_100", 42);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    // Element count < 1
    ASSERT_THROW_HIPDNN_STATUS(engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                                          HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                                          0,
                                                          &knobData),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceNullPointer)
{
    auto engineConfig = getEngineConfigDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                                          HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                                          1,
                                                          nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceNullFlatbufferPointer)
{
    auto engineConfig = getEngineConfigDescriptor();

    hipdnnBackendFlatbufferData_t knobData = {nullptr, 100};

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                                          HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                                          1,
                                                          &knobData),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceZeroSize)
{
    auto engineConfig = getEngineConfigDescriptor();

    auto knobBuffer = createSerializedKnobSetting("test_knob_100", 42);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), 0};

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                                          HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                                          1,
                                                          &knobData),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceSuccess)
{
    auto engineConfig = getEngineConfigDescriptor();

    EXPECT_CALL(*getMockEngine(), isFinalized()).WillRepeatedly(Return(true));
    EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));

    // Set engine first
    ASSERT_NO_THROW(engineConfig->setAttribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper));

    // Now set a knob choice
    auto knobBuffer = createSerializedKnobSetting("test_knob_100", 42);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    ASSERT_NO_THROW(engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                               HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                               1,
                                               &knobData));
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceMultipleKnobs)
{
    auto engineConfig = getEngineConfigDescriptor();

    EXPECT_CALL(*getMockEngine(), isFinalized()).WillRepeatedly(Return(true));
    EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));

    // Set engine first
    ASSERT_NO_THROW(engineConfig->setAttribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper));

    // Create multiple knob settings
    auto knobBuffer1 = createSerializedKnobSetting("test_knob_100", 42);
    auto knobBuffer2 = createSerializedKnobSetting("test_knob_101", 84);

    std::vector<hipdnnBackendFlatbufferData_t> knobDataArray
        = {{knobBuffer1.data(), knobBuffer1.size()}, {knobBuffer2.data(), knobBuffer2.size()}};

    ASSERT_NO_THROW(engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                               HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                               2,
                                               knobDataArray.data()));
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceOnFinalizedDescriptor)
{
    auto engineConfig = getEngineConfigDescriptor();
    makeEngineConfigFinalized();

    auto knobBuffer = createSerializedKnobSetting("test_knob_100", 42);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                                          HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                                          1,
                                                          &knobData),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

// Helper to create a finalized KnobSettingDescriptor
static std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedKnobSettingDescriptor(const std::string& knobId, int64_t value)
{
    auto wrapper = test_utilities::createDescriptor<KnobSettingDescriptor>();
    auto desc = wrapper->asDescriptor<KnobSettingDescriptor>();
    desc->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(knobId.size()),
                       knobId.c_str());
    desc->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_INT64, 1, &value);
    desc->finalize();
    return wrapper;
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceViaDescriptorSuccess)
{
    auto engineConfig = getEngineConfigDescriptor();

    EXPECT_CALL(*getMockEngine(), isFinalized()).WillRepeatedly(Return(true));
    EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));

    ASSERT_NO_THROW(engineConfig->setAttribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper));

    auto knobWrapper = createFinalizedKnobSettingDescriptor("test_knob_100", 42);
    auto* knobPtr = knobWrapper.get();

    ASSERT_NO_THROW(engineConfig->setAttribute(HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                               1,
                                               static_cast<const void*>(&knobPtr)));
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceViaDescriptorMultiple)
{
    auto engineConfig = getEngineConfigDescriptor();

    EXPECT_CALL(*getMockEngine(), isFinalized()).WillRepeatedly(Return(true));
    EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));

    ASSERT_NO_THROW(engineConfig->setAttribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper));

    auto knobWrapper1 = createFinalizedKnobSettingDescriptor("knob_1", 10);
    auto knobWrapper2 = createFinalizedKnobSettingDescriptor("knob_2", 20);
    std::array<HipdnnBackendDescriptor*, 2> knobPtrs = {knobWrapper1.get(), knobWrapper2.get()};

    ASSERT_NO_THROW(engineConfig->setAttribute(HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                               2,
                                               static_cast<const void*>(knobPtrs.data())));
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceViaDescriptorRejectNotFinalized)
{
    auto engineConfig = getEngineConfigDescriptor();

    EXPECT_CALL(*getMockEngine(), isFinalized()).WillRepeatedly(Return(true));
    EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));

    ASSERT_NO_THROW(engineConfig->setAttribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper));

    // Create a non-finalized knob descriptor
    auto knobWrapper = test_utilities::createDescriptor<KnobSettingDescriptor>();
    auto* knobPtr = knobWrapper.get();

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &knobPtr),
        HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceViaDescriptorRejectWrongType)
{
    auto engineConfig = getEngineConfigDescriptor();

    auto knobWrapper = createFinalizedKnobSettingDescriptor("test_knob", 42);
    auto* knobPtr = knobWrapper.get();

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES, HIPDNN_TYPE_INT64, 1, &knobPtr),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceViaDescriptorRejectExceedsMaxCount)
{
    auto engineConfig = getEngineConfigDescriptor();

    EXPECT_CALL(*getMockEngine(), isFinalized()).WillRepeatedly(Return(true));
    EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));

    ASSERT_NO_THROW(engineConfig->setAttribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper));

    auto knobWrapper = createFinalizedKnobSettingDescriptor("test_knob", 42);
    auto* knobPtr = knobWrapper.get();

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                   EngineConfigDescriptor::MAX_KNOB_CHOICES + 1,
                                   &knobPtr),
        HIPDNN_STATUS_BAD_PARAM);
}
