// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnBackendFlatbufferData.h"
#include "HipdnnException.hpp"
#include "TestMacros.hpp"
#include "descriptors/EngineDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockDescriptor.hpp"
#include "mocks/MockEnginePluginResourceManager.hpp"
#include "mocks/MockHandle.hpp"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>

#include <memory>

using namespace hipdnn_backend;
using namespace plugin;
using namespace hipdnn_backend::test_utilities;
using namespace ::testing;

using ::testing::Return;

class TestEngineDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<EngineDescriptor> getEngineDescriptor() const
    {
        return _engineWrapper->asDescriptor<EngineDescriptor>();
    }

    std::shared_ptr<MockGraphDescriptor> getMockGraph() const
    {
        return MockDescriptorUtility::asDescriptorUnsafe<MockGraphDescriptor>(
            _mockGraphWrapper.get());
    }

    std::shared_ptr<MockGraphDescriptor> getMockGraphBadType() const
    {
        return MockDescriptorUtility::asDescriptorUnsafe<MockGraphDescriptor>(
            _mockGraphBadTypeWrapper.get());
    }

    void setGraph() const
    {
        EXPECT_CALL(*getMockGraph(), isFinalized()).WillOnce(Return(true));
        ASSERT_NO_THROW(getEngineDescriptor()->setAttribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                            1,
                                                            &_mockGraphWrapper));
    }

    void setGlobalIndex(int64_t engineId) const
    {
        ASSERT_NO_THROW(getEngineDescriptor()->setAttribute(
            HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, &engineId));
    }

    void makeEngineFinalized() const
    {
        setGraph();
        setGlobalIndex(ENGINE_ID);
        EXPECT_CALL(*getMockGraph(), getHandle()).WillOnce(Return(_mockHandle.get()));
        EXPECT_CALL(*_mockHandle, getPluginResourceManager())
            .WillOnce(Return(_mockEnginePluginResourceManager));
        EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, _))
            .WillOnce(Return(std::vector<int64_t>{ENGINE_ID}));
        EXPECT_CALL(*_mockEnginePluginResourceManager, getEngineDetails(_, _, _))
            .WillOnce(Invoke([this](int64_t, const GraphDescriptor*, hipdnnPluginConstData_t* d) {
                *d = this->_serializedEngineDetails;
            }));
        EXPECT_CALL(*_mockEnginePluginResourceManager, destroyEngineDetails(_, _));
        ASSERT_NO_THROW(getEngineDescriptor()->finalize());
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _engineWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockGraphWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockGraphBadTypeWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockWrongTypeWrapper = nullptr;
    std::unique_ptr<MockHandle> _mockHandle = nullptr;
    std::shared_ptr<MockEnginePluginResourceManager> _mockEnginePluginResourceManager = nullptr;

    void SetUp() override
    {
        _engineWrapper = createDescriptor<EngineDescriptor>();
        _mockGraphWrapper = createDescriptor<MockGraphDescriptor>();
        _mockGraphBadTypeWrapper = createDescriptor<MockGraphDescriptor>();
        _mockWrongTypeWrapper = createDescriptor<MockEngineDescriptor>();
        _mockHandle = std::make_unique<MockHandle>();
        _mockEnginePluginResourceManager = std::make_shared<MockEnginePluginResourceManager>();

        serializeEngineDetails(ENGINE_ID);
    }

    void TearDown() override
    {
        _engineWrapper.reset();
    }

    void serializeEngineDetails(int64_t engineId)
    {
        flatbuffers::FlatBufferBuilder builder;
        hipdnn_flatbuffers_sdk::data_objects::EngineDetailsBuilder engineDetailsBuilder(builder);
        engineDetailsBuilder.add_engine_id(engineId);
        builder.Finish(engineDetailsBuilder.Finish());
        _engineDetailsBuffer = builder.Release();
        _serializedEngineDetails = {_engineDetailsBuffer.data(), _engineDetailsBuffer.size()};
    }

    void serializeEngineDetailsWithBehaviorNotes(int64_t engineId,
                                                 const std::vector<int32_t>& behaviorNotes)
    {
        flatbuffers::FlatBufferBuilder builder;
        auto behaviorNotesVector = builder.CreateVector(behaviorNotes);
        auto engineDetails = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(
            builder, engineId, 0, behaviorNotesVector);
        builder.Finish(engineDetails);
        _engineDetailsBuffer = builder.Release();
        _serializedEngineDetails = {_engineDetailsBuffer.data(), _engineDetailsBuffer.size()};
    }

    static constexpr int64_t ENGINE_ID = 0;
    flatbuffers::DetachedBuffer _engineDetailsBuffer;
    hipdnnPluginConstData_t _serializedEngineDetails;
};

TEST_F(TestEngineDescriptor, CreateEngineDescriptor)
{
    auto engine = getEngineDescriptor();
    ASSERT_NE(engine, nullptr);
    ASSERT_FALSE(engine->isFinalized());
    ASSERT_EQ(engine->getType(), HIPDNN_BACKEND_ENGINE_DESCRIPTOR);
}

TEST_F(TestEngineDescriptor, SetEngineDescriptorGraph)
{
    auto engine = getEngineDescriptor();
    EXPECT_CALL(*getMockGraphBadType(), isFinalized()).Times(1);

    ASSERT_THROW_HIPDNN_STATUS(
        engine->setAttribute(
            HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_INT64, 1, &_mockGraphWrapper),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(engine->setAttribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                    2,
                                                    &_mockGraphWrapper),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engine->setAttribute(
            HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipdnnBackendDescriptor_t graph = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(
        engine->setAttribute(
            HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(engine->setAttribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                    1,
                                                    &_mockGraphBadTypeWrapper),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    ASSERT_THROW_HIPDNN_STATUS(engine->setAttribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                    1,
                                                    &_mockWrongTypeWrapper),
                               HIPDNN_STATUS_BAD_PARAM);

    EXPECT_CALL(*getMockGraph(), isFinalized()).WillOnce(Return(false));
    ASSERT_THROW_HIPDNN_STATUS(engine->setAttribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                    1,
                                                    &_mockGraphWrapper),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    EXPECT_CALL(*getMockGraph(), isFinalized()).WillOnce(Return(true));
    ASSERT_NO_THROW(engine->setAttribute(
        HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockGraphWrapper));
}

TEST_F(TestEngineDescriptor, SetEngineDescriptorGlobalId)
{
    auto engine = getEngineDescriptor();
    int64_t gidx = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        engine->setAttribute(
            HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &gidx),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engine->setAttribute(HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 2, &gidx),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engine->setAttribute(HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(
        engine->setAttribute(HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, &gidx));
}

TEST_F(TestEngineDescriptor, SetAttrOnFinalizedEngineDescriptor)
{
    auto engine = getEngineDescriptor();
    makeEngineFinalized();

    ASSERT_THROW_HIPDNN_STATUS(engine->setAttribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                    1,
                                                    &_mockGraphWrapper),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestEngineDescriptor, FinalizeEngineDescriptor)
{
    auto engine = getEngineDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(engine->finalize(), HIPDNN_STATUS_BAD_PARAM);

    makeEngineFinalized();

    ASSERT_THROW_HIPDNN_STATUS(engine->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineDescriptor, GetAttrOnUnfinalizedEngineDescriptor)
{
    auto engine = getEngineDescriptor();
    hipdnnBackendDescriptor_t dummyGraph = nullptr;

    ASSERT_THROW_HIPDNN_STATUS(engine->getAttribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                    1,
                                                    nullptr,
                                                    &dummyGraph),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestEngineDescriptor, GetEngineDescriptorUnsupportedAttr)
{
    auto engine = getEngineDescriptor();
    int32_t dummy;

    makeEngineFinalized();

    ASSERT_THROW_HIPDNN_STATUS(
        engine->getAttribute(
            HIPDNN_ATTR_ENGINE_CU_COUNT_TARGET_EXT, HIPDNN_TYPE_INT32, 1, nullptr, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestEngineDescriptor, GetBehaviorNotesCountWithNoNotes)
{
    auto engine = getEngineDescriptor();
    makeEngineFinalized();

    int64_t noteCount = -1;
    ASSERT_NO_THROW(engine->getAttribute(
        HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 0, &noteCount, nullptr));
    ASSERT_EQ(noteCount, 0);
}

TEST_F(TestEngineDescriptor, GetBehaviorNotesReturnsNotes)
{
    serializeEngineDetailsWithBehaviorNotes(
        0,
        {static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION),
         static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_EXTERNAL_LIBRARY_DEPENDENCY),
         static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION)});
    auto engine = getEngineDescriptor();
    makeEngineFinalized();

    int64_t noteCount = 0;
    ASSERT_NO_THROW(engine->getAttribute(
        HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 0, &noteCount, nullptr));
    ASSERT_EQ(noteCount, 3);

    std::vector<hipdnnBackendBehaviorNote_t> notes(static_cast<size_t>(noteCount));
    int64_t returnedCount = 0;
    ASSERT_NO_THROW(engine->getAttribute(HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                                         HIPDNN_TYPE_BEHAVIOR_NOTE,
                                         noteCount,
                                         &returnedCount,
                                         notes.data()));
    ASSERT_EQ(returnedCount, 3);
    EXPECT_EQ(notes[0], HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION);
    EXPECT_EQ(notes[1], HIPDNN_BEHAVIOR_NOTE_EXTERNAL_LIBRARY_DEPENDENCY);
    EXPECT_EQ(notes[2], HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION);
}

TEST_F(TestEngineDescriptor, GetBehaviorNotesReturnsNotesWithoutElementCount)
{
    serializeEngineDetailsWithBehaviorNotes(
        0,
        {static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION),
         static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_SUPPORTS_GRAPH_CAPTURE)});
    auto engine = getEngineDescriptor();
    makeEngineFinalized();

    std::vector<hipdnnBackendBehaviorNote_t> notes(2);
    ASSERT_NO_THROW(engine->getAttribute(HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                                         HIPDNN_TYPE_BEHAVIOR_NOTE,
                                         static_cast<int64_t>(notes.size()),
                                         nullptr,
                                         notes.data()));
    EXPECT_EQ(notes[0], HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION);
    EXPECT_EQ(notes[1], HIPDNN_BEHAVIOR_NOTE_SUPPORTS_GRAPH_CAPTURE);
}

TEST_F(TestEngineDescriptor, GetBehaviorNotesNullOutputReturnsCount)
{
    serializeEngineDetailsWithBehaviorNotes(
        0,
        {static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION),
         static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_EXTERNAL_LIBRARY_DEPENDENCY)});
    auto engine = getEngineDescriptor();
    makeEngineFinalized();

    int64_t noteCount = -1;
    ASSERT_NO_THROW(engine->getAttribute(
        HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 2, &noteCount, nullptr));
    ASSERT_EQ(noteCount, 2);
}

TEST_F(TestEngineDescriptor, GetBehaviorNotesZeroRequestedWithOutputReturnsCount)
{
    serializeEngineDetailsWithBehaviorNotes(
        0,
        {static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION),
         static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_EXTERNAL_LIBRARY_DEPENDENCY)});
    auto engine = getEngineDescriptor();
    makeEngineFinalized();

    hipdnnBackendBehaviorNote_t note = HIPDNN_BEHAVIOR_NOTE_TYPE_COUNT;
    int64_t noteCount = -1;
    ASSERT_NO_THROW(engine->getAttribute(
        HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 0, &noteCount, &note));
    ASSERT_EQ(noteCount, 2);
    EXPECT_EQ(note, HIPDNN_BEHAVIOR_NOTE_TYPE_COUNT);
}

TEST_F(TestEngineDescriptor, GetBehaviorNotesCountQueryRequiresElementCount)
{
    auto engine = getEngineDescriptor();
    makeEngineFinalized();

    ASSERT_THROW_HIPDNN_STATUS(
        engine->getAttribute(
            HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 0, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestEngineDescriptor, GetBehaviorNotesInvalidType)
{
    auto engine = getEngineDescriptor();
    makeEngineFinalized();

    int64_t noteCount = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        engine->getAttribute(
            HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_INT64, 0, &noteCount, nullptr),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineDescriptor, GetBehaviorNotesInsufficientOutputCount)
{
    serializeEngineDetailsWithBehaviorNotes(
        0,
        {static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION),
         static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_EXTERNAL_LIBRARY_DEPENDENCY)});
    auto engine = getEngineDescriptor();
    makeEngineFinalized();

    hipdnnBackendBehaviorNote_t note = HIPDNN_BEHAVIOR_NOTE_TYPE_COUNT;
    int64_t returnedCount = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        engine->getAttribute(
            HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 1, &returnedCount, &note),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineDescriptor, FinalizePreservesUnknownBehaviorNote)
{
    constexpr hipdnnBackendBehaviorNote_t UNKNOWN_NOTE = HIPDNN_BEHAVIOR_NOTE_TYPE_COUNT + 1;
    serializeEngineDetailsWithBehaviorNotes(0, {UNKNOWN_NOTE});

    setGraph();
    setGlobalIndex(0);
    EXPECT_CALL(*getMockGraph(), getHandle()).WillOnce(Return(_mockHandle.get()));
    EXPECT_CALL(*_mockHandle, getPluginResourceManager())
        .WillOnce(Return(_mockEnginePluginResourceManager));
    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, _))
        .WillOnce(Return(std::vector<int64_t>{0}));
    EXPECT_CALL(*_mockEnginePluginResourceManager, getEngineDetails(_, _, _))
        .WillOnce(Invoke([this](int64_t, const GraphDescriptor*, hipdnnPluginConstData_t* d) {
            *d = this->_serializedEngineDetails;
        }));
    EXPECT_CALL(*_mockEnginePluginResourceManager, destroyEngineDetails(_, _));

    auto engine = getEngineDescriptor();
    ASSERT_NO_THROW(engine->finalize());

    hipdnnBackendBehaviorNote_t note = HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION;
    int64_t returnedCount = 0;
    ASSERT_NO_THROW(engine->getAttribute(
        HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 1, &returnedCount, &note));
    EXPECT_EQ(returnedCount, 1);
    EXPECT_EQ(note, UNKNOWN_NOTE);
}

TEST_F(TestEngineDescriptor, FinalizeFailsWithNegativeBehaviorNote)
{
    serializeEngineDetailsWithBehaviorNotes(0, {-1});

    setGraph();
    setGlobalIndex(0);
    EXPECT_CALL(*getMockGraph(), getHandle()).WillOnce(Return(_mockHandle.get()));
    EXPECT_CALL(*_mockHandle, getPluginResourceManager())
        .WillOnce(Return(_mockEnginePluginResourceManager));
    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, _))
        .WillOnce(Return(std::vector<int64_t>{0}));
    EXPECT_CALL(*_mockEnginePluginResourceManager, getEngineDetails(_, _, _))
        .WillOnce(Invoke([this](int64_t, const GraphDescriptor*, hipdnnPluginConstData_t* d) {
            *d = this->_serializedEngineDetails;
        }));
    EXPECT_CALL(*_mockEnginePluginResourceManager, destroyEngineDetails(_, _));

    ASSERT_THROW_HIPDNN_STATUS(getEngineDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineDescriptor, GetEngineDescriptorGraph)
{
    auto engine = getEngineDescriptor();
    ScopedDescriptor graph;
    ScopedDescriptor graph2;

    makeEngineFinalized();

    ASSERT_THROW_HIPDNN_STATUS(
        engine->getAttribute(
            HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_INT64, 1, nullptr, graph.getPtr()),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(engine->getAttribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                    2,
                                                    nullptr,
                                                    graph.getPtr()),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(engine->getAttribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                    1,
                                                    nullptr,
                                                    nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(engine->getAttribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                         1,
                                         nullptr,
                                         static_cast<void*>(graph.getPtr())));
    ASSERT_EQ(*graph.get(), *(_mockGraphWrapper.get()));

    int64_t count;
    ASSERT_NO_THROW(engine->getAttribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                         1,
                                         &count,
                                         static_cast<void*>(graph2.getPtr())));
    ASSERT_EQ(count, 1);
}

TEST_F(TestEngineDescriptor, GetEngineDescriptorGlobalId)
{
    auto engine = getEngineDescriptor();
    int64_t gidx = -1;

    makeEngineFinalized();

    ASSERT_THROW_HIPDNN_STATUS(
        engine->getAttribute(
            HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, &gidx),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engine->getAttribute(HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 2, nullptr, &gidx),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engine->getAttribute(
            HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(engine->getAttribute(
        HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, &gidx));
    ASSERT_EQ(gidx, 0);

    int64_t count;
    ASSERT_NO_THROW(
        engine->getAttribute(HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, &count, &gidx));
    ASSERT_EQ(count, 1);
}

TEST_F(TestEngineDescriptor, GetGraphThrowsIfNotFinalized)
{
    auto engine = getEngineDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(engine->getGraph(), HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestEngineDescriptor, GetGraphReturnsPointerIfFinalized)
{
    auto engine = getEngineDescriptor();
    makeEngineFinalized();
    auto graphPtr = engine->getGraph();
    ASSERT_NE(graphPtr, nullptr);
    ASSERT_EQ(static_cast<const IBackendDescriptor*>(graphPtr.get()),
              static_cast<const IBackendDescriptor*>(getMockGraph().get()));
}

TEST_F(TestEngineDescriptor, GetEngineIdThrowsIfNotFinalized)
{
    auto engine = getEngineDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(engine->getEngineId(), HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestEngineDescriptor, GetEngineIdReturnsValueIfFinalized)
{
    auto engine = getEngineDescriptor();
    makeEngineFinalized();
    auto engineId = engine->getEngineId();
    ASSERT_EQ(engineId, 0);
}

// Test fixture for EngineDescriptor with knobs
class TestEngineDescriptorWithKnobs : public TestEngineDescriptor
{
protected:
    void SetUp() override
    {
        TestEngineDescriptor::SetUp();
        // Serialize engine details with knobs for knob tests
        serializeEngineDetailsWithKnobs(0, 2);
    }

    void serializeEngineDetailsWithKnobs(int64_t engineId, size_t knobCount)
    {
        flatbuffers::FlatBufferBuilder builder;

        std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Knob>> knobOffsets;
        for(size_t i = 0; i < knobCount; ++i)
        {
            auto knobIdStr = builder.CreateString("test_knob_" + std::to_string(i));
            auto description = builder.CreateString("Test knob description " + std::to_string(i));

            // Create a default int value
            auto defaultValue = hipdnn_flatbuffers_sdk::data_objects::CreateIntValue(
                builder, static_cast<int64_t>(i * 10));

            auto knob = hipdnn_flatbuffers_sdk::data_objects::CreateKnob(
                builder,
                knobIdStr,
                description,
                hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue,
                defaultValue.Union());
            knobOffsets.push_back(knob);
        }

        auto knobsVector = builder.CreateVector(knobOffsets);
        auto engineDetails = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(
            builder, engineId, knobsVector);
        builder.Finish(engineDetails);

        _engineDetailsWithKnobsBuffer = builder.Release();
        _serializedEngineDetailsWithKnobs
            = {_engineDetailsWithKnobsBuffer.data(), _engineDetailsWithKnobsBuffer.size()};
    }

    void makeEngineFinalizedWithKnobs() const
    {
        EXPECT_CALL(*getMockGraph(), isFinalized()).WillOnce(Return(true));
        ASSERT_NO_THROW(getEngineDescriptor()->setAttribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                            1,
                                                            &_mockGraphWrapper));

        int64_t engineId = 0;
        ASSERT_NO_THROW(getEngineDescriptor()->setAttribute(
            HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, &engineId));

        EXPECT_CALL(*getMockGraph(), getHandle()).WillOnce(Return(_mockHandle.get()));
        EXPECT_CALL(*_mockHandle, getPluginResourceManager())
            .WillOnce(Return(_mockEnginePluginResourceManager));
        EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, _))
            .WillOnce(Return(std::vector<int64_t>{0}));
        EXPECT_CALL(*_mockEnginePluginResourceManager, getEngineDetails(_, _, _))
            .WillOnce(Invoke([this](int64_t, const GraphDescriptor*, hipdnnPluginConstData_t* d) {
                *d = this->_serializedEngineDetailsWithKnobs;
            }));
        EXPECT_CALL(*_mockEnginePluginResourceManager, destroyEngineDetails(_, _));
        ASSERT_NO_THROW(getEngineDescriptor()->finalize());
    }

    flatbuffers::DetachedBuffer _engineDetailsWithKnobsBuffer;
    hipdnnPluginConstData_t _serializedEngineDetailsWithKnobs;
};

TEST_F(TestEngineDescriptor, GetKnobInfoCountWithNoKnobs)
{
    auto engine = getEngineDescriptor();
    makeEngineFinalized();

    int64_t knobCount = -1;
    ASSERT_NO_THROW(engine->getAttribute(HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE,
                                         HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                         0,
                                         &knobCount,
                                         nullptr));
    ASSERT_EQ(knobCount, 0);
}

TEST_F(TestEngineDescriptor, GetKnobInfoInvalidType)
{
    auto engine = getEngineDescriptor();
    makeEngineFinalized();

    int64_t knobCount = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        engine->getAttribute(
            HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE, HIPDNN_TYPE_INT64, 0, &knobCount, nullptr),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineDescriptor, GetKnobInfoNotFinalized)
{
    auto engine = getEngineDescriptor();

    int64_t knobCount = 0;
    ASSERT_THROW_HIPDNN_STATUS(engine->getAttribute(HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE,
                                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                                    0,
                                                    &knobCount,
                                                    nullptr),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestEngineDescriptorWithKnobs, GetKnobInfoCountWithKnobs)
{
    auto engine = getEngineDescriptor();
    makeEngineFinalizedWithKnobs();

    int64_t knobCount = -1;
    ASSERT_NO_THROW(engine->getAttribute(HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE,
                                         HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                         0,
                                         &knobCount,
                                         nullptr));
    ASSERT_EQ(knobCount, 2);
}

TEST_F(TestEngineDescriptorWithKnobs, GetKnobInfoReturnsSerializedKnobs)
{
    auto engine = getEngineDescriptor();
    makeEngineFinalizedWithKnobs();

    // First, get the count
    int64_t knobCount = 0;
    ASSERT_NO_THROW(engine->getAttribute(HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE,
                                         HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                         0,
                                         &knobCount,
                                         nullptr));
    ASSERT_EQ(knobCount, 2);

    // Now get the actual knob data
    std::vector<hipdnnBackendFlatbufferData_t> knobData(static_cast<size_t>(knobCount));
    int64_t returnedCount = 0;
    ASSERT_NO_THROW(engine->getAttribute(HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE,
                                         HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                         knobCount,
                                         &returnedCount,
                                         knobData.data()));
    ASSERT_EQ(returnedCount, 2);

    // Verify the returned data is valid
    for(size_t i = 0; i < static_cast<size_t>(returnedCount); ++i)
    {
        ASSERT_NE(knobData[i].ptr, nullptr);
        ASSERT_GT(knobData[i].size, 0UL);

        // Verify we can parse the flatbuffer
        flatbuffers::Verifier verifier(static_cast<const uint8_t*>(knobData[i].ptr),
                                       knobData[i].size);
        ASSERT_TRUE(verifier.VerifyBuffer<hipdnn_flatbuffers_sdk::data_objects::Knob>());

        auto knob
            = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::Knob>(knobData[i].ptr);
        ASSERT_EQ(knob->knob_id()->str(), "test_knob_" + std::to_string(i));
    }
}

TEST_F(TestEngineDescriptorWithKnobs, GetKnobInfoNullPointerWhenCountNonZero)
{
    auto engine = getEngineDescriptor();
    makeEngineFinalizedWithKnobs();

    int64_t returnedCount = 0;
    ASSERT_THROW_HIPDNN_STATUS(engine->getAttribute(HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE,
                                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                                    1,
                                                    &returnedCount,
                                                    nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}
