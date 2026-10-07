// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnBackendFlatbufferData.h"
#include "HipdnnException.hpp"
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
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_knob_catalog_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_plugin_sdk/KnobFactory.hpp>

#include <array>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

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

    /// Sets every mock expectation finalize() consumes except the engine name
    /// resolution, which the caller pins separately.
    void expectFinalizeCalls(int64_t engineId) const
    {
        setGraph();
        setGlobalIndex(engineId);
        EXPECT_CALL(*getMockGraph(), getHandle()).WillOnce(Return(_mockHandle.get()));
        EXPECT_CALL(*_mockHandle, getPluginResourceManager())
            .WillOnce(Return(_mockEnginePluginResourceManager));
        EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, _))
            .WillOnce(Return(std::vector<int64_t>{engineId}));
        EXPECT_CALL(*_mockEnginePluginResourceManager, getEngineDetails(_, _, _))
            .WillOnce(Invoke([this](int64_t, const GraphDescriptor*, hipdnnPluginConstData_t* d) {
                *d = this->_serializedEngineDetails;
            }));
        EXPECT_CALL(*_mockEnginePluginResourceManager, destroyEngineDetails(_, _));
    }

    void makeEngineFinalized() const
    {
        expectFinalizeCalls(ENGINE_ID);
        EXPECT_CALL(*_mockEnginePluginResourceManager, resolveEngineName(_, _));
        ASSERT_NO_THROW(getEngineDescriptor()->finalize());
    }

    /// Captures the candidate name the descriptor hands to the resolver and
    /// pins the name it answers with. A disengaged @p resolvedName keeps the
    /// resolver's hexadecimal fallback.
    void expectResolveEngineName(int64_t engineId, const std::optional<std::string>& resolvedName)
    {
        EXPECT_CALL(*_mockEnginePluginResourceManager, resolveEngineName(engineId, _))
            .WillOnce(Invoke(
                [this, resolvedName](int64_t id, std::optional<std::string_view> detailsName) {
                    _capturedDetailsName
                        = detailsName ? std::optional<std::string>(*detailsName) : std::nullopt;
                    _resolverCalled = true;
                    return resolvedName.value_or(hipdnn_data_sdk::utilities::formatEngineIdHex(id));
                }));
    }

    /// Two-call read of HIPDNN_ATTR_ENGINE_NAME_EXT: count, then value.
    std::string getEngineName() const
    {
        auto engine = getEngineDescriptor();

        int64_t elementCount = 0;
        EXPECT_NO_THROW(engine->getAttribute(
            HIPDNN_ATTR_ENGINE_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &elementCount, nullptr));
        EXPECT_GT(elementCount, 0);

        std::vector<char> buffer(static_cast<size_t>(elementCount));
        int64_t returnedCount = 0;
        EXPECT_NO_THROW(engine->getAttribute(HIPDNN_ATTR_ENGINE_NAME_EXT,
                                             HIPDNN_TYPE_CHAR,
                                             elementCount,
                                             &returnedCount,
                                             buffer.data()));
        return {buffer.data()};
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

    void serializeEngineDetailsWithName(int64_t engineId, const char* name)
    {
        flatbuffers::FlatBufferBuilder builder;
        auto engineDetails = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetailsDirect(
            builder, engineId, nullptr, nullptr, name);
        builder.Finish(engineDetails);
        _engineDetailsBuffer = builder.Release();
        _serializedEngineDetails = {_engineDetailsBuffer.data(), _engineDetailsBuffer.size()};
    }

    static constexpr int64_t ENGINE_ID = 0;
    /// An engine ID that the static name registry does not know about.
    static constexpr int64_t UNREGISTERED_ENGINE_ID = 0x0123456789ABCDEF;
    flatbuffers::DetachedBuffer _engineDetailsBuffer;
    hipdnnPluginConstData_t _serializedEngineDetails;
    /// Candidate name observed by the resolver: disengaged when the engine has
    /// no details, engaged and empty when the details carry no name.
    std::optional<std::string> _capturedDetailsName;
    bool _resolverCalled = false;
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
    EXPECT_CALL(*_mockEnginePluginResourceManager, resolveEngineName(_, _));

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

// HIPDNN_ATTR_ENGINE_NAME_EXT. The resolver is mocked, so these pin the candidate
// the descriptor derives and the answer it publishes, leaving the resolver's own
// choice of source to TestEnginePluginResourceManager.cpp.

TEST_F(TestEngineDescriptor, GetEngineNameForwardsEngineDetailsNameAsTheCandidate)
{
    serializeEngineDetailsWithName(ENGINE_ID, "EXAMPLE_PROVIDER_RELU_ENGINE");

    expectFinalizeCalls(ENGINE_ID);
    expectResolveEngineName(ENGINE_ID, "EXAMPLE_PROVIDER_RELU_ENGINE");
    ASSERT_NO_THROW(getEngineDescriptor()->finalize());

    EXPECT_TRUE(_resolverCalled);
    ASSERT_TRUE(_capturedDetailsName.has_value());
    EXPECT_EQ(*_capturedDetailsName, "EXAMPLE_PROVIDER_RELU_ENGINE");

    EXPECT_EQ(getEngineName(), "EXAMPLE_PROVIDER_RELU_ENGINE");
}

TEST_F(TestEngineDescriptor, GetEngineNamePublishesResolverAnswerNotItsOwnCandidate)
{
    // An engine ID the static registry already names, so the resolver's answer
    // below differs from every name reachable without the plugin entry point.
    const int64_t registeredEngineId = hipdnn_data_sdk::utilities::MIOPEN_ENGINE_ID;
    serializeEngineDetailsWithName(registeredEngineId, "DETAILS_ENGINE");

    expectFinalizeCalls(registeredEngineId);
    expectResolveEngineName(registeredEngineId, "PACK_SUPPLIED_ENGINE");
    ASSERT_NO_THROW(getEngineDescriptor()->finalize());

    EXPECT_TRUE(_resolverCalled);
    ASSERT_TRUE(_capturedDetailsName.has_value());
    EXPECT_EQ(*_capturedDetailsName, "DETAILS_ENGINE");

    EXPECT_EQ(getEngineName(), "PACK_SUPPLIED_ENGINE");
    EXPECT_NE(getEngineName(), "DETAILS_ENGINE");
    EXPECT_NE(getEngineName(), hipdnn_data_sdk::utilities::MIOPEN_ENGINE_NAME);
}

TEST_F(TestEngineDescriptor, GetEngineNameWithoutNameInEngineDetails)
{
    // Default fixture buffer: engine ID only, so the candidate is present but empty.
    expectFinalizeCalls(ENGINE_ID);
    expectResolveEngineName(ENGINE_ID, std::nullopt);
    ASSERT_NO_THROW(getEngineDescriptor()->finalize());

    EXPECT_TRUE(_resolverCalled);
    ASSERT_TRUE(_capturedDetailsName.has_value());
    EXPECT_TRUE(_capturedDetailsName->empty());

    EXPECT_EQ(getEngineName(), hipdnn_data_sdk::utilities::formatEngineIdHex(ENGINE_ID));
}

TEST_F(TestEngineDescriptor, GetEngineNameForUnknownEngineIdUsesHex)
{
    serializeEngineDetails(UNREGISTERED_ENGINE_ID);

    expectFinalizeCalls(UNREGISTERED_ENGINE_ID);
    expectResolveEngineName(UNREGISTERED_ENGINE_ID, std::nullopt);
    ASSERT_NO_THROW(getEngineDescriptor()->finalize());

    EXPECT_TRUE(_resolverCalled);
    EXPECT_EQ(getEngineName(), "0x0123456789ABCDEF");
}

TEST_F(TestEngineDescriptor, GetEngineNameEmptyNameInEngineDetails)
{
    serializeEngineDetailsWithName(ENGINE_ID, "");

    expectFinalizeCalls(ENGINE_ID);
    expectResolveEngineName(ENGINE_ID, std::nullopt);
    ASSERT_NO_THROW(getEngineDescriptor()->finalize());

    // An unset schema string reads back as an empty one and is forwarded as an
    // engaged candidate.
    EXPECT_TRUE(_resolverCalled);
    ASSERT_TRUE(_capturedDetailsName.has_value());
    EXPECT_TRUE(_capturedDetailsName->empty());

    EXPECT_EQ(getEngineName(), hipdnn_data_sdk::utilities::formatEngineIdHex(ENGINE_ID));
}

TEST_F(TestEngineDescriptor, FinalizeFailsBeforeNameResolutionWhenEngineDetailsMissing)
{
    // The plugin answers with no engine details at all. finalize() declares a
    // disengaged candidate for this case, but never reaches it: EngineDetailsWrapper
    // refuses to construct around an empty buffer, so finalize() fails first.
    setGraph();
    setGlobalIndex(ENGINE_ID);
    EXPECT_CALL(*getMockGraph(), getHandle()).WillOnce(Return(_mockHandle.get()));
    EXPECT_CALL(*_mockHandle, getPluginResourceManager())
        .WillOnce(Return(_mockEnginePluginResourceManager));
    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, _))
        .WillOnce(Return(std::vector<int64_t>{ENGINE_ID}));

    EXPECT_CALL(*_mockEnginePluginResourceManager, getEngineDetails(_, _, _));
    EXPECT_CALL(*_mockEnginePluginResourceManager, resolveEngineName(_, _)).Times(0);

    ASSERT_THROW_HIPDNN_STATUS(getEngineDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
    EXPECT_FALSE(_resolverCalled);
}

TEST_F(TestEngineDescriptor, GetEngineNameInvalidType)
{
    auto engine = getEngineDescriptor();
    makeEngineFinalized();

    int64_t elementCount = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        engine->getAttribute(
            HIPDNN_ATTR_ENGINE_NAME_EXT, HIPDNN_TYPE_INT64, 0, &elementCount, nullptr),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineDescriptor, GetEngineNameNotFinalized)
{
    auto engine = getEngineDescriptor();

    int64_t elementCount = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        engine->getAttribute(
            HIPDNN_ATTR_ENGINE_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &elementCount, nullptr),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestEngineDescriptor, SetEngineNameIsRejected)
{
    auto engine = getEngineDescriptor();
    const char* name = "USER_SUPPLIED_ENGINE";

    ASSERT_THROW_HIPDNN_STATUS(engine->setAttribute(HIPDNN_ATTR_ENGINE_NAME_EXT,
                                                    HIPDNN_TYPE_CHAR,
                                                    static_cast<int64_t>(std::strlen(name)),
                                                    name),
                               HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestEngineDescriptor, ToStringReportsEngineName)
{
    serializeEngineDetailsWithName(ENGINE_ID, "EXAMPLE_PROVIDER_RELU_ENGINE");

    expectFinalizeCalls(ENGINE_ID);
    expectResolveEngineName(ENGINE_ID, "EXAMPLE_PROVIDER_RELU_ENGINE");
    ASSERT_NO_THROW(getEngineDescriptor()->finalize());

    EXPECT_TRUE(_resolverCalled);
    EXPECT_NE(getEngineDescriptor()->toString().find("engineName=EXAMPLE_PROVIDER_RELU_ENGINE"),
              std::string::npos);
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
        EXPECT_CALL(*_mockEnginePluginResourceManager, resolveEngineName(_, _));
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

// =============================================================================
// HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT
// =============================================================================

namespace
{

using hipdnn_flatbuffers_sdk::data_objects::EngineConfigT;
using hipdnn_flatbuffers_sdk::data_objects::EngineKnobCatalogT;
using hipdnn_flatbuffers_sdk::data_objects::KnobSettingT;

using CatalogValue = std::variant<int64_t, std::string>;
using CatalogEntry = std::vector<std::pair<std::string, CatalogValue>>;

constexpr const char* TILE_KNOB = "cat.tile";
constexpr const char* MODE_KNOB = "cat.mode";
constexpr const char* GLOBAL_KNOB = "global.benchmarking";

std::unique_ptr<KnobSettingT> intSetting(const std::string& knobId, int64_t value)
{
    auto setting = std::make_unique<KnobSettingT>();
    setting->knob_id = knobId;
    hipdnn_flatbuffers_sdk::data_objects::IntValueT intValue;
    intValue.value = value;
    setting->value.Set(intValue);
    return setting;
}

std::unique_ptr<KnobSettingT> stringSetting(const std::string& knobId, const std::string& value)
{
    auto setting = std::make_unique<KnobSettingT>();
    setting->knob_id = knobId;
    hipdnn_flatbuffers_sdk::data_objects::StringValueT stringValue;
    stringValue.value = value;
    setting->value.Set(std::move(stringValue));
    return setting;
}

/// One catalog entry over the (tile, mode) knobs the fixture's engine advertises.
std::unique_ptr<EngineConfigT>
    makeEntry(int64_t tile, const std::string& mode, int64_t engineId = 0)
{
    auto entry = std::make_unique<EngineConfigT>();
    entry->engine_id = engineId;
    entry->knobs.push_back(intSetting(TILE_KNOB, tile));
    entry->knobs.push_back(stringSetting(MODE_KNOB, mode));
    return entry;
}

std::vector<std::unique_ptr<EngineConfigT>> entries(std::unique_ptr<EngineConfigT> a)
{
    std::vector<std::unique_ptr<EngineConfigT>> result;
    result.push_back(std::move(a));
    return result;
}

std::vector<std::unique_ptr<EngineConfigT>> entries(std::unique_ptr<EngineConfigT> a,
                                                    std::unique_ptr<EngineConfigT> b)
{
    auto result = entries(std::move(a));
    result.push_back(std::move(b));
    return result;
}

std::vector<std::unique_ptr<EngineConfigT>> entries(std::unique_ptr<EngineConfigT> a,
                                                    std::unique_ptr<EngineConfigT> b,
                                                    std::unique_ptr<EngineConfigT> c)
{
    auto result = entries(std::move(a), std::move(b));
    result.push_back(std::move(c));
    return result;
}

} // namespace

class TestEngineDescriptorKnobCatalog : public TestEngineDescriptor
{
protected:
    /// Finalizes the engine with details advertising (tile, mode) knobs, when
    /// @p engineSpecificKnobs, plus one global knob.
    void finalizeEngine(bool engineSpecificKnobs)
    {
        flatbuffers::FlatBufferBuilder builder;
        std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Knob>> knobs;
        if(engineSpecificKnobs)
        {
            knobs.push_back(hipdnn_plugin_sdk::KnobFactory::createIntKnob(
                builder, TILE_KNOB, "Tile size", 16, 0, 0, 1, {16, 32, 64}));
            knobs.push_back(hipdnn_plugin_sdk::KnobFactory::createStringKnob(
                builder, MODE_KNOB, "Mode", "fast", {"fast", "accurate"}));
        }
        knobs.push_back(hipdnn_plugin_sdk::KnobFactory::createIntKnob(
            builder, GLOBAL_KNOB, "Benchmarking", 0, 0, 1, 1, {}));
        auto details = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(
            builder, ENGINE_ID, builder.CreateVector(knobs));
        builder.Finish(details);
        _engineDetailsBuffer = builder.Release();
        _serializedEngineDetails = {_engineDetailsBuffer.data(), _engineDetailsBuffer.size()};

        makeEngineFinalized();

        // Everything past finalize(): reading the catalog goes back to the plugin, and each
        // entry is handed out as a finalized engine config, which asks for its workspace size.
        EXPECT_CALL(*getMockGraph(), getHandle()).WillRepeatedly(Return(_mockHandle.get()));
        EXPECT_CALL(*_mockHandle, getPluginResourceManager())
            .WillRepeatedly(Return(_mockEnginePluginResourceManager));
        EXPECT_CALL(*_mockEnginePluginResourceManager, getWorkspaceSize(_, _, _))
            .WillRepeatedly(Return(size_t{0}));
    }

    /// The plugin serves @p configs once.
    void serveCatalog(std::vector<std::unique_ptr<EngineConfigT>> configs) const
    {
        EngineKnobCatalogT catalog;
        catalog.engine_id = ENGINE_ID;
        catalog.configs = std::move(configs);
        EXPECT_CALL(*_mockEnginePluginResourceManager, getEngineKnobCatalog(ENGINE_ID, _))
            .WillOnce(Invoke([catalog](int64_t, const GraphDescriptor*) {
                return std::make_unique<EngineKnobCatalogT>(catalog);
            }));
    }

    /// The plugin offers no catalog for the engine.
    void serveNoCatalog() const
    {
        EXPECT_CALL(*_mockEnginePluginResourceManager, getEngineKnobCatalog(ENGINE_ID, _))
            .WillOnce(Invoke([](int64_t, const GraphDescriptor*) {
                return std::unique_ptr<EngineKnobCatalogT>();
            }));
    }

    /// Reads the whole catalog through the public attribute protocol: count, then entries,
    /// then each entry's knob choices.
    std::vector<CatalogEntry> readCatalog() const
    {
        auto engine = getEngineDescriptor();

        int64_t count = -1;
        engine->getAttribute(HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT,
                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                             0,
                             &count,
                             nullptr);

        std::vector<HipdnnBackendDescriptor*> raw(static_cast<size_t>(count));
        int64_t returned = -1;
        engine->getAttribute(HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT,
                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                             count,
                             &returned,
                             static_cast<void*>(raw.data()));
        EXPECT_EQ(returned, count);

        std::vector<std::unique_ptr<HipdnnBackendDescriptor>> owned;
        owned.reserve(raw.size());
        for(auto* descriptor : raw)
        {
            owned.emplace_back(descriptor);
        }

        std::vector<CatalogEntry> catalog;
        catalog.reserve(owned.size());
        for(const auto& entryDescriptor : owned)
        {
            catalog.push_back(readEntry(*entryDescriptor->asDescriptor<EngineConfigDescriptor>()));
        }
        return catalog;
    }

    static std::string readString(const KnobSettingDescriptor& choice,
                                  hipdnnBackendAttributeName_t attribute)
    {
        int64_t length = 0;
        choice.getAttribute(attribute, HIPDNN_TYPE_CHAR, 0, &length, nullptr);
        std::vector<char> buffer(static_cast<size_t>(length));
        choice.getAttribute(attribute, HIPDNN_TYPE_CHAR, length, &length, buffer.data());
        return {buffer.data()};
    }

    static CatalogEntry readEntry(const EngineConfigDescriptor& config)
    {
        int64_t count = -1;
        config.getAttribute(
            HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr);

        std::vector<HipdnnBackendDescriptor*> raw(static_cast<size_t>(count));
        int64_t returned = -1;
        config.getAttribute(HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                            count,
                            &returned,
                            static_cast<void*>(raw.data()));

        CatalogEntry entry;
        for(auto* descriptor : raw)
        {
            const std::unique_ptr<HipdnnBackendDescriptor> owned(descriptor);
            auto choice = owned->asDescriptor<KnobSettingDescriptor>();
            const auto knobId = readString(*choice, HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE);

            // The value is only retrievable in the type it was stored with.
            int64_t intValue = 0;
            int64_t valueCount = 0;
            try
            {
                choice->getAttribute(HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE,
                                     HIPDNN_TYPE_INT64,
                                     1,
                                     &valueCount,
                                     &intValue);
                entry.emplace_back(knobId, intValue);
            }
            catch(const HipdnnException&)
            {
                entry.emplace_back(knobId, readString(*choice, HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE));
            }
        }
        return entry;
    }

    /// Reads the catalog count and requires the read to fail with @p status and a message
    /// containing every one of @p fragments.
    void expectCatalogRejected(hipdnnStatus_t status,
                               const std::vector<std::string>& fragments) const
    {
        int64_t count = -1;
        try
        {
            getEngineDescriptor()->getAttribute(HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT,
                                                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                0,
                                                &count,
                                                nullptr);
            ADD_FAILURE() << "Reading the catalog was expected to fail";
        }
        catch(const HipdnnException& e)
        {
            EXPECT_EQ(e.getStatus(), status) << e.getMessage();
            for(const auto& fragment : fragments)
            {
                EXPECT_NE(e.getMessage().find(fragment), std::string::npos)
                    << "'" << fragment << "' missing from: " << e.getMessage();
            }
        }
    }
};

TEST_F(TestEngineDescriptorKnobCatalog, CatalogRoundTripsIntAndStringKnobsInPluginOrder)
{
    finalizeEngine(true);
    serveCatalog(entries(makeEntry(16, "fast"), makeEntry(64, "accurate"), makeEntry(32, "fast")));

    const auto catalog = readCatalog();

    const std::vector<CatalogEntry> expected = {
        {{TILE_KNOB, int64_t{16}}, {MODE_KNOB, std::string("fast")}},
        {{TILE_KNOB, int64_t{64}}, {MODE_KNOB, std::string("accurate")}},
        {{TILE_KNOB, int64_t{32}}, {MODE_KNOB, std::string("fast")}},
    };
    EXPECT_EQ(catalog, expected);
}

TEST_F(TestEngineDescriptorKnobCatalog, CatalogIsFetchedFromThePluginOnceAcrossReads)
{
    finalizeEngine(true);
    // serveCatalog() expects exactly one plugin call.
    serveCatalog(entries(makeEntry(16, "fast"), makeEntry(32, "fast")));

    const auto first = readCatalog();
    const auto second = readCatalog();

    EXPECT_EQ(first.size(), 2U);
    EXPECT_EQ(first, second);
}

TEST_F(TestEngineDescriptorKnobCatalog, FinalizeDoesNotFetchTheCatalog)
{
    EXPECT_CALL(*_mockEnginePluginResourceManager, getEngineKnobCatalog(_, _)).Times(0);

    finalizeEngine(true);
}

TEST_F(TestEngineDescriptorKnobCatalog, EntryNamingAnotherEngineIsRejected)
{
    finalizeEngine(true);
    serveCatalog(entries(makeEntry(16, "fast"), makeEntry(32, "fast", 7)));

    expectCatalogRejected(HIPDNN_STATUS_INTERNAL_ERROR,
                          {"(ID 0)", "entry 1", "engine ID 7 instead of the queried engine ID 0"});
}

TEST_F(TestEngineDescriptorKnobCatalog, EntryWithKnobTheEngineDoesNotAdvertiseIsRejected)
{
    finalizeEngine(true);
    auto bad = makeEntry(16, "fast");
    bad->knobs.push_back(intSetting("cat.unknown", 1));
    serveCatalog(entries(std::move(bad)));

    expectCatalogRejected(HIPDNN_STATUS_INTERNAL_ERROR,
                          {"(ID 0)", "entry 0", "'cat.unknown' is not advertised"});
}

TEST_F(TestEngineDescriptorKnobCatalog, EntryWithGlobalKnobIsRejectedEvenWhenAdvertised)
{
    finalizeEngine(true);
    auto bad = makeEntry(16, "fast");
    bad->knobs.push_back(intSetting(GLOBAL_KNOB, 1));
    serveCatalog(entries(makeEntry(32, "fast"), std::move(bad)));

    expectCatalogRejected(HIPDNN_STATUS_INTERNAL_ERROR,
                          {"(ID 0)", "entry 1", "'global.benchmarking' is a global knob"});
}

TEST_F(TestEngineDescriptorKnobCatalog, EntryWithValueOutsideTheKnobConstraintIsRejected)
{
    finalizeEngine(true);
    // 48 is not one of the tile knob's valid values {16, 32, 64}.
    serveCatalog(entries(makeEntry(16, "fast"), makeEntry(48, "fast")));

    expectCatalogRejected(
        HIPDNN_STATUS_INTERNAL_ERROR,
        {"(ID 0)", "entry 1", "'cat.tile' has an invalid value", "VALID_VALUES_INT"});
}

TEST_F(TestEngineDescriptorKnobCatalog, EntryWithValueOfTheWrongTypeIsRejected)
{
    finalizeEngine(true);
    auto bad = std::make_unique<EngineConfigT>();
    bad->engine_id = ENGINE_ID;
    bad->knobs.push_back(stringSetting(TILE_KNOB, "16"));
    serveCatalog(entries(std::move(bad)));

    expectCatalogRejected(HIPDNN_STATUS_INTERNAL_ERROR,
                          {"entry 0", "'cat.tile' has an invalid value", "type does not match"});
}

TEST_F(TestEngineDescriptorKnobCatalog, EntrySettingAKnobTwiceIsRejected)
{
    finalizeEngine(true);
    auto bad = makeEntry(16, "fast");
    bad->knobs.push_back(intSetting(TILE_KNOB, 32));
    serveCatalog(entries(std::move(bad)));

    expectCatalogRejected(HIPDNN_STATUS_INTERNAL_ERROR,
                          {"(ID 0)", "entry 0", "'cat.tile' appears more than once"});
}

TEST_F(TestEngineDescriptorKnobCatalog, EntriesWithEqualKnobValuesAreRejectedRegardlessOfKnobOrder)
{
    finalizeEngine(true);
    auto reordered = std::make_unique<EngineConfigT>();
    reordered->engine_id = ENGINE_ID;
    reordered->knobs.push_back(stringSetting(MODE_KNOB, "fast"));
    reordered->knobs.push_back(intSetting(TILE_KNOB, 16));
    serveCatalog(entries(makeEntry(16, "fast"), makeEntry(32, "fast"), std::move(reordered)));

    expectCatalogRejected(HIPDNN_STATUS_INTERNAL_ERROR,
                          {"(ID 0)", "entry 2", "same knob values as entry 0"});
}

TEST_F(TestEngineDescriptorKnobCatalog,
       PluginWithoutCatalogReportsOneEmptyEntryForEngineWithoutKnobs)
{
    // Only the global knob is advertised, and global knobs do not count as engine-specific.
    finalizeEngine(false);
    serveNoCatalog();

    const auto catalog = readCatalog();

    ASSERT_EQ(catalog.size(), 1U);
    EXPECT_TRUE(catalog.front().empty());
}

TEST_F(TestEngineDescriptorKnobCatalog, PluginWithoutCatalogIsNotSupportedForEngineWithKnobs)
{
    finalizeEngine(true);
    serveNoCatalog();

    expectCatalogRejected(HIPDNN_STATUS_NOT_SUPPORTED,
                          {"(ID 0)", "2 engine-specific knob(s)", "supplies no knob catalog"});
}

TEST_F(TestEngineDescriptorKnobCatalog, CatalogReadRequiresFinalizedEngine)
{
    int64_t count = -1;
    ASSERT_THROW_HIPDNN_STATUS(
        getEngineDescriptor()->getAttribute(HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT,
                                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                            0,
                                            &count,
                                            nullptr),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestEngineDescriptorKnobCatalog, CatalogReadRejectsWrongAttributeType)
{
    finalizeEngine(true);

    int64_t count = -1;
    ASSERT_THROW_HIPDNN_STATUS(
        getEngineDescriptor()->getAttribute(
            HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT, HIPDNN_TYPE_INT64, 0, &count, nullptr),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineDescriptorKnobCatalog, CatalogReadRejectsTooSmallBuffer)
{
    finalizeEngine(true);
    serveCatalog(entries(makeEntry(16, "fast"), makeEntry(32, "fast")));

    std::array<HipdnnBackendDescriptor*, 1> out{};
    int64_t count = -1;
    ASSERT_THROW_HIPDNN_STATUS(
        getEngineDescriptor()->getAttribute(HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT,
                                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                            1,
                                            &count,
                                            static_cast<void*>(out.data())),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineDescriptorKnobCatalog, CatalogAttributeCannotBeSet)
{
    int64_t value = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        getEngineDescriptor()->setAttribute(
            HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT, HIPDNN_TYPE_INT64, 1, &value),
        HIPDNN_STATUS_NOT_SUPPORTED);
}
