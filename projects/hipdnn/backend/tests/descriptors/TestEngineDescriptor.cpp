// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
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
#include <hipdnn_sdk/data_objects/engine_details_generated.h>

#include <memory>

using namespace hipdnn_backend;
using namespace plugin;
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
        EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_))
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
        _engineWrapper = hipdnn_sdk::test_utilities::createDescriptor<EngineDescriptor>();
        _mockGraphWrapper = hipdnn_sdk::test_utilities::createDescriptor<MockGraphDescriptor>();
        _mockGraphBadTypeWrapper
            = hipdnn_sdk::test_utilities::createDescriptor<MockGraphDescriptor>();
        _mockWrongTypeWrapper
            = hipdnn_sdk::test_utilities::createDescriptor<MockEngineDescriptor>();
        _mockHandle = std::make_unique<MockHandle>();
        _mockEnginePluginResourceManager = std::make_shared<MockEnginePluginResourceManager>();

        serializeEngineDetails(ENGINE_ID);
    }

    void TearDown() override
    {
        _engineWrapper.reset();
    }

private:
    void serializeEngineDetails(int64_t engineId)
    {
        flatbuffers::FlatBufferBuilder builder;
        hipdnn_sdk::data_objects::EngineDetailsBuilder engineDetailsBuilder(builder);
        engineDetailsBuilder.add_engine_id(engineId);
        builder.Finish(engineDetailsBuilder.Finish());
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
            HIPDNN_ATTR_ENGINE_SM_COUNT_TARGET, HIPDNN_TYPE_INT32, 1, nullptr, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
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
                                         graph.getPtr()));
    ASSERT_EQ(*graph.get(), *(_mockGraphWrapper.get()));

    int64_t count;
    ASSERT_NO_THROW(engine->getAttribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                         1,
                                         &count,
                                         graph2.getPtr()));
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
