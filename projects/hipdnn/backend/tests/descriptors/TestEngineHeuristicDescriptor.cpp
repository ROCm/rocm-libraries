// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TestMacros.hpp"
#include "descriptors/EngineConfigDescriptor.hpp"
#include "descriptors/EngineDescriptor.hpp"
#include "descriptors/EngineHeuristicDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockDescriptor.hpp"
#include "mocks/MockEnginePluginResourceManager.hpp"
#include "mocks/MockHandle.hpp"

#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/engine_details_generated.h>

#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace plugin;
using namespace hipdnn_sdk::test_utilities;
using namespace ::testing;

using ::testing::Return;

class TestEngineHeuristicDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<EngineHeuristicDescriptor> getEngineHeuristicDescriptor() const
    {
        return _engineHeuristicWrapper->asDescriptor<EngineHeuristicDescriptor>();
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

    std::shared_ptr<MockDescriptor<EngineHeuristicDescriptor>> getMockWrongType() const
    {
        return _mockWrongTypeWrapper->asDescriptor<MockDescriptor<EngineHeuristicDescriptor>>();
    }

    void setGraph() const
    {
        EXPECT_CALL(*getMockGraph(), isFinalized()).WillRepeatedly(Return(true));
        EXPECT_CALL(*getMockGraph(), getHandle()).WillRepeatedly(Return(_mockHandle.get()));
        EXPECT_CALL(*_mockHandle, getPluginResourceManager())
            .WillRepeatedly(Return(_mockEnginePluginResourceManager));
        ASSERT_NO_THROW(
            getEngineHeuristicDescriptor()->setAttribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                         1,
                                                         &_mockGraphWrapper));
    }

    void setHeuristicMode() const
    {
        hipdnnBackendHeurMode_t mode = HIPDNN_HEUR_MODE_FALLBACK;
        ASSERT_NO_THROW(getEngineHeuristicDescriptor()->setAttribute(
            HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, &mode));
    }

    void makeEngineHeuristicFinalized() const
    {
        setGraph();
        setHeuristicMode();
        EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_))
            .WillRepeatedly(Return(std::vector<int64_t>{0, 1, 2}));
        ASSERT_NO_THROW(getEngineHeuristicDescriptor()->finalize());
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _engineHeuristicWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockGraphWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockGraphBadTypeWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockWrongTypeWrapper = nullptr;
    std::unique_ptr<MockHandle> _mockHandle = nullptr;
    std::shared_ptr<MockEnginePluginResourceManager> _mockEnginePluginResourceManager = nullptr;

    void SetUp() override
    {
        _engineHeuristicWrapper = createDescriptor<EngineHeuristicDescriptor>();
        _mockGraphWrapper = createDescriptor<MockGraphDescriptor>();
        _mockGraphBadTypeWrapper = createDescriptor<MockGraphDescriptor>();
        _mockWrongTypeWrapper = createDescriptor<MockDescriptor<EngineHeuristicDescriptor>>();
        _mockHandle = std::make_unique<MockHandle>();
        _mockEnginePluginResourceManager = std::make_shared<MockEnginePluginResourceManager>();
    }

    void TearDown() override
    {
        _engineDetailBuffers.clear();
    }

    hipdnnPluginConstData_t serializeEngineDetails(int64_t gidx)
    {
        flatbuffers::FlatBufferBuilder builder;
        hipdnn_sdk::data_objects::EngineDetailsBuilder engineDetailsBuilder(builder);
        engineDetailsBuilder.add_engine_id(gidx);
        builder.Finish(engineDetailsBuilder.Finish());
        auto engineDetailsBuffer = builder.Release();
        hipdnnPluginConstData_t serializedEngineDetails
            = {engineDetailsBuffer.data(), engineDetailsBuffer.size()};
        _engineDetailBuffers.push_back(std::move(engineDetailsBuffer));
        return serializedEngineDetails;
    }

    std::vector<flatbuffers::DetachedBuffer> _engineDetailBuffers;
};

TEST_F(TestEngineHeuristicDescriptor, CreateEngineHeuristicDescriptor)
{
    auto heur = getEngineHeuristicDescriptor();
    ASSERT_NE(heur, nullptr);
    ASSERT_FALSE(heur->isFinalized());
    ASSERT_EQ(heur->getType(), HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
}

TEST_F(TestEngineHeuristicDescriptor, SetEngineHeuristicDescriptorGraph)
{
    auto heur = getEngineHeuristicDescriptor();

    EXPECT_CALL(*getMockGraphBadType(), isFinalized()).Times(1);
    EXPECT_CALL(*getMockGraph(), isFinalized()).WillOnce(Return(false)).WillOnce(Return(true));

    // isFinalized->false
    ASSERT_THROW_HIPDNN_STATUS(heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_mockGraphWrapper),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    // isFinalized()->true
    ASSERT_NO_THROW(heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_mockGraphWrapper));

    ASSERT_THROW_HIPDNN_STATUS(
        heur->setAttribute(
            HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_INT64, 1, &_mockGraphWrapper),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  2,
                                                  &_mockGraphWrapper),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        heur->setAttribute(
            HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipdnnBackendDescriptor_t graph = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(
        heur->setAttribute(
            HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_mockGraphBadTypeWrapper),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    ASSERT_THROW_HIPDNN_STATUS(heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_mockWrongTypeWrapper),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineHeuristicDescriptor, SetEngineHeuristicDescriptorHeurMode)
{
    auto heur = getEngineHeuristicDescriptor();
    hipdnnBackendHeurMode_t mode = HIPDNN_HEUR_MODE_FALLBACK;
    auto unsupportedMode = static_cast<hipdnnBackendHeurMode_t>(999);

    ASSERT_THROW_HIPDNN_STATUS(
        heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &mode),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 2, &mode),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(
        heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, &unsupportedMode),
        HIPDNN_STATUS_NOT_SUPPORTED);

    ASSERT_NO_THROW(
        heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, &mode));
}

TEST_F(TestEngineHeuristicDescriptor, SetEngineHeuristicDescriptorUnsupportedAttr)
{
    auto heur = getEngineHeuristicDescriptor();
    int32_t dummy = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_INT32, 1, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestEngineHeuristicDescriptor, SetAttrOnFinalizedEngineHeuristicDescriptor)
{
    auto heur = getEngineHeuristicDescriptor();
    makeEngineHeuristicFinalized();

    ASSERT_THROW_HIPDNN_STATUS(heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_mockGraphWrapper),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestEngineHeuristicDescriptor, FinalizeEngineHeuristicDescriptor)
{
    auto heur = getEngineHeuristicDescriptor();
    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds)
        .Times(AnyNumber()); //Uninteresting call

    ASSERT_THROW_HIPDNN_STATUS(heur->finalize(), HIPDNN_STATUS_BAD_PARAM);

    setGraph();
    ASSERT_THROW_HIPDNN_STATUS(heur->finalize(), HIPDNN_STATUS_BAD_PARAM);

    setHeuristicMode();
    ASSERT_NO_THROW(heur->finalize());

    ASSERT_THROW_HIPDNN_STATUS(heur->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineHeuristicDescriptor, FinalizeEngineHeuristicDescriptorReverseOrder)
{
    auto heur = getEngineHeuristicDescriptor();
    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds)
        .Times(AnyNumber()); //Uninteresting call

    ASSERT_THROW_HIPDNN_STATUS(heur->finalize(), HIPDNN_STATUS_BAD_PARAM);

    setHeuristicMode();
    ASSERT_THROW_HIPDNN_STATUS(heur->finalize(), HIPDNN_STATUS_BAD_PARAM);

    setGraph();
    ASSERT_NO_THROW(heur->finalize());

    ASSERT_THROW_HIPDNN_STATUS(heur->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineHeuristicDescriptor, GetAttrOnUnfinalizedEngineHeuristicDescriptor)
{
    auto heur = getEngineHeuristicDescriptor();
    hipdnnBackendDescriptor_t dummyGraph = nullptr;

    ASSERT_THROW_HIPDNN_STATUS(heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummyGraph),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestEngineHeuristicDescriptor, GetEngineHeuristicDescriptorUnsupportedAttr)
{
    auto heur = getEngineHeuristicDescriptor();
    hipdnnBackendHeurMode_t dummy;

    makeEngineHeuristicFinalized();

    ASSERT_THROW_HIPDNN_STATUS(
        heur->getAttribute(HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_HEUR_MODE, 1, nullptr, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestEngineHeuristicDescriptor, GetEngineHeuristicDescriptorGraph)
{
    auto heur = getEngineHeuristicDescriptor();
    ScopedDescriptor graph;
    ScopedDescriptor graph2;

    makeEngineHeuristicFinalized();

    ASSERT_THROW_HIPDNN_STATUS(
        heur->getAttribute(
            HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_INT64, 1, nullptr, graph.getPtr()),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  2,
                                                  nullptr,
                                                  graph.getPtr()),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       nullptr,
                                       graph.getPtr()));
    ASSERT_EQ(*graph.get(), *(_mockGraphWrapper.get()));

    int64_t count;
    ASSERT_NO_THROW(heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &count,
                                       graph2.getPtr()));
    ASSERT_EQ(count, 1);
}

TEST_F(TestEngineHeuristicDescriptor, GetEngineHeuristicDescriptorEngineConfigs)
{
    auto heur = getEngineHeuristicDescriptor();
    makeEngineHeuristicFinalized();

    EXPECT_CALL(*getMockGraph(), isFinalized()).WillRepeatedly(Return(true));
    EXPECT_CALL(*_mockEnginePluginResourceManager, getEngineDetails(_, _, _))
        .WillRepeatedly(
            Invoke([this](int64_t engineId, const GraphDescriptor*, hipdnnPluginConstData_t* d) {
                *d = this->serializeEngineDetails(engineId);
            }));
    EXPECT_CALL(*_mockEnginePluginResourceManager, destroyEngineDetails(_, _))
        .WillRepeatedly(Return());

    ASSERT_THROW_HIPDNN_STATUS(
        heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_INT64, 0, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM);

    int64_t count = 0;
    ASSERT_NO_THROW(heur->getAttribute(
        HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr));
    ASSERT_EQ(count, 3);

    ASSERT_THROW_HIPDNN_STATUS(
        heur->getAttribute(
            HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    std::vector<hipdnnBackendDescriptor_t> configs(3);
    for(size_t i = 0; i < 3; ++i)
    {
        configs[i] = createDescriptorPtr<EngineConfigDescriptor>();
    }

    ASSERT_THROW_HIPDNN_STATUS(heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  3,
                                                  nullptr,
                                                  configs.data()),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    count = 0;
    ASSERT_NO_THROW(heur->getAttribute(
        HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 3, &count, configs.data()));
    ASSERT_EQ(count, 3);

    for(auto config : configs)
    {
        delete config;
    }

    configs.clear();

    ScopedDescriptor singleConfig(createDescriptorPtr<EngineConfigDescriptor>());

    count = 0;
    ASSERT_NO_THROW(heur->getAttribute(
        HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count, &singleConfig));
    ASSERT_EQ(count, 1);
}

TEST_F(TestEngineHeuristicDescriptor, GetEngineConfigsWithNullConfig)
{
    auto heur = getEngineHeuristicDescriptor();
    makeEngineHeuristicFinalized();

    EXPECT_CALL(*getMockGraph(), isFinalized()).WillRepeatedly(Return(true));
    EXPECT_CALL(*_mockEnginePluginResourceManager, getEngineDetails(_, _, _))
        .WillRepeatedly(
            Invoke([this](int64_t engineId, const GraphDescriptor*, hipdnnPluginConstData_t* d) {
                *d = this->serializeEngineDetails(engineId);
            }));
    EXPECT_CALL(*_mockEnginePluginResourceManager, destroyEngineDetails(_, _))
        .WillRepeatedly(Return());

    std::vector<hipdnnBackendDescriptor_t> configs(3);
    configs[0] = createDescriptorPtr<EngineConfigDescriptor>();
    configs[1] = nullptr;
    configs[2] = createDescriptorPtr<EngineConfigDescriptor>();

    EXPECT_CALL(*getMockGraph(), isFinalized()).WillRepeatedly(Return(true));
    int64_t count = 0;
    ASSERT_THROW_HIPDNN_STATUS(heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  3,
                                                  &count,
                                                  configs.data()),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    for(auto config : configs)
    {
        delete config;
    }
}

TEST_F(TestEngineHeuristicDescriptor, GetEngineConfigsWithNoEngineIds)
{
    auto heur = getEngineHeuristicDescriptor();
    setGraph();
    setHeuristicMode();

    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_))
        .WillRepeatedly(Return(std::vector<int64_t>{}));

    ASSERT_NO_THROW(heur->finalize());

    std::vector<hipdnnBackendDescriptor_t> configs(3);
    for(size_t i = 0; i < 3; ++i)
    {
        configs[i] = createDescriptorPtr<EngineConfigDescriptor>();
    }

    int64_t count = 0;
    ASSERT_NO_THROW(heur->getAttribute(
        HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 3, &count, configs.data()));
    ASSERT_EQ(count, 0);

    for(auto config : configs)
    {
        delete config;
    }
}

TEST_F(TestEngineHeuristicDescriptor, GetEngineConfigsRequestMoreThanAvailable)
{
    auto heur = getEngineHeuristicDescriptor();
    makeEngineHeuristicFinalized();

    EXPECT_CALL(*getMockGraph(), isFinalized()).WillRepeatedly(Return(true));
    EXPECT_CALL(*_mockEnginePluginResourceManager, getEngineDetails(_, _, _))
        .WillRepeatedly(
            Invoke([this](int64_t engineId, const GraphDescriptor*, hipdnnPluginConstData_t* d) {
                *d = this->serializeEngineDetails(engineId);
            }));
    EXPECT_CALL(*_mockEnginePluginResourceManager, destroyEngineDetails(_, _))
        .WillRepeatedly(Return());

    std::vector<hipdnnBackendDescriptor_t> configs(5);
    for(size_t i = 0; i < 3; ++i)
    {
        configs[i] = createDescriptorPtr<EngineConfigDescriptor>();
    }

    int64_t count = 0;
    ASSERT_NO_THROW(heur->getAttribute(
        HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 5, &count, configs.data()));
    ASSERT_EQ(count, 3);

    for(auto config : configs)
    {
        delete config;
    }
}

TEST_F(TestEngineHeuristicDescriptor, GetEngineConfigsCountOnly)
{
    auto heur = getEngineHeuristicDescriptor();
    makeEngineHeuristicFinalized();

    EXPECT_CALL(*getMockGraph(), isFinalized()).WillRepeatedly(Return(true));

    int64_t count = 0;
    ASSERT_NO_THROW(heur->getAttribute(
        HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr));
    ASSERT_EQ(count, 3);
}

TEST_F(TestEngineHeuristicDescriptor, GetEngineHeuristicDescriptorHeurMode)
{
    auto heur = getEngineHeuristicDescriptor();
    hipdnnBackendHeurMode_t mode = HIPDNN_HEUR_MODE_FALLBACK;

    makeEngineHeuristicFinalized();

    ASSERT_THROW_HIPDNN_STATUS(
        heur->getAttribute(
            HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, &mode),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 2, nullptr, &mode),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(
        heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, nullptr, &mode));
    ASSERT_EQ(mode, HIPDNN_HEUR_MODE_FALLBACK);

    int64_t count = 0;
    ASSERT_NO_THROW(
        heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, &count, &mode));
    ASSERT_EQ(count, 1);
    ASSERT_EQ(mode, HIPDNN_HEUR_MODE_FALLBACK);
}

TEST_F(TestEngineHeuristicDescriptor, GetGraphThrowsIfNotFinalized)
{
    auto heur = getEngineHeuristicDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(heur->getGraph(), HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestEngineHeuristicDescriptor, GetGraphReturnsPointerIfFinalized)
{
    auto heur = getEngineHeuristicDescriptor();
    makeEngineHeuristicFinalized();
    auto graphPtr = heur->getGraph();
    ASSERT_NE(graphPtr, nullptr);
    ASSERT_EQ(static_cast<const IBackendDescriptor*>(graphPtr.get()),
              static_cast<const IBackendDescriptor*>(getMockGraph().get()));
}
