// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "TestUtil.hpp"
#include "hipdnn_backend.h"
#include <test_plugins/TestPluginConstants.hpp>

#include <gtest/gtest.h>

#include <vector>

// TODO - Once PluginManager is implemented, we will need to add a testing plugin for this API.
namespace
{
constexpr hipdnnBackendHeurMode_t FALLBACK_MODE = HIPDNN_HEUR_MODE_FALLBACK;
}

class IntegrationEngineHeuristicApi : public ::testing::Test
{
protected:
    hipdnnBackendDescriptor_t _engineHeuristic = nullptr;
    hipdnnBackendDescriptor_t _graph = nullptr;
    hipdnnHandle_t _handle = nullptr;

    void SetUp() override
    {
        const std::array<const char*, 1> paths
            = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(
            hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &_engineHeuristic),
            HIPDNN_STATUS_SUCCESS);
        ASSERT_NE(_engineHeuristic, nullptr);
    }

    void TearDown() override
    {
        if(_engineHeuristic != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(_engineHeuristic), HIPDNN_STATUS_SUCCESS);
        }
        if(_graph != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(_graph), HIPDNN_STATUS_SUCCESS);
        }
        if(_handle != nullptr)
        {
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
            _handle = nullptr;
        }
    }

    void setHeuristicMode()
    {
        ASSERT_EQ(hipdnnBackendSetAttribute(_engineHeuristic,
                                            HIPDNN_ATTR_ENGINEHEUR_MODE,
                                            HIPDNN_TYPE_HEUR_MODE,
                                            1,
                                            &FALLBACK_MODE),
                  HIPDNN_STATUS_SUCCESS);
    }

    void setOperationGraph()
    {
        test_util::createTestGraph(&_graph, _handle);
        ASSERT_EQ(hipdnnBackendFinalize(_graph), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnBackendSetAttribute(_engineHeuristic,
                                            HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                            1,
                                            &_graph),
                  HIPDNN_STATUS_SUCCESS);
    }

    void populateEngineHeuristic(bool finalize = true)
    {
        setOperationGraph();
        setHeuristicMode();

        if(finalize)
        {
            ASSERT_EQ(hipdnnBackendFinalize(_engineHeuristic), HIPDNN_STATUS_SUCCESS);
        }
    }
};

TEST_F(IntegrationEngineHeuristicApi, SetEngineHeuristicOperationGraph)
{
    hipdnnBackendDescriptor_t nullGraph = nullptr;
    EXPECT_EQ(hipdnnBackendSetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &nullGraph),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    test_util::createTestGraph(&_graph, _handle);
    ASSERT_EQ(hipdnnBackendFinalize(_graph), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendSetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_graph),
              HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationEngineHeuristicApi, SetEngineHeuristicMode)
{
    EXPECT_EQ(hipdnnBackendSetAttribute(
                  _engineHeuristic, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_MODE,
                                        HIPDNN_TYPE_HEUR_MODE,
                                        2,
                                        &FALLBACK_MODE),
              HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_MODE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &FALLBACK_MODE),
              HIPDNN_STATUS_BAD_PARAM);

    auto unsupportedMode = static_cast<hipdnnBackendHeurMode_t>(999);
    EXPECT_EQ(hipdnnBackendSetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_MODE,
                                        HIPDNN_TYPE_HEUR_MODE,
                                        1,
                                        &unsupportedMode),
              HIPDNN_STATUS_NOT_SUPPORTED);

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_MODE,
                                        HIPDNN_TYPE_HEUR_MODE,
                                        1,
                                        &FALLBACK_MODE),
              HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationEngineHeuristicApi, SetUnsupportedAttribute)
{
    int32_t dummy = 0;
    EXPECT_EQ(hipdnnBackendSetAttribute(
                  _engineHeuristic, HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_INT32, 1, &dummy),
              HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(IntegrationEngineHeuristicApi, Finalize)
{
    EXPECT_EQ(hipdnnBackendFinalize(_engineHeuristic), HIPDNN_STATUS_BAD_PARAM);

    setOperationGraph();
    EXPECT_EQ(hipdnnBackendFinalize(_engineHeuristic), HIPDNN_STATUS_BAD_PARAM);

    setHeuristicMode();
    EXPECT_EQ(hipdnnBackendFinalize(_engineHeuristic), HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(_engineHeuristic), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(IntegrationEngineHeuristicApi, SetAttributeOnFinalizedDescriptor)
{
    populateEngineHeuristic(true);

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_MODE,
                                        HIPDNN_TYPE_HEUR_MODE,
                                        1,
                                        &FALLBACK_MODE),
              HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(IntegrationEngineHeuristicApi, GetAttributeOnUnfinalizedDescriptor)
{
    hipdnnBackendDescriptor_t dummyGraph = nullptr;

    EXPECT_EQ(hipdnnBackendGetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        nullptr,
                                        &dummyGraph),
              HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(IntegrationEngineHeuristicApi, GetEngineHeuristicOperationGraph)
{
    populateEngineHeuristic(true);

    hipdnnBackendDescriptor_t retrievedGraph = nullptr;
    hipdnnBackendDescriptor_t retrievedGraph2 = nullptr;

    EXPECT_EQ(hipdnnBackendGetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                        HIPDNN_TYPE_INT64,
                                        1,
                                        nullptr,
                                        &retrievedGraph),
              HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnBackendGetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        2,
                                        nullptr,
                                        &retrievedGraph),
              HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnBackendGetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        nullptr,
                                        nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    EXPECT_EQ(hipdnnBackendGetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        nullptr,
                                        &retrievedGraph),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_NE(retrievedGraph, nullptr);

    hipdnnBackendDestroyDescriptor(retrievedGraph);
    retrievedGraph = nullptr;

    int64_t count = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &count,
                                        &retrievedGraph2),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 1);

    hipdnnBackendDestroyDescriptor(retrievedGraph2);
    retrievedGraph2 = nullptr;
}

TEST_F(IntegrationEngineHeuristicApi, GetEngineHeuristicMode)
{
    populateEngineHeuristic(true);

    hipdnnBackendHeurMode_t mode = HIPDNN_HEUR_MODE_FALLBACK;

    EXPECT_EQ(hipdnnBackendGetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_MODE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        nullptr,
                                        &mode),
              HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnBackendGetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_MODE,
                                        HIPDNN_TYPE_HEUR_MODE,
                                        2,
                                        nullptr,
                                        &mode),
              HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnBackendGetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_MODE,
                                        HIPDNN_TYPE_HEUR_MODE,
                                        1,
                                        nullptr,
                                        nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    EXPECT_EQ(hipdnnBackendGetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_MODE,
                                        HIPDNN_TYPE_HEUR_MODE,
                                        1,
                                        nullptr,
                                        &mode),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(mode, HIPDNN_HEUR_MODE_FALLBACK);

    int64_t count = 0;
    EXPECT_EQ(
        hipdnnBackendGetAttribute(
            _engineHeuristic, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, &count, &mode),
        HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 1);
    EXPECT_EQ(mode, HIPDNN_HEUR_MODE_FALLBACK);
}

TEST_F(IntegrationEngineHeuristicApi, GetUnsupportedAttribute)
{
    populateEngineHeuristic(true);

    hipdnnBackendHeurMode_t dummy;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINE_KNOB_INFO,
                                        HIPDNN_TYPE_HEUR_MODE,
                                        1,
                                        nullptr,
                                        &dummy),
              HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(IntegrationEngineHeuristicApi, GetEngineConfigsCountOnly)
{
    populateEngineHeuristic(true);

    int64_t count = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_INT64,
                                        0,
                                        &count,
                                        nullptr),
              HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnBackendGetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &count,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 1);
}

TEST_F(IntegrationEngineHeuristicApi, GetEngineConfigs)
{
    populateEngineHeuristic(true);

    std::vector<hipdnnBackendDescriptor_t> configs(3);
    EXPECT_EQ(hipdnnBackendGetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        3,
                                        nullptr,
                                        configs.data()),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    int64_t count = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &count,
                                        nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    EXPECT_EQ(hipdnnBackendGetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &count,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 1);

    for(size_t i = 0; i < static_cast<size_t>(count); ++i)
    {
        EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &configs[i]),
                  HIPDNN_STATUS_SUCCESS);
    }

    count = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        3, // Ask for 3, but only one engine avaliable.
                                        &count,
                                        configs.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 1); // Only one returned since there isn't more than that.

    EXPECT_EQ(hipdnnBackendFinalize(configs[0]), HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDescriptor_t engine = nullptr;
    EXPECT_EQ(hipdnnBackendGetAttribute(configs[0],
                                        HIPDNN_ATTR_ENGINECFG_ENGINE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        nullptr,
                                        &engine),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(engine, nullptr);

    int64_t engineId = 0;
    EXPECT_EQ(
        hipdnnBackendGetAttribute(
            engine, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, &engineId),
        HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(engineId, hipdnn_tests::plugin_constants::engineId<GoodPlugin>());

    // Expecting to only need to clean-up 1 engine config, since we only created & requested 1.
    for(auto config : configs)
    {
        if(config != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(config), HIPDNN_STATUS_SUCCESS);
        }
    }

    EXPECT_EQ(hipdnnBackendDestroyDescriptor(engine), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationEngineHeuristicApi, GetEngineConfigsRequestMoreThanAvailable)
{
    populateEngineHeuristic(true);

    std::vector<hipdnnBackendDescriptor_t> configs(5);
    for(size_t i = 0; i < 5; ++i)
    {
        EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &configs[i]),
                  HIPDNN_STATUS_SUCCESS);
    }

    int64_t count = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engineHeuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        5,
                                        &count,
                                        configs.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 1);

    for(auto& config : configs)
    {
        EXPECT_EQ(hipdnnBackendDestroyDescriptor(config), HIPDNN_STATUS_SUCCESS);
    }
}
