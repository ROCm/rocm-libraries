// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "TestUtil.hpp"
#include "hipdnn_backend.h"
#include "test_plugins/TestPluginEngineIdMap.hpp"
#include <test_plugins/TestPluginConstants.hpp>

#include <gtest/gtest.h>

class IntegrationEngineApi : public ::testing::Test
{
protected:
    hipdnnBackendDescriptor_t _engine;
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
        EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, &_engine),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_NE(_engine, nullptr);
    }

    void TearDown() override
    {
        EXPECT_EQ(hipdnnBackendDestroyDescriptor(_engine), HIPDNN_STATUS_SUCCESS);
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
};

TEST_F(IntegrationEngineApi, SetEngineOperationGraph)
{
    EXPECT_EQ(hipdnnBackendSetAttribute(_engine,
                                        HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_graph),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    test_util::createTestGraph(&_graph, _handle);
    ASSERT_EQ(hipdnnBackendFinalize(_graph), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendSetAttribute(_engine,
                                        HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_graph),
              HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationEngineApi, SetEngineGlobalIndex)
{
    int64_t gidx = hipdnn_tests::plugin_constants::engineId<GoodPlugin>();

    EXPECT_EQ(hipdnnBackendSetAttribute(
                  _engine, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    EXPECT_EQ(hipdnnBackendSetAttribute(
                  _engine, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, &gidx),
              HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationEngineApi, SetAttrNotSupported)
{
    EXPECT_EQ(hipdnnBackendSetAttribute(
                  _engine, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, nullptr),
              HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(IntegrationEngineApi, SetAttrEngineAlreadyFinalized)
{
    int64_t gidx = hipdnn_tests::plugin_constants::engineId<GoodPlugin>();

    test_util::populateTestEngine(_engine, &_graph, _handle, gidx, true);
    EXPECT_EQ(hipdnnBackendSetAttribute(
                  _engine, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, &gidx),
              HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(IntegrationEngineApi, Finalize)
{
    int64_t gidx = hipdnn_tests::plugin_constants::engineId<GoodPlugin>();

    EXPECT_EQ(hipdnnBackendFinalize(_engine), HIPDNN_STATUS_BAD_PARAM);
    test_util::populateTestEngine(_engine, &_graph, _handle, gidx);
    EXPECT_EQ(hipdnnBackendFinalize(_engine), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationEngineApi, GetEngineOperationGraph)
{
    hipdnnBackendDescriptor_t graph = nullptr;
    int64_t gidx = hipdnn_tests::plugin_constants::engineId<GoodPlugin>();

    test_util::populateTestEngine(_engine, &_graph, _handle, gidx, true);
    EXPECT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        nullptr,
                                        &graph),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_NE(graph, nullptr);

    hipdnnBackendDestroyDescriptor(graph);
}

TEST_F(IntegrationEngineApi, GetEngineGlobalIndex)
{
    int64_t gidx = hipdnn_tests::plugin_constants::engineId<GoodPlugin>();
    int64_t gidxOut;

    test_util::populateTestEngine(_engine, &_graph, _handle, gidx, true);
    EXPECT_EQ(
        hipdnnBackendGetAttribute(
            _engine, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, &gidxOut),
        HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(gidxOut, gidx);
}
