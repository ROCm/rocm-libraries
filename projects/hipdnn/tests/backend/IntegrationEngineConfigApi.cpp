// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "TestUtil.hpp"
#include "hipdnn_backend.h"
#include <test_plugins/TestPluginConstants.hpp>

#include <gtest/gtest.h>

class IntegrationEngineConfigApi : public ::testing::Test
{
protected:
    hipdnnBackendDescriptor_t _engineConfig;
    hipdnnBackendDescriptor_t _engine = nullptr;
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
            hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
            HIPDNN_STATUS_SUCCESS);
        ASSERT_NE(_engineConfig, nullptr);
    }

    void TearDown() override
    {
        EXPECT_EQ(hipdnnBackendDestroyDescriptor(_engineConfig), HIPDNN_STATUS_SUCCESS);
        if(_engine != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(_engine), HIPDNN_STATUS_SUCCESS);
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
};

TEST_F(IntegrationEngineConfigApi, SetEngineConfigEngine)
{
    int64_t gidx = hipdnn_tests::plugin_constants::engineId<GoodPlugin>();

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_ENGINECFG_ENGINE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_engine),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    test_util::createTestEngine(&_engine, &_graph, _handle, gidx, true);
    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_ENGINECFG_ENGINE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_engine),
              HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationEngineConfigApi, Finalize)
{
    int64_t gidx = hipdnn_tests::plugin_constants::engineId<GoodPlugin>();

    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_BAD_PARAM);
    test_util::populateTestEngineConfig(&_engineConfig, &_engine, &_graph, _handle, gidx);
    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationEngineConfigApi, GetMaxWorkspaceSize)
{
    int64_t gidx = hipdnn_tests::plugin_constants::engineId<GoodPlugin>();
    int64_t maxWorkspaceSize = 0;

    test_util::populateTestEngineConfig(&_engineConfig, &_engine, &_graph, _handle, gidx, true);
    EXPECT_EQ(hipdnnBackendGetAttribute(_engineConfig,
                                        HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE,
                                        HIPDNN_TYPE_INT64,
                                        1,
                                        nullptr,
                                        &maxWorkspaceSize),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(maxWorkspaceSize, 1024);
}
