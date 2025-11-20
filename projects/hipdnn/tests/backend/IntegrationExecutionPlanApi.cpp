// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "TestUtil.hpp"
#include "hipdnn_backend.h"
#include <test_plugins/TestPluginConstants.hpp>

#include <gtest/gtest.h>

class IntegrationExecutionPlanApi : public ::testing::Test
{
protected:
    static constexpr int64_t GIDX = hipdnn_tests::plugin_constants::engineId<GoodPlugin>();
    hipdnnBackendDescriptor_t _plan;
    hipdnnHandle_t _handle = nullptr;
    hipdnnBackendDescriptor_t _engineConfig = nullptr;
    hipdnnBackendDescriptor_t _engine = nullptr;
    hipdnnBackendDescriptor_t _graph = nullptr;

    void SetUp() override
    {
        const std::array<const char*, 1> paths
            = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &_plan),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_NE(_plan, nullptr);
    }

    void TearDown() override
    {
        destroyTestDescriptor(_plan);
        destroyTestHandle();
        destroyTestDescriptor(_engineConfig);
        destroyTestDescriptor(_engine);
        destroyTestDescriptor(_graph);
    }

private:
    void destroyTestHandle()
    {
        if(_handle != nullptr)
        {
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
            _handle = nullptr;
        }
    }

    static void destroyTestDescriptor(hipdnnBackendDescriptor_t descriptor)
    {
        if(descriptor != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(descriptor), HIPDNN_STATUS_SUCCESS);
            descriptor = nullptr;
        }
    }
};

TEST_F(IntegrationExecutionPlanApi, SetExecutionPlanHandle)
{
    EXPECT_EQ(hipdnnBackendSetAttribute(
                  _plan, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    EXPECT_EQ(hipdnnBackendSetAttribute(
                  _plan, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &_handle),
              HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationExecutionPlanApi, SetExecutionPlanEngineConfig)
{
    EXPECT_EQ(hipdnnBackendSetAttribute(_plan,
                                        HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    test_util::createTestEngineConfig(&_engineConfig, &_engine, &_graph, _handle, GIDX, true);

    EXPECT_EQ(hipdnnBackendSetAttribute(_plan,
                                        HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationExecutionPlanApi, SetExecutionPlanAttrNotSupported)
{
    EXPECT_EQ(hipdnnBackendSetAttribute(
                  _plan, HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr),
              HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(IntegrationExecutionPlanApi, GetExecutionPlanWorkSpaceSize)
{
    size_t size = 0;
    EXPECT_EQ(
        hipdnnBackendGetAttribute(
            _plan, HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, &size),
        HIPDNN_STATUS_NOT_INITIALIZED);

    test_util::populateTestExecutionPlan(
        &_plan, &_engineConfig, &_engine, &_graph, _handle, GIDX, true);
    EXPECT_EQ(
        hipdnnBackendGetAttribute(
            _plan, HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, &size),
        HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(size, 2048);
}

TEST_F(IntegrationExecutionPlanApi, Finalize)
{
    EXPECT_EQ(hipdnnBackendFinalize(_plan), HIPDNN_STATUS_BAD_PARAM);

    ASSERT_EQ(hipdnnBackendSetAttribute(
                  _plan, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &_handle),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendFinalize(_plan), HIPDNN_STATUS_BAD_PARAM);

    test_util::populateTestExecutionPlan(&_plan, &_engineConfig, &_engine, &_graph, _handle, GIDX);
    EXPECT_EQ(hipdnnBackendFinalize(_plan), HIPDNN_STATUS_SUCCESS);
}
