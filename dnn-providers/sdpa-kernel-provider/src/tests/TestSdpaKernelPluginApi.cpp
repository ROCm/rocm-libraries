// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "SdpaKernelEngine.hpp"

using namespace sdpa_kernel_provider;

TEST(TestSdpaKernelPluginApi, HipdnnEnginePluginCreate)
{
    hipdnnEnginePluginHandle_t handle = nullptr;

    ASSERT_EQ(hipdnnEnginePluginCreate(&handle),
              hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_SUCCESS);

    EXPECT_NE(handle, nullptr);
}

TEST(TestSdpaKernelPluginApi, HipdnnEnginePluginDestroy)
{
    hipdnnEnginePluginHandle_t handle = nullptr;

    ASSERT_EQ(hipdnnEnginePluginCreate(&handle),
              hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnEnginePluginDestroy(handle),
              hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestSdpaKernelPluginApi, HipdnnEnginePluginGetAllEngineIds)
{
    hipdnnEnginePluginHandle_t handle = nullptr;

    ASSERT_EQ(hipdnnEnginePluginCreate(&handle),
              hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_SUCCESS);

    EXPECT_NE(handle, nullptr);

    std::array<int64_t, 2> engineIds = {-1, -1};
    uint32_t numEngineIdsReturned = 0;

    auto status = hipdnnEnginePluginGetAllEngineIds(engineIds.data(), 2, &numEngineIdsReturned);

    EXPECT_EQ(status, hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(numEngineIdsReturned, 1);
    EXPECT_EQ(engineIds[0], SdpaKernelEngine::staticId());
}

TEST(TestSdpaKernelPluginApi, HipdnnEnginePluginGetApplicableEngineIds)
{
    hipdnnEnginePluginHandle_t handle = nullptr;

    ASSERT_EQ(hipdnnEnginePluginCreate(&handle),
              hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_SUCCESS);

    EXPECT_NE(handle, nullptr);

    std::array<int64_t, 2> engineIds = {-1, -1};
    uint32_t numEngineIdsReturned = 0;

    auto graph = hipdnn_test_sdk::utilities::createValidSdpaFpropGraph();
    auto graphBuffer = graph.Release();
    hipdnnPluginConstData_t opGraph = {graphBuffer.data(), graphBuffer.size()};

    auto status = hipdnnEnginePluginGetApplicableEngineIds(
        handle, &opGraph, engineIds.data(), 2, &numEngineIdsReturned);

    EXPECT_EQ(status, hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(numEngineIdsReturned, 1);
    EXPECT_EQ(engineIds[0], SdpaKernelEngine::staticId());
}
