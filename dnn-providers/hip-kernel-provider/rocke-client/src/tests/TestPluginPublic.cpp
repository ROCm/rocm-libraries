// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "version.h"

#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/PluginApi.h>

TEST(TestPluginPublic, GetNameReturnsRockeClient)
{
    const char* name = nullptr;

    auto status = hipdnnPluginGetName(&name);

    ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(name, "rocke-client");
}

TEST(TestPluginPublic, GetVersionReturnsGeneratedVersion)
{
    const char* version = nullptr;

    auto status = hipdnnPluginGetVersion(&version);

    ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(version, ROCKE_CLIENT_VERSION_STRING);
}

TEST(TestPluginPublic, GetApiVersionReturnsBaselineEngineApi)
{
    const char* version = nullptr;

    auto status = hipdnnPluginGetApiVersion(&version);

    ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(version, "1.0.0");
}

TEST(TestPluginPublic, GetTypeReturnsEnginePlugin)
{
    hipdnnPluginType_t type = HIPDNN_PLUGIN_TYPE_UNSPECIFIED;

    auto status = hipdnnPluginGetType(&type);

    ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(type, HIPDNN_PLUGIN_TYPE_ENGINE);
}

TEST(TestPluginPublic, NullMetadataPointersReturnBadParam)
{
    EXPECT_EQ(hipdnnPluginGetName(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(hipdnnPluginGetVersion(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(hipdnnPluginGetApiVersion(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(hipdnnPluginGetType(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestPluginPublic, GetAllEngineIdsReportsSingleRockeClientEngine)
{
    uint32_t numEngines = 0;

    auto status = hipdnnEnginePluginGetAllEngineIds(nullptr, 0, &numEngines);

    ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(numEngines, 1u);

    int64_t engineId = 0;
    status = hipdnnEnginePluginGetAllEngineIds(&engineId, 1, &numEngines);

    ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(numEngines, 1u);
    EXPECT_EQ(engineId, hipdnn_data_sdk::utilities::ROCKE_ENGINE_ID);
}

TEST(TestPluginPublic, ApplicableEngineQueryRejectsGraph)
{
    hipdnnEnginePluginHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    const hipdnnPluginConstData_t invalidGraph{nullptr, 0};
    uint32_t numEngines = 1;

    const auto status
        = hipdnnEnginePluginGetApplicableEngineIds(handle, &invalidGraph, nullptr, 0, &numEngines);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(numEngines, 0u);
    EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestPluginPublic, GetEngineDetailsRoundTripReleasesBuffer)
{
    hipdnnEnginePluginHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    const hipdnnPluginConstData_t opGraph{nullptr, 0};
    hipdnnPluginConstData_t details{nullptr, 0};

    ASSERT_EQ(hipdnnEnginePluginGetEngineDetails(
                  handle, hipdnn_data_sdk::utilities::ROCKE_ENGINE_ID, &opGraph, &details),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(details.ptr, nullptr);
    EXPECT_GT(details.size, 0u);

    EXPECT_EQ(hipdnnEnginePluginDestroyEngineDetails(handle, &details),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestPluginPublic, SetLogLevelSuccess)
{
    auto status = hipdnnPluginSetLogLevel(HIPDNN_SEV_INFO);

    ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_TRUE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO));
}
