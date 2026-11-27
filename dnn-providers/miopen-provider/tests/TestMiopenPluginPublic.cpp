// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "MiopenPluginDefines.hpp"
#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>

using namespace miopen_plugin;

TEST(TestMiopenPluginPublic, HipdnnPluginGetVersionSuccess)
{
    const char* version;
    auto status = hipdnnPluginGetVersion(&version);

    ASSERT_EQ(status, hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(version, HIPDNN_PLUGIN_VERSION);
}

TEST(TestMiopenPluginPublic, HipdnnPluginGetVersionNullPtr)
{
    auto status = hipdnnPluginGetVersion(nullptr);
    EXPECT_EQ(status, hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestMiopenPluginPublic, HipdnnPluginGetApiVersionSuccess)
{
    const char* version;
    auto status = hipdnnPluginGetApiVersion(&version);

    ASSERT_EQ(status, hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(version, HIPDNN_PLUGIN_API_VERSION);
}

TEST(TestMiopenPluginPublic, HipdnnPluginGetApiVersionNullPtr)
{
    auto status = hipdnnPluginGetApiVersion(nullptr);
    EXPECT_EQ(status, hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestMiopenPluginPublic, HipdnnPluginSetLogLevelSuccess)
{
    auto status = hipdnnPluginSetLogLevel(HIPDNN_SEV_INFO);
    ASSERT_EQ(status, hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_TRUE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO));
}

TEST(TestMiopenPluginPublic, HipdnnPluginSetLogLevelFiltersLowerSeverity)
{
    auto status = hipdnnPluginSetLogLevel(HIPDNN_SEV_WARN);
    ASSERT_EQ(status, hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_TRUE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_WARN));
    EXPECT_FALSE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO));
}

TEST(TestMiopenPluginPublic, ExecuteOpGraphWithOverridesReturnsNotApplicable)
{
    const int64_t uid = 1;
    const uint32_t rank = 4;
    const std::array<int64_t, 4> shape = {1, 3, 4, 4};
    const std::array<int64_t, 4> strides = {48, 16, 4, 1};
    const std::array<const int64_t*, 1> shapes = {shape.data()};
    const std::array<const int64_t*, 1> stridesPerTensor = {strides.data()};

    // Null handle and context: the entry must reject without touching either.
    auto status = hipdnnEnginePluginExecuteOpGraphWithOverrides(nullptr,
                                                                nullptr,
                                                                nullptr,
                                                                nullptr,
                                                                0,
                                                                1,
                                                                &uid,
                                                                &rank,
                                                                shapes.data(),
                                                                stridesPerTensor.data());
    EXPECT_EQ(status, hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE);

    const char* error = nullptr;
    hipdnnPluginGetLastErrorString(&error);
    ASSERT_NE(error, nullptr);
    EXPECT_THAT(error, ::testing::HasSubstr("does not support override-shape execution"));
}
