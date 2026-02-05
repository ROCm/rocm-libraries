// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <mutex>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_data_sdk/logging/Logger.hpp>

// Test SDK logging works in plugin_sdk context using a custom callback

namespace
{
std::vector<std::pair<hipdnnSeverity_t, std::string>> s_capturedLogs; // NOLINT
std::mutex s_logMutex; // NOLINT

void testLoggingCallback(hipdnnSeverity_t severity, const char* msg)
{
    std::lock_guard<std::mutex> lock(s_logMutex);
    if(msg != nullptr)
    {
        s_capturedLogs.emplace_back(severity, msg);
    }
}

} // namespace

class TestSdkLogging : public ::testing::Test
{
protected:
    void SetUp() override
    {
        std::lock_guard<std::mutex> lock(s_logMutex);
        s_capturedLogs.clear();

        hipdnn_data_sdk::logging::resetLogging();
        hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_INFO);
        hipdnn_data_sdk::logging::registerLoggingCallback(testLoggingCallback);
    }

    void TearDown() override
    {
        hipdnn_data_sdk::logging::unregisterLoggingCallback();
        hipdnn_data_sdk::logging::resetLogging();
    }

    static std::vector<std::pair<hipdnnSeverity_t, std::string>> getCapturedLogs()
    {
        std::lock_guard<std::mutex> lock(s_logMutex);
        return s_capturedLogs;
    }
};

TEST_F(TestSdkLogging, SdkLogInfoMessageIsCorrectlyPassedToCallback)
{
    HIPDNN_SDK_LOG_INFO("Plugin SDK info test message");

    auto logs = getCapturedLogs();
    ASSERT_EQ(logs.size(), 1);
    EXPECT_EQ(logs[0].first, HIPDNN_SEV_INFO);
    EXPECT_NE(logs[0].second.find("Plugin SDK info test message"), std::string::npos);
}

TEST_F(TestSdkLogging, SdkLogContainsComponentName)
{
    HIPDNN_SDK_LOG_INFO("Component name check");

    auto logs = getCapturedLogs();
    ASSERT_EQ(logs.size(), 1);

    // SDK uses fixed "hipdnn_sdk" component name
    EXPECT_NE(logs[0].second.find("hipdnn_sdk"), std::string::npos)
        << "Log message did not contain expected component name.\n"
        << "Actual log: " << logs[0].second;
}

TEST_F(TestSdkLogging, SdkLogAllSeverityLevels)
{
    HIPDNN_SDK_LOG_INFO("SDK info message");
    HIPDNN_SDK_LOG_WARN("SDK warn message");
    HIPDNN_SDK_LOG_ERROR("SDK error message");
    HIPDNN_SDK_LOG_FATAL("SDK fatal message");

    auto logs = getCapturedLogs();
    ASSERT_EQ(logs.size(), 4);

    EXPECT_EQ(logs[0].first, HIPDNN_SEV_INFO);
    EXPECT_EQ(logs[1].first, HIPDNN_SEV_WARN);
    EXPECT_EQ(logs[2].first, HIPDNN_SEV_ERROR);
    EXPECT_EQ(logs[3].first, HIPDNN_SEV_FATAL);

    EXPECT_NE(logs[0].second.find("SDK info message"), std::string::npos);
    EXPECT_NE(logs[1].second.find("SDK warn message"), std::string::npos);
    EXPECT_NE(logs[2].second.find("SDK error message"), std::string::npos);
    EXPECT_NE(logs[3].second.find("SDK fatal message"), std::string::npos);
}

TEST_F(TestSdkLogging, SdkLogRespectsLogLevel)
{
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_ERROR);

    HIPDNN_SDK_LOG_INFO("Should not appear");
    HIPDNN_SDK_LOG_WARN("Should not appear");
    HIPDNN_SDK_LOG_ERROR("Error should appear");
    HIPDNN_SDK_LOG_FATAL("Fatal should appear");

    auto logs = getCapturedLogs();
    ASSERT_EQ(logs.size(), 2);
    EXPECT_NE(logs[0].second.find("Error should appear"), std::string::npos);
    EXPECT_NE(logs[1].second.find("Fatal should appear"), std::string::npos);
}
