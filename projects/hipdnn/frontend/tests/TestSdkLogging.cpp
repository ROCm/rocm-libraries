// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_frontend/Utilities.hpp>

class TestSdkLogging : public ::testing::Test
{
protected:
    // Thread-safe log message storage
    static std::mutex sLogMutex;
    static std::vector<std::pair<hipdnnSeverity_t, std::string>> sLogMessages;

    static void testLogCallback(hipdnnSeverity_t severity, const char* message)
    {
        std::lock_guard<std::mutex> lock(sLogMutex);
        sLogMessages.emplace_back(severity, std::string(message));
    }

public:
    void SetUp() override
    {
        // Reset logging state and clear messages
        hipdnn_data_sdk::logging::resetLogging();
        {
            std::lock_guard<std::mutex> lock(sLogMutex);
            sLogMessages.clear();
        }

        // Register our test callback
        hipdnn_data_sdk::logging::registerLoggingCallback(testLogCallback);

        // Start with logging off - tests will enable as needed
        hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_OFF);
    }

    void TearDown() override
    {
        hipdnn_data_sdk::logging::unregisterLoggingCallback();
        hipdnn_data_sdk::logging::resetLogging();
    }

    static std::string getLogContent()
    {
        std::lock_guard<std::mutex> lock(sLogMutex);
        std::ostringstream oss;
        for(const auto& [severity, message] : sLogMessages)
        {
            oss << message << "\n";
        }
        return oss.str();
    }

    static bool hasLogMessage(const std::string& substring)
    {
        std::lock_guard<std::mutex> lock(sLogMutex);
        for(const auto& [severity, message] : sLogMessages)
        {
            if(message.find(substring) != std::string::npos)
            {
                return true;
            }
        }
        return false;
    }
};

// Static member definitions
std::mutex TestSdkLogging::sLogMutex;
std::vector<std::pair<hipdnnSeverity_t, std::string>> TestSdkLogging::sLogMessages;

TEST_F(TestSdkLogging, SdkLogInfoMessageIsCorrectlyPassedToCallback)
{
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_INFO);

    // Initialize frontend logging via component-specific macro
    HIPDNN_FE_LOG_INFO("Frontend initialized");

    // Now SDK logging should work
    HIPDNN_SDK_LOG_INFO("SDK info test message");

    std::string logContent = getLogContent();

    EXPECT_THAT(logContent, ::testing::HasSubstr("Frontend initialized"));
    EXPECT_THAT(logContent, ::testing::HasSubstr("SDK info test message"));
}

TEST_F(TestSdkLogging, SdkLogContainsComponentName)
{
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_INFO);

    HIPDNN_FE_LOG_INFO("Frontend initialized");
    HIPDNN_SDK_LOG_INFO("Component name check");

    std::string logContent = getLogContent();

    // SDK logs should contain the "hipdnn_frontend" component name
    EXPECT_THAT(logContent, ::testing::HasSubstr("hipdnn_frontend"))
        << "Log message did not contain expected component name.\n"
        << "Actual log: " << logContent;
}

TEST_F(TestSdkLogging, SdkLogAllSeverityLevels)
{
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_INFO);

    HIPDNN_FE_LOG_INFO("Frontend initialized");

    HIPDNN_SDK_LOG_INFO("SDK info message");
    HIPDNN_SDK_LOG_WARN("SDK warn message");
    HIPDNN_SDK_LOG_ERROR("SDK error message");
    HIPDNN_SDK_LOG_FATAL("SDK fatal message");

    std::string logContent = getLogContent();

    EXPECT_THAT(logContent, ::testing::HasSubstr("SDK info message"));
    EXPECT_THAT(logContent, ::testing::HasSubstr("SDK warn message"));
    EXPECT_THAT(logContent, ::testing::HasSubstr("SDK error message"));
    EXPECT_THAT(logContent, ::testing::HasSubstr("SDK fatal message"));
}

TEST_F(TestSdkLogging, SdkLogRespectsLogLevel)
{
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_ERROR);

    HIPDNN_FE_LOG_ERROR("Frontend initialized");

    HIPDNN_SDK_LOG_INFO("Should not appear");
    HIPDNN_SDK_LOG_WARN("Should not appear");
    HIPDNN_SDK_LOG_ERROR("Error should appear");
    HIPDNN_SDK_LOG_FATAL("Fatal should appear");

    std::string logContent = getLogContent();

    EXPECT_THAT(logContent, ::testing::Not(::testing::HasSubstr("Should not appear")));
    EXPECT_THAT(logContent, ::testing::HasSubstr("Error should appear"));
    EXPECT_THAT(logContent, ::testing::HasSubstr("Fatal should appear"));
}

TEST_F(TestSdkLogging, SdkLogStreamFormatting)
{
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_INFO);

    HIPDNN_FE_LOG_INFO("Frontend initialized");

    int value = 42;
    std::string text = "formatted";
    HIPDNN_SDK_LOG_INFO("SDK " << text << " message with value " << value);

    std::string logContent = getLogContent();

    EXPECT_THAT(logContent, ::testing::HasSubstr("SDK formatted message with value 42"));
}
