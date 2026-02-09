// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <hipdnn_data_sdk/logging/Logger.hpp>

static std::vector<std::pair<hipdnnSeverity_t, std::string>> s_capturedLogs; //NOLINT
static std::mutex s_logMutex; //NOLINT

// Custom callback for testing.
void testLoggingCallback(hipdnnSeverity_t severity, const char* msg)
{
    std::lock_guard<std::mutex> lock(s_logMutex);
    if(msg != nullptr)
    {
        s_capturedLogs.emplace_back(severity, msg);
    }
}

class TestCallbackLogger : public ::testing::Test
{
protected:
    void SetUp() override
    {
        std::lock_guard<std::mutex> lock(s_logMutex);
        s_capturedLogs.clear();

        // Reset logging and register our test callback
        hipdnn_data_sdk::logging::resetLogging();
        hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_INFO);
        hipdnn_data_sdk::logging::registerLoggingCallback(testLoggingCallback);
    }

    void TearDown() override
    {
        hipdnn_data_sdk::logging::resetLogging();
    }

    static std::vector<std::pair<hipdnnSeverity_t, std::string>> getCapturedLogs()
    {
        std::lock_guard<std::mutex> lock(s_logMutex);
        return s_capturedLogs;
    }
};

TEST_F(TestCallbackLogger, InfoMessageIsCorrectlyPassedToCallback)
{
    std::string testMessage = "Test info message";
    HIPDNN_SDK_LOG_INFO(testMessage);

    auto logs = getCapturedLogs();
    ASSERT_EQ(logs.size(), 1);
    EXPECT_EQ(logs[0].first, HIPDNN_SEV_INFO);
    EXPECT_NE(logs[0].second.find(testMessage), std::string::npos);
}

TEST_F(TestCallbackLogger, WarnMessageIsCorrectlyPassedToCallback)
{
    std::string testMessage = "Test warning message";
    HIPDNN_SDK_LOG_WARN(testMessage);

    auto logs = getCapturedLogs();
    ASSERT_EQ(logs.size(), 1);
    EXPECT_EQ(logs[0].first, HIPDNN_SEV_WARN);
    EXPECT_NE(logs[0].second.find(testMessage), std::string::npos);
}

TEST_F(TestCallbackLogger, ErrorMessageIsCorrectlyPassedToCallback)
{
    std::string testMessage = "Test error message";
    HIPDNN_SDK_LOG_ERROR(testMessage);

    auto logs = getCapturedLogs();
    ASSERT_EQ(logs.size(), 1);
    EXPECT_EQ(logs[0].first, HIPDNN_SEV_ERROR);
    EXPECT_NE(logs[0].second.find(testMessage), std::string::npos);
}

TEST_F(TestCallbackLogger, StreamFormattedMessagesAreCorrectlyPassed)
{
    int value = 42;
    std::string text = "formatted";
    HIPDNN_SDK_LOG_INFO("Test " << text << " message with value " << value);

    auto logs = getCapturedLogs();
    ASSERT_EQ(logs.size(), 1);
    EXPECT_NE(logs[0].second.find("Test formatted message with value 42"), std::string::npos);
}

TEST_F(TestCallbackLogger, LogLevelsAreRespected)
{
    // Set level to error so info and warn should be ignored
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_ERROR);

    HIPDNN_SDK_LOG_INFO("This info should not appear");
    HIPDNN_SDK_LOG_WARN("This warning should not appear");
    HIPDNN_SDK_LOG_ERROR("This error should appear");

    auto logs = getCapturedLogs();
    ASSERT_EQ(logs.size(), 1);
    EXPECT_NE(logs[0].second.find("This error should appear"), std::string::npos);
}

TEST_F(TestCallbackLogger, MultipleMessagesAreLogged)
{
    HIPDNN_SDK_LOG_INFO("First message");
    HIPDNN_SDK_LOG_WARN("Second message");
    HIPDNN_SDK_LOG_ERROR("Third message");

    auto logs = getCapturedLogs();
    ASSERT_EQ(logs.size(), 3);
    EXPECT_NE(logs[0].second.find("First message"), std::string::npos);
    EXPECT_NE(logs[1].second.find("Second message"), std::string::npos);
    EXPECT_NE(logs[2].second.find("Third message"), std::string::npos);
}

TEST_F(TestCallbackLogger, MessageContainsComponentName)
{
    std::string testMessage = "Component check";
    HIPDNN_SDK_LOG_INFO(testMessage);

    auto logs = getCapturedLogs();
    ASSERT_EQ(logs.size(), 1);

    // The message should contain the SDK component name
    std::string componentName = hipdnn_data_sdk::logging::K_COMPONENT_NAME;
    EXPECT_NE(logs[0].second.find(componentName), std::string::npos)
        << "Log message did not contain component name.\n"
        << "Expected to find: " << componentName << "\n"
        << "Actual log: " << logs[0].second;
}

TEST_F(TestCallbackLogger, ParamsAreNotExpandedIfLogLevelIsDisabled)
{
    bool wasCalledForInfo = false;
    bool wasCalledForWarn = false;
    bool wasCalledForError = false;
    bool wasCalledForFatal = false;
    std::string infoMessage("info log message");
    std::string warnMessage("warn log message");
    std::string errorMessage("error log message");
    std::string fatalMessage("fatal log message");
    auto trackingLambda = [](bool& wasCalled, std::string message) {
        wasCalled = true;
        return message;
    };

    // Set level to error so info and warn should be ignored
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_ERROR);

    HIPDNN_SDK_LOG_INFO(trackingLambda(wasCalledForInfo, infoMessage));
    HIPDNN_SDK_LOG_WARN(trackingLambda(wasCalledForWarn, warnMessage));
    HIPDNN_SDK_LOG_ERROR(trackingLambda(wasCalledForError, errorMessage));
    HIPDNN_SDK_LOG_FATAL(trackingLambda(wasCalledForFatal, fatalMessage));

    auto logs = getCapturedLogs();
    ASSERT_EQ(logs.size(), 2);

    // Info and warn lambdas should not have been called due to lazy evaluation
    EXPECT_FALSE(wasCalledForInfo) << "Info lambda should not have been called";
    EXPECT_FALSE(wasCalledForWarn) << "Warn lambda should not have been called";
    EXPECT_TRUE(wasCalledForError) << "Error lambda should have been called";
    EXPECT_TRUE(wasCalledForFatal) << "Fatal lambda should have been called";

    // Check that the logged messages are the error and fatal ones
    EXPECT_NE(logs[0].second.find(errorMessage), std::string::npos);
    EXPECT_NE(logs[1].second.find(fatalMessage), std::string::npos);
}

TEST_F(TestCallbackLogger, LogLevelOffDisablesAllLogs)
{
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_OFF);

    HIPDNN_SDK_LOG_INFO("Info");
    HIPDNN_SDK_LOG_WARN("Warn");
    HIPDNN_SDK_LOG_ERROR("Error");
    HIPDNN_SDK_LOG_FATAL("Fatal");

    auto logs = getCapturedLogs();
    ASSERT_EQ(logs.size(), 0) << "No logs should be captured when log level is OFF";
}

TEST_F(TestCallbackLogger, SeverityLevelsAreCorrect)
{
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_INFO);

    HIPDNN_SDK_LOG_INFO("Info message");
    HIPDNN_SDK_LOG_WARN("Warn message");
    HIPDNN_SDK_LOG_ERROR("Error message");
    HIPDNN_SDK_LOG_FATAL("Fatal message");

    auto logs = getCapturedLogs();
    ASSERT_EQ(logs.size(), 4);

    EXPECT_EQ(logs[0].first, HIPDNN_SEV_INFO);
    EXPECT_EQ(logs[1].first, HIPDNN_SEV_WARN);
    EXPECT_EQ(logs[2].first, HIPDNN_SEV_ERROR);
    EXPECT_EQ(logs[3].first, HIPDNN_SEV_FATAL);
}
