// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <mutex>
#include <regex>
#include <spdlog/spdlog.h>
#include <string>
#include <thread>
#include <vector>

#include <hipdnn_sdk/logging/CallbackTypes.h>
#include <hipdnn_sdk/logging/Logger.hpp>

static std::vector<std::string> s_capturedLogs; //NOLINT
static std::mutex s_logMutex; //NOLINT

// Custom callback for testing. It doesn't fully simulate the real logging behavior,
// but the backend tests use the true callback function. The test could use the backend callback, but then it has to link against the backend.
void testLoggingCallback([[maybe_unused]] hipdnnSeverity_t severity, const char* msg)
{
    std::lock_guard<std::mutex> lock(s_logMutex);
    if(msg != nullptr)
    {
        s_capturedLogs.emplace_back(msg);
    }
}

class TestCallbackLogger : public ::testing::Test
{
protected:
    const std::string _testLoggerName = COMPONENT_NAME;

    void SetUp() override
    {
        s_capturedLogs.clear();

        spdlog::drop_all();

        hipdnn::logging::initializeCallbackLogging(_testLoggerName, testLoggingCallback);

        auto testLogger = spdlog::get(_testLoggerName);
        ASSERT_NE(testLogger, nullptr);
        testLogger->set_level(spdlog::level::trace);
    }

    void TearDown() override
    {
        spdlog::shutdown();
    }

    static std::vector<std::string> getCapturedLogs()
    {
        spdlog::shutdown(); // block until async queue is fully processed
        return s_capturedLogs;
    }
};

TEST_F(TestCallbackLogger, InfoMessageIsCorrectlyPassedToCallback)
{
    std::string testMessage = "Test info message";
    HIPDNN_LOG_INFO(testMessage);

    auto logs = getCapturedLogs();
    ASSERT_EQ(logs.size(), 1);
    EXPECT_NE(logs[0].find(testMessage), std::string::npos);
}

TEST_F(TestCallbackLogger, WarnMessageIsCorrectlyPassedToCallback)
{
    std::string testMessage = "Test warning message";
    HIPDNN_LOG_WARN(testMessage);

    auto logs = getCapturedLogs();
    ASSERT_EQ(logs.size(), 1);
    EXPECT_NE(logs[0].find(testMessage), std::string::npos);
}

TEST_F(TestCallbackLogger, ErrorMessageIsCorrectlyPassedToCallback)
{
    std::string testMessage = "Test error message";
    HIPDNN_LOG_ERROR(testMessage);

    auto logs = getCapturedLogs();
    ASSERT_EQ(logs.size(), 1);
    EXPECT_NE(logs[0].find(testMessage), std::string::npos);
}

TEST_F(TestCallbackLogger, FormattedMessagesAreCorrectlyPassed)
{
    int value = 42;
    std::string text = "formatted";
    HIPDNN_LOG_INFO("Test {} message with value {}", text, value);

    auto logs = getCapturedLogs();
    ASSERT_EQ(logs.size(), 1);
    EXPECT_NE(logs[0].find("Test formatted message with value 42"), std::string::npos);
}

TEST_F(TestCallbackLogger, LogLevelsAreRespected)
{
    auto testLogger = spdlog::get(_testLoggerName);
    ASSERT_NE(testLogger, nullptr);

    // Set level to error so info and warn should be ignored
    testLogger->set_level(spdlog::level::err);

    HIPDNN_LOG_INFO("This info should not appear");
    HIPDNN_LOG_WARN("This warning should not appear");
    HIPDNN_LOG_ERROR("This error should appear");

    auto logs = getCapturedLogs();
    ASSERT_EQ(logs.size(), 1);
    EXPECT_NE(logs[0].find("This error should appear"), std::string::npos);
}

TEST_F(TestCallbackLogger, MultipleMessagesAreLogged)
{
    HIPDNN_LOG_INFO("First message");
    HIPDNN_LOG_WARN("Second message");
    HIPDNN_LOG_ERROR("Third message");

    auto logs = getCapturedLogs();
    ASSERT_EQ(logs.size(), 3);
    EXPECT_NE(logs[0].find("First message"), std::string::npos);
    EXPECT_NE(logs[1].find("Second message"), std::string::npos);
    EXPECT_NE(logs[2].find("Third message"), std::string::npos);
}

TEST_F(TestCallbackLogger, CallbackReceivesFormattedPattern)
{
    std::string testMessage = "Pattern check";
    HIPDNN_LOG_INFO(testMessage);

    auto logs = getCapturedLogs();
    ASSERT_EQ(logs.size(), 1);

    std::regex pattern(R"(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[tid \d+\] \[info\] \[)"
                       + _testLoggerName + R"(\] )" + testMessage);

    EXPECT_TRUE(std::regex_search(logs[0], pattern))
        << "Log message did not match expected pattern.\n"
        << "Actual log: " << logs[0];
}
