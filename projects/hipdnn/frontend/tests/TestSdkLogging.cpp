// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <string>

#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

// Test SDK logging works in frontend context using SharedLogRecorder

TEST(TestSdkLogging, SdkLogInfoMessageIsCorrectlyPassedToCallback)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Initialize frontend logging via component-specific macro
    HIPDNN_FE_LOG_INFO("Frontend initialized");

    // Now SDK logging should work
    HIPDNN_SDK_LOG_INFO("SDK info test message");

    // Verify we got exactly two logs
    ASSERT_EQ(recorder.getRecordedLogCount(), 2) << "Expected 2 logs, but captured:\n"
                                                 << recorder.getRecordedLogsAsString();

    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "Frontend initialized"))
        << "Expected log containing: \"Frontend initialized\"\n"
        << "But captured:\n"
        << recorder.getRecordedLogsAsString();

    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "SDK info test message"))
        << "Expected log containing: \"SDK info test message\"\n"
        << "But captured:\n"
        << recorder.getRecordedLogsAsString();
}

TEST(TestSdkLogging, SdkLogContainsComponentName)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    HIPDNN_FE_LOG_INFO("Frontend initialized");
    HIPDNN_SDK_LOG_INFO("Component name check");

    // Verify we got exactly two logs
    ASSERT_EQ(recorder.getRecordedLogCount(), 2) << "Expected 2 logs, but captured:\n"
                                                 << recorder.getRecordedLogsAsString();

    // SDK logs should contain the "hipdnn_frontend" component name
    const std::string expectedComponentName = "[hipdnn_frontend]";
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO, expectedComponentName))
        << "Expected log containing: \"" << expectedComponentName << "\"\n"
        << "But captured:\n"
        << recorder.getRecordedLogsAsString();
}

TEST(TestSdkLogging, SdkLogAllSeverityLevels)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    HIPDNN_FE_LOG_INFO("Frontend initialized");

    HIPDNN_SDK_LOG_INFO("SDK info message");
    HIPDNN_SDK_LOG_WARN("SDK warn message");
    HIPDNN_SDK_LOG_ERROR("SDK error message");
    HIPDNN_SDK_LOG_FATAL("SDK fatal message");

    // Verify we got exactly five logs
    ASSERT_EQ(recorder.getRecordedLogCount(), 5) << "Expected 5 logs, but captured:\n"
                                                 << recorder.getRecordedLogsAsString();

    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "SDK info message"))
        << "Expected info log containing: \"SDK info message\"\n"
        << "But captured:\n"
        << recorder.getRecordedLogsAsString();
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "SDK warn message"))
        << "Expected warning log containing: \"SDK warn message\"\n"
        << "But captured:\n"
        << recorder.getRecordedLogsAsString();
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "SDK error message"))
        << "Expected error log containing: \"SDK error message\"\n"
        << "But captured:\n"
        << recorder.getRecordedLogsAsString();
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_FATAL, "SDK fatal message"))
        << "Expected fatal log containing: \"SDK fatal message\"\n"
        << "But captured:\n"
        << recorder.getRecordedLogsAsString();
}

TEST(TestSdkLogging, SdkLogRespectsLogLevel)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);

    HIPDNN_FE_LOG_ERROR("Frontend initialized");

    HIPDNN_SDK_LOG_INFO("Should not appear");
    HIPDNN_SDK_LOG_WARN("Should not appear");
    HIPDNN_SDK_LOG_ERROR("Error should appear");
    HIPDNN_SDK_LOG_FATAL("Fatal should appear");

    // Verify we got exactly three logs (FE error + SDK error + SDK fatal)
    ASSERT_EQ(recorder.getRecordedLogCount(), 3) << "Expected 3 logs, but captured:\n"
                                                 << recorder.getRecordedLogsAsString();

    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "Should not appear"))
        << "Did NOT expect info log containing: \"Should not appear\"\n"
        << "But captured:\n"
        << recorder.getRecordedLogsAsString();
    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "Should not appear"))
        << "Did NOT expect warning log containing: \"Should not appear\"\n"
        << "But captured:\n"
        << recorder.getRecordedLogsAsString();
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "Error should appear"))
        << "Expected error log containing: \"Error should appear\"\n"
        << "But captured:\n"
        << recorder.getRecordedLogsAsString();
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_FATAL, "Fatal should appear"))
        << "Expected fatal log containing: \"Fatal should appear\"\n"
        << "But captured:\n"
        << recorder.getRecordedLogsAsString();
}

TEST(TestSdkLogging, SdkLogStreamFormatting)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    HIPDNN_FE_LOG_INFO("Frontend initialized");

    const int value = 42;
    const std::string text = "formatted";
    HIPDNN_SDK_LOG_INFO("SDK " << text << " message with value " << value);

    // Verify we got exactly two logs
    ASSERT_EQ(recorder.getRecordedLogCount(), 2) << "Expected 2 logs, but captured:\n"
                                                 << recorder.getRecordedLogsAsString();

    const std::string expectedMessage = "SDK formatted message with value 42";
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO, expectedMessage))
        << "Expected log containing: \"" << expectedMessage << "\"\n"
        << "But captured:\n"
        << recorder.getRecordedLogsAsString();
}
