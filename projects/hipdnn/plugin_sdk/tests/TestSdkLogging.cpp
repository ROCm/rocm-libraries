// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <mutex>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

// Test SDK logging works in plugin_sdk context using a custom callback

TEST(TestSdkLogging, SdkLogInfoMessageIsCorrectlyPassedToCallback)
{
    // Create RAII log recorder - captures all logs for this test
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    HIPDNN_SDK_LOG_INFO("Plugin SDK info test message");

    // Verify we got exactly one log
    ASSERT_EQ(recorder.getRecordedLogCount(), 1) << "Expected 1 log, but captured:\n"
                                                 << recorder.getRecordedLogsAsString();

    // Build expected message
    const std::string expectedLogSuffix = "Plugin SDK info test message";
    // Verify log contains expected message at INFO level
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO, expectedLogSuffix))
        << "Expected log containing: \"" << expectedLogSuffix << "\"\n"
        << "But captured:\n"
        << recorder.getRecordedLogsAsString();
}

TEST(TestSdkLogging, SdkLogContainsComponentName)
{
    // Create RAII log recorder - captures all logs for this test
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    HIPDNN_SDK_LOG_INFO("Component name check");

    // Verify we got exactly one log
    ASSERT_EQ(recorder.getRecordedLogCount(), 1) << "Expected 1 log, but captured:\n"
                                                 << recorder.getRecordedLogsAsString();

    // Build expected message
    const std::string expectedComponentName = "[hipdnn_sdk]";
    // Verify log contains expected message at INFO level
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO, expectedComponentName))
        << "Expected log containing: \"" << expectedComponentName << "\"\n"
        << "But captured:\n"
        << recorder.getRecordedLogsAsString();
}

TEST(TestSdkLogging, SdkLogAllSeverityLevels)
{
    // Create RAII log recorder - captures all logs for this test
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    HIPDNN_SDK_LOG_INFO("SDK info message");
    HIPDNN_SDK_LOG_WARN("SDK warn message");
    HIPDNN_SDK_LOG_ERROR("SDK error message");
    HIPDNN_SDK_LOG_FATAL("SDK fatal message");

    // Verify we got exactly four logs
    ASSERT_EQ(recorder.getRecordedLogCount(), 4) << "Expected 4 logs, but captured:\n"
                                                 << recorder.getRecordedLogsAsString();

    // Build expected message
    const std::string expectedInfoMessage = "SDK info message";
    const std::string expectedWarnMessage = "SDK warn message";
    const std::string expectedErrorMessage = "SDK error message";
    const std::string expectedFatalMessage = "SDK fatal message";

    // Verify log contains expected messages
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO, expectedInfoMessage))
        << "Expected info log containing: \"" << expectedInfoMessage << "\"\n"
        << "But captured:\n"
        << recorder.getRecordedLogsAsString();
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, expectedWarnMessage))
        << "Expected warning log containing: \"" << expectedWarnMessage << "\"\n"
        << "But captured:\n"
        << recorder.getRecordedLogsAsString();
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, expectedErrorMessage))
        << "Expected error log containing: \"" << expectedErrorMessage << "\"\n"
        << "But captured:\n"
        << recorder.getRecordedLogsAsString();
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_FATAL, expectedFatalMessage))
        << "Expected fatal log containing: \"" << expectedFatalMessage << "\"\n"
        << "But captured:\n"
        << recorder.getRecordedLogsAsString();
}

TEST(TestSdkLogging, SdkLogRespectsLogLevel)
{
    // Create RAII log recorder - captures all logs for this test
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);

    HIPDNN_SDK_LOG_INFO("SDK info message");
    HIPDNN_SDK_LOG_WARN("SDK warn message");
    HIPDNN_SDK_LOG_ERROR("SDK error message");
    HIPDNN_SDK_LOG_FATAL("SDK fatal message");

    // Verify we got exactly two logs
    ASSERT_EQ(recorder.getRecordedLogCount(), 2) << "Expected 2 logs, but captured:\n"
                                                 << recorder.getRecordedLogsAsString();

    // Build expected message
    const std::string expectedInfoMessage = "SDK info message";
    const std::string expectedWarnMessage = "SDK warn message";
    const std::string expectedErrorMessage = "SDK error message";
    const std::string expectedFatalMessage = "SDK fatal message";

    // Verify log contains expected messages
    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_INFO, expectedInfoMessage))
        << "Did NOT expect info log containing: \"" << expectedInfoMessage << "\"\n"
        << "But captured:\n"
        << recorder.getRecordedLogsAsString();
    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_WARN, expectedWarnMessage))
        << "Did NOT expect warning log containing: \"" << expectedWarnMessage << "\"\n"
        << "But captured:\n"
        << recorder.getRecordedLogsAsString();
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, expectedErrorMessage))
        << "Expected error log containing: \"" << expectedErrorMessage << "\"\n"
        << "But captured:\n"
        << recorder.getRecordedLogsAsString();
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_FATAL, expectedFatalMessage))
        << "Expected fatal log containing: \"" << expectedFatalMessage << "\"\n"
        << "But captured:\n"
        << recorder.getRecordedLogsAsString();
}
