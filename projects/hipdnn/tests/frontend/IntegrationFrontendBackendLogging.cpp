// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_test_sdk::utilities;

class IntegrationFrontendBackendLogging : public ::testing::Test
{
protected:
    hipdnnSeverity_t _originalLogLevel = HIPDNN_SEV_OFF;

    void SetUp() override
    {
        // Clear any previous callback
        auto error = setGlobalLoggingCallback(nullptr, false);
        ASSERT_EQ(error.code, ErrorCode::OK);

        // Save original log level.
        error = getGlobalLogLevel(_originalLogLevel);
        ASSERT_EQ(error.code, ErrorCode::OK);

        // Reset log level
        error = setGlobalLogLevel(HIPDNN_SEV_OFF);
        ASSERT_EQ(error.code, ErrorCode::OK);
    }

    void TearDown() override
    {
        auto error = setGlobalLoggingCallback(nullptr, false);
        ASSERT_EQ(error.code, ErrorCode::OK);

        // Restore original log level.
        error = setGlobalLogLevel(_originalLogLevel);
        ASSERT_EQ(error.code, ErrorCode::OK);
    }
};

// Test: Frontend logs are produced on global callback
TEST_F(IntegrationFrontendBackendLogging, FrontendLogsProducedOnGlobalCallback)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Register callback through frontend API
    auto error
        = setGlobalLoggingCallback(IsolatedLogRecorder::getIsolatedRecordingCallback(), false);
    ASSERT_EQ(error.code, ErrorCode::OK);

    error = setGlobalLogLevel(HIPDNN_SEV_INFO);
    ASSERT_EQ(error.code, ErrorCode::OK);

    // Emit frontend logs
    HIPDNN_FE_LOG_INFO("Test info message from frontend");
    HIPDNN_FE_LOG_WARN("Test warning message from frontend");
    HIPDNN_FE_LOG_ERROR("Test error message from frontend");

    // Verify logs were received on callback
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "Test info message"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "Test warning message"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "Test error message"));
}

// Test: Log level API controls which frontend logs are produced
TEST_F(IntegrationFrontendBackendLogging, LogLevelControlsFrontendLogs)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    auto error
        = setGlobalLoggingCallback(IsolatedLogRecorder::getIsolatedRecordingCallback(), false);
    ASSERT_EQ(error.code, ErrorCode::OK);

    // Set to WARN level - INFO should be filtered
    error = setGlobalLogLevel(HIPDNN_SEV_WARN);
    ASSERT_EQ(error.code, ErrorCode::OK);

    HIPDNN_FE_LOG_INFO("Info should be filtered");
    HIPDNN_FE_LOG_WARN("Warning should pass");
    HIPDNN_FE_LOG_ERROR("Error should pass");

    // INFO filtered, WARN and ERROR pass
    EXPECT_FALSE(recorder.hasLogContaining("Info should be filtered"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "Warning should pass"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "Error should pass"));
}

// Test: Setting log level to OFF filters all logs
TEST_F(IntegrationFrontendBackendLogging, LogLevelOffFiltersAllLogs)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    auto error
        = setGlobalLoggingCallback(IsolatedLogRecorder::getIsolatedRecordingCallback(), false);
    ASSERT_EQ(error.code, ErrorCode::OK);

    error = setGlobalLogLevel(HIPDNN_SEV_OFF);
    ASSERT_EQ(error.code, ErrorCode::OK);

    HIPDNN_FE_LOG_INFO("Should be filtered");
    HIPDNN_FE_LOG_WARN("Should be filtered");
    HIPDNN_FE_LOG_ERROR("Should be filtered");

    // All logs filtered
    EXPECT_EQ(recorder.getRecordedLogCount(), 0);
}

// Test: Clearing callback stops log callbacks
TEST_F(IntegrationFrontendBackendLogging, ClearingCallbackStopsCallbacks)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Set callback
    auto error
        = setGlobalLoggingCallback(IsolatedLogRecorder::getIsolatedRecordingCallback(), false);
    ASSERT_EQ(error.code, ErrorCode::OK);
    error = setGlobalLogLevel(HIPDNN_SEV_INFO);
    ASSERT_EQ(error.code, ErrorCode::OK);

    HIPDNN_FE_LOG_INFO("Log before clearing callback");
    size_t logsWithCallback = recorder.getRecordedLogCount();
    EXPECT_GT(logsWithCallback, 0);

    // Clear callback
    error = setGlobalLoggingCallback(nullptr, false);
    ASSERT_EQ(error.code, ErrorCode::OK);

    HIPDNN_FE_LOG_INFO("Log after clearing callback");
    size_t logsAfterClear = recorder.getRecordedLogCount();

    // No new logs should be provided via callback.
    EXPECT_EQ(logsAfterClear, logsWithCallback);
}

// Test: Get/set log level round-trips correctly
TEST_F(IntegrationFrontendBackendLogging, GetSetLogLevelRoundTrip)
{
    hipdnnSeverity_t level = HIPDNN_SEV_OFF;

    // Set and verify each level
    auto error = setGlobalLogLevel(HIPDNN_SEV_WARN);
    ASSERT_EQ(error.code, ErrorCode::OK);
    error = getGlobalLogLevel(level);
    ASSERT_EQ(error.code, ErrorCode::OK);
    EXPECT_EQ(level, HIPDNN_SEV_WARN);

    error = setGlobalLogLevel(HIPDNN_SEV_ERROR);
    ASSERT_EQ(error.code, ErrorCode::OK);
    error = getGlobalLogLevel(level);
    ASSERT_EQ(error.code, ErrorCode::OK);
    EXPECT_EQ(level, HIPDNN_SEV_ERROR);

    error = setGlobalLogLevel(HIPDNN_SEV_INFO);
    ASSERT_EQ(error.code, ErrorCode::OK);
    error = getGlobalLogLevel(level);
    ASSERT_EQ(error.code, ErrorCode::OK);
    EXPECT_EQ(level, HIPDNN_SEV_INFO);
}
