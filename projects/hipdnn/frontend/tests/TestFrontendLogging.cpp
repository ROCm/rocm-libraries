// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_test_sdk::utilities;

class TestFrontendLogging : public ::testing::Test
{
protected:
    std::unique_ptr<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter> _logLevelGuard;

    void SetUp() override
    {
        _logLevelGuard
            = std::make_unique<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter>(
                "HIPDNN_LOG_LEVEL");
    }

    void TearDown() override
    {
        _logLevelGuard.reset();
    }
};

// Scenario: Verify code contains log statement (regardless of configured level)
TEST_F(TestFrontendLogging, VerifyCodeEmitsLogs)
{
    // Use withOverrideLevel() to capture ALL logs
    const auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    HIPDNN_FE_LOG_INFO("Test info message from frontend");
    HIPDNN_FE_LOG_WARN("Test warning message from frontend");
    HIPDNN_FE_LOG_ERROR("Test error message from frontend");

    // All logs are captured
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "Test info message"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "Test warning message"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "Test error message"));

    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_INFO), 1);
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_WARN), 1);
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_ERROR), 1);
}

// Scenario: Verify log is emitted by UUT when UUT log level allows it
TEST_F(TestFrontendLogging, VerifyLogEmittedWhenLevelAllows)
{
    // Set UUT to WARN level
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_WARN);

    // Use withCurrentLevel() to preserve the WARN level
    const auto recorder = SharedLogRecorder::withCurrentLevel();

    HIPDNN_FE_LOG_INFO("Info should be filtered");
    HIPDNN_FE_LOG_WARN("Warning should pass");
    HIPDNN_FE_LOG_ERROR("Error should pass");

    // INFO is filtered by macro (level check before callback)
    EXPECT_FALSE(recorder.hasLogContaining("filtered"));

    // WARN and ERROR pass through
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "Warning should pass"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "Error should pass"));

    EXPECT_EQ(recorder.getRecordedLogCount(), 2);
}

// Scenario: Verify log is not emitted from UUT because level prevents it
TEST_F(TestFrontendLogging, VerifyLogNotEmittedWhenLevelPrevents)
{
    // Set UUT to ERROR level (filters out INFO and WARN)
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_ERROR);

    // Use withCurrentLevel() to preserve the ERROR level
    const auto recorder = SharedLogRecorder::withCurrentLevel();

    HIPDNN_FE_LOG_INFO("Info should be filtered");
    HIPDNN_FE_LOG_WARN("Warn should be filtered");
    HIPDNN_FE_LOG_ERROR("Error should pass");

    // Only ERROR passes through
    EXPECT_FALSE(recorder.hasLogContaining("filtered"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "Error should pass"));

    EXPECT_EQ(recorder.getRecordedLogCount(), 1);
}

// Verify no logs emitted from UUT
TEST_F(TestFrontendLogging, VerifyNoLogsEmitted)
{
    const auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_OFF);

    HIPDNN_FE_LOG_INFO("Info should be filtered");
    HIPDNN_FE_LOG_WARN("Warn should be filtered");
    HIPDNN_FE_LOG_ERROR("Error should be filtered");

    EXPECT_FALSE(recorder.hasLogContaining("filtered"));
    EXPECT_EQ(recorder.getRecordedLogCount(), 0);
}

// === Frontend Logging API Tests ===

// Test: setGlobalLogLevel and getGlobalLogLevel
TEST_F(TestFrontendLogging, SetAndGetLogLevel)
{
    hipdnnSeverity_t level = HIPDNN_SEV_OFF;

    // Invalid severity
    auto error = setGlobalLogLevel(static_cast<hipdnnSeverity_t>(999));
    EXPECT_EQ(error.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_TRUE(error.get_message().find("Failed to set global log level") != std::string::npos);

    // Set to WARN
    error = setGlobalLogLevel(HIPDNN_SEV_WARN);
    EXPECT_EQ(error.code, ErrorCode::OK);
    error = getGlobalLogLevel(level);
    EXPECT_EQ(error.code, ErrorCode::OK);
    EXPECT_EQ(level, HIPDNN_SEV_WARN);
    EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_WARN);

    // Set to ERROR
    error = setGlobalLogLevel(HIPDNN_SEV_ERROR);
    EXPECT_EQ(error.code, ErrorCode::OK);
    error = getGlobalLogLevel(level);
    EXPECT_EQ(error.code, ErrorCode::OK);
    EXPECT_EQ(level, HIPDNN_SEV_ERROR);
    EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_ERROR);

    // Set to INFO
    error = setGlobalLogLevel(HIPDNN_SEV_INFO);
    EXPECT_EQ(error.code, ErrorCode::OK);
    error = getGlobalLogLevel(level);
    EXPECT_EQ(error.code, ErrorCode::OK);
    EXPECT_EQ(level, HIPDNN_SEV_INFO);
    EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_INFO);

    // Set to OFF
    error = setGlobalLogLevel(HIPDNN_SEV_OFF);
    EXPECT_EQ(error.code, ErrorCode::OK);
    error = getGlobalLogLevel(level);
    EXPECT_EQ(error.code, ErrorCode::OK);
    EXPECT_EQ(level, HIPDNN_SEV_OFF);
    EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_OFF);
}

// Test: setGlobalLogLevel affects frontend log output
TEST_F(TestFrontendLogging, SetLogLevelAffectsFrontendLogs)
{
    // Set to ERROR level
    const auto error = setGlobalLogLevel(HIPDNN_SEV_ERROR);
    EXPECT_EQ(error.code, ErrorCode::OK);

    const auto recorder = SharedLogRecorder::withCurrentLevel();

    HIPDNN_FE_LOG_INFO("Info should be filtered");
    HIPDNN_FE_LOG_WARN("Warn should be filtered");
    HIPDNN_FE_LOG_ERROR("Error should pass");

    // Only ERROR passes through
    EXPECT_FALSE(recorder.hasLogContaining("filtered"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "Error should pass"));
    EXPECT_EQ(recorder.getRecordedLogCount(), 1);
}

namespace
{

// The frontend appends the backend's reason after this separator, as for every backend failure.
constexpr const char* K_BACKEND_ERROR_SEPARATOR = "Backend error: ";

// The text after the backend error separator, or an empty string if there is none.
std::string backendReason(const std::string& message)
{
    const auto pos = message.find(K_BACKEND_ERROR_SEPARATOR);
    if(pos == std::string::npos)
    {
        return {};
    }
    return message.substr(pos + std::string(K_BACKEND_ERROR_SEPARATOR).size());
}

} // namespace

// Test: a level the backend rejects reports the backend's reason and changes no level
TEST_F(TestFrontendLogging, RejectedLogLevelReportsReasonAndKeepsLevel)
{
    ASSERT_EQ(setGlobalLogLevel(HIPDNN_SEV_WARN).code, ErrorCode::OK);

    auto error = setGlobalLogLevel(static_cast<hipdnnSeverity_t>(999));
    EXPECT_EQ(error.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_FALSE(backendReason(error.get_message()).empty()) << error.get_message();

    EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_WARN);
    hipdnnSeverity_t backendLevel = HIPDNN_SEV_OFF;
    ASSERT_EQ(getGlobalLogLevel(backendLevel).code, ErrorCode::OK);
    EXPECT_EQ(backendLevel, HIPDNN_SEV_WARN);
}

// Test: a callback the backend rejects reports the backend's reason
TEST_F(TestFrontendLogging, RejectedUserLogCallbackReportsReason)
{
    int token = 0;
    auto error = setUserLogCallback(nullptr, HIPDNN_SEV_INFO, LogCallbackMode::SYNC, &token);
    EXPECT_EQ(error.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_FALSE(backendReason(error.get_message()).empty()) << error.get_message();
}
