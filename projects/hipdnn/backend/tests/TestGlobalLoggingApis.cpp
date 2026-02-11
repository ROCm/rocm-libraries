// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include <chrono>
#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace hipdnn_test_sdk::utilities;

// Test fixture for backend global log output callback API
class IntegrationBackendGlobalLoggingApis : public ::testing::Test
{
protected:
    std::unique_ptr<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter> _logLevelGuard;
    std::unique_ptr<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter> _logFileGuard;
    void SetUp() override
    {
        _logLevelGuard
            = std::make_unique<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter>(
                "HIPDNN_LOG_LEVEL");
        _logFileGuard
            = std::make_unique<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter>(
                "HIPDNN_LOG_FILE");

        // Clear any previous callback
        hipdnnBackendSetGlobalLoggingCallback_ext(nullptr, false);
    }

    void TearDown() override
    {
        // Clear callback after each test
        hipdnnBackendSetGlobalLoggingCallback_ext(nullptr, false);

        _logLevelGuard.reset();
        _logFileGuard.reset();
    }
};

// Test: Set and get log level through backend API
TEST_F(IntegrationBackendGlobalLoggingApis, SetAndGetLogLevel)
{
    hipdnnSeverity_t level = HIPDNN_SEV_OFF;

    // Set to WARN
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_WARN), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendGetGlobalLogLevel_ext(&level), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(level, HIPDNN_SEV_WARN);

    // Set to ERROR
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_ERROR), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendGetGlobalLogLevel_ext(&level), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(level, HIPDNN_SEV_ERROR);

    // Set to INFO
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendGetGlobalLogLevel_ext(&level), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(level, HIPDNN_SEV_INFO);
}

// Test: Global callback receives backend logs
TEST_F(IntegrationBackendGlobalLoggingApis, GlobalCallbackReceivesLogs)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Register test recording callback
    hipdnnBackendSetGlobalLoggingCallback_ext(IsolatedLogRecorder::getIsoaltedRecordingCallback(),
                                              false);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    // Trigger backend logging by creating a handle
    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    // Backend should log handle creation
    EXPECT_TRUE(recorder.hasLogContaining("API success: [hipdnnCreate]"));

    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

// Test: Callback respects log level filtering
TEST_F(IntegrationBackendGlobalLoggingApis, CallbackRespectsLogLevel)
{
    auto recorder = IsolatedLogRecorder::withCurrentLevel();

    // Set log level to ERROR (filters out INFO and WARN)
    hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_ERROR);

    // Register test recording callback
    hipdnnBackendSetGlobalLoggingCallback_ext(IsolatedLogRecorder::getIsoaltedRecordingCallback(),
                                              false);

    // Trigger backend logging
    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);

    // INFO logs should be filtered out by macro level check
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_INFO), 0);
}

// Test: Clearing callback stops log capture
TEST_F(IntegrationBackendGlobalLoggingApis, ClearingCallbackStopsCapture)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Register callback
    hipdnnBackendSetGlobalLoggingCallback_ext(IsolatedLogRecorder::getIsoaltedRecordingCallback(),
                                              false);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    // Trigger some logging
    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    // Clear callback
    hipdnnBackendSetGlobalLoggingCallback_ext(nullptr, false);

    size_t logsAfterCreate = recorder.getRecordedLogCount();
    EXPECT_GT(logsAfterCreate, 0);

    // Further operations should not be captured
    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);

    // Log count should not increase significantly (maybe 1-2 from recorder shutdown)
    size_t finalLogs = recorder.getRecordedLogCount();
    EXPECT_EQ(finalLogs, logsAfterCreate);
}

// Test: Multiple handle operations are logged
TEST_F(IntegrationBackendGlobalLoggingApis, MultipleOperationsLogged)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    hipdnnBackendSetGlobalLoggingCallback_ext(IsolatedLogRecorder::getIsoaltedRecordingCallback(),
                                              false);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    // Create multiple handles
    hipdnnHandle_t handle1 = nullptr;
    hipdnnHandle_t handle2 = nullptr;

    ASSERT_EQ(hipdnnCreate(&handle1), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnCreate(&handle2), HIPDNN_STATUS_SUCCESS);

    size_t logsAfterCreates = recorder.getRecordedLogCount();
    EXPECT_GT(logsAfterCreates, 0);
    EXPECT_TRUE(
        recorder.hasLogContaining(HIPDNN_SEV_INFO, "[hipdnn_backend] API success: [hipdnnCreate]"));

    ASSERT_EQ(hipdnnDestroy(handle1), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnDestroy(handle2), HIPDNN_STATUS_SUCCESS);

    // Should have more logs after destroy operations
    EXPECT_GT(recorder.getRecordedLogCount(), logsAfterCreates);
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO,
                                          "[hipdnn_backend] API success: [hipdnnDestroy]"));
}

// Test: Descriptor operations are logged
TEST_F(IntegrationBackendGlobalLoggingApis, DescriptorOperationsLogged)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    hipdnnBackendSetGlobalLoggingCallback_ext(IsolatedLogRecorder::getIsoaltedRecordingCallback(),
                                              false);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDescriptor_t descriptor = nullptr;

    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &descriptor),
              HIPDNN_STATUS_SUCCESS);

    // Should log descriptor creation
    EXPECT_TRUE(recorder.hasLogContaining("Create"));

    ASSERT_EQ(hipdnnBackendDestroyDescriptor(descriptor), HIPDNN_STATUS_SUCCESS);
}

// Test: Error conditions are logged
TEST_F(IntegrationBackendGlobalLoggingApis, ErrorConditionsLogged)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);

    hipdnnBackendSetGlobalLoggingCallback_ext(IsolatedLogRecorder::getIsoaltedRecordingCallback(),
                                              false);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    // Intentionally cause an error
    auto status = hipdnnCreate(nullptr);
    ASSERT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    // Should log error
    EXPECT_GT(recorder.countLogsAtLevel(HIPDNN_SEV_ERROR), 0);
}

// Test: Callback that throws exception is handled
TEST_F(IntegrationBackendGlobalLoggingApis, CallbackThrowsException)
{
    // Callback that throws exception
    auto throwingCallback = [](hipdnnSeverity_t, const char*) {
        throw std::runtime_error("Test exception from callback");
    };

    // Set throwing callback
    ASSERT_EQ(hipdnnBackendSetGlobalLoggingCallback_ext(throwingCallback, false),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    // Trigger logging - should not crash despite exception
    hipdnnHandle_t handle = nullptr;
    EXPECT_NO_THROW(hipdnnCreate(&handle));

    // Cleanup
    if(handle != nullptr)
    {
        hipdnnDestroy(handle);
    }
}

// Test: Set callback, clear it, then set again
TEST_F(IntegrationBackendGlobalLoggingApis, SetClearSetCallback)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Set callback first time
    hipdnnBackendSetGlobalLoggingCallback_ext(IsolatedLogRecorder::getIsoaltedRecordingCallback(),
                                              false);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle1 = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle1), HIPDNN_STATUS_SUCCESS);
    size_t logsAfterFirst = recorder.getRecordedLogCount();
    EXPECT_GT(logsAfterFirst, 0);

    // Clear callback
    hipdnnBackendSetGlobalLoggingCallback_ext(nullptr, false);

    // Operations should not be captured now
    size_t logsBefore = recorder.getRecordedLogCount();
    ASSERT_EQ(hipdnnDestroy(handle1), HIPDNN_STATUS_SUCCESS);
    size_t logsAfter = recorder.getRecordedLogCount();
    EXPECT_LE(logsAfter - logsBefore, 1); // Minimal or no increase

    // Set callback again
    hipdnnBackendSetGlobalLoggingCallback_ext(IsolatedLogRecorder::getIsoaltedRecordingCallback(),
                                              false);

    hipdnnHandle_t handle2 = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle2), HIPDNN_STATUS_SUCCESS);
    EXPECT_GT(recorder.getRecordedLogCount(), logsAfter); // Should capture again

    ASSERT_EQ(hipdnnDestroy(handle2), HIPDNN_STATUS_SUCCESS);
}

// Test: Setting invalid log level returns error
TEST_F(IntegrationBackendGlobalLoggingApis, InvalidLogLevelReturnsError)
{
    // Invalid log level (not in enum)
    auto invalidLevel = static_cast<hipdnnSeverity_t>(999);

    auto status = hipdnnBackendSetGlobalLogLevel_ext(invalidLevel);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
}

// Test: Async callback behavior
TEST_F(IntegrationBackendGlobalLoggingApis, AsyncCallbackBehavior)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Set callback in ASYNC mode
    hipdnnBackendSetGlobalLoggingCallback_ext(IsolatedLogRecorder::getIsoaltedRecordingCallback(),
                                              true);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    // Trigger logging
    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
    // Logs should eventually be captured (async delivery)
    // Note: May need small delay for async processing
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_GT(recorder.getRecordedLogCount(), 0);
    EXPECT_TRUE(
        recorder.hasLogContaining(HIPDNN_SEV_INFO, "[hipdnn_backend] API success: [hipdnnCreate]"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO,
                                          "[hipdnn_backend] API success: [hipdnnDestroy]"));
}

// Test: Switching between sync and async
TEST_F(IntegrationBackendGlobalLoggingApis, SyncVsAsyncCallbackDifference)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_FATAL);

    // Test sync callback
    hipdnnBackendSetGlobalLoggingCallback_ext(IsolatedLogRecorder::getIsoaltedRecordingCallback(),
                                              false);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    // Sync: logs should be available immediately
    size_t syncLogs = recorder.getRecordedLogCount();
    EXPECT_GT(syncLogs, 0);

    // Switch to async callback
    hipdnnBackendSetGlobalLoggingCallback_ext(IsolatedLogRecorder::getIsoaltedRecordingCallback(),
                                              true);

    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);

    // Async: logs may not be immediately available, but should arrive soon
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    size_t asyncLogs = recorder.getRecordedLogCount();
    EXPECT_GT(asyncLogs, syncLogs);

    // Back to sync callback
    hipdnnBackendSetGlobalLoggingCallback_ext(IsolatedLogRecorder::getIsoaltedRecordingCallback(),
                                              false);

    handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    // Sync: logs should be available immediately
    syncLogs = recorder.getRecordedLogCount();
    EXPECT_GT(syncLogs, asyncLogs);

    // And then back to async callback
    hipdnnBackendSetGlobalLoggingCallback_ext(IsolatedLogRecorder::getIsoaltedRecordingCallback(),
                                              true);

    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);

    // Async: logs may not be immediately available, but should arrive soon
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    asyncLogs = recorder.getRecordedLogCount();
    EXPECT_GT(asyncLogs, syncLogs);
}

// Test: Multiple async callbacks - logs routed to most recent
TEST_F(IntegrationBackendGlobalLoggingApis, MultipleAsyncCallbacksRoutedToMostRecent)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_FATAL);

    // Track which callback received logs
    static int s_callbackACount = 0; // NOLINT(readability-identifier-naming)
    static int s_callbackBCount = 0; // NOLINT(readability-identifier-naming)
    s_callbackACount = 0;
    s_callbackBCount = 0;

    auto callbackA = [](hipdnnSeverity_t severity, const char* message) {
        (void)severity;
        (void)message;
        s_callbackACount++;
    };

    auto callbackB = [](hipdnnSeverity_t severity, const char* message) {
        (void)severity;
        (void)message;
        s_callbackBCount++;
    };

    // Register callback A with async=true
    hipdnnBackendSetGlobalLoggingCallback_ext(callbackA, true);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle1 = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle1), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnDestroy(handle1), HIPDNN_STATUS_SUCCESS);

    // Give async logger time to process
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Verify callback A received logs
    int logsToA = s_callbackACount;
    EXPECT_GT(logsToA, 0) << "Callback A should have received logs";

    // Now register callback B with async=true (same async setting)
    hipdnnBackendSetGlobalLoggingCallback_ext(callbackB, true);

    // Create another handle - logs should go to callback B now
    hipdnnHandle_t handle2 = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle2), HIPDNN_STATUS_SUCCESS);

    // Give async logger time to process
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Verify callback B received NEW logs, not callback A
    int logsToB = s_callbackBCount;
    EXPECT_GT(logsToB, 0) << "Callback B should have received logs after registration";
    EXPECT_EQ(s_callbackACount, logsToA)
        << "Callback A should NOT receive any more logs after B was registered";

    ASSERT_EQ(hipdnnDestroy(handle2), HIPDNN_STATUS_SUCCESS);
}

// Test: Multiple sync callbacks - logs routed to most recent
TEST_F(IntegrationBackendGlobalLoggingApis, MultipleSyncCallbacksRoutedToMostRecent)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_FATAL);

    // Track which callback received logs
    static int s_callbackACount = 0; // NOLINT(readability-identifier-naming)
    static int s_callbackBCount = 0; // NOLINT(readability-identifier-naming)
    s_callbackACount = 0;
    s_callbackBCount = 0;

    auto callbackA = [](hipdnnSeverity_t severity, const char* message) {
        (void)severity;
        (void)message;
        s_callbackACount++;
    };

    auto callbackB = [](hipdnnSeverity_t severity, const char* message) {
        (void)severity;
        (void)message;
        s_callbackBCount++;
    };

    // Register callback A with async=false
    hipdnnBackendSetGlobalLoggingCallback_ext(callbackA, false);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle1 = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle1), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnDestroy(handle1), HIPDNN_STATUS_SUCCESS);

    // Verify callback A received logs (sync, so immediate)
    int logsToA = s_callbackACount;
    EXPECT_GT(logsToA, 0) << "Callback A should have received logs";

    // Now register callback B with async=false (same sync setting)
    hipdnnBackendSetGlobalLoggingCallback_ext(callbackB, false);

    // Create another handle - logs should go to callback B now
    hipdnnHandle_t handle2 = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle2), HIPDNN_STATUS_SUCCESS);

    // Verify callback B received NEW logs, not callback A
    int logsToB = s_callbackBCount;
    EXPECT_GT(logsToB, 0) << "Callback B should have received logs after registration";
    EXPECT_EQ(s_callbackACount, logsToA)
        << "Callback A should NOT receive any more logs after B was registered";

    ASSERT_EQ(hipdnnDestroy(handle2), HIPDNN_STATUS_SUCCESS);
}
