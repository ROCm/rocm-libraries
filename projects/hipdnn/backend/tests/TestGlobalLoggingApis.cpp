// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace hipdnn_test_sdk::utilities;

// Test fixture for backend global log output callback API
class IntegrationBackendGlobalLoggingApis : public ::testing::Test
{
protected:
    std::string _logFile;
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

        // Clean up any test log file
        if(!_logFile.empty())
        {
            std::remove(_logFile.c_str());
        }
    }
};

// Test: Set and get log level through backend API
TEST_F(IntegrationBackendGlobalLoggingApis, SetAndGetLogLevel)
{
    hipdnnSeverity_t level = HIPDNN_SEV_OFF;

    // nullptr doesn't cause exception.
    ASSERT_EQ(hipdnnBackendGetGlobalLogLevel_ext(nullptr), HIPDNN_STATUS_BAD_PARAM);

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

// Test: Log pattern format is correct for global callback
TEST_F(IntegrationBackendGlobalLoggingApis, LogPatternFormatIsCorrect)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Register test recording callback
    hipdnnBackendSetGlobalLoggingCallback_ext(IsolatedLogRecorder::getIsoaltedRecordingCallback(),
                                              false);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    // Trigger backend logging by creating a handle
    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);

    // Get one of the captured log messages
    auto logs = recorder.getRecordedLogs();
    ASSERT_GT(logs.size(), 0) << "Expected at least one log message";

    // [timestamp format] [thread id] [log level] [hipdnn_backend] message
    // Example: [2026-02-18 09:15:30.123] [tid 12345] [info] [hipdnn_backend] API success: [hipdnnCreate]
    std::regex patternRegex(
        R"(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[tid \d+\] \[(info|warn|error|critical)\] \[hipdnn_backend\] .+)");

    bool foundMatchingLog = false;
    for(const auto& log : logs)
    {
        if(std::regex_search(log.message, patternRegex))
        {
            foundMatchingLog = true;
            break;
        }
    }

    EXPECT_TRUE(foundMatchingLog)
        << "Expected at least one log message to match the pattern format.\n"
        << "First log message: " << (logs.empty() ? "(empty)" : logs[0].message);
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

// Test: Concurrent logging with callback toggle between enabled and nullptr
// Disabled for now due to lengthy test time.
TEST_F(IntegrationBackendGlobalLoggingApis, DISABLED_ConcurrentLoggingWithCallbackToggle)
{
    // Redirect default logging to file to avoid console spam when callback is disabled
    _logFile = "concurrent_callback_toggle_test.log";
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_FILE", _logFile.c_str());

    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    constexpr int NUM_LOGGER_THREADS = 4;
    std::atomic<bool> startFlag{false};
    std::atomic<bool> stopFlag{false};
    std::vector<std::thread> threads;
    threads.reserve(NUM_LOGGER_THREADS);

    // Register the callback and set log level
    hipdnnBackendSetGlobalLoggingCallback_ext(IsolatedLogRecorder::getIsoaltedRecordingCallback(),
                                              false);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    // Create threads that generate logs by repeated create/destroy of hipDNN handles
    for(int i = 0; i < NUM_LOGGER_THREADS; ++i)
    {
        threads.emplace_back([&]() {
            while(!startFlag.load())
            {
                std::this_thread::yield();
            }

            while(!stopFlag.load())
            {
                hipdnnHandle_t handle = nullptr;
                hipdnnCreate(&handle);
                if(handle != nullptr)
                {
                    hipdnnDestroy(handle);
                }
                std::this_thread::yield();
            }
        });
    }

    // Start all logger threads
    startFlag.store(true);

    // Control thread behavior: toggle callback on and off
    constexpr int NUM_CYCLES = 4;
    for(int cycle = 0; cycle < NUM_CYCLES; ++cycle)
    {
        // Use async=true for even cycles, async=false for odd cycles
        bool useAsync = (cycle % 2 == 0);

        // With callback registered - logs should be captured
        hipdnnBackendSetGlobalLoggingCallback_ext(
            IsolatedLogRecorder::getIsoaltedRecordingCallback(), useAsync);

        size_t countBefore = recorder.getRecordedLogCount();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        size_t countAfterEnabled = recorder.getRecordedLogCount();

        EXPECT_GT(countAfterEnabled, countBefore)
            << "Log count should increase when callback is registered (cycle " << cycle
            << ", async=" << useAsync << ")";

        // With callback set to nullptr - logs should NOT be captured
        hipdnnBackendSetGlobalLoggingCallback_ext(nullptr, false);

        size_t countBeforeDisabled = recorder.getRecordedLogCount();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        size_t countAfterDisabled = recorder.getRecordedLogCount();

        EXPECT_EQ(countAfterDisabled, countBeforeDisabled)
            << "Log count should NOT increase when callback is nullptr (cycle " << cycle << ")";
    }

    // Stop all threads
    stopFlag.store(true);
    for(auto& t : threads)
    {
        t.join();
    }

    // Final verification - total logs should be > 0
    EXPECT_GT(recorder.getRecordedLogCount(), 0);
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
