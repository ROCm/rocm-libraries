// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

constexpr auto ASYNC_LOG_TIMEOUT = std::chrono::seconds(10);
constexpr auto NEGATIVE_ASSERT_TIMEOUT = std::chrono::milliseconds(100);

using namespace hipdnn_frontend;
using namespace hipdnn_test_sdk::utilities;

class IntegrationFrontendUserLogging : public ::testing::Test
{
protected:
    hipdnnSeverity_t _originalLogLevel = HIPDNN_SEV_OFF;

    void SetUp() override
    {
        // Save original log level
        auto error = getGlobalLogLevel(_originalLogLevel);
        ASSERT_EQ(error.code, ErrorCode::OK);

        // Clear any previous user callback (may fail if not registered — that's OK)
        setIsolatedUserCallback(HIPDNN_SEV_OFF, LogCallbackMode::ASYNC);

        // Reset log level
        error = setGlobalLogLevel(HIPDNN_SEV_OFF);
        ASSERT_EQ(error.code, ErrorCode::OK);
    }

    void TearDown() override
    {
        // Clear user callback (may fail if not registered — that's OK)
        setIsolatedUserCallback(HIPDNN_SEV_OFF, LogCallbackMode::ASYNC);

        // Restore original log level
        auto error = setGlobalLogLevel(_originalLogLevel);
        ASSERT_EQ(error.code, ErrorCode::OK);
    }

    // Set the isolated user callback. Returns the error without asserting.
    Error setIsolatedUserCallback(hipdnnSeverity_t minLevel, LogCallbackMode mode)
    {
        return setUserLogCallback(
            IsolatedLogRecorder::getIsolatedUserRecordingCallback(), minLevel, mode, this);
    }

    // Register/update/unregister the isolated user callback. Asserts success.
    void registerUserCallback(hipdnnSeverity_t minLevel, LogCallbackMode mode)
    {
        ASSERT_EQ(setIsolatedUserCallback(minLevel, mode).code, ErrorCode::OK);
    }
};

// Test: Frontend logs are produced on user callback
TEST_F(IntegrationFrontendUserLogging, FrontendLogsProducedOnUserCallback)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Register user callback through frontend API
    registerUserCallback(HIPDNN_SEV_INFO, LogCallbackMode::ASYNC);

    auto error = setGlobalLogLevel(HIPDNN_SEV_INFO);
    ASSERT_EQ(error.code, ErrorCode::OK);

    // Emit frontend logs
    HIPDNN_FE_LOG_INFO("Test info message from frontend");
    HIPDNN_FE_LOG_WARN("Test warning message from frontend");
    HIPDNN_FE_LOG_ERROR("Test error message from frontend");

    // Wait for async callbacks to deliver all log messages
    EXPECT_TRUE(recorder.waitForLogsContaining(
        {"Test info message", "Test warning message", "Test error message"}, ASYNC_LOG_TIMEOUT))
        << "Timed out waiting for all expected log messages\nRecorded logs:\n"
        << recorder.getRecordedLogsAsString();

    // Verify logs were received on callback with correct severity
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "Test info message"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "Test warning message"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "Test error message"));
}

// Test: Log level API controls which frontend logs are produced
TEST_F(IntegrationFrontendUserLogging, LogLevelControlsFrontendLogs)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    registerUserCallback(HIPDNN_SEV_INFO, LogCallbackMode::ASYNC);

    // Set to WARN level - INFO should be filtered
    auto error = setGlobalLogLevel(HIPDNN_SEV_WARN);
    ASSERT_EQ(error.code, ErrorCode::OK);

    HIPDNN_FE_LOG_INFO("Info should be filtered");
    HIPDNN_FE_LOG_WARN("Warning should pass");
    HIPDNN_FE_LOG_ERROR("Error should pass");

    // Wait for expected logs (WARN + ERROR); INFO should be filtered
    EXPECT_TRUE(recorder.waitForLogsContaining({"Warning should pass", "Error should pass"},
                                               ASYNC_LOG_TIMEOUT))
        << "Timed out waiting for expected WARN and ERROR logs\nRecorded logs:\n"
        << recorder.getRecordedLogsAsString();

    // INFO filtered, WARN and ERROR pass
    EXPECT_FALSE(recorder.hasLogContaining("Info should be filtered"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "Warning should pass"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "Error should pass"));
}

// Test: Setting log level to OFF filters all logs
TEST_F(IntegrationFrontendUserLogging, LogLevelOffFiltersAllLogs)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    registerUserCallback(HIPDNN_SEV_INFO, LogCallbackMode::ASYNC);

    auto error = setGlobalLogLevel(HIPDNN_SEV_OFF);
    ASSERT_EQ(error.code, ErrorCode::OK);

    HIPDNN_FE_LOG_INFO("Should be filtered");
    HIPDNN_FE_LOG_WARN("Should be filtered");
    HIPDNN_FE_LOG_ERROR("Should be filtered");

    // Negative assertion: no logs expected, bounded wait to confirm none arrive
    EXPECT_FALSE(recorder.waitForLogCount(1, NEGATIVE_ASSERT_TIMEOUT))
        << "Unexpected log received despite log level OFF\nRecorded logs:\n"
        << recorder.getRecordedLogsAsString();

    // All logs filtered
    EXPECT_EQ(recorder.getRecordedLogCount(), 0);
}

// Test: Unregistering callback with SEV_OFF stops log callbacks
TEST_F(IntegrationFrontendUserLogging, UnregisterWithSevOffStopsCallbacks)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Set callback
    registerUserCallback(HIPDNN_SEV_INFO, LogCallbackMode::ASYNC);
    auto error = setGlobalLogLevel(HIPDNN_SEV_INFO);
    ASSERT_EQ(error.code, ErrorCode::OK);

    const auto countBefore = recorder.getRecordedLogCount();
    HIPDNN_FE_LOG_INFO("Log before unregistering callback");

    // Wait for at least 1 log to arrive via async callback
    EXPECT_TRUE(recorder.waitForLogCount(countBefore + 1, ASYNC_LOG_TIMEOUT))
        << "Timed out waiting for async log delivery\nRecorded logs:\n"
        << recorder.getRecordedLogsAsString();
    EXPECT_GT(recorder.getRecordedLogCount(), 0);

    // Unregister callback with SEV_OFF
    registerUserCallback(HIPDNN_SEV_OFF, LogCallbackMode::ASYNC);

    // Capture count after unregister — the only point where it's guaranteed
    // that no more logs will be delivered via the callback
    const size_t logsAfterUnregister = recorder.getRecordedLogCount();

    HIPDNN_FE_LOG_INFO("Log after unregistering callback");

    // Negative assertion: no new logs expected after unregister, bounded wait
    EXPECT_FALSE(recorder.waitForLogCount(logsAfterUnregister + 1, NEGATIVE_ASSERT_TIMEOUT))
        << "Unexpected log received after unregistering callback\nRecorded logs:\n"
        << recorder.getRecordedLogsAsString();

    // No new logs should be provided via callback
    EXPECT_EQ(recorder.getRecordedLogCount(), logsAfterUnregister);
}

// Test: Callback level filtering works independently of global level
TEST_F(IntegrationFrontendUserLogging, CallbackLevelFiltersIndependently)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Register callback at WARN level (filters INFO)
    registerUserCallback(HIPDNN_SEV_WARN, LogCallbackMode::ASYNC);

    // Set global level to INFO (allows INFO through)
    auto error = setGlobalLogLevel(HIPDNN_SEV_INFO);
    ASSERT_EQ(error.code, ErrorCode::OK);

    HIPDNN_FE_LOG_INFO("Info should be filtered by callback level");
    HIPDNN_FE_LOG_WARN("Warning should pass");
    HIPDNN_FE_LOG_ERROR("Error should pass");

    // Wait for expected logs (WARN + ERROR); INFO filtered at callback level
    EXPECT_TRUE(recorder.waitForLogsContaining({"Warning should pass", "Error should pass"},
                                               ASYNC_LOG_TIMEOUT))
        << "Timed out waiting for expected WARN and ERROR logs\nRecorded logs:\n"
        << recorder.getRecordedLogsAsString();

    // INFO filtered at callback level, WARN and ERROR pass
    EXPECT_FALSE(recorder.hasLogContaining("Info should be filtered"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "Warning should pass"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "Error should pass"));
}

// Test: Sync callback executes immediately
TEST_F(IntegrationFrontendUserLogging, SyncCallbackExecutesImmediately)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Register callback in SYNC mode
    registerUserCallback(HIPDNN_SEV_INFO, LogCallbackMode::SYNC);

    auto error = setGlobalLogLevel(HIPDNN_SEV_INFO);
    ASSERT_EQ(error.code, ErrorCode::OK);

    HIPDNN_FE_LOG_INFO("Sync callback message");

    // No delay needed for sync - should be immediate
    EXPECT_GT(recorder.getRecordedLogCount(), 0);
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "Sync callback message"));
}

// Test: Async callback queues logs
TEST_F(IntegrationFrontendUserLogging, AsyncCallbackQueuesLogs)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Register callback in ASYNC mode
    registerUserCallback(HIPDNN_SEV_INFO, LogCallbackMode::ASYNC);

    auto error = setGlobalLogLevel(HIPDNN_SEV_INFO);
    ASSERT_EQ(error.code, ErrorCode::OK);

    HIPDNN_FE_LOG_INFO("Async callback message");

    // Wait for async callback to deliver the log
    EXPECT_TRUE(recorder.waitForLogsContaining({"Async callback message"}, ASYNC_LOG_TIMEOUT))
        << "Timed out waiting for log containing 'Async callback message'\nRecorded logs:\n"
        << recorder.getRecordedLogsAsString();

    EXPECT_GT(recorder.getRecordedLogCount(), 0);
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "Async callback message"));
}

// Test: Update callback level
TEST_F(IntegrationFrontendUserLogging, UpdateCallbackLevel)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Register callback at INFO
    registerUserCallback(HIPDNN_SEV_INFO, LogCallbackMode::ASYNC);
    auto error = setGlobalLogLevel(HIPDNN_SEV_INFO);
    ASSERT_EQ(error.code, ErrorCode::OK);

    const auto countBefore = recorder.getRecordedLogCount();
    HIPDNN_FE_LOG_INFO("Message at INFO level");

    // Wait for the INFO log to arrive
    EXPECT_TRUE(recorder.waitForLogCount(countBefore + 1, ASYNC_LOG_TIMEOUT))
        << "Timed out waiting for INFO log delivery\nRecorded logs:\n"
        << recorder.getRecordedLogsAsString();

    const size_t infoLogs = recorder.getRecordedLogCount();
    EXPECT_GT(infoLogs, 0);

    // Update callback to WARN level (same callback, same handle, different level)
    registerUserCallback(HIPDNN_SEV_WARN, LogCallbackMode::ASYNC);

    HIPDNN_FE_LOG_INFO("Info after update - should be filtered");
    HIPDNN_FE_LOG_WARN("Warn after update - should pass");

    // Wait for the WARN log to arrive (INFO should be filtered)
    EXPECT_TRUE(recorder.waitForLogsContaining({"Warn after update"}, ASYNC_LOG_TIMEOUT))
        << "Timed out waiting for log containing 'Warn after update'\nRecorded logs:\n"
        << recorder.getRecordedLogsAsString();

    // INFO should be filtered after updating callback to WARN level
    EXPECT_FALSE(recorder.hasLogContaining("Info after update"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "Warn after update"));
}

// Test: Switch between sync and async mode
TEST_F(IntegrationFrontendUserLogging, SwitchBetweenSyncAndAsync)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Start with SYNC mode
    registerUserCallback(HIPDNN_SEV_INFO, LogCallbackMode::SYNC);
    auto error = setGlobalLogLevel(HIPDNN_SEV_INFO);
    ASSERT_EQ(error.code, ErrorCode::OK);

    HIPDNN_FE_LOG_INFO("Sync mode message");

    // Sync: logs immediately available
    const size_t syncLogs = recorder.getRecordedLogCount();
    EXPECT_GT(syncLogs, 0);

    // Switch to ASYNC mode (same callback, same handle, different mode)
    registerUserCallback(HIPDNN_SEV_INFO, LogCallbackMode::ASYNC);

    HIPDNN_FE_LOG_INFO("Async mode message");

    // Wait for async callback to deliver the log
    EXPECT_TRUE(recorder.waitForLogsContaining({"Async mode message"}, ASYNC_LOG_TIMEOUT))
        << "Timed out waiting for log containing 'Async mode message'\nRecorded logs:\n"
        << recorder.getRecordedLogsAsString();

    const size_t asyncLogs = recorder.getRecordedLogCount();
    EXPECT_GT(asyncLogs, syncLogs);
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "Async mode message"));
}

// Test: Switch from async to sync mode
TEST_F(IntegrationFrontendUserLogging, SwitchBetweenAsyncAndSync)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Start with ASYNC mode
    registerUserCallback(HIPDNN_SEV_INFO, LogCallbackMode::ASYNC);
    auto error = setGlobalLogLevel(HIPDNN_SEV_INFO);
    ASSERT_EQ(error.code, ErrorCode::OK);

    const auto countBefore = recorder.getRecordedLogCount();
    HIPDNN_FE_LOG_INFO("Async mode message");

    // Wait for async callback to deliver the log
    EXPECT_TRUE(recorder.waitForLogCount(countBefore + 1, ASYNC_LOG_TIMEOUT))
        << "Timed out waiting for async log delivery\nRecorded logs:\n"
        << recorder.getRecordedLogsAsString();
    const size_t asyncLogs = recorder.getRecordedLogCount();
    EXPECT_GT(asyncLogs, 0);

    // Switch to SYNC mode (same callback, same handle, different mode)
    registerUserCallback(HIPDNN_SEV_INFO, LogCallbackMode::SYNC);

    HIPDNN_FE_LOG_INFO("Sync mode message");

    // Sync: logs immediately available
    const size_t afterSwitch = recorder.getRecordedLogCount();
    EXPECT_GT(afterSwitch, asyncLogs);
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "Sync mode message"));
}

// Test: Null callback parameter is rejected
TEST_F(IntegrationFrontendUserLogging, RejectsNullCallback)
{
    auto error = setUserLogCallback(nullptr, HIPDNN_SEV_INFO, LogCallbackMode::ASYNC, this);
    EXPECT_NE(error.code, ErrorCode::OK);
}

// Test: Null userHandle parameter is rejected
TEST_F(IntegrationFrontendUserLogging, RejectsNullHandle)
{
    auto error = setUserLogCallback(IsolatedLogRecorder::getIsolatedUserRecordingCallback(),
                                    HIPDNN_SEV_INFO,
                                    LogCallbackMode::ASYNC,
                                    nullptr);
    EXPECT_NE(error.code, ErrorCode::OK);
}

// Test: Synchronous guarantee on unregister
TEST_F(IntegrationFrontendUserLogging, SyncGuaranteeOnUnregister)
{
    // Track callback invocations
    static std::atomic<int> s_callbackCount{0};
    static std::mutex s_callbackMutex;
    static std::condition_variable s_callbackCV;

    auto trackingCallback = [](hipdnnUserLogCallbackHandle_t, hipdnnSeverity_t, const char*) {
        s_callbackCount.fetch_add(1);
        s_callbackCV.notify_all();
        // Simulate some work
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    };

    int handleToken = 0;
    s_callbackCount.store(0);

    // Register async callback
    auto error = setUserLogCallback(
        trackingCallback, HIPDNN_SEV_INFO, LogCallbackMode::ASYNC, &handleToken);
    ASSERT_EQ(error.code, ErrorCode::OK);
    error = setGlobalLogLevel(HIPDNN_SEV_INFO);
    ASSERT_EQ(error.code, ErrorCode::OK);

    // Generate many log messages
    for(int i = 0; i < 50; ++i)
    {
        HIPDNN_FE_LOG_INFO("Test message");
    }

    // Unregister with SEV_OFF - should provide synchronous guarantee
    error = setUserLogCallback(
        trackingCallback, HIPDNN_SEV_OFF, LogCallbackMode::ASYNC, &handleToken);
    ASSERT_EQ(error.code, ErrorCode::OK);

    // Take snapshot AFTER unregister to avoid race with async queue processing
    const int countAfterUnregister = s_callbackCount.load();

    // Generate more logs - callback should NOT receive them
    for(int i = 0; i < 10; ++i)
    {
        HIPDNN_FE_LOG_INFO("After unregister");
    }

    // Negative assertion: bounded wait to confirm no new callbacks arrive
    {
        std::unique_lock<std::mutex> lock(s_callbackMutex);
        s_callbackCV.wait_for(lock, NEGATIVE_ASSERT_TIMEOUT, [&] {
            return s_callbackCount.load() > countAfterUnregister;
        });
    }

    // Count should not have increased after unregister
    EXPECT_EQ(s_callbackCount.load(), countAfterUnregister)
        << "No new callbacks should execute after unregister";
}

// Test: Multiple callbacks (2 async + 2 sync) all receive logs independently
TEST_F(IntegrationFrontendUserLogging, MultipleCallbacksAllReceiveLogs)
{
    struct CounterWithNotify
    {
        std::atomic<int> count{0};
        std::mutex mtx;
        std::condition_variable cv;
    };

    auto countingCallback
        = [](hipdnnUserLogCallbackHandle_t userHandle, hipdnnSeverity_t, const char*) {
              auto* c = static_cast<CounterWithNotify*>(userHandle);
              c->count.fetch_add(1);
              c->cv.notify_all();
          };

    CounterWithNotify asyncCount1;
    CounterWithNotify asyncCount2;
    CounterWithNotify syncCount1;
    CounterWithNotify syncCount2;

    auto error = setGlobalLogLevel(HIPDNN_SEV_INFO);
    ASSERT_EQ(error.code, ErrorCode::OK);

    // Register 4 callbacks: 2 async, 2 sync
    error = setUserLogCallback(
        countingCallback, HIPDNN_SEV_INFO, LogCallbackMode::ASYNC, &asyncCount1);
    ASSERT_EQ(error.code, ErrorCode::OK);
    error = setUserLogCallback(
        countingCallback, HIPDNN_SEV_INFO, LogCallbackMode::ASYNC, &asyncCount2);
    ASSERT_EQ(error.code, ErrorCode::OK);
    error
        = setUserLogCallback(countingCallback, HIPDNN_SEV_INFO, LogCallbackMode::SYNC, &syncCount1);
    ASSERT_EQ(error.code, ErrorCode::OK);
    error
        = setUserLogCallback(countingCallback, HIPDNN_SEV_INFO, LogCallbackMode::SYNC, &syncCount2);
    ASSERT_EQ(error.code, ErrorCode::OK);

    // Trigger logging
    HIPDNN_FE_LOG_INFO("Test message for multiple callbacks");

    // Wait for async callbacks to deliver logs
    {
        std::unique_lock<std::mutex> lock(asyncCount1.mtx);
        asyncCount1.cv.wait_for(
            lock, ASYNC_LOG_TIMEOUT, [&] { return asyncCount1.count.load() > 0; });
    }
    {
        std::unique_lock<std::mutex> lock(asyncCount2.mtx);
        asyncCount2.cv.wait_for(
            lock, ASYNC_LOG_TIMEOUT, [&] { return asyncCount2.count.load() > 0; });
    }

    // All 4 callbacks should have received logs
    EXPECT_GT(asyncCount1.count.load(), 0) << "First async callback should receive logs";
    EXPECT_GT(asyncCount2.count.load(), 0) << "Second async callback should receive logs";
    EXPECT_GT(syncCount1.count.load(), 0) << "First sync callback should receive logs";
    EXPECT_GT(syncCount2.count.load(), 0) << "Second sync callback should receive logs";

    // Unregister all 4
    setUserLogCallback(countingCallback, HIPDNN_SEV_OFF, LogCallbackMode::ASYNC, &asyncCount1);
    setUserLogCallback(countingCallback, HIPDNN_SEV_OFF, LogCallbackMode::ASYNC, &asyncCount2);
    setUserLogCallback(countingCallback, HIPDNN_SEV_OFF, LogCallbackMode::SYNC, &syncCount1);
    setUserLogCallback(countingCallback, HIPDNN_SEV_OFF, LogCallbackMode::SYNC, &syncCount2);

    // Capture counts after unregistering
    const int async1After = asyncCount1.count.load();
    const int async2After = asyncCount2.count.load();
    const int sync1After = syncCount1.count.load();
    const int sync2After = syncCount2.count.load();

    // Trigger more logging — none of the callbacks should be invoked
    HIPDNN_FE_LOG_INFO("Message after all callbacks unregistered");

    // Negative assertion: bounded wait to confirm no new callbacks arrive
    {
        std::unique_lock<std::mutex> lock(asyncCount1.mtx);
        asyncCount1.cv.wait_for(
            lock, NEGATIVE_ASSERT_TIMEOUT, [&] { return asyncCount1.count.load() > async1After; });
    }

    EXPECT_EQ(asyncCount1.count.load(), async1After)
        << "First async callback should not receive logs after unregister";
    EXPECT_EQ(asyncCount2.count.load(), async2After)
        << "Second async callback should not receive logs after unregister";
    EXPECT_EQ(syncCount1.count.load(), sync1After)
        << "First sync callback should not receive logs after unregister";
    EXPECT_EQ(syncCount2.count.load(), sync2After)
        << "Second sync callback should not receive logs after unregister";
}
