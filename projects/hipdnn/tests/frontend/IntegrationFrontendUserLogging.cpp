// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include <atomic>
#include <chrono>
#include <thread>

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

    // Small delay for async callback
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Verify logs were received on callback
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

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

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

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

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

    HIPDNN_FE_LOG_INFO("Log before unregistering callback");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    size_t logsWithCallback = recorder.getRecordedLogCount();
    EXPECT_GT(logsWithCallback, 0);

    // Unregister callback with SEV_OFF
    registerUserCallback(HIPDNN_SEV_OFF, LogCallbackMode::ASYNC);

    HIPDNN_FE_LOG_INFO("Log after unregistering callback");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    size_t logsAfterUnregister = recorder.getRecordedLogCount();

    // No new logs should be provided via callback
    EXPECT_EQ(logsAfterUnregister, logsWithCallback);
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

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

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

    // Async: may not be immediate but should arrive soon
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

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

    HIPDNN_FE_LOG_INFO("Message at INFO level");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    size_t infoLogs = recorder.getRecordedLogCount();
    EXPECT_GT(infoLogs, 0);

    // Update callback to WARN level (same callback, same handle, different level)
    registerUserCallback(HIPDNN_SEV_WARN, LogCallbackMode::ASYNC);

    HIPDNN_FE_LOG_INFO("Info after update - should be filtered");
    HIPDNN_FE_LOG_WARN("Warn after update - should pass");

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Should have exactly 1 new log (the WARN)
    EXPECT_EQ(recorder.getRecordedLogCount(), infoLogs + 1);
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
    size_t syncLogs = recorder.getRecordedLogCount();
    EXPECT_GT(syncLogs, 0);

    // Switch to ASYNC mode (same callback, same handle, different mode)
    registerUserCallback(HIPDNN_SEV_INFO, LogCallbackMode::ASYNC);

    HIPDNN_FE_LOG_INFO("Async mode message");

    // Async: need delay
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    size_t asyncLogs = recorder.getRecordedLogCount();
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

    HIPDNN_FE_LOG_INFO("Async mode message");

    // Async: need delay
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    size_t asyncLogs = recorder.getRecordedLogCount();
    EXPECT_GT(asyncLogs, 0);

    // Switch to SYNC mode (same callback, same handle, different mode)
    registerUserCallback(HIPDNN_SEV_INFO, LogCallbackMode::SYNC);

    HIPDNN_FE_LOG_INFO("Sync mode message");

    // Sync: logs immediately available
    size_t afterSwitch = recorder.getRecordedLogCount();
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
    static std::atomic<bool> s_callbackActive{false};

    auto trackingCallback = [](hipdnnUserLogCallbackHandle_t, hipdnnSeverity_t, const char*) {
        s_callbackActive.store(true);
        s_callbackCount.fetch_add(1);
        // Simulate some work
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        s_callbackActive.store(false);
    };

    int handleToken = 0;
    s_callbackCount.store(0);
    s_callbackActive.store(false);

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
    int countAfterUnregister = s_callbackCount.load();

    // After unregister returns, callback should NOT be active
    EXPECT_FALSE(s_callbackActive.load()) << "Callback should not be active after unregister";

    // Generate more logs - callback should NOT receive them
    for(int i = 0; i < 10; ++i)
    {
        HIPDNN_FE_LOG_INFO("After unregister");
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Count should not have increased after unregister
    EXPECT_EQ(s_callbackCount.load(), countAfterUnregister)
        << "No new callbacks should execute after unregister";
}

// Test: Multiple callbacks (2 async + 2 sync) all receive logs independently
TEST_F(IntegrationFrontendUserLogging, MultipleCallbacksAllReceiveLogs)
{
    auto countingCallback
        = [](hipdnnUserLogCallbackHandle_t userHandle, hipdnnSeverity_t, const char*) {
              static_cast<std::atomic<int>*>(userHandle)->fetch_add(1);
          };

    std::atomic<int> asyncCount1{0};
    std::atomic<int> asyncCount2{0};
    std::atomic<int> syncCount1{0};
    std::atomic<int> syncCount2{0};

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

    // Wait for async callbacks to process
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // All 4 callbacks should have received logs
    EXPECT_GT(asyncCount1.load(), 0) << "First async callback should receive logs";
    EXPECT_GT(asyncCount2.load(), 0) << "Second async callback should receive logs";
    EXPECT_GT(syncCount1.load(), 0) << "First sync callback should receive logs";
    EXPECT_GT(syncCount2.load(), 0) << "Second sync callback should receive logs";

    // Unregister all 4
    setUserLogCallback(countingCallback, HIPDNN_SEV_OFF, LogCallbackMode::ASYNC, &asyncCount1);
    setUserLogCallback(countingCallback, HIPDNN_SEV_OFF, LogCallbackMode::ASYNC, &asyncCount2);
    setUserLogCallback(countingCallback, HIPDNN_SEV_OFF, LogCallbackMode::SYNC, &syncCount1);
    setUserLogCallback(countingCallback, HIPDNN_SEV_OFF, LogCallbackMode::SYNC, &syncCount2);

    // Capture counts after unregistering
    int async1After = asyncCount1.load();
    int async2After = asyncCount2.load();
    int sync1After = syncCount1.load();
    int sync2After = syncCount2.load();

    // Trigger more logging — none of the callbacks should be invoked
    HIPDNN_FE_LOG_INFO("Message after all callbacks unregistered");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    EXPECT_EQ(asyncCount1.load(), async1After)
        << "First async callback should not receive logs after unregister";
    EXPECT_EQ(asyncCount2.load(), async2After)
        << "Second async callback should not receive logs after unregister";
    EXPECT_EQ(syncCount1.load(), sync1After)
        << "First sync callback should not receive logs after unregister";
    EXPECT_EQ(syncCount2.load(), sync2After)
        << "Second sync callback should not receive logs after unregister";
}
