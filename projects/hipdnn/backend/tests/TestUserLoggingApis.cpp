// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <logging/Logging.hpp>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace hipdnn_test_sdk::utilities;

// Test fixture for backend user log callback API
class IntegrationBackendUserLoggingApis : public ::testing::Test
{
protected:
    hipdnnSeverity_t _originalLogLevel = HIPDNN_SEV_OFF;
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

        // Save original log level
        ASSERT_EQ(hipdnnBackendGetGlobalLogLevel_ext(&_originalLogLevel), HIPDNN_STATUS_SUCCESS);

        // Clear any previous callback (may fail if not registered — that's OK)
        setIsolatedUserCallback(HIPDNN_SEV_OFF, HIPDNN_LOG_CALLBACK_ASYNC);
    }

    void TearDown() override
    {
        // Clear callback after each test (may fail if not registered — that's OK)
        setIsolatedUserCallback(HIPDNN_SEV_OFF, HIPDNN_LOG_CALLBACK_ASYNC);

        // Restore original log level
        ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(_originalLogLevel), HIPDNN_STATUS_SUCCESS);

        _logLevelGuard.reset();
        _logFileGuard.reset();

        // Clean up any test log file
        if(!_logFile.empty())
        {
            std::remove(_logFile.c_str());
        }
    }

    // Set the isolated user callback. Returns the status without asserting.
    hipdnnStatus_t setIsolatedUserCallback(hipdnnSeverity_t minLevel, hipdnnLogCallbackMode_t mode)
    {
        return hipdnnSetUserLogCallback_ext(
            IsolatedLogRecorder::getIsolatedUserRecordingCallback(), minLevel, mode, this);
    }

    // Register/update/unregister the isolated user callback. Asserts success.
    void registerIsolatedCallback(hipdnnSeverity_t minLevel, hipdnnLogCallbackMode_t mode)
    {
        ASSERT_EQ(setIsolatedUserCallback(minLevel, mode), HIPDNN_STATUS_SUCCESS);
    }
};

// Test: User callback receives backend logs
TEST_F(IntegrationBackendUserLoggingApis, UserCallbackReceivesLogs)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Register test recording callback
    registerIsolatedCallback(HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_ASYNC);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    // Trigger backend logging by creating a handle
    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    // Small delay for async callback
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Backend should log handle creation
    EXPECT_TRUE(recorder.hasLogContaining("API success: [hipdnnCreate]"));

    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

// Test: Unregister callback with SEV_OFF stops log capture
TEST_F(IntegrationBackendUserLoggingApis, UnregisterWithSevOffStopsCapture)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Register callback
    registerIsolatedCallback(HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_ASYNC);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    // Trigger some logging
    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    // Small delay for async callback
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    const size_t logsAfterCreate = recorder.getRecordedLogCount();
    EXPECT_GT(logsAfterCreate, 0);

    // Unregister callback with SEV_OFF
    registerIsolatedCallback(HIPDNN_SEV_OFF, HIPDNN_LOG_CALLBACK_ASYNC);

    // Further operations should not be captured
    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);

    // Small delay to ensure any async logs would have arrived
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Log count should not increase
    const size_t finalLogs = recorder.getRecordedLogCount();
    EXPECT_EQ(finalLogs, logsAfterCreate);
}

// Test: Sync callback executes immediately
TEST_F(IntegrationBackendUserLoggingApis, SyncCallbackImmediate)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Register callback in SYNC mode
    registerIsolatedCallback(HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_SYNC);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    // Trigger logging
    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    // Sync: logs should be available immediately without delay
    EXPECT_GT(recorder.getRecordedLogCount(), 0);

    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

// Test: Null userHandle is rejected
TEST_F(IntegrationBackendUserLoggingApis, RejectsNullHandle)
{
    auto status
        = hipdnnSetUserLogCallback_ext(IsolatedLogRecorder::getIsolatedUserRecordingCallback(),
                                       HIPDNN_SEV_INFO,
                                       HIPDNN_LOG_CALLBACK_ASYNC,
                                       nullptr);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
}

// Test: Null callback is rejected
TEST_F(IntegrationBackendUserLoggingApis, RejectsNullCallback)
{
    auto status
        = hipdnnSetUserLogCallback_ext(nullptr, HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_ASYNC, this);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
}

// Test: Invalid mode is rejected
TEST_F(IntegrationBackendUserLoggingApis, RejectsInvalidMode)
{
    auto invalidMode = static_cast<hipdnnLogCallbackMode_t>(999);
    auto status
        = hipdnnSetUserLogCallback_ext(IsolatedLogRecorder::getIsolatedUserRecordingCallback(),
                                       HIPDNN_SEV_INFO,
                                       invalidMode,
                                       this);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
}

// Test: Callback respects log level filtering
TEST_F(IntegrationBackendUserLoggingApis, CallbackRespectsLogLevel)
{
    auto recorder = IsolatedLogRecorder::withCurrentLevel();

    // Set log level to ERROR (filters out INFO and WARN)
    hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_ERROR);

    // Register test recording callback at INFO (but global level is ERROR)
    registerIsolatedCallback(HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_ASYNC);

    // Trigger backend logging
    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);

    // Small delay for async callback
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // INFO logs should be filtered out by global level check
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_INFO), 0);
}

// Test: Update callback level
TEST_F(IntegrationBackendUserLoggingApis, UpdateCallbackLevel)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Register callback at INFO
    registerIsolatedCallback(HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_ASYNC);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    // Trigger logging
    hipdnnHandle_t handle1 = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle1), HIPDNN_STATUS_SUCCESS);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const size_t infoLogs = recorder.getRecordedLogCount();
    EXPECT_GT(infoLogs, 0);

    // Update callback to WARN level (same callback, same handle, different level)
    registerIsolatedCallback(HIPDNN_SEV_WARN, HIPDNN_LOG_CALLBACK_ASYNC);

    // Trigger more logging - INFO logs should now be filtered at sink level
    hipdnnHandle_t handle2 = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle2), HIPDNN_STATUS_SUCCESS);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const size_t afterUpdate = recorder.getRecordedLogCount();

    // Verify minimal new logs (only WARN+ would be captured now)
    EXPECT_LE(afterUpdate - infoLogs, 1); // May get at most 1 WARN/ERROR log

    ASSERT_EQ(hipdnnDestroy(handle1), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnDestroy(handle2), HIPDNN_STATUS_SUCCESS);
}

// Test: Async callback queues logs
TEST_F(IntegrationBackendUserLoggingApis, AsyncCallbackQueued)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Register callback in ASYNC mode
    registerIsolatedCallback(HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_ASYNC);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    // Trigger logging
    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);

    // Async: logs may not be immediate but should arrive soon
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_GT(recorder.getRecordedLogCount(), 0);
    EXPECT_TRUE(recorder.hasLogContaining("API success: [hipdnnCreate]"));
}

// Test: Callback that throws exception is handled
TEST_F(IntegrationBackendUserLoggingApis, CallbackThrowsException)
{
    // Callback that throws exception
    static bool s_shouldThrow = true;
    auto throwingCallback = [](hipdnnUserLogCallbackHandle_t, hipdnnSeverity_t, const char*) {
        if(s_shouldThrow)
        {
            throw std::runtime_error("Test exception from callback");
        }
    };

    int handleToken = 0;
    s_shouldThrow = true;

    // Set throwing callback
    ASSERT_EQ(hipdnnSetUserLogCallback_ext(
                  throwingCallback, HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_SYNC, &handleToken),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    // Trigger logging - should not crash despite exception
    hipdnnHandle_t handle = nullptr;
    EXPECT_NO_THROW(hipdnnCreate(&handle));

    // Cleanup
    s_shouldThrow = false;
    if(handle != nullptr)
    {
        hipdnnDestroy(handle);
    }

    // Remove callback
    hipdnnSetUserLogCallback_ext(
        throwingCallback, HIPDNN_SEV_OFF, HIPDNN_LOG_CALLBACK_SYNC, &handleToken);
}

// Test: Synchronous guarantee on unregister
TEST_F(IntegrationBackendUserLoggingApis, SyncGuaranteeOnUnregister)
{
    // Track callback invocations
    static std::atomic<int> s_callbackCount{0};
    static std::atomic<bool> s_callbackActive{false};

    auto trackingCallback = [](hipdnnUserLogCallbackHandle_t, hipdnnSeverity_t, const char*) {
        s_callbackActive.store(true);
        s_callbackCount.fetch_add(1);
        // Simulate some work
        std::this_thread::sleep_for(std::chrono::microseconds(10));
        s_callbackActive.store(false);
    };

    int handleToken = 0;
    s_callbackCount.store(0);
    s_callbackActive.store(false);

    // Register async callback
    ASSERT_EQ(hipdnnSetUserLogCallback_ext(
                  trackingCallback, HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_ASYNC, &handleToken),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    // Generate many log messages to fill async queue
    hipdnnHandle_t handle = nullptr;
    for(int i = 0; i < 100; ++i)
    {
        hipdnnCreate(&handle);
        if(handle != nullptr)
        {
            hipdnnDestroy(handle);
        }
    }

    // Unregister with SEV_OFF - should provide synchronous guarantee
    hipdnnSetUserLogCallback_ext(
        trackingCallback, HIPDNN_SEV_OFF, HIPDNN_LOG_CALLBACK_ASYNC, &handleToken);

    // Take snapshot AFTER unregister to avoid race with async queue processing
    const int countAfterUnregister = s_callbackCount.load();

    // After unregister returns, callback should NOT be active
    EXPECT_FALSE(s_callbackActive.load()) << "Callback should not be active after unregister";

    // Generate more logs - callback should NOT receive them
    for(int i = 0; i < 10; ++i)
    {
        hipdnnCreate(&handle);
        if(handle != nullptr)
        {
            hipdnnDestroy(handle);
        }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Count should not have increased after unregister
    EXPECT_EQ(s_callbackCount.load(), countAfterUnregister)
        << "No new callbacks should execute after unregister";
}

// Test: Switch between sync and async mode
TEST_F(IntegrationBackendUserLoggingApis, SwitchBetweenSyncAndAsync)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Start with SYNC mode
    registerIsolatedCallback(HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_SYNC);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    // Sync: logs immediately available
    const size_t syncLogs = recorder.getRecordedLogCount();
    EXPECT_GT(syncLogs, 0);

    // Update to ASYNC mode (same callback, same handle, different mode)
    registerIsolatedCallback(HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_ASYNC);

    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);

    // Async: need delay
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const size_t asyncLogs = recorder.getRecordedLogCount();
    EXPECT_GT(asyncLogs, syncLogs);
}

// Test: Switch from async to sync mode
TEST_F(IntegrationBackendUserLoggingApis, SwitchBetweenAsyncAndSync)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Start with ASYNC mode
    registerIsolatedCallback(HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_ASYNC);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    // Async: need delay
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const size_t asyncLogs = recorder.getRecordedLogCount();
    EXPECT_GT(asyncLogs, 0);

    // Update to SYNC mode (same callback, same handle, different mode)
    registerIsolatedCallback(HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_SYNC);

    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);

    // Sync: logs immediately available
    const size_t afterSwitch = recorder.getRecordedLogCount();
    EXPECT_GT(afterSwitch, asyncLogs);
}

// Test: Duplicate registration updates existing
TEST_F(IntegrationBackendUserLoggingApis, DuplicateUpdatesExisting)
{
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Register callback at INFO
    registerIsolatedCallback(HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_SYNC);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    // Register again with same callback and handle but different level
    // This should UPDATE, not create second registration
    registerIsolatedCallback(HIPDNN_SEV_WARN, HIPDNN_LOG_CALLBACK_SYNC);

    recorder.clearLogs();

    // Trigger INFO level log
    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Should NOT receive INFO logs (updated to WARN)
    // If there were 2 registrations, we'd get logs from the first one
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_INFO), 0)
        << "Should not receive INFO logs after update to WARN level.\n"
        << "Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

// Test: Concurrent logging with user callback toggle between registered and unregistered
TEST_F(IntegrationBackendUserLoggingApis, ConcurrentLoggingWithCallbackToggle)
{
    // Redirect default logging to file to avoid console spam when callback is disabled
    _logFile = "concurrent_user_callback_toggle_test.log";
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_FILE", _logFile.c_str());

    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    constexpr int NUM_LOGGER_THREADS = 4;
    std::mutex mutex;
    std::condition_variable cvStart; // Main -> logger threads: start/stop signals
    std::condition_variable cvProgress; // Logger threads -> main: iteration progress
    bool startFlag = false;
    bool stopFlag = false;
    uint64_t iterationCount = 0;
    int threadsReady = 0;
    std::vector<std::thread> threads;
    threads.reserve(NUM_LOGGER_THREADS);

    constexpr auto THREAD_SYNC_TIMEOUT = std::chrono::seconds(10);

    // Helper to wait until at least one log-generating thread has completed a full loop.
    auto waitForIterations
        = [&](uint64_t snapshot) {
              std::unique_lock<std::mutex> lock(mutex);
              // The iteration count will need to increment by the number of threads plus one
              // to ensure that at least one thread ran through the log generating section of code.
              const uint64_t targetCount = snapshot + static_cast<uint64_t>(NUM_LOGGER_THREADS) + 1;
              const bool completed = cvProgress.wait_for(
                  lock, THREAD_SYNC_TIMEOUT, [&] { return iterationCount >= targetCount; });
              EXPECT_TRUE(completed)
                  << "Timeout waiting for log-generating threads to complete iterations";
              return iterationCount;
          };

    // Register the user callback and set log level
    registerIsolatedCallback(HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_ASYNC);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    // Create threads that generate logs by repeated create/destroy of hipDNN handles
    for(int i = 0; i < NUM_LOGGER_THREADS; ++i)
    {
        threads.emplace_back([&]() {
            {
                std::unique_lock<std::mutex> lock(mutex);
                ++threadsReady;
                cvProgress.notify_one(); // Notify main thread that this thread is ready
                cvStart.wait(lock, [&] { return startFlag; }); // Wait for start.
            }

            hipdnnHandle_t handle = nullptr;

            while(true)
            {
                if(handle == nullptr)
                {
                    hipdnnCreate(&handle);
                }
                else
                {
                    hipdnnDestroy(handle);
                    handle = nullptr;
                }

                {
                    const std::lock_guard<std::mutex> lock(mutex);
                    ++iterationCount;
                    if(stopFlag && handle == nullptr)
                    {
                        cvProgress.notify_all();
                        break;
                    }
                }
                cvProgress.notify_all();
            }
        });
    }

    // Wait for all log-generating threads to be ready, then start them
    {
        std::unique_lock<std::mutex> lock(mutex);
        const bool allReady = cvProgress.wait_for(
            lock, THREAD_SYNC_TIMEOUT, [&] { return threadsReady == NUM_LOGGER_THREADS; });
        ASSERT_TRUE(allReady) << "Timeout waiting for log-generating threads to become ready";
        startFlag = true;
    }
    cvStart.notify_all();

    // Control thread behavior: toggle callback on and off
    constexpr int NUM_CYCLES = 8;
    constexpr auto LOG_WAIT_TIMEOUT = std::chrono::milliseconds(500);

    for(int cycle = 0; cycle < NUM_CYCLES; ++cycle)
    {
        // Use async mode for even cycles, sync mode for odd cycles
        auto mode = (cycle % 2 == 0) ? HIPDNN_LOG_CALLBACK_ASYNC : HIPDNN_LOG_CALLBACK_SYNC;

        // With callback registered - logs should be captured
        registerIsolatedCallback(HIPDNN_SEV_INFO, mode);

        const size_t countBefore = recorder.getRecordedLogCount();
        uint64_t iterSnapshot;
        {
            const std::lock_guard<std::mutex> lock(mutex);
            iterSnapshot = iterationCount;
        }
        waitForIterations(iterSnapshot);

        // For async mode, wait for logs to be delivered from the queue.
        // For sync mode, logs are immediate so no wait needed.
        if(mode == HIPDNN_LOG_CALLBACK_ASYNC)
        {
            const bool logReceived = recorder.waitForLogCount(countBefore + 1, LOG_WAIT_TIMEOUT);
            EXPECT_TRUE(logReceived)
                << "Timed out waiting for logs (cycle " << cycle
                << ", mode=" << (mode == HIPDNN_LOG_CALLBACK_ASYNC ? "async" : "sync") << ")";
        }

        const size_t countAfterEnabled = recorder.getRecordedLogCount();
        EXPECT_GT(countAfterEnabled, countBefore)
            << "Log count should increase when callback is registered (cycle " << cycle
            << ", mode=" << (mode == HIPDNN_LOG_CALLBACK_ASYNC ? "async" : "sync") << ")";

        // With callback unregistered (SEV_OFF) - logs should NOT be captured
        registerIsolatedCallback(HIPDNN_SEV_OFF, mode);

        const size_t countBeforeDisabled = recorder.getRecordedLogCount();
        uint64_t iterSnapshotDisabled;
        {
            const std::lock_guard<std::mutex> lock(mutex);
            iterSnapshotDisabled = iterationCount;
        }
        waitForIterations(iterSnapshotDisabled);

        // For async mode, a short wait for any potential logs to arrive.
        if(mode == HIPDNN_LOG_CALLBACK_ASYNC)
        {
            recorder.waitForLogCount(countBeforeDisabled + 1, std::chrono::milliseconds(50));
        }

        const size_t countAfterDisabled = recorder.getRecordedLogCount();
        EXPECT_EQ(countAfterDisabled, countBeforeDisabled)
            << "Log count should NOT increase when callback is unregistered (cycle " << cycle
            << ", mode=" << (mode == HIPDNN_LOG_CALLBACK_ASYNC ? "async" : "sync") << ")";
    }

    // Stop all threads - logger threads will see stopFlag after destroying their handle
    {
        const std::lock_guard<std::mutex> lock(mutex);
        stopFlag = true;
    }
    for(auto& t : threads)
    {
        t.join();
    }

    // Final verification - total logs should be > 0
    EXPECT_GT(recorder.getRecordedLogCount(), 0);
}

// Test: Reentrant logging from sync callback is prevented (no stack overflow)
TEST_F(IntegrationBackendUserLoggingApis, ReentrantLoggingPrevented)
{
    // Counter to track recursive attempts
    std::atomic<int> recursiveAttempts{0};
    std::atomic<int> callbackInvocations{0};

    // Create a callback that tries to log (which would cause infinite recursion without fix)
    struct RecursiveContext
    {
        std::atomic<int>* attempts;
        std::atomic<int>* invocations;
    };

    RecursiveContext recursiveCtx{&recursiveAttempts, &callbackInvocations};

    auto recursiveCallback = [](hipdnnUserLogCallbackHandle_t userHandle,
                                hipdnnSeverity_t /*severity*/,
                                const char* /*message*/) {
        auto* ctx = static_cast<RecursiveContext*>(userHandle);

        // Count this invocation
        ctx->invocations->fetch_add(1);

        // Try to log from within the callback
        // Without the fix, this would cause infinite recursion and stack overflow
        // With the fix, recursion is detected and the recursive log is dropped
        HIPDNN_BACKEND_LOG_INFO("Recursive log attempt");
        ctx->attempts->fetch_add(1);
    };

    // Register the recursive callback in SYNC mode (where recursion is most problematic)
    ASSERT_EQ(hipdnnSetUserLogCallback_ext(
                  recursiveCallback, HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_SYNC, &recursiveCtx),
              HIPDNN_STATUS_SUCCESS);

    // Set log level to INFO to enable logging
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    // Trigger logging by creating a hipDNN handle (generates backend logs)
    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    // Verify the callback was invoked at least once
    const int invocations = callbackInvocations.load();
    EXPECT_GT(invocations, 0) << "Callback should be invoked for backend logs";

    // Verify recursive attempts were made
    const int attempts = recursiveAttempts.load();
    EXPECT_GT(attempts, 0) << "Should have attempted recursive logging";

    // Invocations should equal attempts (if recursion was NOT prevented there would
    // be more invocations than attempts, but more likely a stack overflow)
    EXPECT_EQ(invocations, attempts);

    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnSetUserLogCallback_ext(
                  recursiveCallback, HIPDNN_SEV_OFF, HIPDNN_LOG_CALLBACK_SYNC, &recursiveCtx),
              HIPDNN_STATUS_SUCCESS);
}

// Test: Multiple callbacks (2 async + 2 sync) all receive logs independently
TEST_F(IntegrationBackendUserLoggingApis, MultipleCallbacksAllReceiveLogs)
{
    auto countingCallback
        = [](hipdnnUserLogCallbackHandle_t userHandle, hipdnnSeverity_t, const char*) {
              static_cast<std::atomic<int>*>(userHandle)->fetch_add(1);
          };

    std::atomic<int> asyncCount1{0};
    std::atomic<int> asyncCount2{0};
    std::atomic<int> syncCount1{0};
    std::atomic<int> syncCount2{0};

    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    // Register 4 callbacks: 2 async, 2 sync
    ASSERT_EQ(hipdnnSetUserLogCallback_ext(
                  countingCallback, HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_ASYNC, &asyncCount1),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnSetUserLogCallback_ext(
                  countingCallback, HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_ASYNC, &asyncCount2),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnSetUserLogCallback_ext(
                  countingCallback, HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_SYNC, &syncCount1),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnSetUserLogCallback_ext(
                  countingCallback, HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_SYNC, &syncCount2),
              HIPDNN_STATUS_SUCCESS);

    // Trigger logging
    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    // Wait for async callbacks to process
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // All 4 callbacks should have received logs
    EXPECT_GT(asyncCount1.load(), 0) << "First async callback should receive logs";
    EXPECT_GT(asyncCount2.load(), 0) << "Second async callback should receive logs";
    EXPECT_GT(syncCount1.load(), 0) << "First sync callback should receive logs";
    EXPECT_GT(syncCount2.load(), 0) << "Second sync callback should receive logs";

    // Unregister all 4
    hipdnnSetUserLogCallback_ext(
        countingCallback, HIPDNN_SEV_OFF, HIPDNN_LOG_CALLBACK_ASYNC, &asyncCount1);
    hipdnnSetUserLogCallback_ext(
        countingCallback, HIPDNN_SEV_OFF, HIPDNN_LOG_CALLBACK_ASYNC, &asyncCount2);
    hipdnnSetUserLogCallback_ext(
        countingCallback, HIPDNN_SEV_OFF, HIPDNN_LOG_CALLBACK_SYNC, &syncCount1);
    hipdnnSetUserLogCallback_ext(
        countingCallback, HIPDNN_SEV_OFF, HIPDNN_LOG_CALLBACK_SYNC, &syncCount2);

    // Capture counts after unregistering
    const int async1After = asyncCount1.load();
    const int async2After = asyncCount2.load();
    const int sync1After = syncCount1.load();
    const int sync2After = syncCount2.load();

    // Trigger more logging — none of the callbacks should be invoked
    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
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
