// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Test for shutdown safety with multiple active threads and user callbacks.
 *
 * This program tests that hipDNN handles graceful shutdown when:
 * 1. Multiple threads are actively using hipDNN APIs during normal operation
 * 2. Some threads have registered user log callbacks (async mode)
 * 3. Threads are NOT joined before main() returns
 * 4. Threads are joined during static destruction (in TestLoggerShutdown destructor)
 * 5. User callbacks are NOT explicitly deregistered before threads exit
 * 6. New threads are spawned during static destruction (before and after joining workers)
 *
 * Important: main() does NOT call any hipDNN functions. All hipDNN interaction
 * occurs from worker threads and the TestLoggerShutdown destructor. This ensures
 * the main thread never acquires a thread_local s_tlRef to BackendLogState,
 * meaning BackendLogState is truly destroyed once all worker threads are joined
 * (no lingering main-thread reference keeping it alive).
 *
 * Static destruction order (key to understanding this test):
 *
 *   sTestLoggerShutdown is a file-scope static, constructed BEFORE main() starts.
 *   BackendLogState lives in a function-local static (s_state in getBackendLogState()),
 *   constructed by the first worker thread that triggers logging.
 *
 *   C++ destroys statics in reverse construction order, so:
 *     1. atexit handlers run first (sets s_loggingShutdown flag via atomic)
 *     2. s_state (BackendLogState's shared_ptr) is destroyed
 *        - The shared_ptr object is destroyed (UB to dereference after this)
 *        - BackendLogState object survives only if thread_local s_tlRef copies
 *          still hold references (refcount > 0)
 *        - The main thread has NO s_tlRef (never called hipDNN functions)
 *     3. sTestLoggerShutdown destructor runs
 *        - Worker threads may still be running (they hold s_tlRef copies)
 *        - Spawns new threads, signals workers to stop, joins everything
 *        - When each worker thread terminates, its thread_local s_tlRef is
 *          destroyed, decrementing BackendLogState's refcount
 *        - After the last worker is joined, BackendLogState is truly destroyed
 *          (no remaining references — main thread never had one)
 *
 *   The atexit handler is the primary safety mechanism:
 *     - Sets s_loggingShutdown flag (an independent std::atomic<bool>)
 *     - Both initialize() and backendLoggingCallback() check this flag first
 *       and return early if set, before touching getBackendLogState()
 *     - This prevents any thread (existing or new) from accessing the
 *       destroyed s_state shared_ptr or BackendLogState infrastructure
 *
 * Expected behavior: No crashes, no deadlocks, clean exit.
 */

#include <hipdnn_backend.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace
{

constexpr size_t NUM_PLAIN_WORKERS = 2;
constexpr size_t NUM_CALLBACK_WORKERS = 4;
constexpr size_t TOTAL_WORKERS = NUM_PLAIN_WORKERS + NUM_CALLBACK_WORKERS;

// Delay after receiving stop signal, before final log attempt.
// This exercises the window where the thread is winding down while
// static destruction may be progressing on the main thread.
constexpr auto POST_STOP_DELAY = std::chrono::milliseconds(20);

// Delay after joining all workers, before spawning the post-join thread.
constexpr auto POST_JOIN_DELAY = std::chrono::milliseconds(50);

// Synchronization for ensuring all workers have produced at least one log
// before main() returns. Without this, on heavily loaded systems, workers
// may not get scheduled before the atexit handler disables logging.
std::atomic<int> gWorkersReady{0};
std::mutex gReadyMutex;
std::condition_variable gReadyCV;

// --- User callback infrastructure ---

// Number of callback workers that use the slow (10ms delay) callback.
// The remaining callback workers use the fast (no delay) callback.
constexpr size_t NUM_SLOW_CALLBACK_WORKERS = 2;

struct CallbackData
{
    std::atomic<int> callCount{0};
    int threadId = 0;
};

void testUserCallback(hipdnnUserLogCallbackHandle_t userHandle,
                      hipdnnSeverity_t /*severity*/,
                      const char* /*message*/)
{
    auto* data = static_cast<CallbackData*>(userHandle);
    data->callCount.fetch_add(1, std::memory_order_relaxed);
}

// Simulates a poorly behaved user callback that takes 10ms per invocation.
// This exercises the shutdown path when async callbacks are still in-progress:
// loggerShutdownLocked() must wait for these slow callbacks to complete via
// waitForIdle() before destroying the callback infrastructure.
void slowUserCallback(hipdnnUserLogCallbackHandle_t userHandle,
                      hipdnnSeverity_t /*severity*/,
                      const char* /*message*/)
{
    auto* data = static_cast<CallbackData*>(userHandle);
    data->callCount.fetch_add(1, std::memory_order_relaxed);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

// --- Helpers ---

// Create and immediately destroy a hipDNN handle. This triggers backend
// logging internally, which is the operation we want to exercise during
// various stages of shutdown.
void createAndDestroyHandle(const char* context)
{
    hipdnnHandle_t handle = nullptr;
    const hipdnnStatus_t status = hipdnnCreate(&handle);
    if(status == HIPDNN_STATUS_SUCCESS && handle != nullptr)
    {
        hipdnnDestroy(handle);
    }
    else
    {
        std::cout << "[Test] " << context << ": hipdnnCreate returned " << status << '\n';
    }
}

// Run a worker loop that continuously creates/destroys handles until the stop
// flag is set. Signals readiness after the first API call completes.
void runWorkerLoop(std::atomic<bool>& stopFlag, const char* context)
{
    bool signaled = false;
    while(!stopFlag.load(std::memory_order_acquire))
    {
        createAndDestroyHandle(context);
        if(!signaled)
        {
            gWorkersReady.fetch_add(1, std::memory_order_release);
            gReadyCV.notify_one();
            signaled = true;
        }
        std::this_thread::yield();
    }
}

// Spawn a new thread that makes a hipDNN API call, then join it.
// The new thread has never called getBackendLogState(), so it has no
// thread_local s_tlRef. The atexit handler (log level OFF) prevents it
// from reaching getBackendLogState() via isLogLevelEnabled() early return.
void spawnAndJoinOneShot(const char* label)
{
    std::cout << "[Test] Spawning " << label << " thread...\n";
    std::thread t([label]() {
        std::cout << "[Test]   " << label << " thread: calling hipdnnCreate\n";
        createAndDestroyHandle(label);
        std::cout << "[Test]   " << label << " thread: done\n";
    });
    t.join();
    std::cout << "[Test] " << label << " thread joined\n";
}

// --- Worker thread functions ---

void plainWorker(std::atomic<bool>& stopFlag, int threadId)
{
    std::cout << "[Test] Plain worker " << threadId << " started\n";

    // Set log level from this thread. This is idempotent — multiple workers
    // may call it concurrently, all setting the same value.
    hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO);

    runWorkerLoop(stopFlag, "plain worker (loop)");

    // After receiving the stop signal, delay and make one more API call.
    // By this point the atexit handler has set log level to OFF, so the
    // logging path in hipdnnCreate will early-return. This verifies that
    // the API call itself still completes without crashing.
    std::this_thread::sleep_for(POST_STOP_DELAY);
    createAndDestroyHandle("plain worker (post-stop)");

    std::cout << "[Test] Plain worker " << threadId << " exiting\n";
}

void callbackWorker(std::atomic<bool>& stopFlag,
                    int threadId,
                    CallbackData* cbData,
                    hipdnnUserLogCallback_t callback)
{
    std::cout << "[Test] Callback worker " << threadId << " started\n";

    // Set log level from this thread. Idempotent if another worker already set it.
    hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO);

    // Register user callback at INFO level, async mode.
    hipdnnSetUserLogCallback_ext(callback,
                                 HIPDNN_SEV_INFO,
                                 HIPDNN_LOG_CALLBACK_ASYNC,
                                 static_cast<hipdnnUserLogCallbackHandle_t>(cbData));

    runWorkerLoop(stopFlag, "callback worker (loop)");

    // Intentionally do NOT deregister the user callback before exiting.
    // This tests that BackendLogState::~BackendLogState (via loggerShutdownLocked)
    // properly handles callbacks that were never explicitly removed by the user:
    //   1. Atomically disables each callback (stores nullptr in callbackHolder)
    //   2. Waits for any in-progress callback invocation (waitForIdle)
    //   3. Clears the callback tracking map
    //   4. Destroys the async thread pool (blocks until queued work completes)

    // Delay and make one more API call after receiving the stop signal.
    std::this_thread::sleep_for(POST_STOP_DELAY);
    createAndDestroyHandle("callback worker (post-stop)");

    std::cout << "[Test] Callback worker " << threadId << " exiting (callback NOT deregistered)\n";
}

} // namespace

// --- Global shutdown test object ---
//
// File-scope static: constructed BEFORE main() runs.
// BackendLogState (function-local static in getBackendLogState()) is
// constructed DURING main() on first logging call.
//
// Destruction is reverse construction order:
//   1. atexit handlers (log level → OFF)
//   2. ~s_state (BackendLogState's shared_ptr destroyed)
//   3. ~TestLoggerShutdown (this destructor — joins threads, spawns new ones)

struct TestLoggerShutdown
{
    std::vector<std::thread> workers;
    std::atomic<bool> stopFlag{false};
    std::array<CallbackData, NUM_CALLBACK_WORKERS> callbackData;
    bool testStarted = false; // Set to true when the test actually runs

    ~TestLoggerShutdown()
    {
        // If the test never started (e.g., --gtest_list_tests was passed),
        // skip all shutdown logic to avoid aborting on zero callback counts.
        if(!testStarted)
        {
            return;
        }

        std::cout << "\n[Test] === TestLoggerShutdown destructor running ===\n";
        std::cout << "[Test] At this point:\n";
        std::cout << "[Test]   - atexit handler has set log level to OFF\n";
        std::cout << "[Test]   - BackendLogState's s_state (function-local static shared_ptr)\n";
        std::cout << "[Test]     has been destroyed (constructed AFTER sTestLoggerShutdown,\n";
        std::cout << "[Test]     therefore destroyed BEFORE sTestLoggerShutdown)\n";
        std::cout << "[Test]   - " << workers.size() << " worker threads may still be running\n";
        std::cout << "[Test]   - Worker threads' thread_local s_tlRef copies are the ONLY\n";
        std::cout << "[Test]     references keeping BackendLogState alive (main has none)\n\n";

        // Step 1: Spawn a new thread BEFORE stopping workers.
        // This thread has never called getBackendLogState() and has no
        // thread_local s_tlRef. The atexit handler (log level OFF) prevents
        // it from reaching getBackendLogState() — isLogLevelEnabled() returns
        // false, so backendLoggingCallback() returns early.
        spawnAndJoinOneShot("pre-join");

        // Step 2: Signal all worker threads to stop.
        // They will delay and make one more API call before exiting.
        std::cout << "\n[Test] Signaling " << workers.size() << " worker threads to stop...\n";
        stopFlag.store(true, std::memory_order_release);

        // Step 3: Join all worker threads.
        // When each thread terminates, its thread_local s_tlRef (shared_ptr
        // to BackendLogState) is destroyed, decrementing the refcount.
        // The 4 callback workers exit without deregistering their callbacks.
        for(size_t i = 0; i < workers.size(); ++i)
        {
            if(workers[i].joinable())
            {
                workers[i].join();
                std::cout << "[Test] Worker " << i << " joined\n";
            }
        }
        workers.clear();

        std::cout << "\n[Test] All worker threads joined\n";

        // Report callback invocation counts and verify each worker received at least one.
        bool allCallbacksReceived = true;
        for(size_t i = 0; i < NUM_CALLBACK_WORKERS; ++i)
        {
            const int count = callbackData[i].callCount.load();
            std::cout << "[Test] Callback worker " << i << " received " << count
                      << " log callbacks\n";
            if(count == 0)
            {
                std::cerr << "[Test] FAIL: Callback worker " << i << " received no log callbacks\n";
                allCallbacksReceived = false;
            }
        }
        if(!allCallbacksReceived)
        {
            std::cerr << "[Test] FAIL: One or more callback workers received no callbacks\n";
            // Use std::abort() to signal test failure. We intentionally avoid exit() and
            // _Exit() here: std::abort() is C++ standard and produces a clear crash signal.
            std::abort();
        }

        // Step 4: Delay after joining all workers.
        // All worker threads' thread_local s_tlRef copies have been destroyed.
        // Since main() never called any hipDNN functions, the main thread has
        // no s_tlRef. BackendLogState is now truly destroyed (refcount reached 0
        // when the last worker thread was joined).
        std::cout << "\n[Test] Delaying " << POST_JOIN_DELAY.count() << "ms after joining...\n";
        std::this_thread::sleep_for(POST_JOIN_DELAY);

        // Step 5: Spawn another thread AFTER all workers have exited.
        // At this point, BackendLogState is fully destroyed (not just s_state,
        // but the object itself — no remaining shared_ptr references).
        // The atexit handler (log level OFF) is the only thing preventing
        // this thread from accessing the destroyed infrastructure.
        spawnAndJoinOneShot("post-join");

        std::cout << "\n[Test] === TestLoggerShutdown destructor finished ===\n";
    }
};

static TestLoggerShutdown sTestLoggerShutdown;

int main(int argc, char* argv[])
{
    // Handle --gtest_list_tests for compatibility with test_name_validator.py.
    // This test is not a gtest harness, but the validator script expects all test
    // executables to respond to --gtest_list_tests with a list of test names.
    for(int i = 1; i < argc; ++i)
    {
        if(std::strcmp(argv[i], "--gtest_list_tests") == 0)
        {
            std::cout << "TestBackendLogging.\n";
            std::cout << "  ShutdownSafety\n";
            return 0;
        }
    }

    // Set log level via environment variable so the backend logger will produce output.
    // Must be set before any hipDNN function is called (i.e., before worker threads start).
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");

#ifdef __linux__
    // Create a temporary file for HIPDNN_LOG_FILE. The file is removed before main()
    // returns, while spdlog's file sink still has the fd open. On POSIX, unlink()
    // removes the directory entry but the inode remains valid as long as spdlog holds
    // the fd — writes silently succeed to the orphaned inode. This tests that the
    // file sink doesn't crash when the underlying file is deleted during operation.
    // Skipped on Windows: remove() would fail because the file is still open (no
    // FILE_SHARE_DELETE with standard fopen).
    //
    // This MUST be set before any hipDNN API call (including plugin path setup
    // below), because the first API call triggers logging initialization via
    // LOG_API_ENTRY → getBackendLogState(). If HIPDNN_LOG_FILE is not yet set
    // at that point, spdlog creates a console sink instead of a file sink.
    std::string logFilePath = "./hipdnn_shutdown_test_XXXXXX";
    const int logFd = mkstemp(logFilePath.data());
    if(logFd < 0)
    {
        std::cerr << "[Test] FAIL: mkstemp failed\n";
        return 1;
    }
    close(logFd); // Close the fd; spdlog will reopen it by path.
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_FILE", logFilePath.c_str());
    std::cout << "[Test] Created temp log file: " << logFilePath << "\n";
#endif

    // Constrain plugin loading to the known-safe test_good_default_plugin.
    // Without this, the default plugin discovery may load MIOpen or other plugins
    // whose static objects can cause segfaults during the static destruction
    // stress test. Using ABSOLUTE mode ensures only the specified plugin is loaded.
    //
    // The API call is made from a helper thread rather than main() because
    // hipdnnSetEnginePluginPaths_ext() triggers logging initialization via
    // LOG_API_ENTRY, which calls getBackendLogState(). If called from main(),
    // main() would acquire a thread_local s_tlRef to BackendLogState, keeping
    // it alive longer than intended during static destruction. By using a
    // short-lived helper thread, the s_tlRef is destroyed when the thread
    // exits, and BackendLogState's lifetime depends only on worker threads.
    {
        const std::string pluginPath
            = (std::filesystem::path(".") / "test_plugins" / "default"
               / hipdnn_data_sdk::utilities::getLibraryName(TEST_GOOD_DEFAULT_PLUGIN_NAME))
                  .string();
        hipdnnStatus_t pluginStatus = HIPDNN_STATUS_SUCCESS;
        std::thread pluginSetup([&] {
            const std::array<const char*, 1> paths = {pluginPath.c_str()};
            pluginStatus = hipdnnSetEnginePluginPaths_ext(
                paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE);
        });
        pluginSetup.join();
        if(pluginStatus != HIPDNN_STATUS_SUCCESS)
        {
            std::cerr << "[Test] FAIL: hipdnnSetEnginePluginPaths_ext returned " << pluginStatus
                      << '\n';
            return 1;
        }
        std::cout << "[Test] Plugin path set to: " << pluginPath << '\n';
    }

    // Mark the test as started so the destructor knows to run shutdown logic.
    sTestLoggerShutdown.testStarted = true;

    std::cout << "[Test] main() starting\n";
    std::cout << "[Test] main() will NOT call any hipDNN functions directly.\n";
    std::cout << "[Test] This ensures the main thread has no thread_local s_tlRef\n";
    std::cout << "[Test] to BackendLogState, so it won't keep the object alive.\n\n";

    // Spawn 2 plain worker threads (no user callback registration).
    // These threads set the log level and continuously create/destroy handles,
    // generating backend logs. The first worker to call a hipDNN function will
    // trigger construction of BackendLogState's s_state (function-local static).
    for(size_t i = 0; i < NUM_PLAIN_WORKERS; ++i)
    {
        sTestLoggerShutdown.workers.emplace_back(
            plainWorker, std::ref(sTestLoggerShutdown.stopFlag), static_cast<int>(i));
    }

    // Spawn 4 callback worker threads.
    // Each sets the log level, registers an async user callback at INFO level,
    // then continuously creates/destroys handles. The callbacks are never
    // explicitly deregistered.
    // The first NUM_SLOW_CALLBACK_WORKERS use slowUserCallback (10ms delay per
    // invocation) to simulate a poorly behaved callback that is still in-progress
    // when shutdown begins. The remainder use the fast testUserCallback.
    for(size_t i = 0; i < NUM_CALLBACK_WORKERS; ++i)
    {
        sTestLoggerShutdown.callbackData[i].threadId = static_cast<int>(i);
        auto* callback = (i < NUM_SLOW_CALLBACK_WORKERS) ? &slowUserCallback : &testUserCallback;
        sTestLoggerShutdown.workers.emplace_back(callbackWorker,
                                                 std::ref(sTestLoggerShutdown.stopFlag),
                                                 static_cast<int>(i),
                                                 &sTestLoggerShutdown.callbackData[i],
                                                 callback);
    }

    std::cout << "[Test] Spawned " << TOTAL_WORKERS << " worker threads (" << NUM_PLAIN_WORKERS
              << " plain + " << NUM_CALLBACK_WORKERS << " with async callbacks)\n";
    std::cout << "[Test] main() returning WITHOUT joining threads\n\n";
    std::cout << "[Test] Shutdown sequence will be:\n";
    std::cout << "[Test]   1. atexit handler runs → log level set to OFF\n";
    std::cout << "[Test]   2. ~s_state runs → BackendLogState's shared_ptr destroyed\n";
    std::cout << "[Test]      (main thread holds no s_tlRef — never called hipDNN)\n";
    std::cout << "[Test]   3. ~TestLoggerShutdown runs → joins threads, spawns new ones\n";
    std::cout << "[Test]      After last worker joined, BackendLogState is truly destroyed\n";
    std::cout << "\n";

    // Wait for all worker threads to complete at least one hipDNN API call cycle,
    // ensuring they have produced log output and (for callback workers) their
    // registered callbacks have been invoked at least once. This replaces a fixed
    // delay which could be insufficient on heavily loaded systems.
    {
        std::unique_lock<std::mutex> lock(gReadyMutex);
        if(!gReadyCV.wait_for(lock, std::chrono::seconds(30), [&] {
               return gWorkersReady.load(std::memory_order_acquire)
                      >= static_cast<int>(TOTAL_WORKERS);
           }))
        {
            std::cerr << "[Test] FAIL: Not all workers produced logs within 30 seconds ("
                      << gWorkersReady.load() << "/" << TOTAL_WORKERS << " ready)\n";
            return 1;
        }
    }

#ifdef __linux__
    // Remove the temp log file while spdlog's file sink still has it open.
    // On POSIX, this removes the directory entry but the inode stays valid —
    // spdlog continues writing to the orphaned fd without error. The inode
    // is reclaimed when spdlog closes the fd during shutdown.
    std::remove(logFilePath.c_str());
    std::cout << "[Test] Removed temp log file (spdlog still has fd open)\n";
#endif

    return 0;
}
