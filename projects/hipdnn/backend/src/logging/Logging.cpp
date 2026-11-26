// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Logging.hpp"
#include "GraphLogger.hpp"
#include "LastErrorManager.hpp"
#include "PlatformUtils.hpp"
#include "UserCallbackSink.hpp"
#include "plugin/EnginePluginResourceManager.hpp"

#include <atomic>
#include <cstdint>
#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/DeviceQuery.hpp>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/dist_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <hip/hip_runtime.h>
#include <shared_mutex>

namespace hipdnn_backend
{
namespace logging
{
namespace
{

// Independent shutdown flag. Set by the atexit handler to prevent any thread from accessing
// BackendLogState after static destruction has begun. This is checked at the top of both
// initialize() and backendLoggingCallback() to short-circuit before touching the state object.
std::atomic<bool> sLoggingShutdown{false};

// Set (release) as the last write under loggerStateMutex once logger state is built, and
// cleared (release) as the first write in loggerShutdown() so concurrent initialize() calls
// wait for teardown. The acquire load at the top of initialize() pairs with both stores.
std::atomic<bool> sLoggerInitialized{false};

constexpr const char* S_BACKEND_ASYNC_LOGGER_NAME = "hipdnn_backend_async";
constexpr const char* S_BACKEND_SYNC_LOGGER_NAME = "hipdnn_backend_sync";

// Pattern string for the backend logger.
// Component name is already included in messages (e.g., "[hipdnn_backend] ..."),
// so the pattern includes timestamp, thread ID, and log level, but not a component name.
constexpr const char* BACKEND_LOGGER_PATTERN = "[%Y-%m-%d %H:%M:%S.%e] [tid %t] [%l] %v";

// Global backend log output callback state
struct BackendLogState
{
    // Shared mutex protects all logger state. Using shared_mutex allows concurrent reads
    // (multiple threads logging simultaneously) while still providing exclusive access
    // for modifications (setting callbacks, shutdown).
    std::shared_mutex loggerStateMutex;

    // Shared thread pool for async logger
    std::shared_ptr<spdlog::details::thread_pool> sharedThreadPool;

    // Shared async logger with two sinks:
    //   - asyncSharedDistSink: the console sink, or the file sink if HIPDNN_LOG_FILE is set,
    //     created the first time logging is enabled (see lockedCreateOutputSink())
    //   - asyncCallbackSinks: user callback sinks (async mode)
    std::shared_ptr<spdlog::async_logger> asyncLogger;
    std::shared_ptr<spdlog::sinks::dist_sink_mt> asyncSharedDistSink;
    std::shared_ptr<UserCallbackDispatchSink> asyncCallbackSinks;

    std::shared_ptr<spdlog::sinks::sink> consoleSink; // May be null
    std::shared_ptr<spdlog::sinks::sink> fileSink; // May be null

    // Set once the console or file sink has been created (or its creation has failed and
    // was reported); cleared by loggerShutdown().
    std::atomic<bool> outputSinkCreated{false};

    // Sync user callback logger
    std::shared_ptr<spdlog::logger> syncLogger;
    std::shared_ptr<UserCallbackDispatchSink> syncCallbackSinks;
    std::atomic<int> syncSinkCount{0}; // Number of sinks in syncCallbackSinks
    // Number of sinks in asyncSharedDistSink and asyncCallbackSinks
    std::atomic<int> asyncSinkCount{0};

    struct UserCallbackInfo
    {
        hipdnnUserLogCallback_t callback;
        hipdnnUserLogCallbackHandle_t userHandle;
        std::shared_ptr<std::atomic<hipdnnUserLogCallback_t>> callbackHolder;
        std::shared_ptr<UserCallbackSink> sink;
        hipdnnLogCallbackMode_t mode;
    };

    // Track user callbacks by composite key (callback, userHandle)
    struct CallbackKey
    {
        hipdnnUserLogCallback_t callback;
        hipdnnUserLogCallbackHandle_t userHandle;

        bool operator<(const CallbackKey& other) const
        {
            // Use reinterpret_cast to compare function pointers as integers
            if(callback != other.callback)
            {
                return reinterpret_cast<uintptr_t>(callback)
                       < reinterpret_cast<uintptr_t>(other.callback);
            }
            return reinterpret_cast<uintptr_t>(userHandle)
                   < reinterpret_cast<uintptr_t>(other.userHandle);
        }
    };

    std::map<CallbackKey, UserCallbackInfo> userCallbacks;

    BackendLogState()
    {
        // Register atexit handler to disable logging before static destruction.
        // Sets sLoggingShutdown to prevent any thread from accessing BackendLogState
        std::atexit([]() {
            hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_OFF);
            sLoggingShutdown.store(true, std::memory_order_release);
        });
    }

    ~BackendLogState()
    {
        // This destructor will only be called after all shared pointers from
        // threads' local storage are destroyed (see getBackendLogState()).
        loggerShutdown();
    }
};

BackendLogState& getBackendLogState()
{
    // Thread-local shared_ptr ensures BackendLogState survives until all threads release references.
    // LIMITATION: Threads must not call this function for the first time during static destruction,
    // as the mutex may already be destroyed. Ensure every thread that logs has emitted at least
    // one message before program exit handlers run. initialize() takes no reference on its
    // lock-free fast path, so only a message that reaches backendLoggingCallback() does.
    static const auto s_state = std::make_shared<BackendLogState>();
    thread_local const auto s_tlRef = s_state; // Each thread holds a reference
    return *s_state;
}

// Creates the console sink, or the file sink if HIPDNN_LOG_FILE is set. Called under the unique
// state lock once logging is enabled, so no log file is opened while the level is OFF. Never
// throws: a file that cannot be opened is reported once on stderr and the console sink is used
// instead. Failures are written with std::cerr because the backend log macros would re-enter
// initialize().
void lockedCreateOutputSink(BackendLogState& state)
{
    // Set first so that a failure below is reported once, not on every message.
    state.outputSinkCreated.store(true, std::memory_order_release);

    try
    {
        const std::string logFilePath = hipdnn_data_sdk::utilities::trim(
            hipdnn_data_sdk::utilities::getEnv("HIPDNN_LOG_FILE", ""));

        std::shared_ptr<spdlog::sinks::sink> sink;
        if(!logFilePath.empty())
        {
            try
            {
                state.fileSink
                    = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logFilePath, false);
                sink = state.fileSink;
            }
            catch(const std::exception& e)
            {
                std::cerr << "[hipDNN] WARNING: Cannot open HIPDNN_LOG_FILE '" << logFilePath
                          << "': " << e.what() << ". Logging to stderr instead.\n";
            }
        }

        if(sink == nullptr)
        {
            state.consoleSink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
            sink = state.consoleSink;
        }

        sink->set_level(spdlog::level::trace);
        sink->set_pattern(BACKEND_LOGGER_PATTERN);
        state.asyncSharedDistSink->add_sink(sink);
        state.asyncSinkCount.fetch_add(1, std::memory_order_release);
    }
    catch(const std::exception& e)
    {
        std::cerr << "[hipDNN] WARNING: Failed to create the log sink; console and file logging "
                     "is disabled: "
                  << e.what() << '\n';
    }
}

// Creates the console or file sink when logging was initialized while the level was OFF and
// the level is no longer OFF.
void ensureOutputSink(BackendLogState& state)
{
    const std::unique_lock<std::shared_mutex> lock(state.loggerStateMutex);
    if(!sLoggerInitialized.load(std::memory_order_relaxed)
       || state.outputSinkCreated.load(std::memory_order_relaxed)
       || hipdnn_data_sdk::logging::getLogLevel() == HIPDNN_SEV_OFF)
    {
        return;
    }

    lockedCreateOutputSink(state);
}

} // namespace

// NOLINTNEXTLINE(misc-no-recursion) Intentional: logging macros trigger lazy initialization
void logHipDeviceInfo(hipStream_t stream)
{
    int deviceId = 0;
    auto err = hipdnn_plugin_sdk::getDeviceFromStream(stream, &deviceId);
    if(err != hipSuccess)
    {
        HIPDNN_BACKEND_LOG_WARN("Failed to get device from stream: {}", hipGetErrorString(err));
        return;
    }

    hipDeviceProp_t props;
    err = hipGetDeviceProperties(&props, deviceId);
    if(err != hipSuccess)
    {
        HIPDNN_BACKEND_LOG_WARN(
            "Failed to get properties for device {}: {}", deviceId, hipGetErrorString(err));
        return;
    }

    HIPDNN_BACKEND_LOG_INFO(
        "HIP Device Information: {{Device: {}, Name: {}, Global Mem: {} bytes, Compute: {}.{}, "
        "MPs: {}, Clock: {} kHz}}",
        deviceId,
        props.name,
        props.totalGlobalMem,
        props.major,
        props.minor,
        props.multiProcessorCount,
        props.clockRate);
}

// NOLINTNEXTLINE(misc-no-recursion) Intentional: lazy init may trigger logging which re-enters
void initialize()
{
    if(sLoggingShutdown.load(std::memory_order_acquire))
    {
        // Program is shutting down; do not initialize the logger.
        return;
    }

    if(sLoggerInitialized.load(std::memory_order_acquire))
    {
        return;
    }

    auto& state = getBackendLogState();

    // initialize() runs inside API calls and log calls, so a failure never propagates from
    // here: it is reported once on stderr and logging stays disabled until loggerShutdown().
    const auto disableLogging = [&state](const char* reason) {
        std::cerr << "[hipDNN] WARNING: Failed to initialize logger; logging is disabled: "
                  << reason << '\n';
        loggerShutdown();
        const std::unique_lock<std::shared_mutex> lock(state.loggerStateMutex);
        state.outputSinkCreated.store(true, std::memory_order_release);
        sLoggerInitialized.store(true, std::memory_order_release);
    };

    try
    {
        const std::unique_lock<std::shared_mutex> lock(state.loggerStateMutex);
        if(sLoggerInitialized.load(std::memory_order_relaxed)) // Check again - race protection
        {
            return;
        }

        // Register the backend logging callback with the backend's data SDK based logger.
        hipdnn_data_sdk::logging::registerLoggingCallback(backendLoggingCallback);

        if(!state.sharedThreadPool)
        {
            state.sharedThreadPool
                = std::make_shared<spdlog::details::thread_pool>(8192, // queue size
                                                                 1 // worker threads
                );
        }

        // Create async shared dist_sink, async callback sinks and logger
        state.asyncSharedDistSink = std::make_shared<spdlog::sinks::dist_sink_mt>();
        state.asyncCallbackSinks = std::make_shared<UserCallbackDispatchSink>();

        state.asyncLogger = std::make_shared<spdlog::async_logger>(
            S_BACKEND_ASYNC_LOGGER_NAME,
            spdlog::sinks_init_list{state.asyncSharedDistSink, state.asyncCallbackSinks},
            state.sharedThreadPool,
            spdlog::async_overflow_policy::block);
        state.asyncLogger->set_pattern(BACKEND_LOGGER_PATTERN);
        state.asyncLogger->set_level(spdlog::level::trace);

        // Create sync callback sinks and logger
        state.syncCallbackSinks = std::make_shared<UserCallbackDispatchSink>();

        state.syncLogger
            = std::make_shared<spdlog::logger>(S_BACKEND_SYNC_LOGGER_NAME, state.syncCallbackSinks);
        state.syncLogger->set_pattern(BACKEND_LOGGER_PATTERN);
        state.syncLogger->set_level(spdlog::level::trace);

        // getLogLevel() will read from environment on first call, or return cached value
        // if setLogLevel() was called programmatically. While the level is OFF no console or
        // file sink exists; ensureOutputSink() creates it once the level is raised.
        const hipdnnSeverity_t logLevel = hipdnn_data_sdk::logging::getLogLevel();
        if(logLevel != HIPDNN_SEV_OFF)
        {
            lockedCreateOutputSink(state);
        }

        // Set global log level in data_sdk
        hipdnn_data_sdk::logging::setLogLevel(logLevel);

        sLoggerInitialized.store(true, std::memory_order_release);
    }
    catch(const std::exception& e)
    {
        disableLogging(e.what());
        return;
    }
    catch(...)
    {
        disableLogging("unknown exception");
        return;
    }
    // Backend logger is running. Log some system info details.
    HIPDNN_BACKEND_LOG_INFO("{}", platform_utilities::getSystemInfo());
    logHipDeviceInfo(nullptr);
}

void loggerShutdown()
{
    auto& state = getBackendLogState();

    // On Windows, loggerShutdown() also runs from ~BackendLogState while the loader detaches
    // the library at process exit. ExitProcess has already terminated every other thread,
    // including the pool worker and any thread that was inside a callback or held
    // loggerStateMutex, so taking that lock or waiting for those threads could never return:
    // the pool's destructor enqueues a terminate message for a worker that no longer exists.
    // This thread is the only one left, so the pool is read without the lock and leaked;
    // queued messages are dropped and nothing else is reset.
    if(platform_utilities::isProcessTerminating())
    {
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory) - deliberately never freed
        static_cast<void>(
            new std::shared_ptr<spdlog::details::thread_pool>(std::move(state.sharedThreadPool)));
        return;
    }

    std::vector<std::shared_ptr<UserCallbackSink>> removedSinks;
    std::shared_ptr<spdlog::details::thread_pool> threadPool;
    {
        const std::unique_lock<std::shared_mutex> lock(state.loggerStateMutex);
        sLoggerInitialized.store(false, std::memory_order_release);

        // Clear all user callbacks first (atomically disable)
        removedSinks.reserve(state.userCallbacks.size());
        for(auto& [key, info] : state.userCallbacks)
        {
            info.callbackHolder->store(nullptr, std::memory_order_release);
            removedSinks.push_back(std::move(info.sink));
        }
        state.userCallbacks.clear();

        // Destroy loggers
        state.syncLogger.reset();
        state.asyncLogger.reset();

        // Destroy dist_sink and callback sinks
        state.syncCallbackSinks.reset();
        state.asyncCallbackSinks.reset();
        state.asyncSharedDistSink.reset();
        state.syncSinkCount.store(0, std::memory_order_release);
        state.asyncSinkCount.store(0, std::memory_order_release);

        // Clear console/file sink references
        state.consoleSink.reset();
        state.fileSink.reset();
        state.outputSinkCreated.store(false, std::memory_order_release);

        // Joined below, after the lock is released
        threadPool = std::move(state.sharedThreadPool);

        // Logging may be re-started after loggerShutdown() is called; Clear the cached log
        // level so that if/when the logger is restarted it will reread the value from the
        // environment, following the original start-up behavior.
        hipdnn_data_sdk::logging::resetLogLevelCache();
        GraphLogger::resetCache();
    }

    // Ensure that any in-progress callbacks have completed before this function returns. The
    // wait happens outside loggerStateMutex, so an in-flight callback that calls
    // setUserLogCallback() cannot deadlock against it.
    for(const auto& sink : removedSinks)
    {
        if(sink != nullptr)
        {
            sink->waitForIdle();
        }
    }

    // Destroy thread pool (drains the queue and joins workers)
    threadPool.reset();
}

namespace
{
// Helper to convert hipdnnSeverity_t to spdlog level
spdlog::level::level_enum toSpdlogLevel(hipdnnSeverity_t severity)
{
    switch(severity)
    {
    case HIPDNN_SEV_FATAL:
        return spdlog::level::critical;
    case HIPDNN_SEV_ERROR:
        return spdlog::level::err;
    case HIPDNN_SEV_WARN:
        return spdlog::level::warn;
    case HIPDNN_SEV_INFO:
        return spdlog::level::info;
    case HIPDNN_SEV_OFF:
    default:
        return spdlog::level::off;
    }
}

// Helper: Add new user callback
hipdnnStatus_t lockedAddUserCallback(BackendLogState& state,
                                     const BackendLogState::CallbackKey& key,
                                     hipdnnUserLogCallback_t callback,
                                     hipdnnUserLogCallbackHandle_t userHandle,
                                     hipdnnSeverity_t minLevel,
                                     hipdnnLogCallbackMode_t mode)
{
    // Create atomic callback holder
    const auto callbackHolder = std::make_shared<std::atomic<hipdnnUserLogCallback_t>>(callback);

    // Create sink with user handle
    const auto sink = std::make_shared<UserCallbackSink>(callbackHolder, userHandle);
    sink->set_level(toSpdlogLevel(minLevel));
    sink->set_pattern(BACKEND_LOGGER_PATTERN);

    const bool isAsync = (mode == HIPDNN_LOG_CALLBACK_ASYNC);

    // Add to appropriate callback sinks (loggers already created in initialize())
    if(isAsync)
    {
        state.asyncCallbackSinks->addSink(sink);
        state.asyncSinkCount.fetch_add(1, std::memory_order_release);
    }
    else
    {
        state.syncCallbackSinks->addSink(sink);
        state.syncSinkCount.fetch_add(1, std::memory_order_release);
    }

    // Track callback
    state.userCallbacks[key] = {callback, userHandle, callbackHolder, sink, mode};

    return HIPDNN_STATUS_SUCCESS;
}

// Helper: Update existing user callback
hipdnnStatus_t lockedUpdateUserCallback(BackendLogState& state,
                                        BackendLogState::UserCallbackInfo& info,
                                        hipdnnSeverity_t newMinLevel,
                                        hipdnnLogCallbackMode_t newMode)
{

    // Check if mode changed (requires sink migration)
    const bool wasSyncNowAsync
        = (info.mode == HIPDNN_LOG_CALLBACK_SYNC && newMode == HIPDNN_LOG_CALLBACK_ASYNC);
    const bool wasAsyncNowSync
        = (info.mode == HIPDNN_LOG_CALLBACK_ASYNC && newMode == HIPDNN_LOG_CALLBACK_SYNC);

    if(wasSyncNowAsync || wasAsyncNowSync)
    {
        // Remove from old callback sinks
        if(info.mode == HIPDNN_LOG_CALLBACK_ASYNC)
        {
            state.asyncCallbackSinks->removeSink(info.sink);
            state.asyncSinkCount.fetch_sub(1, std::memory_order_release);
        }
        else
        {
            state.syncCallbackSinks->removeSink(info.sink);
            state.syncSinkCount.fetch_sub(1, std::memory_order_release);
        }

        // Add to new callback sinks (loggers already created in initialize())
        if(newMode == HIPDNN_LOG_CALLBACK_ASYNC)
        {
            state.asyncCallbackSinks->addSink(info.sink);
            state.asyncSinkCount.fetch_add(1, std::memory_order_release);
        }
        else
        {
            state.syncCallbackSinks->addSink(info.sink);
            state.syncSinkCount.fetch_add(1, std::memory_order_release);
        }

        info.mode = newMode;
    }

    // Update level
    info.sink->set_level(toSpdlogLevel(newMinLevel));

    return HIPDNN_STATUS_SUCCESS;
}

// Helper: Remove user callback (SEV_OFF). The removed sink is returned in removedSink; the
// caller waits for it to go idle after releasing loggerStateMutex.
hipdnnStatus_t lockedRemoveUserCallback(BackendLogState& state,
                                        const BackendLogState::CallbackKey& key,
                                        std::shared_ptr<UserCallbackSink>& removedSink)
{
    const auto it = state.userCallbacks.find(key);
    if(it == state.userCallbacks.end())
    {
        return HIPDNN_STATUS_BAD_PARAM; // Not registered
    }

    const auto& info = it->second;

    // Atomically disable callback FIRST
    info.callbackHolder->store(nullptr, std::memory_order_release);

    // Remove from callback sinks
    const bool isAsync = (info.mode == HIPDNN_LOG_CALLBACK_ASYNC);
    if(isAsync)
    {
        state.asyncCallbackSinks->removeSink(info.sink);
        state.asyncSinkCount.fetch_sub(1, std::memory_order_release);
    }
    else
    {
        state.syncCallbackSinks->removeSink(info.sink);
        state.syncSinkCount.fetch_sub(1, std::memory_order_release);
    }

    removedSink = std::move(info.sink);

    // Remove from tracking
    state.userCallbacks.erase(it);

    return HIPDNN_STATUS_SUCCESS;
}

} // namespace

hipdnnStatus_t setUserLogCallback(hipdnnUserLogCallback_t callback,
                                  hipdnnSeverity_t minLevel,
                                  hipdnnLogCallbackMode_t mode,
                                  hipdnnUserLogCallbackHandle_t userHandle)
{
    // Validate parameters
    if(callback == nullptr || userHandle == nullptr)
    {
        return LastErrorManager::setLastError(
            HIPDNN_STATUS_BAD_PARAM,
            "hipdnnSetUserLogCallback_ext: callback and userHandle must not be null");
    }

    if(mode != HIPDNN_LOG_CALLBACK_SYNC && mode != HIPDNN_LOG_CALLBACK_ASYNC)
    {
        return LastErrorManager::setLastError(
            HIPDNN_STATUS_BAD_PARAM,
            "hipdnnSetUserLogCallback_ext: mode " + std::to_string(static_cast<int>(mode))
                + " is neither HIPDNN_LOG_CALLBACK_SYNC nor HIPDNN_LOG_CALLBACK_ASYNC");
    }

    // Ensure logging initialized
    initialize();

    auto& state = getBackendLogState();
    hipdnnStatus_t status = HIPDNN_STATUS_SUCCESS;
    std::shared_ptr<UserCallbackSink> removedSink;
    {
        const std::unique_lock<std::shared_mutex> lock(state.loggerStateMutex);

        // Create composite key
        const BackendLogState::CallbackKey key{callback, userHandle};

        if(!sLoggerInitialized.load(std::memory_order_relaxed) || state.syncLogger == nullptr)
        {
            status = HIPDNN_STATUS_NOT_INITIALIZED;
        }
        else if(minLevel == HIPDNN_SEV_OFF)
        {
            status = lockedRemoveUserCallback(state, key, removedSink);
        }
        else
        {
            // Update an existing registration, or add a new one
            auto it = state.userCallbacks.find(key);
            status = it != state.userCallbacks.end()
                         ? lockedUpdateUserCallback(state, it->second, minLevel, mode)
                         : lockedAddUserCallback(state, key, callback, userHandle, minLevel, mode);
        }
    }

    // Wait until any in-progress callback completes, so the user can safely destroy data
    // structures after this function returns. The wait happens outside loggerStateMutex: an
    // in-flight callback may itself call this function. A callback that removes itself does
    // not wait for its own invocation.
    if(removedSink != nullptr)
    {
        removedSink->waitForIdle();
    }

    // setLastError() logs through the backend logger, which takes loggerStateMutex, so the
    // reason is recorded only after the lock is released.
    if(status == HIPDNN_STATUS_NOT_INITIALIZED)
    {
        return LastErrorManager::setLastError(
            status, "hipdnnSetUserLogCallback_ext: the backend logger is not initialized");
    }
    if(status == HIPDNN_STATUS_BAD_PARAM)
    {
        return LastErrorManager::setLastError(
            status,
            "hipdnnSetUserLogCallback_ext: cannot remove a callback that is not registered "
            "for this userHandle");
    }
    return status;
}

void backendLoggingCallback(hipdnnSeverity_t severity, const char* msg)
{
    // Check the shutdown flag before accessing BackendLogState. The atexit handler sets this
    // flag to prevent any thread from accessing logging infrastructure during static destruction.
    if(sLoggingShutdown.load(std::memory_order_acquire))
    {
        return;
    }

    // Also check that this log's log level is enabled.
    if(!hipdnn_data_sdk::logging::isLogLevelEnabled(severity))
    {
        return;
    }

    // Detect and prevent reentrant logging to avoid stack overflow. This can occur when
    // a user's synchronous log callback triggers another log message (e.g., by calling a
    // hipDNN API that logs, or by directly invoking the logger from within the callback).
    static thread_local int s_recursionDepth = 0;

    if(s_recursionDepth > 0)
    {
        // Recursion detected - print to stderr instead of recursing further
        std::cerr << "[hipDNN] WARNING: Recursive logging detected. Message dropped to prevent "
                     "stack overflow: "
                  << msg << '\n';
        return;
    }

    // Increment recursion counter for this thread
    ++s_recursionDepth;

    // RAII guard to ensure recursion depth is always decremented
    struct RecursionGuard
    {
        ~RecursionGuard()
        {
            --s_recursionDepth;
        }
    } const guard;

    // Lazy-init; ensure backend logger is initialized.
    initialize();

    auto& state = getBackendLogState();

    // Logging initialized while the level was OFF has no console or file sink yet.
    if(!state.outputSinkCreated.load(std::memory_order_acquire))
    {
        ensureOutputSink(state);
    }

    const auto spdlogLevel = toSpdlogLevel(severity);

    // Copy logger shared_ptrs under read lock, then release lock before logging.
    // This prevents deadlock if the callback re-enters (sync mode) and allows
    // concurrent readers. The copied shared_ptr keeps the shared logger alive.
    std::shared_ptr<spdlog::logger> asyncLogger;
    std::shared_ptr<spdlog::logger> syncLogger;

    {
        const std::shared_lock<std::shared_mutex> lock(state.loggerStateMutex);

        if(state.asyncSinkCount.load(std::memory_order_acquire) > 0)
        {
            asyncLogger = state.asyncLogger;
        }
        if(state.syncSinkCount.load(std::memory_order_acquire) > 0)
        {
            syncLogger = state.syncLogger;
        }
    } // Lock released here

    if(asyncLogger)
    {
        asyncLogger->log(spdlogLevel, msg);
    }

    if(syncLogger)
    {
        syncLogger->log(spdlogLevel, msg);
    }
}

hipdnnStatus_t setGlobalLogLevel(hipdnnSeverity_t level)
{
    // Set the global log level in data_sdk cache (backend's copy)
    hipdnn_data_sdk::logging::setLogLevel(level);

    // Logging initialized while the level was OFF has no console or file sink yet; create it
    // now that the level is raised.
    if(level != HIPDNN_SEV_OFF && !sLoggingShutdown.load(std::memory_order_acquire)
       && sLoggerInitialized.load(std::memory_order_acquire))
    {
        auto& state = getBackendLogState();
        if(!state.outputSinkCreated.load(std::memory_order_acquire))
        {
            ensureOutputSink(state);
        }
    }

    // Notify all loaded plugins of the log level change
    plugin::EnginePluginResourceManager::setPluginLogLevel(level);

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t getGlobalLogLevel(hipdnnSeverity_t& level)
{
    // Get global log level from data_sdk cache (backend's copy)
    level = hipdnn_data_sdk::logging::getLogLevel();

    return HIPDNN_STATUS_SUCCESS;
}

} // namespace logging
} // namespace hipdnn_backend
