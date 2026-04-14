// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Logging.hpp"
#include "GraphLogger.hpp"
#include "PlatformUtils.hpp"
#include "UserCallbackSink.hpp"
#include "plugin/EnginePluginResourceManager.hpp"

#include <cstdint>
#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <iostream>
#include <map>

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
    bool loggerInitialized = false;

    // Shared thread pool for async logger
    std::shared_ptr<spdlog::details::thread_pool> sharedThreadPool;

    // Shared async logger with dist_sink containing:
    //   - Console sink (if HIPDNN_LOG_LEVEL env var set, no HIPDNN_LOG_FILE)
    //   - File sink (if HIPDNN_LOG_FILE env var set)
    //   - User callback sinks (async mode)
    std::shared_ptr<spdlog::async_logger> asyncLogger;
    std::shared_ptr<spdlog::sinks::dist_sink_mt> asyncSharedDistSink;

    std::shared_ptr<spdlog::sinks::sink> consoleSink; // May be null
    std::shared_ptr<spdlog::sinks::sink> fileSink; // May be null

    // Sync user callback logger with dist_sink
    std::shared_ptr<spdlog::logger> syncLogger;
    std::shared_ptr<spdlog::sinks::dist_sink_mt> syncSharedDistSink;
    std::atomic<int> syncSinkCount{0}; // Number of sinks in syncSharedDistSink
    std::atomic<int> asyncSinkCount{0}; // Number of sinks in asyncSharedDistSink

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
    // as the mutex may already be destroyed. Ensure all logging threads have logged at least once
    // before program exit handlers run.
    static auto s_state = std::make_shared<BackendLogState>();
    thread_local auto s_tlRef = s_state; // Each thread holds a reference
    return *s_state;
}

} // namespace

// NOLINTNEXTLINE(misc-no-recursion) Intentional: logging macros trigger lazy initialization
void logHipDeviceInfo(hipStream_t stream)
{
    int deviceId = 0;
    hipError_t err = hipStreamGetDevice(stream, &deviceId);
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

    auto& state = getBackendLogState();
    // Fast path: check if already initialized with read lock (allows concurrent read access)
    {
        const std::shared_lock<std::shared_mutex> lock(state.loggerStateMutex);
        if(state.loggerInitialized)
        {
            return;
        }
    }

    // Slow path: actually initialize with write lock (first call only)
    try
    {
        const std::unique_lock<std::shared_mutex> lock(state.loggerStateMutex);
        if(state.loggerInitialized) // Check again - race protection
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

        // Create async shared dist_sink and logger
        state.asyncSharedDistSink = std::make_shared<spdlog::sinks::dist_sink_mt>();

        state.asyncLogger
            = std::make_shared<spdlog::async_logger>(S_BACKEND_ASYNC_LOGGER_NAME,
                                                     state.asyncSharedDistSink,
                                                     state.sharedThreadPool,
                                                     spdlog::async_overflow_policy::block);
        state.asyncLogger->set_pattern(BACKEND_LOGGER_PATTERN);
        state.asyncLogger->set_level(spdlog::level::trace);

        // Create sync shared dist_sink and logger
        state.syncSharedDistSink = std::make_shared<spdlog::sinks::dist_sink_mt>();

        state.syncLogger = std::make_shared<spdlog::logger>(
            S_BACKEND_SYNC_LOGGER_NAME, // Use same name as async logger
            state.syncSharedDistSink);
        state.syncLogger->set_pattern(BACKEND_LOGGER_PATTERN);
        state.syncLogger->set_level(spdlog::level::trace);

        // Add console or file sink to async dist_sink if logging enabled via env vars
        // getLogLevel() will read from environment on first call, or return cached value
        // if setLogLevel() was called programmatically
        const hipdnnSeverity_t logLevel = hipdnn_data_sdk::logging::getLogLevel();
        const std::string logFilePath = hipdnn_data_sdk::utilities::trim(
            hipdnn_data_sdk::utilities::getEnv("HIPDNN_LOG_FILE", ""));

        if(logLevel != HIPDNN_SEV_OFF)
        {
            if(!logFilePath.empty())
            {
                state.fileSink
                    = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logFilePath, false);
                state.fileSink->set_level(spdlog::level::trace);
                state.fileSink->set_pattern(BACKEND_LOGGER_PATTERN);
                state.asyncSharedDistSink->add_sink(state.fileSink);
                state.asyncSinkCount.fetch_add(1, std::memory_order_release);
            }
            else
            {
                state.consoleSink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
                state.consoleSink->set_level(spdlog::level::trace);
                state.consoleSink->set_pattern(BACKEND_LOGGER_PATTERN);
                state.asyncSharedDistSink->add_sink(state.consoleSink);
                state.asyncSinkCount.fetch_add(1, std::memory_order_release);
            }
        }

        // Set global log level in data_sdk
        hipdnn_data_sdk::logging::setLogLevel(logLevel);

        state.loggerInitialized = true;
    }
    catch(const std::exception& e)
    {
        // Log to stderr since the logging infrastructure failed
        std::cerr << "Failed to initialize logger: " << e.what() << '\n';
        loggerShutdown();
        // Either throw, or continue with logging disabled
        throw;
    }
    // Backend logger is running. Log some system info details.
    HIPDNN_BACKEND_LOG_INFO("{}", platform_utilities::getSystemInfo());
    logHipDeviceInfo(nullptr);
}

void loggerShutdown()
{
    auto& state = getBackendLogState();
    const std::unique_lock<std::shared_mutex> lock(state.loggerStateMutex);

    // Clear all user callbacks first (atomically disable)
    for(auto& [key, info] : state.userCallbacks)
    {
        info.callbackHolder->store(nullptr, std::memory_order_release);
    }

    // Ensure that any in-progress callbacks have completed before this function returns.
    for(auto& [key, info] : state.userCallbacks)
    {
        if(info.sink != nullptr)
        {
            info.sink->waitForIdle();
        }
    }

    state.userCallbacks.clear();

    // Destroy loggers
    state.syncLogger.reset();
    state.asyncLogger.reset();

    // Destroy dist_sinks
    state.syncSharedDistSink.reset();
    state.asyncSharedDistSink.reset();

    // Clear console/file sink references
    state.consoleSink.reset();
    state.fileSink.reset();

    // Destroy thread pool (joins workers)
    state.sharedThreadPool.reset();

    // Logging may be re-started after loggerShutdown() is called; Clear the cached log
    // level so that if/when the logger is restarted it will reread the value from the
    // environment, following the original start-up behavior.
    hipdnn_data_sdk::logging::resetLogLevelCache();
    GraphLogger::resetCache();

    state.loggerInitialized = false;
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
    auto callbackHolder = std::make_shared<std::atomic<hipdnnUserLogCallback_t>>(callback);

    // Create sink with user handle
    auto sink = std::make_shared<UserCallbackSink>(callbackHolder, userHandle);
    sink->set_level(toSpdlogLevel(minLevel));
    sink->set_pattern(BACKEND_LOGGER_PATTERN);

    const bool isAsync = (mode == HIPDNN_LOG_CALLBACK_ASYNC);

    // Add to appropriate dist_sink (loggers already created in initialize())
    if(isAsync)
    {
        state.asyncSharedDistSink->add_sink(sink);
        state.asyncSinkCount.fetch_add(1, std::memory_order_release);
    }
    else
    {
        state.syncSharedDistSink->add_sink(sink);
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
        // Remove from old dist_sink
        if(info.mode == HIPDNN_LOG_CALLBACK_ASYNC)
        {
            state.asyncSharedDistSink->remove_sink(info.sink);
            state.asyncSinkCount.fetch_sub(1, std::memory_order_release);
        }
        else
        {
            state.syncSharedDistSink->remove_sink(info.sink);
            state.syncSinkCount.fetch_sub(1, std::memory_order_release);
        }

        // Add to new dist_sink (loggers already created in initialize())
        if(newMode == HIPDNN_LOG_CALLBACK_ASYNC)
        {
            state.asyncSharedDistSink->add_sink(info.sink);
            state.asyncSinkCount.fetch_add(1, std::memory_order_release);
        }
        else
        {
            state.syncSharedDistSink->add_sink(info.sink);
            state.syncSinkCount.fetch_add(1, std::memory_order_release);
        }

        info.mode = newMode;
    }

    // Update level
    info.sink->set_level(toSpdlogLevel(newMinLevel));

    return HIPDNN_STATUS_SUCCESS;
}

// Helper: Remove user callback (SEV_OFF)
hipdnnStatus_t lockedRemoveUserCallback(BackendLogState& state,
                                        const BackendLogState::CallbackKey& key)
{
    auto it = state.userCallbacks.find(key);
    if(it == state.userCallbacks.end())
    {
        return HIPDNN_STATUS_BAD_PARAM; // Not registered
    }

    auto& info = it->second;

    // Atomically disable callback FIRST
    info.callbackHolder->store(nullptr, std::memory_order_release);

    // Wait until any in-progress callback completes
    // This ensures user can safely destroy data structures after this function returns
    if(info.sink != nullptr)
    {
        info.sink->waitForIdle();
    }

    // Remove from dist_sink
    const bool isAsync = (info.mode == HIPDNN_LOG_CALLBACK_ASYNC);
    if(isAsync)
    {
        state.asyncSharedDistSink->remove_sink(info.sink);
        state.asyncSinkCount.fetch_sub(1, std::memory_order_release);
    }
    else
    {
        state.syncSharedDistSink->remove_sink(info.sink);
        state.syncSinkCount.fetch_sub(1, std::memory_order_release);
    }

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
        return HIPDNN_STATUS_BAD_PARAM;
    }

    if(mode != HIPDNN_LOG_CALLBACK_SYNC && mode != HIPDNN_LOG_CALLBACK_ASYNC)
    {
        return HIPDNN_STATUS_BAD_PARAM;
    }

    // Ensure logging initialized
    initialize();

    auto& state = getBackendLogState();
    const std::unique_lock<std::shared_mutex> lock(state.loggerStateMutex);

    if(!state.loggerInitialized)
    {
        return HIPDNN_STATUS_NOT_INITIALIZED;
    }

    // Create composite key
    const BackendLogState::CallbackKey key{callback, userHandle};

    // Check if removing (SEV_OFF)
    if(minLevel == HIPDNN_SEV_OFF)
    {
        return lockedRemoveUserCallback(state, key);
    }

    // Check if updating existing registration
    auto it = state.userCallbacks.find(key);
    if(it != state.userCallbacks.end())
    {
        return lockedUpdateUserCallback(state, it->second, minLevel, mode);
    }

    // Add new registration
    return lockedAddUserCallback(state, key, callback, userHandle, minLevel, mode);
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

    auto spdlogLevel = toSpdlogLevel(severity);

    // Copy logger shared_ptrs under read lock, then release lock before logging.
    // This prevents deadlock if the callback re-enters (sync mode) and allows
    // concurrent readers. The copied shared_ptr keeps the shared logger alive.
    std::shared_ptr<spdlog::logger> asyncLogger;
    std::shared_ptr<spdlog::logger> syncLogger;

    {
        auto& state = getBackendLogState();
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
