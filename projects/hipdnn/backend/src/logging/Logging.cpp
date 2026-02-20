// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Logging.hpp"
#include "BackendLogOutputSink.hpp"
#include "PlatformUtils.hpp"

#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <iostream>

#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
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

const std::string S_BACKEND_LOGGER_NAME = "hipdnn_backend";
const std::string S_GLOBAL_CALLBACK_SYNC_LOGGER_NAME = "hipdnn_backend_global_callback_sync";
const std::string S_GLOBAL_CALLBACK_ASYNC_LOGGER_NAME = "hipdnn_backend_global_callback_async";

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
    // The loggers are created on-demand and destroyed during shutdown or when replaced.
    std::shared_ptr<spdlog::logger> consoleFileLogger; // Default logger; when no global callback.
    std::shared_ptr<spdlog::logger> callbackLogger; // Dynamic callback logger (sync or async).
    std::shared_ptr<spdlog::details::thread_pool> sharedThreadPool;

    BackendLogState()
    {
        // Register atexit handler to disable logging before static destruction.
        // This prevents log messages during cleanup, avoiding potential issues
        // with accessing the logger infrastructure after mutexes are destroyed.
        std::atexit([]() { hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_OFF); });
    }

    ~BackendLogState()
    {
        // This destructor will only be called after all shared pointers from
        // threads' local storage are destroyed (see getBackendLogState()).

        // Use explicit shutdown ordering; remove all loggers first.
        consoleFileLogger.reset();
        callbackLogger.reset();
        // Remove thread pool last (thread pool is used by loggers).
        sharedThreadPool.reset();
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

void initialize()
{
    auto& state = getBackendLogState();
    // Fast path: check if already initialized with read lock (allows concurrent read access)
    {
        std::shared_lock<std::shared_mutex> lock(state.loggerStateMutex);
        if(state.loggerInitialized)
        {
            return;
        }
    }

    // Slow path: actually initialize with write lock (first call only)
    {
        std::unique_lock<std::shared_mutex> lock(state.loggerStateMutex);
        if(state.loggerInitialized) // Check again - race protection
        {
            return;
        }

        // Register the backend logging callback with the backend's data SDK based logger.
        hipdnn_data_sdk::logging::registerLoggingCallback(hipdnnLoggingCallback);

        if(!state.sharedThreadPool)
        {
            state.sharedThreadPool
                = std::make_shared<spdlog::details::thread_pool>(8192, // queue size
                                                                 1 // worker threads
                );
        }

        std::string logFilePath = hipdnn_data_sdk::utilities::trim(
            hipdnn_data_sdk::utilities::getEnv("HIPDNN_LOG_FILE"));

        std::shared_ptr<spdlog::sinks::sink> sink;
        if(!logFilePath.empty())
        {
            sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logFilePath, false);
        }
        else
        {
            sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
        }

        // Create the backend console/file logger (the specific type of sink -- console
        // or file sink -- was chosen above based on environment variable settings)
        state.consoleFileLogger
            = std::make_shared<spdlog::async_logger>(S_BACKEND_LOGGER_NAME,
                                                     sink,
                                                     state.sharedThreadPool,
                                                     spdlog::async_overflow_policy::block);

        state.consoleFileLogger->set_pattern(BACKEND_LOGGER_PATTERN);

        // Set spdlog to accept all messages (trace is most verbose)
        // Actual filtering is done in HIPDNN_BACKEND_LOG*() macro via data SDK log API.
        state.consoleFileLogger->set_level(spdlog::level::trace);

        // Set the backend's data_sdk logger's cached log level, used by the
        // HIPDNN_BACKEND_LOG*() macros.
        hipdnn_data_sdk::logging::setLogLevel(
            hipdnn_data_sdk::logging::detail::stringToSeverityOrOff(
                hipdnn_data_sdk::utilities::getEnv("HIPDNN_LOG_LEVEL", "off")));

        state.loggerInitialized = true;
    }

    // Backend logger is running. Log some system info details.
    HIPDNN_BACKEND_LOG_INFO("{}", platform_utilities::getSystemInfo());
    logHipDeviceInfo(nullptr);
}

void loggerShutdown()
{
    auto& state = getBackendLogState();
    std::unique_lock<std::shared_mutex> lock(state.loggerStateMutex);

    state.consoleFileLogger.reset();
    state.callbackLogger.reset();
    // Do not reset sharedThreadPool here - let it be destroyed when BackendLogState
    // destructor runs. This ensures the thread pool remains alive until all threads
    // that have attempted to use the logger have exited.

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
} // namespace

void hipdnnLoggingCallback(hipdnnSeverity_t severity, const char* msg)
{
    // Lazy-init; ensure backend logger is initialized.
    initialize();

    // Some integration tests use log levels that are different from what the user
    // set in the environment; check the log level again here to filter-out logs
    // the user doesn't want to see.
    if(!hipdnn_data_sdk::logging::isLogLevelEnabled(severity))
    {
        return;
    }

    // Copy logger shared_ptr under read lock, then release lock before logging.
    // This prevents deadlock if the callback re-enters (sync mode) and allows
    // concurrent readers. The copied shared_ptr keeps the shared logger alive.
    std::shared_ptr<spdlog::logger> logger;
    {
        auto& state = getBackendLogState();
        std::shared_lock<std::shared_mutex> lock(state.loggerStateMutex);

        // Select logger: callback logger if set, otherwise console/file
        logger = state.callbackLogger ? state.callbackLogger : state.consoleFileLogger;
    } // Lock released here

    // Safe to log outside lock (prevents re-entrance deadlock)
    // Confirm logger is valid before using (since state.loggerInitialized was not checked above).
    if(logger)
    {
        logger->log(toSpdlogLevel(severity), msg);
    }
}

hipdnnStatus_t initializeGlobalOutputCallbackLogger(hipdnnBackendLogOutputCallback_t callback,
                                                    bool async)
{
    if(callback != nullptr)
    {
        // Ensure base infrastructure is initialized
        initialize();
    }

    auto& state = getBackendLogState();
    std::unique_lock<std::shared_mutex> lock(state.loggerStateMutex);

    if(callback != nullptr && !state.loggerInitialized)
    {
        return HIPDNN_STATUS_NOT_INITIALIZED;
    }

    // Destroy existing callback logger if present
    if(state.callbackLogger)
    {
        // Best-effort flush for async logger (non-blocking)
        try
        {
            state.callbackLogger->flush();
        }
        catch(...)
        {
            // Ignore flush failures (callback may have thrown)
        }
        state.callbackLogger.reset();
    }

    // Create new callback logger if callback provided
    if(callback != nullptr)
    {
        auto sink = std::make_shared<BackendLogOutputSink>(callback);
        sink->set_pattern(BACKEND_LOGGER_PATTERN);

        if(async)
        {
            state.callbackLogger
                = std::make_shared<spdlog::async_logger>(S_GLOBAL_CALLBACK_ASYNC_LOGGER_NAME,
                                                         sink,
                                                         state.sharedThreadPool,
                                                         spdlog::async_overflow_policy::block);
        }
        else
        {
            state.callbackLogger
                = std::make_shared<spdlog::logger>(S_GLOBAL_CALLBACK_SYNC_LOGGER_NAME, sink);
        }

        state.callbackLogger->set_level(spdlog::level::trace);
    }

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t setGlobalLogLevel(hipdnnSeverity_t level)
{
    // Set the global log level in data_sdk cache (backend's copy)
    hipdnn_data_sdk::logging::setLogLevel(level);

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t getGlobalLogLevel(hipdnnSeverity_t* level)
{
    // Get global log level from data_sdk cache (backend's copy)
    *level = hipdnn_data_sdk::logging::getLogLevel();

    return HIPDNN_STATUS_SUCCESS;
}

} // namespace logging
} // namespace hipdnn_backend
