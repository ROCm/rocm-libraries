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
    std::shared_mutex loggerInitMutex;
    bool loggerInitialized = false;
    bool programIsExiting = false;
    // The loggers, once created, are retained until the hipdnn library is shutdown.
    // This avoids a number of race conditions that can arise in dealing with shutting down
    // and flushing the async thread pools for individual loggers when the user switches
    // between enabling and disabling the global callback logger and switching between
    // sync and async logging modes.
    std::shared_ptr<spdlog::logger> consoleFileLogger; // Default logger; when no global callback.
    std::shared_ptr<spdlog::logger> callbackSyncLogger; // Sync global callback logger.
    std::shared_ptr<spdlog::logger> callbackAsyncLogger; // Async global callback logger.
    std::shared_ptr<spdlog::details::thread_pool> sharedThreadPool;

    std::shared_mutex callbackFnStateMutex;
    // A unique_lock for both loggerInitMutex and callbackFnStateMutex must be taken
    // (in that order) before modifying callbackFn, but callbackFn can be read while
    // holding only one of either mutex.
    hipdnnBackendLogOutputCallback_t callbackFn = nullptr;
    bool async = false; // tracks which mode is currently active
};

BackendLogState& getBackendLogState();

struct ProgramIsExitingSentinel
{
    ~ProgramIsExitingSentinel()
    {
        {
            // System is shutting down -- prevent lazy init from re-initializing the logger.
            auto& state = getBackendLogState();
            std::unique_lock loggerInitLock(state.loggerInitMutex);
            state.programIsExiting = true;
        }

        loggerShutdown();
    }
};

BackendLogState& getBackendLogState()
{
    // Never destroyed -- intentional leak -- to ensure threads won't access
    // deallocated logger state details while the system is shutting down.
    static auto s_state = new BackendLogState();
    // ProgramIsExitingSentinel s_sentinel is used to notify BackendLogState when
    // static objects are being cleaned-up -- the program is exiting. This
    // allows the logger to put itself into a quiescent so as not to interfere
    // while appllication threads which may still attempt to use the logger
    // are being closed.
    static ProgramIsExitingSentinel s_sentinel;
    return *s_state;
}

// Wrapper callback that both sync and async global log callback sinks use
void globalCallbackWrapper(hipdnnSeverity_t severity, const char* message)
{
    auto& state = getBackendLogState();
    // Only take callbackFnStateMutex here as loggerInitMutex may already be taken
    // by loggerShutdown() or hipdnnLoggingCallback().
    std::shared_lock loggerAvailableLock(state.callbackFnStateMutex);
    if(state.callbackFn != nullptr)
    {
        state.callbackFn(severity, message);
    }
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
    {
        // Try first with read lock to see if alrelady initialized.
        auto& state = getBackendLogState();
        std::shared_lock loggerInitLock(state.loggerInitMutex);
        if(state.loggerInitialized || state.programIsExiting)
        {
            return;
        }
    }

    {
        // Potentially not initialized, check again with unique lock and init if needed.
        auto& state = getBackendLogState();
        std::unique_lock loggerInitLock(state.loggerInitMutex);
        if(state.loggerInitialized || state.programIsExiting)
        {
            return;
        }
        // Unique lock sequence is loggerInitMutex then callbackFnStateMutex
        std::unique_lock loggerAvailableLock(state.callbackFnStateMutex);

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
        // or file sink -- was  chosen above based on environment variable settings)
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

        // Create async global callback logger
        sink = std::make_shared<BackendLogOutputSink>(globalCallbackWrapper);
        sink->set_pattern(BACKEND_LOGGER_PATTERN);

        state.callbackAsyncLogger
            = std::make_shared<spdlog::async_logger>(S_GLOBAL_CALLBACK_ASYNC_LOGGER_NAME,
                                                     sink,
                                                     state.sharedThreadPool,
                                                     spdlog::async_overflow_policy::block);

        // Set spdlog to accept all messages (trace is most verbose)
        // Filtering is done by data_sdk log level before calling hipdnnLoggingCallback()
        state.callbackAsyncLogger->set_level(spdlog::level::trace);

        // Create sync global callback logger
        sink = std::make_shared<BackendLogOutputSink>(globalCallbackWrapper);
        sink->set_pattern(BACKEND_LOGGER_PATTERN);

        state.callbackSyncLogger
            = std::make_shared<spdlog::logger>(S_GLOBAL_CALLBACK_SYNC_LOGGER_NAME, sink);

        // Set spdlog to accept all messages (trace is most verbose)
        // Filtering is done by data_sdk log level before calling hipdnnLoggingCallback()
        state.callbackSyncLogger->set_level(spdlog::level::trace);

        state.callbackFn = nullptr;
        state.loggerInitialized = true;
    }

    // Backend logger is running. Log some system info details.
    HIPDNN_BACKEND_LOG_INFO("{}", platform_utilities::getSystemInfo());
    logHipDeviceInfo(nullptr);
}

void loggerShutdown()
{
    auto& state = getBackendLogState();
    std::unique_lock loggerInitLock(state.loggerInitMutex);

    {
        // Unique lock order is loggerInitMutex then callbackFnStateMutex
        std::unique_lock loggerAvailableLock(state.callbackFnStateMutex);
        state.callbackFn = nullptr;
    }

    state.consoleFileLogger.reset();
    state.callbackSyncLogger.reset();
    state.callbackAsyncLogger.reset();
    // Do not destory state.sharedThreadPool to avoid race conditions where loggers
    // could attempt to use the thread pool after the thread pool is destroyed.

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

    auto& state = getBackendLogState();
    // Only take the loggerInitMutex read lock here as the callbackWrapper() will take
    // the callbackFnStateMutex read lock, which may occur in the same thread if using the
    // global callback in sync mode.
    std::shared_lock loggerInitLock(state.loggerInitMutex);
    if(state.programIsExiting)
    {
        // For now, the policy is to drop logs that are generated while the program is exiting.
        return;
    }

    std::shared_ptr<spdlog::logger> logger;
    if(state.callbackFn != nullptr)
    {
        // Select the appropriate callback logger based on current sync / async mode
        logger = state.async ? state.callbackAsyncLogger : state.callbackSyncLogger;
    }
    else
    {
        // Use the default file / console logger
        logger = state.consoleFileLogger;
    }

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
        // Start the loggers if they aren't already running.
        initialize();
    }

    auto& state = getBackendLogState();
    std::unique_lock loggerInitLock(state.loggerInitMutex);
    if(callback != nullptr && (!state.loggerInitialized || state.programIsExiting))
    {
        return HIPDNN_STATUS_NOT_INITIALIZED;
    }

    // Unique lock order is loggerInitMutex then callbackFnStateMutex.
    std::unique_lock loggerAvailableLock(state.callbackFnStateMutex);

    // Update callback pointer and current mode
    state.callbackFn = callback;
    state.async = async;

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
