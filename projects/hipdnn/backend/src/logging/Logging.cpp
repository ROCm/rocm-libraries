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
#include <mutex>

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
    std::recursive_mutex backendLogStateMutex;
    bool loggerInitialized = false;
    bool loggerAvailable = false;
    hipdnnBackendLogOutputCallback_t callback = nullptr;
    bool async = false; // tracks which mode is currently active
    std::shared_ptr<spdlog::logger> syncLogger;
    std::shared_ptr<spdlog::logger> asyncLogger;
    std::shared_ptr<spdlog::details::thread_pool> asyncThreadPool;
};

BackendLogState& getBackendLogState()
{
    static BackendLogState s_state;
    return s_state;
}

// Wrapper callback that both sync and async global log callback sinks use
// This indirection allows updating the callback without recreating loggers/sinks
void globalCallbackWrapper(hipdnnSeverity_t severity, const char* message)
{
    auto& state = getBackendLogState();
    std::lock_guard<std::recursive_mutex> lock(state.backendLogStateMutex);
    if(state.callback != nullptr)
    {
        state.callback(severity, message);
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
        auto& state = getBackendLogState();
        std::lock_guard<std::recursive_mutex> lock(state.backendLogStateMutex);
        if(state.loggerInitialized)
        {
            return;
        }

        try
        {
            // Register the backend logging callback with the backend's data SDK based logger.
            hipdnn_data_sdk::logging::registerLoggingCallback(hipdnnLoggingCallback);

            if(!spdlog::thread_pool())
            {
                spdlog::init_thread_pool(8192, 1);
            }

            std::string logFilePath = hipdnn_data_sdk::utilities::getEnv("HIPDNN_LOG_FILE");

            std::shared_ptr<spdlog::sinks::sink> sharedSink;
            if(!logFilePath.empty())
            {
                sharedSink
                    = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logFilePath, false);
            }
            else
            {
                sharedSink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
            }

            auto backendLogger = std::make_shared<spdlog::async_logger>(
                S_BACKEND_LOGGER_NAME, sharedSink, spdlog::thread_pool());

            // Use a simple pattern formatter for the single unified logger
            // Component name is already included in the message (e.g., "[hipdnn_backend] ...")
            backendLogger->set_pattern(BACKEND_LOGGER_PATTERN);

            // Set spdlog to accept all messages (trace is most verbose)
            // Actual filtering is done in HIPDNN_BACKEND_LOG*() macro via data SDK log API.
            backendLogger->set_level(spdlog::level::trace);

            spdlog::register_logger(backendLogger);

            // Set the backend's data_sdk logger's cached log level for use by the
            // HIPDNN_BACKEND_LOG*() macros.
            hipdnn_data_sdk::logging::setLogLevel(
                hipdnn_data_sdk::logging::detail::stringToSeverityOrOff(
                    hipdnn_data_sdk::utilities::getEnv("HIPDNN_LOG_LEVEL", "off")));

            state.loggerAvailable = true;
            state.loggerInitialized = true;
        }
        catch(const spdlog::spdlog_ex& ex)
        {
            state.loggerAvailable = false;
            spdlog::shutdown();
            std::cerr << "Logging initialization failed: " << ex.what() << "\n";
            return;
        }
    }

    // Backend logger is running. Log some system info details.
    HIPDNN_BACKEND_LOG_INFO("{}", platform_utilities::getSystemInfo());
    logHipDeviceInfo(nullptr);
}

void loggerShutdown()
{
    auto& state = getBackendLogState();
    std::lock_guard<std::recursive_mutex> lock(state.backendLogStateMutex);

    state.loggerAvailable = false;
    state.callback = nullptr;

    spdlog::shutdown();

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

    auto& state = getBackendLogState();
    std::lock_guard<std::recursive_mutex> lock(state.backendLogStateMutex);

    if(state.loggerAvailable)
    {
        std::shared_ptr<spdlog::logger> logger;
        if(state.callback != nullptr)
        {
            // Return the appropriate logger based on current sync / async mode
            logger = state.async ? state.asyncLogger : state.syncLogger;
        }
        else
        {
            // File / console logger
            logger = spdlog::get(S_BACKEND_LOGGER_NAME);
        }

        if(logger)
        {
            // Assumes msg already contains component name.
            logger->log(toSpdlogLevel(severity), msg);
        }
    }
}

hipdnnStatus_t initializeGlobalOutputCallbackLogger(hipdnnBackendLogOutputCallback_t callback,
                                                    bool async)
{
    try
    {
        if(callback != nullptr)
        {
            // The global callback logger is tied to the backend logger. The backend
            // logger can operate without the global callback logger, but the callback
            // logger requires the backend logger state to be initialized before
            // the callback logger function can be used.
            initialize();
        }

        auto& state = getBackendLogState();
        std::lock_guard<std::recursive_mutex> lock(state.backendLogStateMutex);

        if(callback != nullptr)
        {
            if(!state.loggerAvailable)
            {
                // Backend logging init failed or is shutting down; do not enable global callback log output.
                return HIPDNN_STATUS_INTERNAL_ERROR;
            }

            if(async)
            {
                // Create async logger if it doesn't exist
                if(!state.asyncLogger)
                {
                    auto sink = std::make_shared<BackendLogOutputSink>(globalCallbackWrapper);
                    sink->set_pattern("%v");

                    // Create dedicated async thread pool for this callback
                    state.asyncThreadPool
                        = std::make_shared<spdlog::details::thread_pool>(8192, // queue size
                                                                         1 // worker threads
                        );

                    state.asyncLogger = std::make_shared<spdlog::async_logger>(
                        S_GLOBAL_CALLBACK_ASYNC_LOGGER_NAME,
                        sink,
                        state.asyncThreadPool,
                        spdlog::async_overflow_policy::block);

                    // Set spdlog to accept all messages (trace is most verbose)
                    // Filtering is done by data_sdk log level before calling hipdnnLoggingCallback()
                    state.asyncLogger->set_level(spdlog::level::trace);
                }
            }
            else
            {
                // Create sync logger if it doesn't exist
                if(!state.syncLogger)
                {
                    auto sink = std::make_shared<BackendLogOutputSink>(globalCallbackWrapper);
                    sink->set_pattern("%v");

                    state.syncLogger = std::make_shared<spdlog::logger>(
                        S_GLOBAL_CALLBACK_SYNC_LOGGER_NAME, sink);

                    // Set spdlog to accept all messages (trace is most verbose)
                    // Filtering is done by data_sdk log level before calling hipdnnLoggingCallback()
                    state.syncLogger->set_level(spdlog::level::trace);
                }
            }
        }

        // Update callback pointer and current mode
        state.callback = callback;
        state.async = async;

        return HIPDNN_STATUS_SUCCESS;
    }
    catch(...)
    {
        return HIPDNN_STATUS_INTERNAL_ERROR;
    }
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
