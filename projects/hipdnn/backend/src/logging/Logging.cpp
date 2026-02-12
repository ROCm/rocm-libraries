// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Logging.hpp"
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

// Could refactor this to a class with a single static instance.
// The benefit would be a destructor to cleanup logging.
std::mutex s_loggingInitMutex; // NOLINT(readability-identifier-naming)
bool s_loggingInitialized = false; // NOLINT(readability-identifier-naming)
const std::string S_BACKEND_LOGGER_NAME = "hipdnn_backend";

// Pattern string for the backend logger.
// Component name is already included in messages (e.g., "[hipdnn_backend] ..."),
// so the pattern includes timestamp, thread ID, and log level, but not a component name.
constexpr const char* BACKEND_LOGGER_PATTERN = "[%Y-%m-%d %H:%M:%S.%e] [tid %t] [%l] %v";

std::shared_ptr<spdlog::logger> getBackendLogger()
{
    return spdlog::get(S_BACKEND_LOGGER_NAME);
}

} // namespace

void logSystemInfo()
{
    auto logger = getBackendLogger();
    if(!logger)
    {
        return;
    }

    logger->info(platform_utilities::getSystemInfo());
}

void logHipDeviceInfo(hipStream_t stream)
{
    auto logger = getBackendLogger();
    if(!logger)
    {
        return;
    }

    int deviceId = 0;
    hipError_t err = hipStreamGetDevice(stream, &deviceId);
    if(err != hipSuccess)
    {
        logger->warn("Failed to get device from stream: {}", hipGetErrorString(err));
        return;
    }

    hipDeviceProp_t props;
    err = hipGetDeviceProperties(&props, deviceId);
    if(err != hipSuccess)
    {
        logger->warn(
            "Failed to get properties for device {}: {}", deviceId, hipGetErrorString(err));
        return;
    }

    logger->info(
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
    try
    {
        std::lock_guard<std::mutex> lock(s_loggingInitMutex);
        if(s_loggingInitialized)
        {
            return;
        }

        // Initialize the log level from environment variable
        hipdnn_data_sdk::logging::initializeLogLevel();

        // Register the global callback so non-backend components can send logs to the backend
        hipdnn_data_sdk::logging::registerLoggingCallback(hipdnnLoggingCallback);

        std::string logLevelStr = hipdnn_data_sdk::utilities::getEnv("HIPDNN_LOG_LEVEL", "off");
        hipdnnSeverity_t logLevel
            = hipdnn_data_sdk::logging::detail::stringToSeverityOrOff(logLevelStr);

        // It doesn't need to return if logLevel == off, but it avoids unnecessary initialization
        if(logLevel == HIPDNN_SEV_OFF)
        {
            s_loggingInitialized = true;
            return;
        }

        if(!spdlog::thread_pool())
        {
            spdlog::init_thread_pool(8192, 1);
        }

        std::string logFilePath = hipdnn_data_sdk::utilities::getEnv("HIPDNN_LOG_FILE");

        std::shared_ptr<spdlog::sinks::sink> sharedSink;
        if(!logFilePath.empty())
        {
            sharedSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logFilePath, false);
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
        // Actual filtering is done in HIPDNN_BACKEND_LOG*() macro via isLogLevelEnabled()
        backendLogger->set_level(spdlog::level::trace);

        spdlog::register_logger(backendLogger);

        // Update the data_sdk log level cache for use by the
        // HIPDNN_BACKEND_LOG*() macros to filter-out logs based on level.
        hipdnn_data_sdk::logging::setLogLevel(logLevel);

        s_loggingInitialized = true;

        logSystemInfo();
        logHipDeviceInfo(nullptr);

        return;
    }
    catch(const spdlog::spdlog_ex& ex)
    {
        cleanup();
        std::cerr << "Logging initialization failed: " << ex.what() << "\n";
    }
}

void cleanup()
{
    std::lock_guard<std::mutex> lock(s_loggingInitMutex);
    spdlog::shutdown();
    s_loggingInitialized = false;
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

void logMessage(hipdnnSeverity_t severity, const std::string& message)
{
    // Check log level using data_sdk infrastructure
    if(!hipdnn_data_sdk::logging::isLogLevelEnabled(severity))
    {
        return;
    }

    if(auto logger = getBackendLogger())
    {
        logger->log(toSpdlogLevel(severity), message);
    }
}

void hipdnnLoggingCallback(hipdnnSeverity_t severity, const char* msg)
{
    // Message already includes component name from source (frontend/plugins)
    // Route through central backend logMessage function
    logMessage(severity, msg);
}

} // namespace logging
} // namespace hipdnn_backend
