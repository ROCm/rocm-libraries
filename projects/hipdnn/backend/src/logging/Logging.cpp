// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Logging.hpp"

#include <hipdnn_sdk/logging/ComponentFormatter.hpp>
#include <hipdnn_sdk/logging/LoggingUtils.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <iostream>

#include <hipdnn_sdk/logging/CallbackTypes.h>
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

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
const std::string S_CALLBACK_RECEIVER_LOGGER_NAME = "hipdnn_callback_receiver";

} // namespace

void initialize()
{
    try
    {
        std::lock_guard<std::mutex> lock(s_loggingInitMutex);
        if(s_loggingInitialized)
        {
            return;
        }

        // It doesn't need to return if logLevel == off, but it avoids unnecessary initialization
        if(!hipdnn_sdk::logging::isLoggingEnabled())
        {
            s_loggingInitialized = true;
            return;
        }

        if(!spdlog::thread_pool())
        {
            spdlog::init_thread_pool(8192, 1);
        }

        std::string logLevel = hipdnn_sdk::utilities::getEnv("HIPDNN_LOG_LEVEL", "off");
        std::string logFilePath = hipdnn_sdk::utilities::getEnv("HIPDNN_LOG_FILE");

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

        // In spdlog, the formatting is a property of the underlying sink, not the logger.
        // However, we need one destination sink for thread safety because the mutex is attached to the sink.
        // Therefore, we implement a custom formatter to have distinct formatting for the backend, which does not use a callback sink.
        backendLogger->set_formatter(std::make_unique<hipdnn_sdk::logging::ComponentFormatter>());
        spdlog::register_logger(backendLogger);

        auto callbackReceiverLogger = std::make_shared<spdlog::async_logger>(
            S_CALLBACK_RECEIVER_LOGGER_NAME, sharedSink, spdlog::thread_pool());
        spdlog::register_logger(callbackReceiverLogger);

        setLogLevel(logLevel);

        s_loggingInitialized = true;

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

void setLogLevel(const std::string& level)
{
    if(level == "off")
    {
        spdlog::set_level(spdlog::level::off);
    }
    else if(level == "info")
    {
        spdlog::set_level(spdlog::level::info);
    }
    else if(level == "warn")
    {
        spdlog::set_level(spdlog::level::warn);
    }
    else if(level == "error")
    {
        spdlog::set_level(spdlog::level::err);
    }
    else if(level == "fatal")
    {
        spdlog::set_level(spdlog::level::critical);
    }
}

std::shared_ptr<spdlog::logger> getCallbackReceiverLogger()
{
    return spdlog::get(S_CALLBACK_RECEIVER_LOGGER_NAME);
}

std::shared_ptr<spdlog::logger> getBackendLogger()
{
    return spdlog::get(S_BACKEND_LOGGER_NAME);
}

void hipdnnLoggingCallback(hipdnnSeverity_t severity, const char* msg)
{
    initialize();

    if(auto logger = getCallbackReceiverLogger())
    {
        switch(severity)
        {
        case HIPDNN_SEV_FATAL:
            logger->critical(msg);
            break;
        case HIPDNN_SEV_ERROR:
            logger->error(msg);
            break;
        case HIPDNN_SEV_WARN:
            logger->warn(msg);
            break;
        case HIPDNN_SEV_OFF:
            break;
        default:
            logger->info(msg);
            break;
        }
    }
}

} // namespace logging
} // namespace hipdnn_backend
