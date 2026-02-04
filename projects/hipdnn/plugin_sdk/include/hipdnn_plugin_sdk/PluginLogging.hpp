// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file PluginLogging.hpp
 * @brief Plugin logging macros with dual-mode support
 *
 * This header provides HIPDNN_PLUGIN_LOG_* macros for use in plugins.
 *
 * Two modes are supported based on HIPDNN_PLUGIN_USE_SPDLOG:
 *
 * 1. HIPDNN_PLUGIN_USE_SPDLOG defined:
 *    - Uses spdlog/fmt style logging with format strings
 *    - Usage: HIPDNN_PLUGIN_LOG_INFO("Value: {}", value);
 *    - Requires linking against spdlog
 *    - Used by production plugins (miopen-provider, hipblaslt-provider)
 *
 * 2. HIPDNN_PLUGIN_USE_SPDLOG not defined (default):
 *    - Delegates to HIPDNN_SDK_LOG_* stream-style macros
 *    - Usage: HIPDNN_PLUGIN_LOG_INFO("Value: " << value);
 *    - No spdlog dependency required
 *    - Used by test plugins and lightweight integrations
 *
 * Before using these macros, define COMPONENT_NAME to identify the plugin:
 *   #define COMPONENT_NAME "my_plugin"
 */

// Always include the SDK logging infrastructure
#include <hipdnn_data_sdk/logging/Logger.hpp>

#ifdef HIPDNN_PLUGIN_USE_SPDLOG
// ============================================================================
// Spdlog-style Plugin Logging (HIPDNN_PLUGIN_LOG_*)
// ============================================================================
// Usage: HIPDNN_PLUGIN_LOG_INFO("Value: {}", someValue);

#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_plugin_sdk/logging/CallbackSink.hpp>

#include <iostream>
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#ifndef COMPONENT_NAME
#define _HIPDNN_SPDLOG_ACTION(level, ...) \
    do                                    \
    {                                     \
    } while(0)
#else
#define _HIPDNN_SPDLOG_ACTION(spdlog_level, ...)       \
    do                                                 \
    {                                                  \
        auto logger = spdlog::get(COMPONENT_NAME);     \
        if(logger && logger->should_log(spdlog_level)) \
        {                                              \
            logger->log(spdlog_level, __VA_ARGS__);    \
        }                                              \
    } while(0)
#endif // COMPONENT_NAME

#define HIPDNN_PLUGIN_LOG_TRACE(...) \
    _HIPDNN_SPDLOG_ACTION(spdlog::level::level_enum::trace, __VA_ARGS__)
#define HIPDNN_PLUGIN_LOG_INFO(...) \
    _HIPDNN_SPDLOG_ACTION(spdlog::level::level_enum::info, __VA_ARGS__)
#define HIPDNN_PLUGIN_LOG_WARN(...) \
    _HIPDNN_SPDLOG_ACTION(spdlog::level::level_enum::warn, __VA_ARGS__)
#define HIPDNN_PLUGIN_LOG_ERROR(...) \
    _HIPDNN_SPDLOG_ACTION(spdlog::level::level_enum::err, __VA_ARGS__)
#define HIPDNN_PLUGIN_LOG_FATAL(...) \
    _HIPDNN_SPDLOG_ACTION(spdlog::level::level_enum::critical, __VA_ARGS__)

namespace hipdnn::logging
{

/**
 * @brief Initialize spdlog-based callback logging for plugins
 *
 * Creates an async spdlog logger that forwards messages to the callback.
 */
inline void initializeCallbackLogging(const std::string& componentName,
                                      hipdnnCallback_t callbackFunction)
{
    try
    {
        static std::mutex s_callbackInitMutex;
        std::lock_guard<std::mutex> lock(s_callbackInitMutex);

        if(spdlog::get(componentName))
        {
            return;
        }

        if(!spdlog::thread_pool())
        {
            spdlog::init_thread_pool(8192, 1);
        }

        auto callbackLogger = hipdnn_plugin_sdk::logging::detail::createAsyncCallbackLoggerMt(
            callbackFunction, componentName);
        spdlog::register_logger(callbackLogger);
    }
    catch(const spdlog::spdlog_ex& ex)
    {
        std::cerr << "hipDNN SDK: Failed to initialize callback logger for component '"
                  << componentName << "'. Error: " << ex.what() << "\n";
    }
}

} // namespace hipdnn::logging

#else
// ============================================================================
// Stream-style Plugin Logging (HIPDNN_PLUGIN_LOG_*)
// ============================================================================
// Delegates to SDK logging macros.
// Usage: HIPDNN_PLUGIN_LOG_INFO("Value: " << someValue);

#define HIPDNN_PLUGIN_LOG_TRACE(msg) HIPDNN_SDK_LOG_INFO(msg)
#define HIPDNN_PLUGIN_LOG_INFO(msg) HIPDNN_SDK_LOG_INFO(msg)
#define HIPDNN_PLUGIN_LOG_WARN(msg) HIPDNN_SDK_LOG_WARN(msg)
#define HIPDNN_PLUGIN_LOG_ERROR(msg) HIPDNN_SDK_LOG_ERROR(msg)
#define HIPDNN_PLUGIN_LOG_FATAL(msg) HIPDNN_SDK_LOG_FATAL(msg)

namespace hipdnn::logging
{

/**
 * @brief Initialize stream-based callback logging for plugins
 *
 * Registers the callback and initializes log levels.
 */
inline void initializeCallbackLogging([[maybe_unused]] const std::string& componentName,
                                      hipdnnCallback_t callbackFunction)
{
    hipdnn_data_sdk::logging::initializeLogLevel();
    hipdnn_data_sdk::logging::registerLoggingCallback(callbackFunction);
}

} // namespace hipdnn::logging

#endif // HIPDNN_PLUGIN_USE_SPDLOG
