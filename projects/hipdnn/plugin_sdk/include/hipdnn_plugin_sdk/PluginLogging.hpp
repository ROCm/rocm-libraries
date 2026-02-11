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
 *    - Uses stream-style logging
 *    - Usage: HIPDNN_PLUGIN_LOG_INFO("Value: " << value);
 *    - No spdlog dependency required
 *    - Used by test plugins and lightweight integrations
 *
 * The component name is stored at runtime when initializeCallbackLogging() is called.
 */

// Always include the SDK logging infrastructure
#include <hipdnn_data_sdk/logging/Logger.hpp>

#include <shared_mutex>
#include <string>

namespace hipdnn_plugin_sdk::logging
{

/// Default component name for plugins if not initialized
inline constexpr const char* K_DEFAULT_PLUGIN_COMPONENT_NAME =
#ifdef HIPDNN_COMPONENT_NAME
    HIPDNN_COMPONENT_NAME;
#else
    "hipdnn_plugin";
#endif

namespace detail
{

/// Thread-safe storage for the plugin component name
/// HIPDNN_HIDDEN ensures each shared object has its own copy of the static variable
HIPDNN_HIDDEN inline std::string& getStoredComponentName()
{
    static std::string s_componentName{K_DEFAULT_PLUGIN_COMPONENT_NAME};
    return s_componentName;
}

HIPDNN_HIDDEN inline std::shared_mutex& getComponentNameMutex()
{
    static std::shared_mutex s_mutex;
    return s_mutex;
}

inline void setComponentName(const std::string& name)
{
    std::unique_lock<std::shared_mutex> lock(getComponentNameMutex());
    getStoredComponentName() = name;
}

inline std::string getComponentName()
{
    std::shared_lock<std::shared_mutex> lock(getComponentNameMutex());
    return getStoredComponentName();
}

} // namespace detail

} // namespace hipdnn_plugin_sdk::logging

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

// Check if the global log level would allow this message before
// retrieving component name (which requires mutex lock + string allocation).
// Note: spdlog's should_log checks the logger's level, but we can short-circuit
// earlier by checking the SDK log level.
#define _HIPDNN_SPDLOG_ACTION(spdlog_level, hipdnn_severity, ...)                          \
    do                                                                                     \
    {                                                                                      \
        if(::hipdnn_data_sdk::logging::isLogLevelEnabled(hipdnn_severity))                 \
        {                                                                                  \
            auto componentName = ::hipdnn_plugin_sdk::logging::detail::getComponentName(); \
            auto logger = spdlog::get(componentName);                                      \
            if(logger && logger->should_log(spdlog_level))                                 \
            {                                                                              \
                logger->log(spdlog_level, __VA_ARGS__);                                    \
            }                                                                              \
        }                                                                                  \
    } while(0)

#define HIPDNN_PLUGIN_LOG_TRACE(...) \
    _HIPDNN_SPDLOG_ACTION(spdlog::level::level_enum::trace, HIPDNN_SEV_INFO, __VA_ARGS__)
#define HIPDNN_PLUGIN_LOG_INFO(...) \
    _HIPDNN_SPDLOG_ACTION(spdlog::level::level_enum::info, HIPDNN_SEV_INFO, __VA_ARGS__)
#define HIPDNN_PLUGIN_LOG_WARN(...) \
    _HIPDNN_SPDLOG_ACTION(spdlog::level::level_enum::warn, HIPDNN_SEV_WARN, __VA_ARGS__)
#define HIPDNN_PLUGIN_LOG_ERROR(...) \
    _HIPDNN_SPDLOG_ACTION(spdlog::level::level_enum::err, HIPDNN_SEV_ERROR, __VA_ARGS__)
#define HIPDNN_PLUGIN_LOG_FATAL(...) \
    _HIPDNN_SPDLOG_ACTION(spdlog::level::level_enum::critical, HIPDNN_SEV_FATAL, __VA_ARGS__)

namespace hipdnn_plugin_sdk::logging
{

/**
 * @brief Initialize spdlog-based callback logging for plugins
 *
 * Creates an async spdlog logger that forwards messages to the callback.
 * The component name is stored and used by logging macros.
 */
// HIPDNN_HIDDEN ensures each shared object has its own copy of the static mutex
HIPDNN_HIDDEN inline void initializeCallbackLogging(const std::string& componentName,
                                                    hipdnnCallback_t callbackFunction)
{
    // Store the component name for use by macros
    detail::setComponentName(componentName);

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

} // namespace hipdnn_plugin_sdk::logging

#else
// ============================================================================
// Stream-style Plugin Logging (HIPDNN_PLUGIN_LOG_*)
// ============================================================================
// Uses stream-style logging with runtime component name.
// Usage: HIPDNN_PLUGIN_LOG_INFO("Value: " << someValue);
//
// NOTE: Log level is checked BEFORE retrieving component name because
// getComponentName() requires a mutex lock and string copy.

#define HIPDNN_PLUGIN_LOG_TRACE(msg)                                                      \
    do                                                                                    \
    {                                                                                     \
        if(::hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO))                \
        {                                                                                 \
            auto _hipdnn_comp = ::hipdnn_plugin_sdk::logging::detail::getComponentName(); \
            HIPDNN_SDK_LOG_INFO_WITH_COMPONENT(_hipdnn_comp.c_str(), msg);                \
        }                                                                                 \
    } while(0)

#define HIPDNN_PLUGIN_LOG_INFO(msg)                                                       \
    do                                                                                    \
    {                                                                                     \
        if(::hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO))                \
        {                                                                                 \
            auto _hipdnn_comp = ::hipdnn_plugin_sdk::logging::detail::getComponentName(); \
            HIPDNN_SDK_LOG_INFO_WITH_COMPONENT(_hipdnn_comp.c_str(), msg);                \
        }                                                                                 \
    } while(0)

#define HIPDNN_PLUGIN_LOG_WARN(msg)                                                       \
    do                                                                                    \
    {                                                                                     \
        if(::hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_WARN))                \
        {                                                                                 \
            auto _hipdnn_comp = ::hipdnn_plugin_sdk::logging::detail::getComponentName(); \
            HIPDNN_SDK_LOG_WARN_WITH_COMPONENT(_hipdnn_comp.c_str(), msg);                \
        }                                                                                 \
    } while(0)

#define HIPDNN_PLUGIN_LOG_ERROR(msg)                                                      \
    do                                                                                    \
    {                                                                                     \
        if(::hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_ERROR))               \
        {                                                                                 \
            auto _hipdnn_comp = ::hipdnn_plugin_sdk::logging::detail::getComponentName(); \
            HIPDNN_SDK_LOG_ERROR_WITH_COMPONENT(_hipdnn_comp.c_str(), msg);               \
        }                                                                                 \
    } while(0)

#define HIPDNN_PLUGIN_LOG_FATAL(msg)                                                      \
    do                                                                                    \
    {                                                                                     \
        if(::hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_FATAL))               \
        {                                                                                 \
            auto _hipdnn_comp = ::hipdnn_plugin_sdk::logging::detail::getComponentName(); \
            HIPDNN_SDK_LOG_FATAL_WITH_COMPONENT(_hipdnn_comp.c_str(), msg);               \
        }                                                                                 \
    } while(0)

namespace hipdnn_plugin_sdk::logging
{

/**
 * @brief Initialize stream-based callback logging for plugins
 *
 * Registers the callback and initializes log levels.
 * The component name is stored and used by logging macros.
 */
inline void initializeCallbackLogging(const std::string& componentName,
                                      hipdnnCallback_t callbackFunction)
{
    // Store the component name for use by macros
    detail::setComponentName(componentName);

    hipdnn_data_sdk::logging::initializeLogLevel();
    hipdnn_data_sdk::logging::registerLoggingCallback(callbackFunction);
}

} // namespace hipdnn_plugin_sdk::logging

#endif // HIPDNN_PLUGIN_USE_SPDLOG
