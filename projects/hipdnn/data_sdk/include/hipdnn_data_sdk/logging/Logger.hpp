// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "CallbackTypes.h"
#include "LogLevel.hpp"

#include <atomic>
#include <mutex>
#include <sstream>
#include <string>
#include <type_traits>

// ============================================================================
// Logging Infrastructure
// ============================================================================
// This header provides logging macros for frontend, plugins, and SDKs.
//
// Two modes are supported:
// 1. Stream-style (default): HIPDNN_LOG_INFO("msg " << value)
// 2. Spdlog-style (opt-in): HIPDNN_LOG_INFO("msg {}", value)
//
// To use spdlog-style, define HIPDNN_PLUGIN_USE_SPDLOG before including.
// Plugin providers (miopen, hipblaslt) typically define this via CMake.

namespace hipdnn_data_sdk::logging
{

namespace detail
{

// Global callback registry for stream-based logging
inline std::atomic<hipdnnCallback_t>& getGlobalCallback()
{
    static std::atomic<hipdnnCallback_t> s_callback{nullptr};
    return s_callback;
}

inline void dispatchMessage(hipdnnSeverity_t severity,
                            const char* componentName,
                            const std::string& message)
{
    auto callback = getGlobalCallback().load(std::memory_order_acquire);
    if(callback != nullptr && !message.empty())
    {
        // Use bracketed format for consistency with backend: [component] message
        std::string formattedMsg = "[" + std::string(componentName) + "] " + message;
        callback(severity, formattedMsg.c_str());
    }
}

/**
 * @brief Stream-based logger that accumulates a message and dispatches on destruction
 */
class LogStream
{
public:
    LogStream(hipdnnSeverity_t severity, const char* componentName)
        : _severity(severity)
        , _componentName(componentName)
    {
    }

    ~LogStream()
    {
        std::string msg = _stream.str();
        if(!msg.empty())
        {
            dispatchMessage(_severity, _componentName, msg);
        }
    }

    LogStream(const LogStream&) = delete;
    LogStream& operator=(const LogStream&) = delete;
    LogStream(LogStream&&) = delete;
    LogStream& operator=(LogStream&&) = delete;

    template <typename T,
              typename = decltype(std::declval<std::ostringstream&>() << std::declval<const T&>())>
    LogStream& operator<<(const T& value)
    {
        _stream << value;
        return *this;
    }

private:
    hipdnnSeverity_t _severity;
    const char* _componentName;
    std::ostringstream _stream;
};

} // namespace detail

/**
 * @brief Register a global callback to receive log messages
 */
inline void registerLoggingCallback(hipdnnCallback_t callback)
{
    detail::getGlobalCallback().store(callback, std::memory_order_release);
}

/**
 * @brief Unregister the global logging callback
 */
inline void unregisterLoggingCallback()
{
    detail::getGlobalCallback().store(nullptr, std::memory_order_release);
}

/**
 * @brief Check if a logging callback is registered
 */
inline bool isLoggingCallbackRegistered()
{
    return detail::getGlobalCallback().load(std::memory_order_acquire) != nullptr;
}

} // namespace hipdnn_data_sdk::logging

// ============================================================================
// Internal Data SDK Logging Macros
// ============================================================================
//
// These macros are for use ONLY within data_sdk headers. They are always
// stream-style and are never affected by HIPDNN_PLUGIN_USE_SPDLOG.
// Use HIPDNN_SDK_LOG_* in data_sdk code, not HIPDNN_LOG_*.
//
// Usage:
//   HIPDNN_SDK_LOG_WARN("Warning: " << someValue);
//   HIPDNN_SDK_LOG_ERROR("Error in " << functionName);

#ifdef COMPONENT_NAME
#define HIPDNN_SDK_LOG_INFO(msg)                                                                   \
    do                                                                                             \
    {                                                                                              \
        if(::hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO))                         \
        {                                                                                          \
            ::hipdnn_data_sdk::logging::detail::LogStream(HIPDNN_SEV_INFO, COMPONENT_NAME) << msg; \
        }                                                                                          \
    } while(0)

#define HIPDNN_SDK_LOG_WARN(msg)                                                                   \
    do                                                                                             \
    {                                                                                              \
        if(::hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_WARN))                         \
        {                                                                                          \
            ::hipdnn_data_sdk::logging::detail::LogStream(HIPDNN_SEV_WARN, COMPONENT_NAME) << msg; \
        }                                                                                          \
    } while(0)

#define HIPDNN_SDK_LOG_ERROR(msg)                                                           \
    do                                                                                      \
    {                                                                                       \
        if(::hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_ERROR))                 \
        {                                                                                   \
            ::hipdnn_data_sdk::logging::detail::LogStream(HIPDNN_SEV_ERROR, COMPONENT_NAME) \
                << msg;                                                                     \
        }                                                                                   \
    } while(0)

#define HIPDNN_SDK_LOG_FATAL(msg)                                                           \
    do                                                                                      \
    {                                                                                       \
        if(::hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_FATAL))                 \
        {                                                                                   \
            ::hipdnn_data_sdk::logging::detail::LogStream(HIPDNN_SEV_FATAL, COMPONENT_NAME) \
                << msg;                                                                     \
        }                                                                                   \
    } while(0)
#else
#define HIPDNN_SDK_LOG_INFO(msg) \
    do                           \
    {                            \
    } while(0)
#define HIPDNN_SDK_LOG_WARN(msg) \
    do                           \
    {                            \
    } while(0)
#define HIPDNN_SDK_LOG_ERROR(msg) \
    do                            \
    {                             \
    } while(0)
#define HIPDNN_SDK_LOG_FATAL(msg) \
    do                            \
    {                             \
    } while(0)
#endif // COMPONENT_NAME

// ============================================================================
// Logging Macros
// ============================================================================

#ifdef HIPDNN_PLUGIN_USE_SPDLOG
// ============================================================================
// Spdlog-style logging (for plugin providers that opt-in)
// ============================================================================
// Usage: HIPDNN_LOG_INFO("Value: {}", someValue);

#include "CallbackSink.hpp"
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

#define HIPDNN_LOG_TRACE(...) _HIPDNN_SPDLOG_ACTION(spdlog::level::level_enum::trace, __VA_ARGS__)
#define HIPDNN_LOG_INFO(...) _HIPDNN_SPDLOG_ACTION(spdlog::level::level_enum::info, __VA_ARGS__)
#define HIPDNN_LOG_WARN(...) _HIPDNN_SPDLOG_ACTION(spdlog::level::level_enum::warn, __VA_ARGS__)
#define HIPDNN_LOG_ERROR(...) _HIPDNN_SPDLOG_ACTION(spdlog::level::level_enum::err, __VA_ARGS__)
#define HIPDNN_LOG_FATAL(...) \
    _HIPDNN_SPDLOG_ACTION(spdlog::level::level_enum::critical, __VA_ARGS__)

namespace hipdnn::logging
{

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

        auto callbackLogger = hipdnn_data_sdk::logging::createAsyncCallbackLoggerMt(
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
// Stream-style logging (default for frontend, SDKs, and their tests)
// ============================================================================
// Usage: HIPDNN_LOG_INFO("Value: " << someValue);

#ifndef COMPONENT_NAME
#define HIPDNN_LOG_INFO(msg) \
    do                       \
    {                        \
    } while(0)
#define HIPDNN_LOG_WARN(msg) \
    do                       \
    {                        \
    } while(0)
#define HIPDNN_LOG_ERROR(msg) \
    do                        \
    {                         \
    } while(0)
#define HIPDNN_LOG_FATAL(msg) \
    do                        \
    {                         \
    } while(0)
#else
#define HIPDNN_LOG_INFO(msg)                                                                       \
    do                                                                                             \
    {                                                                                              \
        if(::hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO))                         \
        {                                                                                          \
            ::hipdnn_data_sdk::logging::detail::LogStream(HIPDNN_SEV_INFO, COMPONENT_NAME) << msg; \
        }                                                                                          \
    } while(0)

#define HIPDNN_LOG_WARN(msg)                                                                       \
    do                                                                                             \
    {                                                                                              \
        if(::hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_WARN))                         \
        {                                                                                          \
            ::hipdnn_data_sdk::logging::detail::LogStream(HIPDNN_SEV_WARN, COMPONENT_NAME) << msg; \
        }                                                                                          \
    } while(0)

#define HIPDNN_LOG_ERROR(msg)                                                               \
    do                                                                                      \
    {                                                                                       \
        if(::hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_ERROR))                 \
        {                                                                                   \
            ::hipdnn_data_sdk::logging::detail::LogStream(HIPDNN_SEV_ERROR, COMPONENT_NAME) \
                << msg;                                                                     \
        }                                                                                   \
    } while(0)

#define HIPDNN_LOG_FATAL(msg)                                                               \
    do                                                                                      \
    {                                                                                       \
        if(::hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_FATAL))                 \
        {                                                                                   \
            ::hipdnn_data_sdk::logging::detail::LogStream(HIPDNN_SEV_FATAL, COMPONENT_NAME) \
                << msg;                                                                     \
        }                                                                                   \
    } while(0)
#endif // COMPONENT_NAME

namespace hipdnn::logging
{

inline void initializeCallbackLogging([[maybe_unused]] const std::string& componentName,
                                      hipdnnCallback_t callbackFunction)
{
    hipdnn_data_sdk::logging::initializeLogLevel();
    hipdnn_data_sdk::logging::registerLoggingCallback(callbackFunction);
}

} // namespace hipdnn::logging

#endif // HIPDNN_PLUGIN_USE_SPDLOG
