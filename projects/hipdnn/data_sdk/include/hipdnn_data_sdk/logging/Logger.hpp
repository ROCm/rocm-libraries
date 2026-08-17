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
// SDK Logging Infrastructure
// ============================================================================
// This header provides the core logging infrastructure for hipDNN.
//
// HIPDNN_SDK_LOG_* macros:
// - Stream-style only: HIPDNN_SDK_LOG_INFO("msg " << value)
// - Assumes callback is already registered (no auto-init)
// - Uses fixed "hipdnn_sdk" component name
//
// Other components should use their scoped macros:
// - Frontend: HIPDNN_FE_LOG_* (auto-inits, uses "hipdnn_frontend")
// - Plugins: HIPDNN_PLUGIN_LOG_*
// - Backend: Has its own logging implementation

namespace hipdnn_data_sdk::logging
{

namespace detail
{

// Global callback registry for stream-based logging
// HIPDNN_HIDDEN ensures each shared object has its own copy of the static variable
HIPDNN_HIDDEN inline std::atomic<hipdnnCallback_t>& getGlobalCallback()
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
        const std::string formattedMsg = "[" + std::string(componentName) + "] " + message;
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
        const std::string msg = _stream.str();
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

/// Component name constant for SDK logging
inline constexpr const char* K_COMPONENT_NAME = "hipdnn_sdk";

} // namespace hipdnn_data_sdk::logging

// ============================================================================
// SDK Logging Macros with Component Name (HIPDNN_SDK_LOG_*_WITH_COMPONENT)
// ============================================================================
// These macros allow specifying a custom component name for logging.
// Intended for use by Frontend and Plugin logging macros to avoid duplication.
//
// Usage:
//   HIPDNN_SDK_LOG_INFO_WITH_COMPONENT("my_component", "Message " << value);

#define HIPDNN_SDK_LOG_INFO_WITH_COMPONENT(component, msg)                                  \
    do                                                                                      \
    {                                                                                       \
        if(::hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO))                  \
        {                                                                                   \
            ::hipdnn_data_sdk::logging::detail::LogStream(HIPDNN_SEV_INFO, component)       \
                << msg; /* NOLINT(bugprone-macro-parentheses) msg is a stream expression */ \
        }                                                                                   \
    } while(0)

#define HIPDNN_SDK_LOG_WARN_WITH_COMPONENT(component, msg)                                  \
    do                                                                                      \
    {                                                                                       \
        if(::hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_WARN))                  \
        {                                                                                   \
            ::hipdnn_data_sdk::logging::detail::LogStream(HIPDNN_SEV_WARN, component)       \
                << msg; /* NOLINT(bugprone-macro-parentheses) msg is a stream expression */ \
        }                                                                                   \
    } while(0)

#define HIPDNN_SDK_LOG_ERROR_WITH_COMPONENT(component, msg)                                 \
    do                                                                                      \
    {                                                                                       \
        if(::hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_ERROR))                 \
        {                                                                                   \
            ::hipdnn_data_sdk::logging::detail::LogStream(HIPDNN_SEV_ERROR, component)      \
                << msg; /* NOLINT(bugprone-macro-parentheses) msg is a stream expression */ \
        }                                                                                   \
    } while(0)

#define HIPDNN_SDK_LOG_FATAL_WITH_COMPONENT(component, msg)                                 \
    do                                                                                      \
    {                                                                                       \
        if(::hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_FATAL))                 \
        {                                                                                   \
            ::hipdnn_data_sdk::logging::detail::LogStream(HIPDNN_SEV_FATAL, component)      \
                << msg; /* NOLINT(bugprone-macro-parentheses) msg is a stream expression */ \
        }                                                                                   \
    } while(0)

// ============================================================================
// SDK Logging Macros (HIPDNN_SDK_LOG_*)
// ============================================================================
// These macros are the core stream-style logging for hipDNN SDK.
// - Always stream-style: HIPDNN_SDK_LOG_INFO("msg " << value)
// - Assumes callback is registered (no auto-init)
// - Uses fixed "hipdnn_sdk" component name
//
// Usage in data_sdk code:
//   HIPDNN_SDK_LOG_WARN("Warning: " << someValue);
//   HIPDNN_SDK_LOG_ERROR("Error in " << functionName);

#define HIPDNN_SDK_LOG_INFO(msg) \
    HIPDNN_SDK_LOG_INFO_WITH_COMPONENT(::hipdnn_data_sdk::logging::K_COMPONENT_NAME, msg)

#define HIPDNN_SDK_LOG_WARN(msg) \
    HIPDNN_SDK_LOG_WARN_WITH_COMPONENT(::hipdnn_data_sdk::logging::K_COMPONENT_NAME, msg)

#define HIPDNN_SDK_LOG_ERROR(msg) \
    HIPDNN_SDK_LOG_ERROR_WITH_COMPONENT(::hipdnn_data_sdk::logging::K_COMPONENT_NAME, msg)

#define HIPDNN_SDK_LOG_FATAL(msg) \
    HIPDNN_SDK_LOG_FATAL_WITH_COMPONENT(::hipdnn_data_sdk::logging::K_COMPONENT_NAME, msg)
