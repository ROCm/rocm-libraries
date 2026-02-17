// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file PluginLogging.hpp
 * @brief Plugin logging macros using stream-style syntax
 *
 * This header provides HIPDNN_PLUGIN_LOG_* macros for use in plugins.
 *
 * Usage: HIPDNN_PLUGIN_LOG_INFO("Value: " << value);
 *
 * The component name is stored at runtime when initializeCallbackLogging() is called.
 */

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
