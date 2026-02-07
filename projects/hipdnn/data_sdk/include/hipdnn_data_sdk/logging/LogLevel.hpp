// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "CallbackTypes.h"
#include <hipdnn_data_sdk/Visibility.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

#include <atomic>
#include <string>

namespace hipdnn_data_sdk::logging
{

namespace detail
{
// Cached log level for fast lookups
// HIPDNN_HIDDEN ensures each shared object has its own copy of the static variable
HIPDNN_HIDDEN inline std::atomic<hipdnnSeverity_t>& getLogLevelCache()
{
    static std::atomic<hipdnnSeverity_t> s_logLevel{HIPDNN_SEV_OFF};
    return s_logLevel;
}

HIPDNN_HIDDEN inline std::atomic<bool>& getLogLevelInitialized()
{
    static std::atomic<bool> s_initialized{false};
    return s_initialized;
}

inline hipdnnSeverity_t stringToSeverity(const std::string& level)
{
    if(level == "info")
    {
        return HIPDNN_SEV_INFO;
    }
    if(level == "warn")
    {
        return HIPDNN_SEV_WARN;
    }
    if(level == "error")
    {
        return HIPDNN_SEV_ERROR;
    }
    if(level == "fatal")
    {
        return HIPDNN_SEV_FATAL;
    }
    return HIPDNN_SEV_OFF;
}

} // namespace detail

/**
 * @brief Initialize log level from environment variable HIPDNN_LOG_LEVEL
 *
 * Should be called once at startup. Safe to call multiple times.
 */
inline void initializeLogLevel()
{
    if(detail::getLogLevelInitialized().exchange(true))
    {
        return; // Already initialized
    }

    std::string logLevel = hipdnn_data_sdk::utilities::getEnv("HIPDNN_LOG_LEVEL", "off");
    detail::getLogLevelCache().store(detail::stringToSeverity(logLevel), std::memory_order_relaxed);
}

/**
 * @brief Set the current log level
 *
 * @param level The severity level to set
 */
inline void setLogLevel(hipdnnSeverity_t level)
{
    detail::getLogLevelCache().store(level, std::memory_order_relaxed);
    detail::getLogLevelInitialized().store(true, std::memory_order_relaxed);
}

/**
 * @brief Get the current log level
 *
 * @return The current severity level
 */
inline hipdnnSeverity_t getLogLevel()
{
    if(!detail::getLogLevelInitialized().load(std::memory_order_relaxed))
    {
        initializeLogLevel();
    }
    return detail::getLogLevelCache().load(std::memory_order_relaxed);
}

/**
 * @brief Check if a severity level is enabled for logging
 *
 * @param severity The severity level to check
 * @return true if the severity level is enabled, false otherwise
 */
inline bool isLogLevelEnabled(hipdnnSeverity_t severity)
{
    hipdnnSeverity_t currentLevel = getLogLevel();
    if(currentLevel == HIPDNN_SEV_OFF)
    {
        return false;
    }
    // Lower enum values = more verbose (INFO=0, OFF=4)
    // A message should be logged if its severity >= current level
    return severity >= currentLevel;
}

/**
 * @brief Enable or disable logging globally
 *
 * @param enabled If true, sets log level to INFO; if false, sets to OFF
 */
inline void setLoggingEnabled(bool enabled)
{
    setLogLevel(enabled ? HIPDNN_SEV_INFO : HIPDNN_SEV_OFF);
}

/**
 * @brief Reset logging state (for testing purposes)
 */
inline void resetLogging()
{
    detail::getLogLevelInitialized().store(false, std::memory_order_relaxed);
    detail::getLogLevelCache().store(HIPDNN_SEV_OFF, std::memory_order_relaxed);
}

} // namespace hipdnn_data_sdk::logging
