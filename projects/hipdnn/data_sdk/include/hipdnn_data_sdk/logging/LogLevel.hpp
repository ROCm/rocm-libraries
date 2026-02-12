// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "CallbackTypes.h"
#include <hipdnn_data_sdk/Visibility.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

#include <atomic>
#include <optional>
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

/**
 * @brief Convert a log level string to severity enum
 *
 * Valid levels are: "off", "info", "warn", "error", "fatal" (case-insensitive, whitespace-tolerant)
 *
 * @param level The log level string to convert
 * @return The severity enum if valid, std::nullopt if the string is not a valid log level
 */
inline std::optional<hipdnnSeverity_t> stringToSeverity(const std::string& level)
{
    // Normalize input: trim whitespace and convert to lowercase
    std::string normalized = utilities::toLower(utilities::trim(level));

    if(normalized == "off")
    {
        return HIPDNN_SEV_OFF;
    }
    if(normalized == "info")
    {
        return HIPDNN_SEV_INFO;
    }
    if(normalized == "warn")
    {
        return HIPDNN_SEV_WARN;
    }
    if(normalized == "error")
    {
        return HIPDNN_SEV_ERROR;
    }
    if(normalized == "fatal")
    {
        return HIPDNN_SEV_FATAL;
    }
    return std::nullopt;
}

/**
 * @brief Convert a log level string to severity enum, defaulting to OFF for invalid input
 *
 * This is a convenience wrapper around stringToSeverity() that treats invalid input as OFF.
 *
 * @param level The log level string to convert
 * @return The severity enum, or HIPDNN_SEV_OFF if the string is not a valid log level
 */
inline hipdnnSeverity_t stringToSeverityOrOff(const std::string& level)
{
    return stringToSeverity(level).value_or(HIPDNN_SEV_OFF);
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
    detail::getLogLevelCache().store(detail::stringToSeverityOrOff(logLevel),
                                     std::memory_order_relaxed);
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
