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
    const std::string normalized = utilities::toLower(utilities::trim(level));

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
    // Multiple threads may execute this concurrently. The initialized flag is set after
    // the cache has been initialized. Consequently, depending on timing, multiple threads
    // may see initialzied=false and proceed to read the value from getEnv() (which is reentrant)
    // and set the cache value (to the same value) before initialized is set to true. This
    // is considered an acceptable trade-off to avoid the overhead of having a mutex to guard
    // the log level cache initialization.
    if(detail::getLogLevelInitialized().load(std::memory_order_acquire))
    {
        return; // Already initialized
    }

    const std::string logLevel = hipdnn_data_sdk::utilities::getEnv("HIPDNN_LOG_LEVEL", "off");
    detail::getLogLevelCache().store(detail::stringToSeverityOrOff(logLevel),
                                     std::memory_order_release);
    detail::getLogLevelInitialized().store(true, std::memory_order_release);
}

/**
 * @brief Set the current log level
 *
 * @param level The severity level to set
 */
inline void setLogLevel(hipdnnSeverity_t level)
{
    detail::getLogLevelCache().store(level, std::memory_order_release);
    detail::getLogLevelInitialized().store(true, std::memory_order_release);
}

/**
 * @brief Get the current log level
 *
 * @return The current severity level
 */
inline hipdnnSeverity_t getLogLevel()
{
    if(!detail::getLogLevelInitialized().load(std::memory_order_acquire))
    {
        initializeLogLevel();
    }
    return detail::getLogLevelCache().load(std::memory_order_acquire);
}

/**
 * @brief Check if a severity level is enabled for logging
 *
 * @param severity The severity level to check
 * @return true if the severity level is enabled, false otherwise
 */
inline bool isLogLevelEnabled(hipdnnSeverity_t severity)
{
    const hipdnnSeverity_t currentLevel = getLogLevel();
    if(currentLevel == HIPDNN_SEV_OFF)
    {
        return false;
    }
    // Lower enum values = more verbose (INFO=0, OFF=4)
    // A message should be logged if its severity >= current level
    return severity >= currentLevel;
}

/**
 * @brief Reset logging state
 *
 * This resets the log level initialization flag, causing the next getLogLevel()
 * call to re-read from the environment variable.
 *
 * @note This will effectively erase any log level that may have been set
 * using setLogLevel() since the log level was first initialized, and the system will
 * revert to the value set in the environment unless another call to setLogLevel()
 * is made after calling this function.
 */
inline void resetLogLevelCache()
{
    detail::getLogLevelInitialized().store(false, std::memory_order_release);
}

} // namespace hipdnn_data_sdk::logging
