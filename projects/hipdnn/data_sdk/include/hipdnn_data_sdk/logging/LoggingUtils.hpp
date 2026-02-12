// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "LogLevel.hpp"
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <string>

namespace hipdnn_data_sdk::logging
{

/**
 * @brief Check if a log level string is valid
 *
 * Valid levels are: "off", "info", "warn", "error", "fatal" (case-insensitive, whitespace-tolerant)
 *
 * @param level The log level string to validate
 * @return true if the level is valid, false otherwise
 */
inline bool isValidLogLevel(const std::string& level)
{
    return detail::stringToSeverity(level).has_value();
}

/**
 * @brief Check if logging is currently enabled based on HIPDNN_LOG_LEVEL environment variable
 *
 * @return true if logging is enabled (level != "off"), false otherwise
 */
inline bool isLoggingEnabled()
{
    auto logLevel = hipdnn_data_sdk::utilities::getEnv("HIPDNN_LOG_LEVEL", "off");
    return detail::stringToSeverityOrOff(logLevel) != HIPDNN_SEV_OFF;
}

} // namespace hipdnn_data_sdk::logging
