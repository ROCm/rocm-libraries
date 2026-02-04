// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <array>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <string>

namespace hipdnn_data_sdk::logging
{

inline constexpr std::array<const char*, 5> VALID_LOG_LEVELS
    = {"off", "info", "warn", "error", "fatal"};

inline bool isValidLogLevel(const std::string& level)
{
    return std::find(VALID_LOG_LEVELS.begin(), VALID_LOG_LEVELS.end(), level)
           != VALID_LOG_LEVELS.end();
}

inline bool isLoggingEnabled()
{
    auto logLevel = hipdnn_data_sdk::utilities::getEnv("HIPDNN_LOG_LEVEL", "off");
    return isValidLogLevel(logLevel) && logLevel != "off";
}

/**
 * @brief Generate the spdlog pattern string for callback-based logging
 *
 * This pattern includes only the component name prefix in brackets for consistency
 * with backend logging format. The backend adds timestamp, thread ID, and log level
 * when receiving the callback message.
 *
 * @param componentName The name of the component
 * @return The spdlog pattern string with bracketed component prefix
 */
inline std::string generatePatternString(const std::string& componentName)
{
    return "[" + componentName + "] %v";
}

} // namespace hipdnn_data_sdk::logging
