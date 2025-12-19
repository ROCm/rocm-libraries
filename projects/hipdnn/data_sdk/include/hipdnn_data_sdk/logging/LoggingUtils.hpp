// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <array>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <string>

namespace hipdnn_sdk::logging
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
    auto logLevel = hipdnn_sdk::utilities::getEnv("HIPDNN_LOG_LEVEL", "off");
    return isValidLogLevel(logLevel) && logLevel != "off";
}

inline std::string generatePatternString(const std::string& componentName)
{
    return "[%Y-%m-%d %H:%M:%S.%e] [tid %t] [%l] [" + componentName + "] %v";
}

} // namespace hipdnn_sdk::logging
