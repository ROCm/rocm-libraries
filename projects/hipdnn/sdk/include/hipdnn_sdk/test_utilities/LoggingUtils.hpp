// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/logging/CallbackTypes.h>
#include <hipdnn_sdk/logging/ComponentFormatter.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>

#include <cstdlib>
#include <iostream>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <string>

namespace hipdnn_sdk::test_utilities
{

inline hipdnnSeverity_t stringToSeverity(const std::string& levelStr)
{
    if(levelStr == "info")
    {
        return HIPDNN_SEV_INFO;
    }
    if(levelStr == "warn")
    {
        return HIPDNN_SEV_WARN;
    }
    if(levelStr == "error")
    {
        return HIPDNN_SEV_ERROR;
    }
    if(levelStr == "fatal")
    {
        return HIPDNN_SEV_FATAL;
    }
    return HIPDNN_SEV_OFF;
}

inline void testLoggingCallback(hipdnnSeverity_t severity, const char* message)
{
#ifndef DISABLE_TEST_LOGGING
    std::string logLevelStr = hipdnn_sdk::utilities::getEnv("HIPDNN_LOG_LEVEL", "off");

    if(logLevelStr == "off")
    {
        return;
    }

    hipdnnSeverity_t configuredLevel = stringToSeverity(logLevelStr);

    if(severity >= configuredLevel)
    {
        std::cerr << message << '\n';
    }
#endif
}

inline void initializeSpdlogDefaultLogger(const std::string& componentName)
{
#ifndef DISABLE_TEST_LOGGING
    spdlog::drop_all();
    auto logger = spdlog::stdout_color_mt(componentName);
    logger->set_formatter(std::make_unique<hipdnn_sdk::logging::ComponentFormatter>());
    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::info); // Set default log level
#endif
}

} // namespace hipdnn_sdk::test_utilities
