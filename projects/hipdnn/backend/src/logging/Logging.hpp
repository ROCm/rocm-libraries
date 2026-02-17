// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "EnumFormatters.hpp"
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <memory>
#include <spdlog/spdlog.h>

// Backend-specific logging macros
// These are separate from HIPDNN_LOG_* used by frontend/plugins to avoid conflicts
#ifdef HIPDNN_BACKEND_COMPILATION
#include <hipdnn_data_sdk/logging/LogLevel.hpp>

#define HIPDNN_BACKEND_LOG_INFO(...)                                                            \
    do                                                                                          \
    {                                                                                           \
        hipdnn_backend::logging::initialize();                                                  \
        if(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO))                        \
        {                                                                                       \
            hipdnn_backend::logging::logMessage(                                                \
                HIPDNN_SEV_INFO, fmt::format("[hipdnn_backend] {}", fmt::format(__VA_ARGS__))); \
        }                                                                                       \
    } while(0)

#define HIPDNN_BACKEND_LOG_WARN(...)                                                            \
    do                                                                                          \
    {                                                                                           \
        hipdnn_backend::logging::initialize();                                                  \
        if(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_WARN))                        \
        {                                                                                       \
            hipdnn_backend::logging::logMessage(                                                \
                HIPDNN_SEV_WARN, fmt::format("[hipdnn_backend] {}", fmt::format(__VA_ARGS__))); \
        }                                                                                       \
    } while(0)

#define HIPDNN_BACKEND_LOG_ERROR(...)                                                            \
    do                                                                                           \
    {                                                                                            \
        hipdnn_backend::logging::initialize();                                                   \
        if(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_ERROR))                        \
        {                                                                                        \
            hipdnn_backend::logging::logMessage(                                                 \
                HIPDNN_SEV_ERROR, fmt::format("[hipdnn_backend] {}", fmt::format(__VA_ARGS__))); \
        }                                                                                        \
    } while(0)

#define HIPDNN_BACKEND_LOG_FATAL(...)                                                            \
    do                                                                                           \
    {                                                                                            \
        hipdnn_backend::logging::initialize();                                                   \
        if(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_FATAL))                        \
        {                                                                                        \
            hipdnn_backend::logging::logMessage(                                                 \
                HIPDNN_SEV_FATAL, fmt::format("[hipdnn_backend] {}", fmt::format(__VA_ARGS__))); \
        }                                                                                        \
    } while(0)
#endif // HIPDNN_BACKEND_COMPILATION

namespace hipdnn_backend::logging
{

void initialize();

void cleanup();

void logMessage(hipdnnSeverity_t severity, const std::string& message);

void hipdnnLoggingCallback(hipdnnSeverity_t severity, const char* msg);

void logSystemInfo();

void logHipDeviceInfo(hipStream_t stream);

} // namespace hipdnn_backend::logging
