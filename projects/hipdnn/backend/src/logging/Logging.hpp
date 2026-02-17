// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <memory>
#include <sstream>
#include <string>

// Backend-specific logging macros
// These are separate from HIPDNN_LOG_* used by frontend/plugins to avoid conflicts
#ifdef HIPDNN_BACKEND_COMPILATION
#include <hipdnn_data_sdk/logging/LogLevel.hpp>

#define HIPDNN_BACKEND_LOG_INFO(msg)                                                          \
    do                                                                                        \
    {                                                                                         \
        hipdnn_backend::logging::initialize();                                                \
        if(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO))                      \
        {                                                                                     \
            std::ostringstream _hipdnn_log_oss;                                               \
            _hipdnn_log_oss << msg;                                                           \
            hipdnn_backend::logging::logMessage(HIPDNN_SEV_INFO,                              \
                                                "[hipdnn_backend] " + _hipdnn_log_oss.str()); \
        }                                                                                     \
    } while(0)

#define HIPDNN_BACKEND_LOG_WARN(msg)                                                          \
    do                                                                                        \
    {                                                                                         \
        hipdnn_backend::logging::initialize();                                                \
        if(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_WARN))                      \
        {                                                                                     \
            std::ostringstream _hipdnn_log_oss;                                               \
            _hipdnn_log_oss << msg;                                                           \
            hipdnn_backend::logging::logMessage(HIPDNN_SEV_WARN,                              \
                                                "[hipdnn_backend] " + _hipdnn_log_oss.str()); \
        }                                                                                     \
    } while(0)

#define HIPDNN_BACKEND_LOG_ERROR(msg)                                                         \
    do                                                                                        \
    {                                                                                         \
        hipdnn_backend::logging::initialize();                                                \
        if(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_ERROR))                     \
        {                                                                                     \
            std::ostringstream _hipdnn_log_oss;                                               \
            _hipdnn_log_oss << msg;                                                           \
            hipdnn_backend::logging::logMessage(HIPDNN_SEV_ERROR,                             \
                                                "[hipdnn_backend] " + _hipdnn_log_oss.str()); \
        }                                                                                     \
    } while(0)

#define HIPDNN_BACKEND_LOG_FATAL(msg)                                                         \
    do                                                                                        \
    {                                                                                         \
        hipdnn_backend::logging::initialize();                                                \
        if(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_FATAL))                     \
        {                                                                                     \
            std::ostringstream _hipdnn_log_oss;                                               \
            _hipdnn_log_oss << msg;                                                           \
            hipdnn_backend::logging::logMessage(HIPDNN_SEV_FATAL,                             \
                                                "[hipdnn_backend] " + _hipdnn_log_oss.str()); \
        }                                                                                     \
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
