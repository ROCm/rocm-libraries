// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <memory>
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>

// Backend-specific logging macros
// These are separate from HIPDNN_LOG_* used by frontend/plugins to avoid conflicts
#ifdef HIPDNN_BACKEND_COMPILATION

#define HIPDNN_BACKEND_LOG_INFO(msg)                                              \
    do                                                                            \
    {                                                                             \
        hipdnn_backend::logging::initialize();                                    \
        auto _logger = hipdnn_backend::logging::getBackendLogger();               \
        if(_logger && _logger->should_log(spdlog::level::level_enum::info))       \
        {                                                                         \
            std::ostringstream _hipdnn_log_oss;                                   \
            _hipdnn_log_oss << msg;                                               \
            _logger->log(spdlog::level::level_enum::info, _hipdnn_log_oss.str()); \
        }                                                                         \
    } while(0)

#define HIPDNN_BACKEND_LOG_WARN(msg)                                              \
    do                                                                            \
    {                                                                             \
        hipdnn_backend::logging::initialize();                                    \
        auto _logger = hipdnn_backend::logging::getBackendLogger();               \
        if(_logger && _logger->should_log(spdlog::level::level_enum::warn))       \
        {                                                                         \
            std::ostringstream _hipdnn_log_oss;                                   \
            _hipdnn_log_oss << msg;                                               \
            _logger->log(spdlog::level::level_enum::warn, _hipdnn_log_oss.str()); \
        }                                                                         \
    } while(0)

#define HIPDNN_BACKEND_LOG_ERROR(msg)                                            \
    do                                                                           \
    {                                                                            \
        hipdnn_backend::logging::initialize();                                   \
        auto _logger = hipdnn_backend::logging::getBackendLogger();              \
        if(_logger && _logger->should_log(spdlog::level::level_enum::err))       \
        {                                                                        \
            std::ostringstream _hipdnn_log_oss;                                  \
            _hipdnn_log_oss << msg;                                              \
            _logger->log(spdlog::level::level_enum::err, _hipdnn_log_oss.str()); \
        }                                                                        \
    } while(0)

#define HIPDNN_BACKEND_LOG_FATAL(msg)                                                 \
    do                                                                                \
    {                                                                                 \
        hipdnn_backend::logging::initialize();                                        \
        auto _logger = hipdnn_backend::logging::getBackendLogger();                   \
        if(_logger && _logger->should_log(spdlog::level::level_enum::critical))       \
        {                                                                             \
            std::ostringstream _hipdnn_log_oss;                                       \
            _hipdnn_log_oss << msg;                                                   \
            _logger->log(spdlog::level::level_enum::critical, _hipdnn_log_oss.str()); \
        }                                                                             \
    } while(0)

#endif // HIPDNN_BACKEND_COMPILATION

namespace hipdnn_backend::logging
{

void initialize();

void cleanup();

void setLogLevel(const std::string& level);

std::shared_ptr<spdlog::logger> getBackendLogger();

std::shared_ptr<spdlog::logger> getCallbackReceiverLogger();

void hipdnnLoggingCallback(hipdnnSeverity_t severity, const char* msg);

void logSystemInfo();

void logHipDeviceInfo(hipStream_t stream);

} // namespace hipdnn_backend::logging
