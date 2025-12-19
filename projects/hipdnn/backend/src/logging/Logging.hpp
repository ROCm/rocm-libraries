// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "EnumFormatters.hpp"
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <memory>
#include <spdlog/spdlog.h>

#ifdef HIPDNN_BACKEND_COMPILATION
#define _HIPDNN_BACKEND_LOG_ACTION(spdlog_level, ...)               \
    do                                                              \
    {                                                               \
        hipdnn_backend::logging::initialize();                      \
        auto _logger = hipdnn_backend::logging::getBackendLogger(); \
        if(_logger && _logger->should_log(spdlog_level))            \
        {                                                           \
            _logger->log(spdlog_level, __VA_ARGS__);                \
        }                                                           \
    } while(0)

#define HIPDNN_LOG_INFO(...) \
    _HIPDNN_BACKEND_LOG_ACTION(spdlog::level::level_enum::info, __VA_ARGS__)
#define HIPDNN_LOG_WARN(...) \
    _HIPDNN_BACKEND_LOG_ACTION(spdlog::level::level_enum::warn, __VA_ARGS__)
#define HIPDNN_LOG_ERROR(...) \
    _HIPDNN_BACKEND_LOG_ACTION(spdlog::level::level_enum::err, __VA_ARGS__)
#define HIPDNN_LOG_FATAL(...) \
    _HIPDNN_BACKEND_LOG_ACTION(spdlog::level::level_enum::critical, __VA_ARGS__)
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
