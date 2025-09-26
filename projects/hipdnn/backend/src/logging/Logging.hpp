// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/logging/CallbackTypes.h>
#include <memory>
#include <spdlog/spdlog.h>

#ifdef HIPDNN_BACKEND_COMPILATION
#define _HIPDNN_BACKEND_LOG_ACTION(level, ...)                         \
    do                                                                 \
    {                                                                  \
        hipdnn_backend::logging::initialize();                         \
        if(auto _logger = hipdnn_backend::logging::getBackendLogger()) \
        {                                                              \
            _logger->level(__VA_ARGS__);                               \
        }                                                              \
    } while(0)

#define HIPDNN_LOG_INFO(...) _HIPDNN_BACKEND_LOG_ACTION(info, __VA_ARGS__)
#define HIPDNN_LOG_WARN(...) _HIPDNN_BACKEND_LOG_ACTION(warn, __VA_ARGS__)
#define HIPDNN_LOG_ERROR(...) _HIPDNN_BACKEND_LOG_ACTION(error, __VA_ARGS__)
#define HIPDNN_LOG_FATAL(...) _HIPDNN_BACKEND_LOG_ACTION(critical, __VA_ARGS__)
#endif // HIPDNN_BACKEND_COMPILATION

namespace hipdnn_backend
{
namespace logging
{

void initialize();

void cleanup();

void setLogLevel(const std::string& level);

std::shared_ptr<spdlog::logger> getBackendLogger();

std::shared_ptr<spdlog::logger> getCallbackReceiverLogger();

void hipdnnLoggingCallback(hipdnnSeverity_t severity, const char* msg);

} // namespace logging
} // namespace hipdnn_backend
