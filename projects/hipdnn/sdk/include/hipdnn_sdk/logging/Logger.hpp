// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "CallbackSink.hpp"
#include "CallbackTypes.h"
#include <iostream>
#include <memory>
#include <mutex>
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#ifndef HIPDNN_BACKEND_COMPILATION
#ifndef COMPONENT_NAME
#define _HIPDNN_INTERNAL_LOG_ACTION(level, ...) \
    do                                          \
    {                                           \
    } while(0)
#else
#define _HIPDNN_INTERNAL_LOG_ACTION(level, ...)       \
    do                                                \
    {                                                 \
        if(auto logger = spdlog::get(COMPONENT_NAME)) \
        {                                             \
            logger->level(__VA_ARGS__);               \
        }                                             \
    } while(0)
#endif // COMPONENT_NAME

#define HIPDNN_LOG_INFO(...) _HIPDNN_INTERNAL_LOG_ACTION(info, __VA_ARGS__)
#define HIPDNN_LOG_WARN(...) _HIPDNN_INTERNAL_LOG_ACTION(warn, __VA_ARGS__)
#define HIPDNN_LOG_ERROR(...) _HIPDNN_INTERNAL_LOG_ACTION(error, __VA_ARGS__)
#define HIPDNN_LOG_FATAL(...) _HIPDNN_INTERNAL_LOG_ACTION(critical, __VA_ARGS__)
#endif // HIPDNN_BACKEND_COMPILATION

namespace hipdnn::logging
{
inline void initializeCallbackLogging(const std::string& componentName,
                                      hipdnnCallback_t callbackFunction)
{
    try
    {
        static std::mutex s_callbackInitMutex;
        std::lock_guard<std::mutex> lock(s_callbackInitMutex);

        if(spdlog::get(componentName))
        {
            return;
        }

        if(!spdlog::thread_pool())
        {
            spdlog::init_thread_pool(8192, 1);
        }

        auto callbackLogger
            = hipdnn_sdk::logging::createAsyncCallbackLoggerMt(callbackFunction, componentName);
        spdlog::register_logger(callbackLogger);
    }
    catch(const spdlog::spdlog_ex& ex)
    {
        std::cerr << "hipDNN SDK: Failed to initialize callback logger for component '"
                  << componentName << "'. Error: " << ex.what() << "\n";
    }
}

} // namespace hipdnn::logging
