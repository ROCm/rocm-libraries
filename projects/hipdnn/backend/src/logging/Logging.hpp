// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "EnumFormatters.hpp"
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <memory>

// Backend-specific logging macros
// These are separate from HIPDNN_LOG_* used by frontend/plugins to avoid conflicts
#ifdef HIPDNN_BACKEND_COMPILATION
#include <hipdnn_data_sdk/logging/LogLevel.hpp>

#define HIPDNN_BACKEND_LOG_INFO(...)                                             \
    do                                                                           \
    {                                                                            \
        hipdnn_backend::logging::initialize();                                   \
        HIPDNN_SDK_LOG_INFO_WITH_COMPONENT(                                      \
            hipdnn_backend::logging::detail::K_BACKEND_LOGGER_COMPONENT_NAME,    \
            hipdnn_backend::logging::detail::formatBackendMessage(__VA_ARGS__)); \
    } while(0)

#define HIPDNN_BACKEND_LOG_WARN(...)                                             \
    do                                                                           \
    {                                                                            \
        hipdnn_backend::logging::initialize();                                   \
        HIPDNN_SDK_LOG_WARN_WITH_COMPONENT(                                      \
            hipdnn_backend::logging::detail::K_BACKEND_LOGGER_COMPONENT_NAME,    \
            hipdnn_backend::logging::detail::formatBackendMessage(__VA_ARGS__)); \
    } while(0)

#define HIPDNN_BACKEND_LOG_ERROR(...)                                            \
    do                                                                           \
    {                                                                            \
        hipdnn_backend::logging::initialize();                                   \
        HIPDNN_SDK_LOG_ERROR_WITH_COMPONENT(                                     \
            hipdnn_backend::logging::detail::K_BACKEND_LOGGER_COMPONENT_NAME,    \
            hipdnn_backend::logging::detail::formatBackendMessage(__VA_ARGS__)); \
    } while(0)

#define HIPDNN_BACKEND_LOG_FATAL(...)                                            \
    do                                                                           \
    {                                                                            \
        hipdnn_backend::logging::initialize();                                   \
        HIPDNN_SDK_LOG_FATAL_WITH_COMPONENT(                                     \
            hipdnn_backend::logging::detail::K_BACKEND_LOGGER_COMPONENT_NAME,    \
            hipdnn_backend::logging::detail::formatBackendMessage(__VA_ARGS__)); \
    } while(0)

// DEBUG logging - use INFO level since DEBUG doesn't exist in SDK
#define HIPDNN_BACKEND_LOG_DEBUG(...)                                            \
    do                                                                           \
    {                                                                            \
        hipdnn_backend::logging::initialize();                                   \
        HIPDNN_SDK_LOG_INFO_WITH_COMPONENT(                                      \
            hipdnn_backend::logging::detail::K_BACKEND_LOGGER_COMPONENT_NAME,    \
            hipdnn_backend::logging::detail::formatBackendMessage(__VA_ARGS__)); \
    } while(0)
#endif // HIPDNN_BACKEND_COMPILATION

namespace hipdnn_backend::logging
{

void initialize();

void loggerShutdown();

void backendLoggingCallback(hipdnnSeverity_t severity, const char* msg);

void logHipDeviceInfo(hipStream_t stream);

hipdnnStatus_t setUserLogCallback(hipdnnUserLogCallback_t callback,
                                  hipdnnSeverity_t minLevel,
                                  hipdnnLogCallbackMode_t mode,
                                  hipdnnUserLogCallbackHandle_t userHandle);

hipdnnStatus_t setGlobalLogLevel(hipdnnSeverity_t level);

hipdnnStatus_t getGlobalLogLevel(hipdnnSeverity_t& level);

namespace detail
{

inline constexpr const char* K_BACKEND_LOGGER_COMPONENT_NAME = "hipdnn_backend";

// Overload for log call with a single parameter -- for safety, do not treat as a format string.
inline std::string formatBackendMessage(const std::string& msg)
{
    return msg;
}
// Overload log call that has format string + arguments -- typical use.
template <typename... Args>
inline std::string formatBackendMessage(const char* fmtStr, Args&&... args)
{
    return fmt::format(fmt::runtime(fmtStr), std::forward<Args>(args)...);
}

} // namespace detail

} // namespace hipdnn_backend::logging
