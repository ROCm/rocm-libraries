// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_data_sdk/logging/Logger.hpp>

namespace hipdnn_frontend
{

inline constexpr const char* K_COMPONENT_NAME = "hipdnn_frontend";

// HIPDNN_HIDDEN ensures each shared object has its own copy of the static variable
HIPDNN_HIDDEN inline int32_t initializeFrontendLogging(hipdnnCallback_t fn
                                                       = hipdnnLoggingCallback_ext)
{
    if(fn == nullptr)
    {
        return -1;
    }

    static bool s_loggingInitialized = false;

    if(s_loggingInitialized)
    {
        return 0;
    }

    // Initialize log level from environment variable
    hipdnn_data_sdk::logging::initializeLogLevel();

    // Register the callback so log messages get routed to the backend
    hipdnn_data_sdk::logging::registerLoggingCallback(fn);

    s_loggingInitialized = true;

    // Use this logging macro directly to avoid re-entrant logging call.
    HIPDNN_SDK_LOG_INFO_WITH_COMPONENT(K_COMPONENT_NAME, "Frontend logging initialized");

    return 0;
}

// ============================================================================
// Frontend Logging Macros (HIPDNN_FE_LOG_*)
// ============================================================================
// These macros auto-initialize logging on first use, then log with "hipdnn_frontend"
// as the component name.
// Usage: HIPDNN_FE_LOG_INFO("Message " << value);

#define HIPDNN_FE_LOG_INFO(msg)                                                     \
    do                                                                              \
    {                                                                               \
        hipdnn_frontend::initializeFrontendLogging();                               \
        HIPDNN_SDK_LOG_INFO_WITH_COMPONENT(hipdnn_frontend::K_COMPONENT_NAME, msg); \
    } while(0)

#define HIPDNN_FE_LOG_WARN(msg)                                                     \
    do                                                                              \
    {                                                                               \
        hipdnn_frontend::initializeFrontendLogging();                               \
        HIPDNN_SDK_LOG_WARN_WITH_COMPONENT(hipdnn_frontend::K_COMPONENT_NAME, msg); \
    } while(0)

#define HIPDNN_FE_LOG_ERROR(msg)                                                     \
    do                                                                               \
    {                                                                                \
        hipdnn_frontend::initializeFrontendLogging();                                \
        HIPDNN_SDK_LOG_ERROR_WITH_COMPONENT(hipdnn_frontend::K_COMPONENT_NAME, msg); \
    } while(0)

#define HIPDNN_FE_LOG_FATAL(msg)                                                     \
    do                                                                               \
    {                                                                                \
        hipdnn_frontend::initializeFrontendLogging();                                \
        HIPDNN_SDK_LOG_FATAL_WITH_COMPONENT(hipdnn_frontend::K_COMPONENT_NAME, msg); \
    } while(0)
} // namespace hipdnn_frontend
