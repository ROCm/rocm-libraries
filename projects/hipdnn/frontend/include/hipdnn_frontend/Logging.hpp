// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file Logging.hpp
 * @brief Frontend logging initialization and convenience macros
 *
 * hipDNN uses a callback-based logging system (similar to Python's
 * `logging` module). The macros in this file — HIPDNN_FE_LOG_INFO,
 * HIPDNN_FE_LOG_WARN, etc. — auto-initialize on first use and tag every
 * message with "hipdnn_frontend" so you can filter frontend output from
 * backend or plugin messages.
 *
 * Log verbosity is controlled by the `HIPDNN_LOG_LEVEL` environment
 * variable (e.g. `HIPDNN_LOG_LEVEL=info`).
 */

#pragma once

#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_data_sdk/logging/Logger.hpp>

namespace hipdnn_frontend
{

/// @brief Component name used for all frontend log messages
inline constexpr const char* K_COMPONENT_NAME = "hipdnn_frontend";

/**
 * @brief Initialize the frontend logging subsystem
 *
 * Registers a logging callback and reads the log level from the environment.
 * Subsequent calls are no-ops (initialization happens at most once per
 * shared object). The HIPDNN_FE_LOG_* macros call this automatically.
 *
 * @param fn Logging callback function (defaults to the backend-provided callback)
 * @return 0 on success or if already initialized, -1 if fn is null
 */
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

/**
 * @name Frontend Logging Macros
 * @brief Auto-initializing logging macros for the hipDNN frontend
 *
 * These macros initialize logging on first use and emit messages with
 * "hipdnn_frontend" as the component name. Supports streaming syntax:
 * @code{.cpp}
 * HIPDNN_FE_LOG_INFO("value = " << x);
 * @endcode
 * @{
 */

/** @def HIPDNN_FE_LOG_INFO
 *  @brief Log an informational message */
#define HIPDNN_FE_LOG_INFO(msg)                                                     \
    do                                                                              \
    {                                                                               \
        hipdnn_frontend::initializeFrontendLogging();                               \
        HIPDNN_SDK_LOG_INFO_WITH_COMPONENT(hipdnn_frontend::K_COMPONENT_NAME, msg); \
    } while(0)

/** @def HIPDNN_FE_LOG_WARN
 *  @brief Log a warning message */
#define HIPDNN_FE_LOG_WARN(msg)                                                     \
    do                                                                              \
    {                                                                               \
        hipdnn_frontend::initializeFrontendLogging();                               \
        HIPDNN_SDK_LOG_WARN_WITH_COMPONENT(hipdnn_frontend::K_COMPONENT_NAME, msg); \
    } while(0)

/** @def HIPDNN_FE_LOG_ERROR
 *  @brief Log an error message */
#define HIPDNN_FE_LOG_ERROR(msg)                                                     \
    do                                                                               \
    {                                                                                \
        hipdnn_frontend::initializeFrontendLogging();                                \
        HIPDNN_SDK_LOG_ERROR_WITH_COMPONENT(hipdnn_frontend::K_COMPONENT_NAME, msg); \
    } while(0)

/** @def HIPDNN_FE_LOG_FATAL
 *  @brief Log a fatal error message */
#define HIPDNN_FE_LOG_FATAL(msg)                                                     \
    do                                                                               \
    {                                                                                \
        hipdnn_frontend::initializeFrontendLogging();                                \
        HIPDNN_SDK_LOG_FATAL_WITH_COMPONENT(hipdnn_frontend::K_COMPONENT_NAME, msg); \
    } while(0)
/** @} */ // end of Frontend Logging Macros group
} // namespace hipdnn_frontend
