// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file Logging.hpp
 * @brief Frontend logging configuration and log-level control
 *
 * Log verbosity is controlled by the `HIPDNN_LOG_LEVEL` environment
 * variable. Valid values (case-insensitive):
 * | Value   | Effect                                      |
 * |---------|---------------------------------------------|
 * | `info`  | Informational messages and above             |
 * | `warn`  | Warnings and above                          |
 * | `error` | Errors and fatal messages only               |
 * | `fatal` | Fatal messages only                          |
 * | `off`   | Disable all logging (default)                |
 *
 * Example: `HIPDNN_LOG_LEVEL=warn`
 */

#pragma once

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/detail/HipdnnBackendInterface.hpp>

#ifdef HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND
#include <atomic>

#include <hipdnn_frontend/detail/DynamicBackendLibrary.hpp>
#endif

namespace hipdnn_frontend
{

/// @cond INTERNAL
namespace detail
{
// The logging callback is resolved here (not through hipdnnBackend()) because
// initializeFrontendLogging() fires from every HIPDNN_FE_LOG_* macro, including
// during hipdnnBackend() construction. Going through the instance would re-enter.
#ifdef HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND

HIPDNN_HIDDEN inline hipdnnCallback_t resolveBackendLoggingCallback()
{
    static std::atomic<void*> s_cache{nullptr};
    return resolveBackendSymbol<hipdnnCallback_t>(s_cache, "hipdnnLoggingCallback_ext");
}

#else

HIPDNN_HIDDEN inline hipdnnCallback_t resolveBackendLoggingCallback()
{
    return hipdnnLoggingCallback_ext;
}

#endif

// Lazily initializes and returns the backend instance. Defined in
// BackendWrapper.hpp (which can't be included here due to a circular
// dependency). The definition is always available before any call site.
static std::shared_ptr<IHipdnnBackend> hipdnnBackend();

} // namespace detail

/// @brief Component name used for all frontend log messages
inline constexpr const char* K_COMPONENT_NAME = "hipdnn_frontend";

// Pass nullptr (the default) to route frontend logs to the backend's logging
// callback, resolved according to the build's backend-loading mode. Returns -1
// when no callback is available (e.g. the backend could not be loaded).
HIPDNN_HIDDEN inline int32_t initializeFrontendLogging(hipdnnCallback_t fn = nullptr)
{
    if(fn == nullptr)
    {
        fn = detail::resolveBackendLoggingCallback();
    }

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
/// @endcond

/// @cond INTERNAL
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
/// @endcond

// === Logging Callback and Log Level APIs ===

/**
 * @brief User log callback modes (sync vs async).
 */
enum class LogCallbackMode
{
    SYNC = HIPDNN_LOG_CALLBACK_SYNC, ///< Callback invoked on logging thread (synchronous)
    ASYNC = HIPDNN_LOG_CALLBACK_ASYNC ///< Callback invoked on worker thread (asynchronous)
};

/**
 * @brief Set or update a user log callback.
 *
 * This API allows registering multiple user callbacks with individual log levels and sync/async modes.
 * Each callback is uniquely identified by the composite key (callback, userHandle).
 *
 * @note When a synchronous callback is registered, the synchronous callbacks will delay hipDNN
 *       until the callback returns, regardless of any async log callbacks also being registered.
 *
 * Behavior:
 * - If (callback, userHandle) already registered: UPDATES settings (level and/or mode)
 * - If (callback, userHandle) new: ADDS new registration
 * - If minLevel == SEV_OFF: REMOVES callback
 * - userHandle must be non-null
 *
 * Callback Removal (minLevel == SEV_OFF):
 * - Callback will be atomically disabled and no further logs will be received
 * - Any pending async logs for this callback will be abandoned
 * - After this function returns, user can safely destroy data referenced by userHandle
 *
 * @param callback   The callback function to invoke
 * @param minLevel   Minimum severity level (SEV_OFF removes the callback). Note that
 * the logs produced on this callback will be limited by the global log level set either by
 * the HIPDNN_LOG_LEVEL environment variable or the setGlobalLogLevel() API function.
 * @param mode       Sync or async invocation mode
 * @param userHandle Non-null user data (also serves as unique callback ID)
 * @return Error object indicating success or failure
 */
inline Error setUserLogCallback(hipdnnUserLogCallback_t callback,
                                hipdnnSeverity_t minLevel,
                                LogCallbackMode mode,
                                hipdnnUserLogCallbackHandle_t userHandle)
{
    auto status = detail::hipdnnBackend()->setUserLogCallbackExt(
        callback, minLevel, static_cast<hipdnnLogCallbackMode_t>(mode), userHandle);
    if(status != HIPDNN_STATUS_SUCCESS)
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to set user log callback"};
    }
    return {};
}

/**
 * @brief Set the global log level.
 *
 * This controls which log messages are output to console/file AND to the global backend log output callback.
 * Updates BOTH frontend and backend log levels.
 *
 * @param level   The severity level to set
 * @return Error object indicating success or failure
 */
inline Error setGlobalLogLevel(hipdnnSeverity_t level)
{
    // Update frontend's cache (in user executable)
    hipdnn_data_sdk::logging::setLogLevel(level);

    auto status = detail::hipdnnBackend()->backendSetGlobalLogLevelExt(level);
    if(status != HIPDNN_STATUS_SUCCESS)
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to set global log level"};
    }
    return {};
}

/**
 * @brief Get the global log level.
 *
 * @param[out] level   The current severity level
 * @return Error object indicating success or failure
 */
inline Error getGlobalLogLevel(hipdnnSeverity_t& level)
{
    auto status = detail::hipdnnBackend()->backendGetGlobalLogLevelExt(&level);
    if(status != HIPDNN_STATUS_SUCCESS)
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to get global log level"};
    }
    return {};
}
} // namespace hipdnn_frontend
