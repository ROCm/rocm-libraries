// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// This file's definitions are duplicated by HipdnnBackendCallbackTypes.h.
// Please ensure any updates are synced between the two files!

#ifndef HIPDNN_CALLBACK_TYPES_DEFINED
#define HIPDNN_CALLBACK_TYPES_DEFINED

/**
 * @brief Severity levels for logging in hipDNN
 */
typedef enum
{
    HIPDNN_SEV_INFO = 0,
    HIPDNN_SEV_WARN,
    HIPDNN_SEV_ERROR,
    HIPDNN_SEV_FATAL,
    HIPDNN_SEV_OFF
} hipdnnSeverity_t;

/**
 * @brief Callback function type used by hipDNN sinks to relay messages to clients.
 *
 * @param severity The severity level of the log message.
 * @param message The log message, formatted by the logger.
 */
typedef void (*hipdnnCallback_t)(hipdnnSeverity_t severity, const char* message);

/**
 * @brief User-defined opaque handle passed to callback and used as unique ID.
 *
 * The userHandle serves two purposes:
 * 1. Passed back as parameter to the user callback function
 * 2. Used as part of the unique identifier for the callback (callback, userHandle)
 */
typedef void* hipdnnUserLogCallbackHandle_t;

/**
 * @brief User-logging callback function type.
 *
 * @param[in] userHandle User-provided context handle
 * @param[in] severity   Log message severity level
 * @param[in] message    The log message (null-terminated string, includes component name)
 *
 * @note Callback should return promptly. For blocking operations (network, disk I/O),
 *       queue work to a separate thread to avoid impacting hipDNN performance.
 */
typedef void (*hipdnnUserLogCallback_t)(hipdnnUserLogCallbackHandle_t userHandle,
                                        hipdnnSeverity_t severity,
                                        const char* message);

#endif // HIPDNN_CALLBACK_TYPES_DEFINED

#ifdef __cplusplus
}
#endif
