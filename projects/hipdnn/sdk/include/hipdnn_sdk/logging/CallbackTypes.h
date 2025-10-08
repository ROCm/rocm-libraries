// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif
