// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

/**
 * @file PluginApiDataTypes.h
 * @brief Header file for the hipDNN Plugin API data types.
 *
 * This file contains the definitions of the data types used in the hipDNN Plugin API.
 */

#include <stddef.h>
#include <stdint.h>

// NOLINTBEGIN
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup PluginDataTypes Plugin Data Types
 * @brief Data types used in the Plugin API.
 * @{
 */

/**
 * @brief Enumeration for the status of plugin operations.
 *
 * This enumeration defines the different status codes that can be returned by
 * the hipDNN Plugin API functions.
 */
typedef enum
{
    HIPDNN_PLUGIN_STATUS_SUCCESS = 0,
    HIPDNN_PLUGIN_STATUS_BAD_PARAM = 1,
    HIPDNN_PLUGIN_STATUS_INVALID_VALUE = 2,
    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR = 3,
    HIPDNN_PLUGIN_STATUS_ALLOC_FAILED = 4,
} hipdnnPluginStatus_t;

/**
 * @brief Enumeration for the type of a plugin.
 *
 * This enumeration defines the different types of plugins.
 */
typedef enum
{
    HIPDNN_PLUGIN_TYPE_UNSPECIFIED = 0, // Unspecified plugin type
    HIPDNN_PLUGIN_TYPE_ENGINE = 1, // Plugin with engines
} hipdnnPluginType_t;

/**
 * @brief Structure for describing a constant data buffer.
 *
 * This structure provides a way to pass buffer information (a pointer and a size) into and out of functions.
 */
typedef struct
{
    const void* ptr;
    size_t size;
} hipdnnPluginConstData_t;

/**
 * @brief Represents a device buffer.
 *
 * This structure encapsulates information about a device buffer, including its unique identifier
 * and a pointer to the memory location on the device.
 */
typedef struct
{
    int64_t uid;
    void* ptr;
} hipdnnPluginDeviceBuffer_t;

/**
 * @brief Opaque handle for an engine plugin.
 *
 * This handle is used to represent an engine plugin.
 */
typedef struct HipdnnEnginePluginHandle* hipdnnEnginePluginHandle_t;

/**
 * @brief Opaque handle for an engine execution context.
 *
 * This handle is used to represent the execution context of an engine.
 */
typedef struct HipdnnEnginePluginExecutionContext* hipdnnEnginePluginExecutionContext_t;

/** @} */ // End of PluginDataTypes group

#ifdef __cplusplus
}
#endif
// NOLINTEND
