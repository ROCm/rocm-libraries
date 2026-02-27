// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file HipdnnStatus.h
 * @brief Status codes returned by hipDNN backend API functions
 *
 * This file defines the status codes used throughout the hipDNN backend API.
 * All hipDNN backend functions return a hipdnnStatus_t value to indicate
 * success or the type of error that occurred.
 */

#pragma once

/**
 * @enum hipdnnStatus_t
 * @brief Return status codes for hipDNN backend API functions
 *
 * Each hipDNN backend function returns a status code indicating the result
 * of the operation. HIPDNN_STATUS_SUCCESS indicates successful completion,
 * while other values indicate specific error conditions.
 *
 * Error codes are organized into categories:
 * - **Success** (0): Operation completed successfully
 * - **Parameter errors** (1-7): Invalid inputs from the caller
 * - **Not supported** (8): Requested functionality unavailable
 * - **Internal errors** (9-13): Library internal failures
 * - **Plugin errors** (14): Errors from loaded plugins
 */
typedef enum
{
    /** @brief Operation completed successfully */
    HIPDNN_STATUS_SUCCESS = 0,

    /** @brief Data or descriptor not properly initialized before use */
    HIPDNN_STATUS_NOT_INITIALIZED = 1,

    /**
     * @brief Invalid parameter error (category)
     *
     * An incorrect value or parameter was passed to the function.
     * Check the function documentation for valid parameter ranges.
     */
    HIPDNN_STATUS_BAD_PARAM = 2,

    /**
     * @brief Null pointer received where non-null was required
     *
     * The API received a null pointer from the user where a valid
     * pointer was expected.
     */
    HIPDNN_STATUS_BAD_PARAM_NULL_POINTER = 3,

    /**
     * @brief Descriptor not finalized before use
     *
     * The backend descriptor must be finalized by calling
     * hipdnnBackendFinalize() before it can be used.
     */
    HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED = 4,

    /**
     * @brief Array index or count out of valid range
     *
     * An out-of-bound value was passed to the API, such as an
     * invalid array index or element count.
     */
    HIPDNN_STATUS_BAD_PARAM_OUT_OF_BOUND = 5,

    /**
     * @brief Buffer too small for requested operation
     *
     * The provided memory buffer has insufficient space for the
     * requested data. Query the required size first.
     */
    HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT = 6,

    /**
     * @brief Stream mismatch between handle and execution
     *
     * The HIP stream passed to the API does not match the stream
     * associated with the hipDNN handle.
     */
    HIPDNN_STATUS_BAD_PARAM_STREAM_MISMATCH = 7,

    /**
     * @brief Functionality not supported
     *
     * The requested functionality is not currently supported by hipDNN.
     * This may be due to unsupported data types, tensor layouts,
     * or operation configurations.
     */
    HIPDNN_STATUS_NOT_SUPPORTED = 8,

    /**
     * @brief Internal library error (category)
     *
     * An internal hipDNN operation failed. This typically indicates
     * a bug in the library.
     */
    HIPDNN_STATUS_INTERNAL_ERROR = 9,

    /**
     * @brief Memory allocation failed
     *
     * A memory allocation operation failed. This could be either
     * host or device memory allocation.
     */
    HIPDNN_STATUS_ALLOC_FAILED = 10,

    /**
     * @brief Host memory allocation failed
     *
     * An internal host (CPU) memory allocation failed inside
     * the hipDNN library.
     */
    HIPDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED = 11,

    /**
     * @brief Device memory allocation failed
     *
     * An internal device (GPU) memory allocation failed inside
     * the hipDNN library.
     */
    HIPDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED = 12,

    /**
     * @brief GPU kernel execution failed
     *
     * The GPU program failed to execute. This could indicate
     * invalid tensor data, incorrect dimensions, or a GPU error.
     */
    HIPDNN_STATUS_EXECUTION_FAILED = 13,

    /**
     * @brief Plugin error
     *
     * An error occurred in a loaded plugin. Check plugin-specific
     * error messages for details.
     */
    HIPDNN_STATUS_PLUGIN_ERROR = 14,
} hipdnnStatus_t;
