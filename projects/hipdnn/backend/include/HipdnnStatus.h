// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

typedef enum
{
    HIPDNN_STATUS_SUCCESS = 0, // The operation was completed successfully.
    HIPDNN_STATUS_NOT_INITIALIZED = 1, // Data not initialized.
    HIPDNN_STATUS_BAD_PARAM
    = 2, // This is an error category code. An incorrect value or parameter was passed to the function.
    HIPDNN_STATUS_BAD_PARAM_NULL_POINTER
    = 3, // The hipDNN API has unexpectedly received a null pointer from the user.
    HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED = 4, // The backend descriptor has not been finalized.
    HIPDNN_STATUS_BAD_PARAM_OUT_OF_BOUND = 5, // The hipDNN API has received an out-of-bound value.
    HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT
    = 6, // The hipDNN API has received a memory buffer with insufficient space.
    HIPDNN_STATUS_BAD_PARAM_STREAM_MISMATCH
    = 7, // The stream passed to the hipDNN API is not the same as the one used to create the handle.
    HIPDNN_STATUS_NOT_SUPPORTED
    = 8, // This is an error category code. The functionality requested is not currently supported by hipDNN.
    HIPDNN_STATUS_INTERNAL_ERROR
    = 9, // This is an error category code. An internal hipDNN operation failed.
    HIPDNN_STATUS_ALLOC_FAILED = 10,
    HIPDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED
    = 11, // An internal host memory allocation failed inside the hipDNN library.
    HIPDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED
    = 12, // Resource allocation failed inside the hipDNN library.
    HIPDNN_STATUS_EXECUTION_FAILED
    = 13, // This is an error category code. The GPU program failed to execute.
    HIPDNN_STATUS_PLUGIN_ERROR
    = 14, // This is an error category code. An error occurred in the plugin.
} hipdnnStatus_t;
