// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file status_translation.h
 * @brief Shim-internal translation between cuDNN and hipDNN status codes.
 *
 * The cuDNN C-API stub entry points in `cudnn.h` use these helpers to translate
 * return codes to / from the hipDNN backend. The mapping is explicit and named,
 * never a numeric cast between the enum families.
 *
 * @note Self-contained: pulls in `cudnn_status.h` and `<hipdnn_backend.h>`
 *       directly rather than `cudnn.h`, to avoid an include cycle. Lives under
 *       `detail/` and is not part of the shim's public surface.
 */

#pragma once

#include <hipdnn_backend.h>

#include <hipdnn_compatibility/cudnn/cudnn_status.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail
{

/// @brief Map a hipDNN backend status to the closest cuDNN status code.
inline cudnnStatus_t toCudnnStatus(hipdnnStatus_t status)
{
    switch(status)
    {
    case HIPDNN_STATUS_SUCCESS:
        return CUDNN_STATUS_SUCCESS;
    case HIPDNN_STATUS_NOT_INITIALIZED:
        return CUDNN_STATUS_NOT_INITIALIZED;
    case HIPDNN_STATUS_BAD_PARAM:
    case HIPDNN_STATUS_BAD_PARAM_NULL_POINTER:
    case HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED:
    case HIPDNN_STATUS_BAD_PARAM_OUT_OF_BOUND:
    case HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT:
    case HIPDNN_STATUS_BAD_PARAM_STREAM_MISMATCH:
        return CUDNN_STATUS_BAD_PARAM;
    case HIPDNN_STATUS_NOT_SUPPORTED:
        return CUDNN_STATUS_NOT_SUPPORTED;
    case HIPDNN_STATUS_ALLOC_FAILED:
    case HIPDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED:
    case HIPDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED:
        // Allocation failures (generic, host, and device) all map to the cuDNN
        // allocation-failure code, which fits closer than INTERNAL_ERROR.
        return CUDNN_STATUS_ALLOC_FAILED;
    case HIPDNN_STATUS_EXECUTION_FAILED:
        return CUDNN_STATUS_EXECUTION_FAILED;
    case HIPDNN_STATUS_INTERNAL_ERROR:
    case HIPDNN_STATUS_PLUGIN_ERROR:
    default:
        return CUDNN_STATUS_INTERNAL_ERROR;
    }
}

/// @brief Map a cuDNN status code to the closest hipDNN backend status.
inline hipdnnStatus_t toHipdnnStatus(cudnnStatus_t status)
{
    switch(status)
    {
    case CUDNN_STATUS_SUCCESS:
        return HIPDNN_STATUS_SUCCESS;
    case CUDNN_STATUS_NOT_INITIALIZED:
        return HIPDNN_STATUS_NOT_INITIALIZED;
    case CUDNN_STATUS_ALLOC_FAILED:
        return HIPDNN_STATUS_ALLOC_FAILED;
    case CUDNN_STATUS_BAD_PARAM:
    case CUDNN_STATUS_INVALID_VALUE:
        return HIPDNN_STATUS_BAD_PARAM;
    case CUDNN_STATUS_NOT_SUPPORTED:
    case CUDNN_STATUS_ARCH_MISMATCH:
        return HIPDNN_STATUS_NOT_SUPPORTED;
    case CUDNN_STATUS_EXECUTION_FAILED:
        return HIPDNN_STATUS_EXECUTION_FAILED;
    case CUDNN_STATUS_INTERNAL_ERROR:
    case CUDNN_STATUS_MAPPING_ERROR:
    case CUDNN_STATUS_LICENSE_ERROR:
    case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
    case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
    case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
    case CUDNN_STATUS_VERSION_MISMATCH:
    default:
        return HIPDNN_STATUS_INTERNAL_ERROR;
    }
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail
