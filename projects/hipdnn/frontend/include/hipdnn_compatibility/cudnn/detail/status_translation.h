// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Internal: explicit mapping between cuDNN and hipDNN status codes.

#pragma once

#include <hipdnn_backend.h>

#include <hipdnn_compatibility/cudnn/cudnn_status.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail
{

// Maps a hipDNN backend status to the closest cuDNN status.
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
        // ALLOC_FAILED fits closer than INTERNAL_ERROR for host and device failures.
        return CUDNN_STATUS_ALLOC_FAILED;
    case HIPDNN_STATUS_EXECUTION_FAILED:
        return CUDNN_STATUS_EXECUTION_FAILED;
    case HIPDNN_STATUS_VERSION_MISMATCH:
        return CUDNN_STATUS_VERSION_MISMATCH;
    case HIPDNN_STATUS_INTERNAL_ERROR:
    case HIPDNN_STATUS_PLUGIN_ERROR:
    default:
        return CUDNN_STATUS_INTERNAL_ERROR;
    }
}

// Maps a cuDNN status to the closest hipDNN backend status.
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
    case CUDNN_STATUS_VERSION_MISMATCH:
        return HIPDNN_STATUS_VERSION_MISMATCH;
    case CUDNN_STATUS_INTERNAL_ERROR:
    case CUDNN_STATUS_MAPPING_ERROR:
    case CUDNN_STATUS_LICENSE_ERROR:
    case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
    case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
    case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
    default:
        return HIPDNN_STATUS_INTERNAL_ERROR;
    }
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail
