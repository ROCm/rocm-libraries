// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN, used under the MIT license.

/**
 * @file cudnn_status.h
 * @brief cuDNN library status/return codes for the hipDNN compatibility shim.
 *
 * Split out of `cudnn.h` so that the status enum can be included without pulling
 * in the C entry points. `cudnn.h` includes this for its public `cudnnStatus_t`
 * type, and `detail/status_translation.h` includes it directly — letting the
 * translation header be genuinely self-contained instead of depending on
 * `cudnn.h` (which would otherwise create an include cycle).
 */

#pragma once

/// @brief cuDNN library status/return codes (mirrors NVIDIA `cudnnStatus_t`).
typedef enum
{
    CUDNN_STATUS_SUCCESS = 0,
    CUDNN_STATUS_NOT_INITIALIZED = 1,
    CUDNN_STATUS_ALLOC_FAILED = 2,
    CUDNN_STATUS_BAD_PARAM = 3,
    CUDNN_STATUS_INTERNAL_ERROR = 4,
    CUDNN_STATUS_INVALID_VALUE = 5,
    CUDNN_STATUS_ARCH_MISMATCH = 6,
    CUDNN_STATUS_MAPPING_ERROR = 7,
    CUDNN_STATUS_EXECUTION_FAILED = 8,
    CUDNN_STATUS_NOT_SUPPORTED = 9,
    CUDNN_STATUS_LICENSE_ERROR = 10,
    CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11,
    CUDNN_STATUS_RUNTIME_IN_PROGRESS = 12,
    CUDNN_STATUS_RUNTIME_FP_OVERFLOW = 13,
    CUDNN_STATUS_VERSION_MISMATCH = 14,
} cudnnStatus_t;
