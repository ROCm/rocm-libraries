// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_HIPBLASLT_ERRORS_HPP
#define GUARD_MIOPEN_HIPBLASLT_ERRORS_HPP

#include <hipblaslt/hipblaslt.h>
#include <string>

namespace miopen {

// hipBLASLt exposes no status-to-string function; it returns the hipblasStatus_t enum shared with
// hipBLAS (hipblas-common.h), whose 0..11 values and names have been stable across releases.
// This mirrors that header for readability only. The numeric value is always printed too, so an
// unrecognized status still identifies itself.
inline const char* HipblasLtStatusString(hipblasStatus_t status)
{
    switch(status)
    {
    case HIPBLAS_STATUS_SUCCESS: return "HIPBLAS_STATUS_SUCCESS";
    case HIPBLAS_STATUS_NOT_INITIALIZED: return "HIPBLAS_STATUS_NOT_INITIALIZED";
    case HIPBLAS_STATUS_ALLOC_FAILED: return "HIPBLAS_STATUS_ALLOC_FAILED";
    case HIPBLAS_STATUS_INVALID_VALUE: return "HIPBLAS_STATUS_INVALID_VALUE";
    case HIPBLAS_STATUS_MAPPING_ERROR: return "HIPBLAS_STATUS_MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED: return "HIPBLAS_STATUS_EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR: return "HIPBLAS_STATUS_INTERNAL_ERROR";
    case HIPBLAS_STATUS_NOT_SUPPORTED: return "HIPBLAS_STATUS_NOT_SUPPORTED";
    case HIPBLAS_STATUS_ARCH_MISMATCH: return "HIPBLAS_STATUS_ARCH_MISMATCH";
    case HIPBLAS_STATUS_HANDLE_IS_NULLPTR: return "HIPBLAS_STATUS_HANDLE_IS_NULLPTR";
    case HIPBLAS_STATUS_INVALID_ENUM: return "HIPBLAS_STATUS_INVALID_ENUM";
    case HIPBLAS_STATUS_UNKNOWN: return "HIPBLAS_STATUS_UNKNOWN";
    }
    return "unrecognized hipBLASLt status";
}

inline std::string HipblasLtErrorMessage(hipblasStatus_t status, const char* operation)
{
    return std::string(operation) + " failed with " + HipblasLtStatusString(status) + " (" +
           std::to_string(static_cast<int>(status)) + ")";
}

} // namespace miopen

#endif
