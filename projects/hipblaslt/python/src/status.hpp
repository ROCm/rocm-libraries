// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <stdexcept>
#include <string>

namespace hipblaslt_py {

class HipblasLtError : public std::runtime_error
{
public:
    explicit HipblasLtError(const std::string& msg) : std::runtime_error(msg) {}
};

inline const char* status_name(hipblasStatus_t s)
{
    switch(s)
    {
    case HIPBLAS_STATUS_SUCCESS:          return "HIPBLAS_STATUS_SUCCESS";
    case HIPBLAS_STATUS_NOT_INITIALIZED:  return "HIPBLAS_STATUS_NOT_INITIALIZED";
    case HIPBLAS_STATUS_ALLOC_FAILED:     return "HIPBLAS_STATUS_ALLOC_FAILED";
    case HIPBLAS_STATUS_INVALID_VALUE:    return "HIPBLAS_STATUS_INVALID_VALUE";
    case HIPBLAS_STATUS_MAPPING_ERROR:    return "HIPBLAS_STATUS_MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED: return "HIPBLAS_STATUS_EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR:   return "HIPBLAS_STATUS_INTERNAL_ERROR";
    case HIPBLAS_STATUS_NOT_SUPPORTED:    return "HIPBLAS_STATUS_NOT_SUPPORTED";
    case HIPBLAS_STATUS_ARCH_MISMATCH:    return "HIPBLAS_STATUS_ARCH_MISMATCH";
    case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:return "HIPBLAS_STATUS_HANDLE_IS_NULLPTR";
    case HIPBLAS_STATUS_INVALID_ENUM:     return "HIPBLAS_STATUS_INVALID_ENUM";
    case HIPBLAS_STATUS_UNKNOWN:          return "HIPBLAS_STATUS_UNKNOWN";
    default:                              return "HIPBLAS_STATUS_<unmapped>";
    }
}

inline void check_status(hipblasStatus_t s, const char* call)
{
    if(s != HIPBLAS_STATUS_SUCCESS)
        throw HipblasLtError(std::string(call) + " failed: " + status_name(s)
                             + " (" + std::to_string(static_cast<int>(s)) + ")");
}

inline void check_hip(hipError_t e, const char* call)
{
    if(e != hipSuccess)
        throw HipblasLtError(std::string(call) + " failed: " + hipGetErrorString(e));
}

} // namespace hipblaslt_py

#define HIPBLASLT_CHECK(expr) ::hipblaslt_py::check_status((expr), #expr)
#define HIP_CHECK(expr)       ::hipblaslt_py::check_hip((expr), #expr)
