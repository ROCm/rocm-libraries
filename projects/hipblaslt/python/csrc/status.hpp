// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>

#include <stdexcept>
#include <string>

namespace hipblaslt_py
{

    class HipblasLtError : public std::runtime_error
    {
    public:
        explicit HipblasLtError(const std::string& msg)
            : std::runtime_error(msg)
        {
        }
    };

    // By number, not by enumerator, so a PR that adds a status to hipblas-common.h can name it
    // here before the installed ROCm the bindings compile against has it. test_errors.py checks
    // every name and number against the checkout's header. It takes an int, so a value outside
    // the installed header's enumerators is never stored in a hipblasStatus_t.
    inline const char* status_name(int code)
    {
        struct Name
        {
            int         value;
            const char* name;
        };
        static constexpr Name names[] = {
            {0, "HIPBLAS_STATUS_SUCCESS"},
            {1, "HIPBLAS_STATUS_NOT_INITIALIZED"},
            {2, "HIPBLAS_STATUS_ALLOC_FAILED"},
            {3, "HIPBLAS_STATUS_INVALID_VALUE"},
            {4, "HIPBLAS_STATUS_MAPPING_ERROR"},
            {5, "HIPBLAS_STATUS_EXECUTION_FAILED"},
            {6, "HIPBLAS_STATUS_INTERNAL_ERROR"},
            {7, "HIPBLAS_STATUS_NOT_SUPPORTED"},
            {8, "HIPBLAS_STATUS_ARCH_MISMATCH"},
            {9, "HIPBLAS_STATUS_HANDLE_IS_NULLPTR"},
            {10, "HIPBLAS_STATUS_INVALID_ENUM"},
            {11, "HIPBLAS_STATUS_UNKNOWN"},
        };
        for(const Name& n : names)
            if(n.value == code)
                return n.name;
        return "HIPBLAS_STATUS_<unmapped>";
    }

    inline void check_status_code(int code, const char* call)
    {
        if(code != HIPBLAS_STATUS_SUCCESS)
            throw HipblasLtError(std::string(call) + " failed: " + status_name(code) + " ("
                                 + std::to_string(code) + ")");
    }

    inline void check_status(hipblasStatus_t s, const char* call)
    {
        check_status_code(static_cast<int>(s), call);
    }

    inline void check_hip(hipError_t e, const char* call)
    {
        if(e != hipSuccess)
            throw HipblasLtError(std::string(call) + " failed: " + hipGetErrorString(e));
    }

} // namespace hipblaslt_py

#define HIPBLASLT_CHECK(expr) ::hipblaslt_py::check_status((expr), #expr)
#define HIP_CHECK(expr) ::hipblaslt_py::check_hip((expr), #expr)
