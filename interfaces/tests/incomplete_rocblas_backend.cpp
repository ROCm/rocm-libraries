// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <rocblas/rocblas.h>

extern "C" const char* rocblas_status_to_string(rocblas_status) {
    return "incomplete backend";
}
